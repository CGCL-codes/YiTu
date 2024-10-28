#include "ops.cuh"
#include <torch/extension.h>

template <typename index_t, typename value_t>
__global__ void sddmm_forward_kernel(
    index_t n_nodes, index_t n_heads, const index_t *indptr,
    const index_t *indices, const value_t *query, const value_t *key,
    value_t *attn_values) {
    for (index_t row = blockIdx.x; row < n_nodes; row += gridDim.x) {
        for (index_t h = threadIdx.x; h < n_heads; h += blockDim.x) {
            value_t exp_sum[16] = {0.0};
            for (index_t i = indptr[row]; i < indptr[row + 1]; i += 1) {
                index_t bid = indices[i] >> 24;
                index_t col = 0xffffff & indices[i];
                value_t coeff = expf(
                    leaky_relu(query[bid * n_nodes * n_heads +
                                     row * n_heads + h] +
                                   key[col * n_heads + h],
                               0.2));
                attn_values[i * n_heads + h] = coeff;
                exp_sum[bid] += coeff;
            }
            //
            for (index_t i = 0; i < 16; i += 1) {
                exp_sum[i] = 1.0 / exp_sum[i];
            }
            for (index_t i = indptr[row]; i < indptr[row + 1]; i += 1) {
                index_t bid = indices[i] >> 24;
                attn_values[i * n_heads + h] *= exp_sum[bid];
            }
        }
    }
}

template <typename index_t, typename value_t>
__global__ void sddmm_backward_kernel(
    index_t n_nodes, index_t n_heads, const index_t *indptr,
    const index_t *indices, const value_t *query, const value_t *key,
    const value_t *attn_values, const value_t *grad_out,
    value_t *grad_q, value_t *grad_k) {
    for (index_t row = blockIdx.x; row < n_nodes; row += gridDim.x) {
        for (index_t h = threadIdx.x; h < n_heads; h += blockDim.x) {
            value_t grad_cache[16] = {0.0};
            for (index_t j = indptr[row]; j < indptr[row + 1]; j += 1) {
                index_t bid = indices[j] >> 24;
                grad_cache[bid] -= attn_values[j * n_heads + h] *
                                   grad_out[j * n_heads + h];
            }
            for (index_t i = indptr[row]; i < indptr[row + 1]; i += 1) {
                index_t bid = indices[i] >> 24;
                index_t col = 0xffffff & indices[i];
                value_t grad_softmax = attn_values[i * n_heads + h] *
                                       (grad_cache[bid] + grad_out[i * n_heads + h]);
                value_t grad_val = grad_softmax *
                                   grad_leaky_relu(
                                       query[bid * n_nodes * n_heads +
                                             row * n_heads + h] +
                                           key[col * n_heads + h],
                                       0.2);
                atomicAdd(&grad_q[bid * n_nodes * n_heads +
                                  row * n_heads + h],
                          grad_val);
                atomicAdd(&grad_k[col * n_heads + h], grad_val);
            }
        }
    }
}

torch::Tensor _sddmm_forward_cuda(const torch::Tensor &indptr,
                                  const torch::Tensor &indices,
                                  const torch::Tensor &query,
                                  const torch::Tensor &key) {
    TORCH_CHECK(indptr.dim() == 1);
    TORCH_CHECK(indices.dim() == 1);
    TORCH_CHECK(query.dim() == 3);
    TORCH_CHECK(key.dim() == 2);
    TORCH_CHECK(query.size(-1) == key.size(-1));
    TORCH_CHECK(query.size(1) == indptr.size(0) - 1);
    int32_t n_heads = query.size(-1);
    int32_t n_edges = indices.size(0);
    int32_t n_nodes = indptr.size(0) - 1;
    auto attn_values = torch::zeros({n_edges, n_heads}, query.options());

    sddmm_forward_kernel<<<n_nodes, min(32, n_heads)>>>(
        n_nodes, n_heads, indptr.data_ptr<int32_t>(), indices.data_ptr<int32_t>(),
        query.data_ptr<float>(), key.data_ptr<float>(), attn_values.data_ptr<float>());

    return attn_values;
}

std::vector<torch::Tensor> _sddmm_backward_cuda(const torch::Tensor &indptr,
                                                const torch::Tensor &indices,
                                                const torch::Tensor &query,
                                                const torch::Tensor &key,
                                                const torch::Tensor &attn_values,
                                                const torch::Tensor &grad_out) {
    TORCH_CHECK(indptr.dim() == 1);
    TORCH_CHECK(indices.dim() == 1);
    TORCH_CHECK(query.dim() == 3);
    TORCH_CHECK(key.dim() == 2);
    TORCH_CHECK(attn_values.dim() == 2);
    TORCH_CHECK(grad_out.dim() == 2);
    TORCH_CHECK(query.size(-1) == key.size(-1));
    TORCH_CHECK(indices.size(0) == attn_values.size(0));
    TORCH_CHECK(attn_values.sizes() == grad_out.sizes());

    int32_t n_heads = query.size(-1);
    int32_t n_nodes = indptr.size(0) - 1;
    auto grad_query = torch::zeros_like(query);
    auto grad_key = torch::zeros_like(key);

    sddmm_backward_kernel<<<n_nodes, min(32, n_heads)>>>(
        n_nodes, n_heads, indptr.data_ptr<int32_t>(), indices.data_ptr<int32_t>(),
        query.data_ptr<float>(), key.data_ptr<float>(), attn_values.data_ptr<float>(),
        grad_out.data_ptr<float>(), grad_query.data_ptr<float>(), grad_key.data_ptr<float>());

    return {grad_query, grad_key};
}
