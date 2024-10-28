#include "ops.cuh"
#include <torch/extension.h>

template <typename index_t, typename value_t>
__global__ void hfused_forward_kernel(
    index_t n_nodes, index_t n_heads, index_t n_features,
    const value_t *spmm_values, const index_t *spmm_indptr,
    const index_t *spmm_indices, const value_t *spmm_features,
    const index_t *sddmm_indptr, const index_t *sddmm_indices,
    const value_t *sddmm_query, const value_t *sddmm_key,
    value_t *spmm_output, value_t *attn_values) {
    for (index_t row = blockIdx.x * blockDim.x + threadIdx.x;
         row < n_nodes; row += gridDim.x * blockDim.x) {
        for (index_t h = threadIdx.y; h < n_heads; h += blockDim.y) {
            // TODO: inner loop can be merged and improved
            value_t exp_sum[16] = {0.0};
            for (index_t i = sddmm_indptr[row]; i < sddmm_indptr[row + 1]; i += 1) {
                index_t bid = sddmm_indices[i] >> 24;
                index_t col = 0xffffff & sddmm_indices[i];
                value_t coeff = expf(
                    leaky_relu(sddmm_query[bid * n_nodes * n_heads +
                                           row * n_heads + h] +
                                   sddmm_key[col * n_heads + h],
                               0.2));
                attn_values[i * n_heads + h] = coeff;
                exp_sum[bid] += coeff;
            }
            for (index_t i = 0; i < 16; i += 1) {
                exp_sum[i] = 1.0 / exp_sum[i];
            }
            for (index_t i = sddmm_indptr[row]; i < sddmm_indptr[row + 1]; i += 1) {
                index_t bid = sddmm_indices[i] >> 24;
                attn_values[i * n_heads + h] *= exp_sum[bid];
            }
            for (index_t i = spmm_indptr[row]; i < spmm_indptr[row + 1]; i += 1) {
                index_t col = 0xffffff & spmm_indices[i];
                for (index_t k = 0; k < n_features; k += 1) {
                    value_t out = spmm_values[i * n_heads + h] *
                                  spmm_features[col * n_heads * n_features + h * n_features + k];
                    spmm_output[row * n_heads * n_features + h * n_features + k] += out;
                }
            }
        }
    }
}

template <typename index_t, typename value_t>
__global__ void hfused_backward_kernel(
    index_t n_nodes, index_t n_heads, index_t n_features,
    const value_t *spmm_values, const index_t *spmm_indptr,
    const index_t *spmm_indices, const value_t *spmm_features,
    const value_t *spmm_grad, const index_t *sddmm_indptr,
    const index_t *sddmm_indices, const value_t *sddmm_query,
    const value_t *sddmm_key, const value_t *sddmm_attn,
    const value_t *sddmm_grad, value_t *grad_a,
    value_t *grad_x, value_t *grad_q, value_t *grad_k) {
    for (index_t row = blockIdx.x * blockDim.x + threadIdx.x;
         row < n_nodes; row += gridDim.x * blockDim.x) {
        for (index_t h = threadIdx.x; h < n_heads; h += blockDim.x) {
            value_t grad_cache[16] = {0.0};
            for (index_t j = sddmm_indptr[row]; j < sddmm_indptr[row + 1]; j += 1) {
                index_t bid = sddmm_indices[j] >> 24;
                grad_cache[bid] -= sddmm_attn[j * n_heads + h] *
                                   sddmm_grad[j * n_heads + h];
            }
            for (index_t i = sddmm_indptr[row]; i < sddmm_indptr[row + 1]; i += 1) {
                index_t bid = sddmm_indices[i] >> 24;
                index_t col = 0xffffff & sddmm_indices[i];
                value_t grad_softmax = sddmm_attn[i * n_heads + h] *
                                       (grad_cache[bid] + sddmm_grad[i * n_heads + h]);
                value_t grad_val = grad_softmax *
                                   grad_leaky_relu(
                                       sddmm_query[bid * n_nodes * n_heads +
                                                   row * n_heads + h] +
                                           sddmm_key[col * n_heads + h],
                                       0.2);
                atomicAdd(&grad_q[bid * n_nodes * n_heads +
                                  row * n_heads + h],
                          grad_val);
                atomicAdd(&grad_k[col * n_heads + h], grad_val);
            }
            for (index_t i = spmm_indptr[row]; i < spmm_indptr[row + 1]; i += 1) {
                for (index_t k = 0; k < n_features; k += 1) {
                    index_t col = 0xffffff & spmm_indices[i];
                    value_t grad_row = spmm_grad[row * n_heads * n_features + h * n_features + k];
                    atomicAdd(&grad_a[i * n_heads + h],
                              grad_row * spmm_features[col * n_heads * n_features + h * n_features + k]);
                    atomicAdd(&grad_x[col * n_heads * n_features + h * n_features + k],
                              grad_row * spmm_values[i * n_heads + h]);
                }
            }
        }
    }
}

std::vector<torch::Tensor> _hfused_forward_cuda(const torch::Tensor &spmm_values,
                                                const torch::Tensor &spmm_indptr,
                                                const torch::Tensor &spmm_indices,
                                                const torch::Tensor &spmm_features,
                                                const torch::Tensor &sddmm_indptr,
                                                const torch::Tensor &sddmm_indices,
                                                const torch::Tensor &sddmm_query,
                                                const torch::Tensor &sddmm_key) {
    TORCH_CHECK(spmm_values.dim() == 2);
    TORCH_CHECK(spmm_indptr.dim() == 1);
    TORCH_CHECK(spmm_indices.dim() == 1);
    TORCH_CHECK(spmm_features.dim() == 3);
    TORCH_CHECK(sddmm_indptr.dim() == 1);
    TORCH_CHECK(sddmm_indices.dim() == 1);
    TORCH_CHECK(sddmm_query.dim() == 3);
    TORCH_CHECK(sddmm_key.dim() == 2);
    TORCH_CHECK(sddmm_query.size(-1) == sddmm_key.size(-1));
    TORCH_CHECK(sddmm_query.size(1) == sddmm_indptr.size(0) - 1);
    TORCH_CHECK(spmm_indptr.size(0) == sddmm_indptr.size(0));

    int32_t n_heads = sddmm_query.size(-1);
    int32_t n_edges = sddmm_indices.size(0);
    int32_t n_nodes = sddmm_indptr.size(0) - 1;
    int32_t n_features = spmm_features.size(2);
    auto spmm_output = torch::zeros({n_nodes, n_heads, n_features},
                                    spmm_features.options());
    auto attn_values = torch::zeros({n_edges, n_heads},
                                    sddmm_query.options());

    // sddmm: <n_nodes, min(32, n_heads)
    // spmm: dim3(n_heads, n_nodes), min(32, n_features)
    int32_t n_split = max(1, 32 / n_heads);
    hfused_forward_kernel<<<n_nodes / n_split, dim3(n_split, n_heads)>>>(
        n_nodes, n_heads, n_features,
        spmm_values.data_ptr<float>(), spmm_indptr.data_ptr<int32_t>(),
        spmm_indices.data_ptr<int32_t>(), spmm_features.data_ptr<float>(),
        sddmm_indptr.data_ptr<int32_t>(), sddmm_indices.data_ptr<int32_t>(),
        sddmm_query.data_ptr<float>(), sddmm_key.data_ptr<float>(),
        spmm_output.data_ptr<float>(), attn_values.data_ptr<float>());

    return {spmm_output, attn_values};
}

std::vector<torch::Tensor> _hfused_backward_cuda(const torch::Tensor &spmm_values,
                                                 const torch::Tensor &spmm_indptr,
                                                 const torch::Tensor &spmm_indices,
                                                 const torch::Tensor &spmm_features,
                                                 const torch::Tensor &spmm_grad,
                                                 const torch::Tensor &sddmm_indptr,
                                                 const torch::Tensor &sddmm_indices,
                                                 const torch::Tensor &sddmm_query,
                                                 const torch::Tensor &sddmm_key,
                                                 const torch::Tensor &sddmm_attn,
                                                 const torch::Tensor &sddmm_grad) {
    TORCH_CHECK(spmm_values.dim() == 2);
    TORCH_CHECK(spmm_indptr.dim() == 1);
    TORCH_CHECK(spmm_indices.dim() == 1);
    TORCH_CHECK(spmm_features.dim() == 3);
    TORCH_CHECK(spmm_grad.dim() == 3);
    TORCH_CHECK(sddmm_indptr.dim() == 1);
    TORCH_CHECK(sddmm_indices.dim() == 1);
    TORCH_CHECK(sddmm_query.dim() == 3);
    TORCH_CHECK(sddmm_key.dim() == 2);
    TORCH_CHECK(sddmm_attn.dim() == 2);
    TORCH_CHECK(sddmm_grad.dim() == 2);
    TORCH_CHECK(sddmm_query.size(-1) == sddmm_key.size(-1));
    TORCH_CHECK(sddmm_indices.size(0) == sddmm_attn.size(0));
    TORCH_CHECK(sddmm_attn.sizes() == sddmm_grad.sizes());
    TORCH_CHECK(spmm_indptr.size(0) == sddmm_indptr.size(0));

    int32_t n_heads = sddmm_query.size(-1);
    int32_t n_nodes = sddmm_indptr.size(0) - 1;
    int32_t n_features = spmm_grad.size(2);
    auto grad_a = torch::zeros_like(spmm_values);
    auto grad_x = torch::zeros_like(spmm_features);
    auto grad_query = torch::zeros_like(sddmm_query);
    auto grad_key = torch::zeros_like(sddmm_key);

    int32_t n_split = max(1, 32 / n_heads);
    hfused_backward_kernel<<<n_nodes / n_split, dim3(n_split, n_heads)>>>(
        n_nodes, n_heads, n_features,
        spmm_values.data_ptr<float>(), spmm_indptr.data_ptr<int32_t>(),
        spmm_indices.data_ptr<int32_t>(), spmm_features.data_ptr<float>(),
        spmm_grad.data_ptr<float>(), sddmm_indptr.data_ptr<int32_t>(),
        sddmm_indices.data_ptr<int32_t>(), sddmm_query.data_ptr<float>(),
        sddmm_key.data_ptr<float>(), sddmm_attn.data_ptr<float>(),
        sddmm_grad.data_ptr<float>(), grad_a.data_ptr<float>(),
        grad_x.data_ptr<float>(), grad_query.data_ptr<float>(),
        grad_key.data_ptr<float>());

    return {grad_query, grad_key};
}
