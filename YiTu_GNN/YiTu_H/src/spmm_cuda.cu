#include <torch/extension.h>

template <typename index_t, typename value_t>
__global__ void _spmm_forward_kernel(
    index_t n_nodes, index_t n_heads, index_t n_features,
    const value_t *values, const index_t *indptr, const index_t *indices,
    const value_t *features, value_t *output) {
    for (index_t row = blockIdx.y; row < n_nodes; row += gridDim.y) {
        for (index_t h = blockIdx.x; h < n_heads; h += gridDim.x) {
            for (index_t k = threadIdx.x; k < n_features; k += blockDim.x) {
                value_t out = 0.0;
                for (index_t i = indptr[row]; i < indptr[row + 1]; i += 1) {
                    index_t col = 0xffffff & indices[i];
                    out += values[i * n_heads + h] *
                           features[col * n_heads * n_features + h * n_features + k];
                }
                output[row * n_heads * n_features + h * n_features + k] = out;
            }
        }
    }
}

template <typename index_t, typename value_t>
__global__ void _spmm_backward_kernel(
    index_t n_nodes, index_t n_heads, index_t n_features,
    const value_t *values, const index_t *indptr, const index_t *indices,
    const value_t *features, const value_t *grad_out, value_t *grad_a, value_t *grad_x) {
    for (index_t row = blockIdx.y; row < n_nodes; row += gridDim.y) {
        for (index_t h = blockIdx.x; h < n_heads; h += gridDim.x) {
            for (index_t k = threadIdx.x; k < n_features; k += blockDim.x) {
                value_t grad_row = grad_out[row * n_heads * n_features + h * n_features + k];
                for (index_t i = indptr[row]; i < indptr[row + 1]; i += 1) {
                    index_t col = 0xffffff & indices[i];
                    atomicAdd(&grad_a[i * n_heads + h],
                              grad_row * features[col * n_heads * n_features + h * n_features + k]);
                    atomicAdd(&grad_x[col * n_heads * n_features + h * n_features + k],
                              grad_row * values[i * n_heads + h]);
                }
            }
        }
    }
}

torch::Tensor _spmm_forward_cuda(const torch::Tensor &values,
                                 const torch::Tensor &indptr,
                                 const torch::Tensor &indices,
                                 const torch::Tensor &features) {
    TORCH_CHECK(values.dim() == 2);
    TORCH_CHECK(indptr.dim() == 1);
    TORCH_CHECK(indices.dim() == 1);
    TORCH_CHECK(features.dim() == 3);
    int32_t n_nodes = indptr.size(0) - 1;
    int32_t n_heads = features.size(1);
    int32_t n_features = features.size(2);
    auto output = torch::zeros({n_nodes, n_heads, n_features}, features.options());

    _spmm_forward_kernel<int32_t, float><<<dim3(n_heads, n_nodes), min(32, n_features)>>>(
        n_nodes, n_heads, n_features, values.data_ptr<float>(), indptr.data_ptr<int32_t>(),
        indices.data_ptr<int32_t>(), features.data_ptr<float>(), output.data_ptr<float>());

    return output;
}

std::vector<torch::Tensor> _spmm_backward_cuda(const torch::Tensor &values,
                                               const torch::Tensor &indptr,
                                               const torch::Tensor &indices,
                                               const torch::Tensor &features,
                                               const torch::Tensor &grad_out) {
    TORCH_CHECK(values.dim() == 2);
    TORCH_CHECK(indptr.dim() == 1);
    TORCH_CHECK(indices.dim() == 1);
    TORCH_CHECK(features.dim() == 3);
    TORCH_CHECK(grad_out.dim() == 3);
    int32_t n_nodes = indptr.size(0) - 1;
    int32_t n_heads = grad_out.size(1);
    int32_t n_features = grad_out.size(2);
    auto grad_a = torch::zeros_like(values);
    auto grad_x = torch::zeros_like(features);

    _spmm_backward_kernel<int32_t, float><<<dim3(n_heads, n_nodes), min(32, n_features)>>>(
        n_nodes, n_heads, n_features, values.data_ptr<float>(), indptr.data_ptr<int32_t>(),
        indices.data_ptr<int32_t>(), features.data_ptr<float>(), grad_out.data_ptr<float>(),
        grad_a.data_ptr<float>(), grad_x.data_ptr<float>());

    return {grad_a, grad_x};
}
