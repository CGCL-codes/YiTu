#include "common.h"

std::vector<torch::Tensor> _b2gemm_cuda(const torch::Tensor &x,
                                        const torch::Tensor &w_a,
                                        const torch::Tensor &w_b);

std::vector<torch::Tensor> _b2gemm_backward_cuda(const torch::Tensor &x,
                                                 const torch::Tensor &w_a,
                                                 const torch::Tensor &w_b,
                                                 const torch::Tensor &grad_a,
                                                 const torch::Tensor &grad_b);

torch::Tensor _spmm_forward_cuda(const torch::Tensor &values,
                                 const torch::Tensor &indptr,
                                 const torch::Tensor &indices,
                                 const torch::Tensor &features);

std::vector<torch::Tensor> _spmm_backward_cuda(const torch::Tensor &values,
                                               const torch::Tensor &indptr,
                                               const torch::Tensor &indices,
                                               const torch::Tensor &features,
                                               const torch::Tensor &grad_out);

torch::Tensor _sddmm_forward_cuda(const torch::Tensor &indptr,
                                  const torch::Tensor &indices,
                                  const torch::Tensor &query,
                                  const torch::Tensor &key);

std::vector<torch::Tensor> _sddmm_backward_cuda(const torch::Tensor &indptr,
                                                const torch::Tensor &indices,
                                                const torch::Tensor &query,
                                                const torch::Tensor &key,
                                                const torch::Tensor &attn_values,
                                                const torch::Tensor &grad_out);

std::vector<torch::Tensor> _hfused_forward_cuda(const torch::Tensor &spmm_values,
                                                const torch::Tensor &spmm_indptr,
                                                const torch::Tensor &spmm_indices,
                                                const torch::Tensor &spmm_feature,
                                                const torch::Tensor &sddmm_indptr,
                                                const torch::Tensor &sddmm_indices,
                                                const torch::Tensor &sddmm_query,
                                                const torch::Tensor &sddmm_key);

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
                                                 const torch::Tensor &sddmm_grad);
auto b2gemm(const torch::Tensor &x,
            const torch::Tensor &w_a,
            const torch::Tensor &w_b) {
    CHECK_INPUT(x, torch::kFloat32);
    CHECK_INPUT(w_a, torch::kFloat32);
    CHECK_INPUT(w_b, torch::kFloat32);

    return _b2gemm_cuda(x, w_a, w_b);
}

auto b2gemm_backward(const torch::Tensor &x,
                     const torch::Tensor &w_a,
                     const torch::Tensor &w_b,
                     const torch::Tensor &grad_a,
                     const torch::Tensor &grad_b) {
    CHECK_INPUT(x, torch::kFloat32);
    CHECK_INPUT(w_a, torch::kFloat32);
    CHECK_INPUT(w_b, torch::kFloat32);
    CHECK_INPUT(grad_a, torch::kFloat32);
    CHECK_INPUT(grad_b, torch::kFloat32);

    return _b2gemm_backward_cuda(x, w_a, w_b, grad_a, grad_b);
}

auto spmm_forward(const torch::Tensor &values,
                  const torch::Tensor &indptr,
                  const torch::Tensor &indices,
                  const torch::Tensor &features) {
    CHECK_INPUT(values, torch::kFloat32);
    CHECK_INPUT(indptr, torch::kInt32);
    CHECK_INPUT(indices, torch::kInt32);
    CHECK_INPUT(features, torch::kFloat32);

    return _spmm_forward_cuda(values, indptr, indices, features);
}

auto spmm_backward(const torch::Tensor &values,
                   const torch::Tensor &indptr,
                   const torch::Tensor &indices,
                   const torch::Tensor &features,
                   const torch::Tensor &grad_out) {
    CHECK_INPUT(values, torch::kFloat32);
    CHECK_INPUT(indptr, torch::kInt32);
    CHECK_INPUT(indices, torch::kInt32);
    CHECK_INPUT(features, torch::kFloat32);
    CHECK_INPUT(grad_out, torch::kFloat32);

    return _spmm_backward_cuda(values, indptr, indices, features, grad_out);
}

auto sddmm_forward(const torch::Tensor &indptr,
                   const torch::Tensor &indices,
                   const torch::Tensor &query,
                   const torch::Tensor &key) {
    CHECK_INPUT(indptr, torch::kInt32);
    CHECK_INPUT(indices, torch::kInt32);
    CHECK_INPUT(query, torch::kFloat32);
    CHECK_INPUT(key, torch::kFloat32);

    return _sddmm_forward_cuda(indptr, indices, query, key);
}

auto sddmm_backward(const torch::Tensor &indptr,
                    const torch::Tensor &indices,
                    const torch::Tensor &query,
                    const torch::Tensor &key,
                    const torch::Tensor &attn_values,
                    const torch::Tensor &grad_out) {
    CHECK_INPUT(indptr, torch::kInt32);
    CHECK_INPUT(indices, torch::kInt32);
    CHECK_INPUT(query, torch::kFloat32);
    CHECK_INPUT(key, torch::kFloat32);
    CHECK_INPUT(attn_values, torch::kFloat32);
    CHECK_INPUT(grad_out, torch::kFloat32);

    return _sddmm_backward_cuda(indptr, indices,
                                query, key,
                                attn_values, grad_out);
}

auto hfused_forward(const torch::Tensor &spmm_values,
                    const torch::Tensor &spmm_indptr,
                    const torch::Tensor &spmm_indices,
                    const torch::Tensor &spmm_features,
                    const torch::Tensor &sddmm_indptr,
                    const torch::Tensor &sddmm_indices,
                    const torch::Tensor &sddmm_query,
                    const torch::Tensor &sddmm_key) {
    CHECK_INPUT(spmm_values, torch::kFloat32);
    CHECK_INPUT(spmm_indptr, torch::kInt32);
    CHECK_INPUT(spmm_indices, torch::kInt32);
    CHECK_INPUT(spmm_features, torch::kFloat32);
    CHECK_INPUT(sddmm_indptr, torch::kInt32);
    CHECK_INPUT(sddmm_indices, torch::kInt32);
    CHECK_INPUT(sddmm_query, torch::kFloat32);
    CHECK_INPUT(sddmm_key, torch::kFloat32);

    return _hfused_forward_cuda(spmm_values, spmm_indptr, spmm_indices, spmm_features,
                                sddmm_indptr, sddmm_indices, sddmm_query, sddmm_key);
}

auto hfused_backward(const torch::Tensor &spmm_values,
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
    CHECK_INPUT(spmm_values, torch::kFloat32);
    CHECK_INPUT(spmm_indptr, torch::kInt32);
    CHECK_INPUT(spmm_indices, torch::kInt32);
    CHECK_INPUT(spmm_features, torch::kFloat32);
    CHECK_INPUT(spmm_grad, torch::kFloat32);
    CHECK_INPUT(sddmm_indptr, torch::kInt32);
    CHECK_INPUT(sddmm_indices, torch::kInt32);
    CHECK_INPUT(sddmm_query, torch::kFloat32);
    CHECK_INPUT(sddmm_key, torch::kFloat32);
    CHECK_INPUT(sddmm_attn, torch::kFloat32);
    CHECK_INPUT(sddmm_grad, torch::kFloat32);

    return _hfused_backward_cuda(spmm_values, spmm_indptr, spmm_indices, spmm_features, spmm_grad,
                                 sddmm_indptr, sddmm_indices, sddmm_query, sddmm_key, sddmm_attn, sddmm_grad);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("b2gemm", &b2gemm, "B2GEMM forward (CUDA)");
    m.def("b2gemm_backward", &b2gemm_backward, "backward (CUDA)");
    m.def("spmm_forward", &spmm_forward, "SPMM forward (CUDA)");
    m.def("spmm_backward", &spmm_backward, "SPMM backward (CUDA)");
    m.def("sddmm_forward", &sddmm_forward, "SDDMM forward (CUDA)");
    m.def("sddmm_backward", &sddmm_backward, "SDDMM backward (CUDA)");
    m.def("hfused_forward", &hfused_forward, "HFUSED forward (CUDA)");
    m.def("hfused_backward", &hfused_backward, "HFUSED backward (CUDA)");
}
