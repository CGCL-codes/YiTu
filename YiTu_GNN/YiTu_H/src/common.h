#include <cublas_v2.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>


#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x, t) \
    CHECK_CUDA(x);        \
    CHECK_CONTIGUOUS(x);  \
    TORCH_CHECK(x.scalar_type() == t)

#define CUDA_CHECK(err)                                                        \
    do {                                                                       \
        cudaError_t err_ = (err);                                              \
        if (err_ != cudaSuccess) {                                             \
            std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__); \
            throw std::runtime_error("CUDA error");                            \
        }                                                                      \
    } while (0)

// cublas API error checking
#define CUBLAS_CHECK(err)                                                        \
    do {                                                                         \
        cublasStatus_t err_ = (err);                                             \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                     \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__); \
            throw std::runtime_error("cublas error");                            \
        }                                                                        \
    } while (0)
