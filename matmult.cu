#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <torch/types.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

// TODO: is there a better way to get the handle?
cublasHandle_t get_handle() {
	return at::cuda::getCurrentCUDABlasHandle();
}

// C = alpha AB + beta C
void matmult(torch::Tensor A, torch::Tensor B, torch::Tensor C, 
		bool transpose_A, bool transpose_B, float alpha, float beta) {

	cublasOperation_t op_A = CUBLAS_OP_N;
	cublasOperation_t op_B = CUBLAS_OP_N;
    int m = A.size(0); int k = B.size(0); int n = B.size(1);
	if (transpose_A) {
		op_A = CUBLAS_OP_T;
		m = A.size(1);
	}
	if (transpose_B) {
		op_B = CUBLAS_OP_T;
		k = B.size(1);
		n = B.size(0);
	}

	// Depending on the tensor precision, call cuBLAS with appropriate parameters.
	// Small but important detail: notice how we use CUBLAS_COMPUTE_32F for fp16.
	//  This is for the numerical stability of vector dot-products (another reason why
	//  it's called *mixed* precision.
    if (A.dtype() == torch::kFloat32) {
        cublasGemmEx(get_handle(), op_B, op_A, n, m, k, &alpha,
                     B.data_ptr<float>(), CUDA_R_32F, B.size(1), 
                     A.data_ptr<float>(), CUDA_R_32F, A.size(1), 
                     &beta, C.data_ptr<float>(), CUDA_R_32F, C.size(1),
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    } else if (A.dtype() == torch::kFloat16) {
        cublasGemmEx(get_handle(), op_B, op_A, n, m, k, &alpha,
                     B.data_ptr<at::Half>(), CUDA_R_16F, B.size(1), 
                     A.data_ptr<at::Half>(), CUDA_R_16F, A.size(1), 
                     &beta, C.data_ptr<at::Half>(), CUDA_R_16F, C.size(1),
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
}
