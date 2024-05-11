#include <torch/extension.h>

void matmult(torch::Tensor A, torch::Tensor B, torch::Tensor C, bool transpose_A, bool transpose_B, float alpha, float beta);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmult", torch::wrap_pybind_function(matmult), "matmult");
}