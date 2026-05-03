#include <torch/extension.h>

torch::Tensor saxpy_cuda(torch::Tensor x, torch::Tensor y, double a);

torch::Tensor saxpy(torch::Tensor x, torch::Tensor y, double a) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
  TORCH_CHECK(y.scalar_type() == torch::kFloat32, "y must be float32");
  TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have the same shape");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(y.is_contiguous(), "y must be contiguous");

  return saxpy_cuda(x, y, a);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("saxpy", &saxpy, "SAXPY custom CUDA op");
}
