#include <torch/extension.h>

__global__ void saxpy_kernel(const float* x, const float* y, float* out,
                             float a, int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = a * x[idx] + y[idx];
  }
}

torch::Tensor saxpy_cuda(torch::Tensor x, torch::Tensor y, double a) {
  auto out = torch::empty_like(x);
  int64_t n = x.numel();

  constexpr int threads = 256;
  int blocks = static_cast<int>((n + threads - 1) / threads);

  saxpy_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(),
                                    out.data_ptr<float>(),
                                    static_cast<float>(a), n);

  return out;
}
