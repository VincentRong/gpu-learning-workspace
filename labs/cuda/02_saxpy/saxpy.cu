#include <cmath>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

inline void check_cuda(cudaError_t result, const char* call) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", call, cudaGetErrorString(result));
        std::exit(1);
    }
}

__global__ void saxpy_kernel(float alpha, const float* x, float* y, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        y[index] = alpha * x[index] + y[index];
    }
}

int main() {
    constexpr int n = 1 << 20;
    constexpr int threads = 256;
    constexpr float alpha = 2.0f;
    const int blocks = (n + threads - 1) / threads;

    std::vector<float> host_x(n, 1.25f);
    std::vector<float> host_y(n, 0.75f);
    std::vector<float> cpu_y = host_y;

    float* dev_x = nullptr;
    float* dev_y = nullptr;

    check_cuda(cudaMalloc(&dev_x, n * sizeof(float)), "cudaMalloc(dev_x)");
    check_cuda(cudaMalloc(&dev_y, n * sizeof(float)), "cudaMalloc(dev_y)");

    check_cuda(cudaMemcpy(dev_x, host_x.data(), n * sizeof(float), cudaMemcpyHostToDevice), "copy x");
    check_cuda(cudaMemcpy(dev_y, host_y.data(), n * sizeof(float), cudaMemcpyHostToDevice), "copy y");

    saxpy_kernel<<<blocks, threads>>>(alpha, dev_x, dev_y, n);
    check_cuda(cudaGetLastError(), "saxpy launch");
    check_cuda(cudaDeviceSynchronize(), "saxpy sync");

    check_cuda(cudaMemcpy(host_y.data(), dev_y, n * sizeof(float), cudaMemcpyDeviceToHost), "copy y back");

    float max_error = 0.0f;
    for (int i = 0; i < n; ++i) {
        cpu_y[i] = alpha * host_x[i] + cpu_y[i];
        max_error = std::max(max_error, std::fabs(host_y[i] - cpu_y[i]));
    }

    std::printf("saxpy max_error=%.8f blocks=%d threads=%d\n", max_error, blocks, threads);

    cudaFree(dev_x);
    cudaFree(dev_y);
    return max_error > 1e-5f ? 1 : 0;
}

