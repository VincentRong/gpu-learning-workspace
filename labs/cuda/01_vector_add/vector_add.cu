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

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    constexpr int n = 1 << 20;
    constexpr int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    std::vector<float> host_a(n, 1.5f);
    std::vector<float> host_b(n, 2.5f);
    std::vector<float> host_c(n, 0.0f);

    float* dev_a = nullptr;
    float* dev_b = nullptr;
    float* dev_c = nullptr;

    check_cuda(cudaMalloc(&dev_a, n * sizeof(float)), "cudaMalloc(dev_a)");
    check_cuda(cudaMalloc(&dev_b, n * sizeof(float)), "cudaMalloc(dev_b)");
    check_cuda(cudaMalloc(&dev_c, n * sizeof(float)), "cudaMalloc(dev_c)");

    check_cuda(cudaMemcpy(dev_a, host_a.data(), n * sizeof(float), cudaMemcpyHostToDevice), "copy a");
    check_cuda(cudaMemcpy(dev_b, host_b.data(), n * sizeof(float), cudaMemcpyHostToDevice), "copy b");

    vector_add_kernel<<<blocks, threads>>>(dev_a, dev_b, dev_c, n);
    check_cuda(cudaGetLastError(), "vector_add_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "vector_add sync");

    check_cuda(cudaMemcpy(host_c.data(), dev_c, n * sizeof(float), cudaMemcpyDeviceToHost), "copy c");

    float max_error = 0.0f;
    for (int i = 0; i < n; ++i) {
        max_error = std::max(max_error, std::fabs(host_c[i] - (host_a[i] + host_b[i])));
    }

    std::printf("vector_add max_error=%.8f blocks=%d threads=%d\n", max_error, blocks, threads);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    return max_error > 1e-5f ? 1 : 0;
}

