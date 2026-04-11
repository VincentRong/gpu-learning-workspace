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

__global__ void matmul_naive_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

int main() {
    constexpr int m = 128;
    constexpr int n = 128;
    constexpr int k = 128;

    std::vector<float> host_a(m * k, 1.0f);
    std::vector<float> host_b(k * n, 2.0f);
    std::vector<float> host_c(m * n, 0.0f);

    float* dev_a = nullptr;
    float* dev_b = nullptr;
    float* dev_c = nullptr;

    check_cuda(cudaMalloc(&dev_a, host_a.size() * sizeof(float)), "cudaMalloc(dev_a)");
    check_cuda(cudaMalloc(&dev_b, host_b.size() * sizeof(float)), "cudaMalloc(dev_b)");
    check_cuda(cudaMalloc(&dev_c, host_c.size() * sizeof(float)), "cudaMalloc(dev_c)");

    check_cuda(cudaMemcpy(dev_a, host_a.data(), host_a.size() * sizeof(float), cudaMemcpyHostToDevice), "copy a");
    check_cuda(cudaMemcpy(dev_b, host_b.data(), host_b.size() * sizeof(float), cudaMemcpyHostToDevice), "copy b");

    dim3 threads(16, 16);
    dim3 blocks((n + threads.x - 1) / threads.x, (m + threads.y - 1) / threads.y);

    matmul_naive_kernel<<<blocks, threads>>>(dev_a, dev_b, dev_c, m, n, k);
    check_cuda(cudaGetLastError(), "matmul_naive launch");
    check_cuda(cudaDeviceSynchronize(), "matmul_naive sync");
    check_cuda(cudaMemcpy(host_c.data(), dev_c, host_c.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy c");

    float max_error = 0.0f;
    const float expected = static_cast<float>(k) * 2.0f;
    for (float value : host_c) {
        max_error = std::max(max_error, std::fabs(value - expected));
    }

    std::printf("matmul_naive max_error=%.6f expected=%.1f\n", max_error, expected);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    return max_error > 1e-4f ? 1 : 0;
}

