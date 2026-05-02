#include <cmath>
#include <cstdio>
#include <cstdlib>
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

int main(int argc, char** argv) {
    const int m = argc > 1 ? std::atoi(argv[1]) : 1024;
    const int n = argc > 2 ? std::atoi(argv[2]) : m;
    const int k = argc > 3 ? std::atoi(argv[3]) : m;
    constexpr int warmup_iterations = 5;
    constexpr int benchmark_iterations = 20;

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

    for (int i = 0; i < warmup_iterations; ++i) {
        matmul_naive_kernel<<<blocks, threads>>>(dev_a, dev_b, dev_c, m, n, k);
    }
    check_cuda(cudaGetLastError(), "matmul_naive warmup launch");
    check_cuda(cudaDeviceSynchronize(), "matmul_naive warmup sync");

    cudaEvent_t start;
    cudaEvent_t stop;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    check_cuda(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int i = 0; i < benchmark_iterations; ++i) {
        matmul_naive_kernel<<<blocks, threads>>>(dev_a, dev_b, dev_c, m, n, k);
    }
    check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
    check_cuda(cudaGetLastError(), "matmul_naive benchmark launch");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

    float elapsed_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
    const float avg_elapsed_ms = elapsed_ms / benchmark_iterations;

    check_cuda(cudaMemcpy(host_c.data(), dev_c, host_c.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy c");

    float max_error = 0.0f;
    const float expected = static_cast<float>(k) * 2.0f;
    for (float value : host_c) {
        max_error = std::max(max_error, std::fabs(value - expected));
    }

    std::printf(
        "matmul_naive shape=%dx%dx%d block=16x16 avg_elapsed_ms=%.4f max_error=%.6f expected=%.1f iterations=%d\n",
        m,
        n,
        k,
        avg_elapsed_ms,
        max_error,
        expected,
        benchmark_iterations);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    return max_error > 1e-4f ? 1 : 0;
}
