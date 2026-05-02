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

template <int TileSize>
__global__ void matmul_tiled_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
    __shared__ float tile_a[TileSize][TileSize];
    __shared__ float tile_b[TileSize][TileSize];

    int row = blockIdx.y * TileSize + threadIdx.y;
    int col = blockIdx.x * TileSize + threadIdx.x;
    float sum = 0.0f;

    for (int tile = 0; tile < (k + TileSize - 1) / TileSize; ++tile) {
        int tiled_col = tile * TileSize + threadIdx.x;
        int tiled_row = tile * TileSize + threadIdx.y;

        tile_a[threadIdx.y][threadIdx.x] = (row < m && tiled_col < k)
            ? a[row * k + tiled_col]
            : 0.0f;
        tile_b[threadIdx.y][threadIdx.x] = (tiled_row < k && col < n)
            ? b[tiled_row * n + col]
            : 0.0f;

        __syncthreads();

        for (int i = 0; i < TileSize; ++i) {
            sum += tile_a[threadIdx.y][i] * tile_b[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = sum;
    }
}

int main(int argc, char** argv) {
    const int m = argc > 1 ? std::atoi(argv[1]) : 1024;
    const int n = argc > 2 ? std::atoi(argv[2]) : m;
    const int k = argc > 3 ? std::atoi(argv[3]) : m;
    constexpr int tile = 16;
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

    dim3 threads(tile, tile);
    dim3 blocks((n + tile - 1) / tile, (m + tile - 1) / tile);

    for (int i = 0; i < warmup_iterations; ++i) {
        matmul_tiled_kernel<tile><<<blocks, threads>>>(dev_a, dev_b, dev_c, m, n, k);
    }
    check_cuda(cudaGetLastError(), "matmul_tiled warmup launch");
    check_cuda(cudaDeviceSynchronize(), "matmul_tiled warmup sync");

    cudaEvent_t start;
    cudaEvent_t stop;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    check_cuda(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int i = 0; i < benchmark_iterations; ++i) {
        matmul_tiled_kernel<tile><<<blocks, threads>>>(dev_a, dev_b, dev_c, m, n, k);
    }
    check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
    check_cuda(cudaGetLastError(), "matmul_tiled benchmark launch");
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
        "matmul_tiled shape=%dx%dx%d tile=%d avg_elapsed_ms=%.4f max_error=%.6f expected=%.1f iterations=%d\n",
        m,
        n,
        k,
        tile,
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
