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

int main() {
    constexpr int m = 256;
    constexpr int n = 256;
    constexpr int k = 256;
    constexpr int tile = 16;

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

    matmul_tiled_kernel<tile><<<blocks, threads>>>(dev_a, dev_b, dev_c, m, n, k);
    check_cuda(cudaGetLastError(), "matmul_tiled launch");
    check_cuda(cudaDeviceSynchronize(), "matmul_tiled sync");
    check_cuda(cudaMemcpy(host_c.data(), dev_c, host_c.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy c");

    float max_error = 0.0f;
    const float expected = static_cast<float>(k) * 2.0f;
    for (float value : host_c) {
        max_error = std::max(max_error, std::fabs(value - expected));
    }

    std::printf("matmul_tiled max_error=%.6f expected=%.1f tile=%d\n", max_error, expected, tile);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    return max_error > 1e-4f ? 1 : 0;
}

