#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>
#include <cuda_runtime.h>

inline void check_cuda(cudaError_t result, const char* call) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", call, cudaGetErrorString(result));
        std::exit(1);
    }
}

__global__ void reduce_sum_kernel(const float* input, float* partial_sums, int n) {
    extern __shared__ float shared[];
    unsigned int tid = threadIdx.x;
    unsigned int global = blockIdx.x * blockDim.x + threadIdx.x;

    shared[tid] = global < static_cast<unsigned int>(n) ? input[global] : 0.0f;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = shared[0];
    }
}

int main() {
    constexpr int n = 1 << 20;
    constexpr int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    std::vector<float> host_input(n, 1.0f);
    std::vector<float> host_partial(blocks, 0.0f);

    float* dev_input = nullptr;
    float* dev_partial = nullptr;

    check_cuda(cudaMalloc(&dev_input, n * sizeof(float)), "cudaMalloc(dev_input)");
    check_cuda(cudaMalloc(&dev_partial, blocks * sizeof(float)), "cudaMalloc(dev_partial)");
    check_cuda(cudaMemcpy(dev_input, host_input.data(), n * sizeof(float), cudaMemcpyHostToDevice), "copy input");

    reduce_sum_kernel<<<blocks, threads, threads * sizeof(float)>>>(dev_input, dev_partial, n);
    check_cuda(cudaGetLastError(), "reduce_sum launch");
    check_cuda(cudaDeviceSynchronize(), "reduce_sum sync");
    check_cuda(cudaMemcpy(host_partial.data(), dev_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost), "copy partial");

    const float gpu_sum = std::accumulate(host_partial.begin(), host_partial.end(), 0.0f);
    const float cpu_sum = std::accumulate(host_input.begin(), host_input.end(), 0.0f);
    const float diff = std::fabs(gpu_sum - cpu_sum);

    std::printf("reduction gpu_sum=%.1f cpu_sum=%.1f diff=%.6f\n", gpu_sum, cpu_sum, diff);

    cudaFree(dev_input);
    cudaFree(dev_partial);
    return diff > 1e-3f ? 1 : 0;
}

