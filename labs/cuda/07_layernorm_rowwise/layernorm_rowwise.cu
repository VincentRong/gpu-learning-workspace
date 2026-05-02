#include <algorithm>
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

__global__ void layernorm_rowwise_kernel(const float* input, float* output, int rows, int cols, float eps) {
    extern __shared__ float shared[];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) {
        return;
    }

    const float* row_input = input + row * cols;
    float* row_output = output + row * cols;

    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        local_sum += row_input[col];
    }
    shared[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    const float mean = shared[0] / static_cast<float>(cols);

    float local_sq_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        const float centered = row_input[col] - mean;
        local_sq_sum += centered * centered;
    }
    shared[tid] = local_sq_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    const float variance = shared[0] / static_cast<float>(cols);
    const float inv_std = rsqrtf(variance + eps);

    for (int col = tid; col < cols; col += blockDim.x) {
        row_output[col] = (row_input[col] - mean) * inv_std;
    }
}

static void layernorm_cpu(const std::vector<float>& input, std::vector<float>& output, int rows, int cols, float eps) {
    for (int row = 0; row < rows; ++row) {
        const float* row_input = input.data() + row * cols;
        float* row_output = output.data() + row * cols;

        float mean = 0.0f;
        for (int col = 0; col < cols; ++col) {
            mean += row_input[col];
        }
        mean /= static_cast<float>(cols);

        float variance = 0.0f;
        for (int col = 0; col < cols; ++col) {
            const float centered = row_input[col] - mean;
            variance += centered * centered;
        }
        variance /= static_cast<float>(cols);
        const float inv_std = 1.0f / std::sqrt(variance + eps);

        for (int col = 0; col < cols; ++col) {
            row_output[col] = (row_input[col] - mean) * inv_std;
        }
    }
}

int main(int argc, char** argv) {
    const int rows = argc > 1 ? std::atoi(argv[1]) : 8;
    const int cols = argc > 2 ? std::atoi(argv[2]) : 256;
    constexpr int threads = 256;
    constexpr float eps = 1e-5f;
    constexpr int warmup_iterations = 5;
    constexpr int benchmark_iterations = 50;

    if (cols > 1024) {
        std::fprintf(stderr, "starter version expects cols <= 1024, got %d\n", cols);
        return 1;
    }

    std::vector<float> host_input(rows * cols);
    for (int i = 0; i < rows * cols; ++i) {
        host_input[i] = static_cast<float>((i % 29) - 13) * 0.1f;
    }

    std::vector<float> host_output(rows * cols, 0.0f);
    std::vector<float> cpu_output(rows * cols, 0.0f);

    float* dev_input = nullptr;
    float* dev_output = nullptr;
    check_cuda(cudaMalloc(&dev_input, host_input.size() * sizeof(float)), "cudaMalloc(dev_input)");
    check_cuda(cudaMalloc(&dev_output, host_output.size() * sizeof(float)), "cudaMalloc(dev_output)");
    check_cuda(cudaMemcpy(dev_input, host_input.data(), host_input.size() * sizeof(float), cudaMemcpyHostToDevice), "copy input");

    for (int i = 0; i < warmup_iterations; ++i) {
        layernorm_rowwise_kernel<<<rows, threads, threads * sizeof(float)>>>(dev_input, dev_output, rows, cols, eps);
    }
    check_cuda(cudaGetLastError(), "layernorm warmup launch");
    check_cuda(cudaDeviceSynchronize(), "layernorm warmup sync");

    cudaEvent_t start;
    cudaEvent_t stop;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    check_cuda(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int i = 0; i < benchmark_iterations; ++i) {
        layernorm_rowwise_kernel<<<rows, threads, threads * sizeof(float)>>>(dev_input, dev_output, rows, cols, eps);
    }
    check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
    check_cuda(cudaGetLastError(), "layernorm benchmark launch");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

    float elapsed_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
    const float avg_elapsed_ms = elapsed_ms / benchmark_iterations;

    check_cuda(cudaMemcpy(host_output.data(), dev_output, host_output.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy output");

    layernorm_cpu(host_input, cpu_output, rows, cols, eps);
    float max_error = 0.0f;
    for (int i = 0; i < rows * cols; ++i) {
        max_error = std::max(max_error, std::fabs(host_output[i] - cpu_output[i]));
    }

    std::printf(
        "layernorm_rowwise rows=%d cols=%d threads=%d avg_elapsed_ms=%.4f max_error=%.8f iterations=%d\n",
        rows,
        cols,
        threads,
        avg_elapsed_ms,
        max_error,
        benchmark_iterations);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dev_input);
    cudaFree(dev_output);
    return max_error > 1e-4f ? 1 : 0;
}
