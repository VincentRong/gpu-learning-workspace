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

__global__ void softmax_rowwise_kernel(const float* input, float* output, int rows, int cols) {
    extern __shared__ float shared[];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) {
        return;
    }

    const float* row_input = input + row * cols;
    float* row_output = output + row * cols;

    float local_max = -INFINITY;
    for (int col = tid; col < cols; col += blockDim.x) {
        local_max = fmaxf(local_max, row_input[col]);
    }
    shared[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }
    const float row_max = shared[0];

    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        local_sum += expf(row_input[col] - row_max);
    }
    shared[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    const float row_sum = shared[0];

    for (int col = tid; col < cols; col += blockDim.x) {
        row_output[col] = expf(row_input[col] - row_max) / row_sum;
    }
}

static void softmax_cpu(const std::vector<float>& input, std::vector<float>& output, int rows, int cols) {
    for (int row = 0; row < rows; ++row) {
        const float* row_input = input.data() + row * cols;
        float* row_output = output.data() + row * cols;

        float max_value = row_input[0];
        for (int col = 1; col < cols; ++col) {
            max_value = std::max(max_value, row_input[col]);
        }

        float sum = 0.0f;
        for (int col = 0; col < cols; ++col) {
            row_output[col] = std::exp(row_input[col] - max_value);
            sum += row_output[col];
        }

        for (int col = 0; col < cols; ++col) {
            row_output[col] /= sum;
        }
    }
}

int main(int argc, char** argv) {
    const int rows = argc > 1 ? std::atoi(argv[1]) : 8;
    const int cols = argc > 2 ? std::atoi(argv[2]) : 256;
    constexpr int threads = 256;
    constexpr int warmup_iterations = 5;
    constexpr int benchmark_iterations = 50;

    if (cols > 1024) {
        std::fprintf(stderr, "starter version expects cols <= 1024, got %d\n", cols);
        return 1;
    }

    std::vector<float> host_input(rows * cols);
    for (int i = 0; i < rows * cols; ++i) {
        host_input[i] = static_cast<float>((i % 17) - 8) * 0.25f;
    }

    std::vector<float> host_output(rows * cols, 0.0f);
    std::vector<float> cpu_output(rows * cols, 0.0f);

    float* dev_input = nullptr;
    float* dev_output = nullptr;
    check_cuda(cudaMalloc(&dev_input, host_input.size() * sizeof(float)), "cudaMalloc(dev_input)");
    check_cuda(cudaMalloc(&dev_output, host_output.size() * sizeof(float)), "cudaMalloc(dev_output)");
    check_cuda(cudaMemcpy(dev_input, host_input.data(), host_input.size() * sizeof(float), cudaMemcpyHostToDevice), "copy input");

    for (int i = 0; i < warmup_iterations; ++i) {
        softmax_rowwise_kernel<<<rows, threads, threads * sizeof(float)>>>(dev_input, dev_output, rows, cols);
    }
    check_cuda(cudaGetLastError(), "softmax warmup launch");
    check_cuda(cudaDeviceSynchronize(), "softmax warmup sync");

    cudaEvent_t start;
    cudaEvent_t stop;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    check_cuda(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int i = 0; i < benchmark_iterations; ++i) {
        softmax_rowwise_kernel<<<rows, threads, threads * sizeof(float)>>>(dev_input, dev_output, rows, cols);
    }
    check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
    check_cuda(cudaGetLastError(), "softmax benchmark launch");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

    float elapsed_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
    const float avg_elapsed_ms = elapsed_ms / benchmark_iterations;

    check_cuda(cudaMemcpy(host_output.data(), dev_output, host_output.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy output");

    softmax_cpu(host_input, cpu_output, rows, cols);
    float max_error = 0.0f;
    for (int i = 0; i < rows * cols; ++i) {
        max_error = std::max(max_error, std::fabs(host_output[i] - cpu_output[i]));
    }

    std::printf(
        "softmax_rowwise rows=%d cols=%d threads=%d avg_elapsed_ms=%.4f max_error=%.8f iterations=%d\n",
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
