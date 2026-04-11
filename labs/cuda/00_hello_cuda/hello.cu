#include <cstdio>
#include <cuda_runtime.h>

inline void check_cuda(cudaError_t result, const char* call) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", call, cudaGetErrorString(result));
        std::exit(1);
    }
}

__global__ void hello_kernel() {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("Hello from the GPU. block=%d thread=%d\n", blockIdx.x, threadIdx.x);
    }
}

int main() {
    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    std::printf("Visible CUDA devices: %d\n", device_count);
    if (device_count == 0) {
        std::fprintf(stderr, "No CUDA devices found.\n");
        return 1;
    }

    hello_kernel<<<1, 32>>>();
    check_cuda(cudaGetLastError(), "kernel launch");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    return 0;
}

