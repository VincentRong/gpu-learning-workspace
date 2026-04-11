# CUDA Lab 00: Hello CUDA

## Goal

Verify that `nvcc` works and that a kernel can launch successfully.

## Build

```bash
nvcc -O2 -std=c++17 hello.cu -o hello
./hello
```

## What to look for

- device count is non-zero
- the kernel prints from the GPU
- `cudaDeviceSynchronize()` succeeds

