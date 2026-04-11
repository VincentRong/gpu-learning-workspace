# CUDA Lab 04: Naive Matmul

## Goal

Implement one thread per output element and feel the pain of repeated global
memory loads.

## Build

```bash
nvcc -O2 -std=c++17 matmul_naive.cu -o matmul_naive
./matmul_naive
```

## Study prompt

After it works, explain why this version reloads the same `A` and `B` values
many times.

