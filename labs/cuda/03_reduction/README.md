# CUDA Lab 03: Block Reduction

## Goal

Write a reduction kernel that uses shared memory and produces block-level
partial sums.

## Build

```bash
nvcc -O2 -std=c++17 reduction_sum.cu -o reduction_sum
./reduction_sum
```

## Focus points

- shared memory as a staging area
- synchronization with `__syncthreads()`
- why reductions need a different mindset than elementwise kernels

