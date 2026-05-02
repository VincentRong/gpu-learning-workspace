# CUDA Lab 05: Tiled Matmul

## Goal

Use shared memory to reuse matrix tiles and compare against the naive version.

## Build

```bash
nvcc -O2 -std=c++17 matmul_tiled.cu -o matmul_tiled
./matmul_tiled
```

The default run benchmarks a `1024x1024x1024` multiply using CUDA events. You
can pass a custom shape as `m n k`:

```bash
./matmul_tiled 512 512 512
```

## Study prompt

Compare this kernel with `matmul_naive.cu` and explain where the reuse comes
from.
