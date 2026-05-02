# CUDA Lab 06: Row-wise Softmax

## Goal

Implement a numerically stable row-wise softmax using one block per row.

## Build

```bash
nvcc -O2 -std=c++17 softmax_rowwise.cu -o softmax_rowwise
./softmax_rowwise
```

The default run uses `8x256`. You can pass a custom shape as `rows cols`:

```bash
./softmax_rowwise 4096 1024
```

The starter kernel keeps one block per row, 256 threads per block, and expects
`cols <= 1024`.

## Constraints

- keep `cols <= 1024` for the starter version
- compare against a CPU reference
- test more than one row width
