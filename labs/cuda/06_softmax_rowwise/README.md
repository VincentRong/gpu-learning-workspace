# CUDA Lab 06: Row-wise Softmax

## Goal

Implement a numerically stable row-wise softmax using one block per row.

## Build

```bash
nvcc -O2 -std=c++17 softmax_rowwise.cu -o softmax_rowwise
./softmax_rowwise
```

## Constraints

- keep `cols <= 1024` for the starter version
- compare against a CPU reference
- test more than one row width

