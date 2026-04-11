# CUDA Lab 07: Row-wise LayerNorm

## Goal

Implement row-wise layer normalization and compare against a CPU reference.

## Build

```bash
nvcc -O2 -std=c++17 layernorm_rowwise.cu -o layernorm_rowwise
./layernorm_rowwise
```

## Constraints

- starter version assumes one block per row
- use a small epsilon such as `1e-5`
- inspect how row width changes runtime

