# CUDA Lab 07: Row-wise LayerNorm

## Goal

Implement row-wise layer normalization and compare against a CPU reference.

## Build

```bash
nvcc -O2 -std=c++17 layernorm_rowwise.cu -o layernorm_rowwise
./layernorm_rowwise
```

The default run uses `8x256`. You can pass a custom shape as `rows cols`:

```bash
./layernorm_rowwise 4096 1024
```

The starter kernel keeps one block per row, 256 threads per block, and expects
`cols <= 1024`.

## Constraints

- starter version assumes one block per row
- use a small epsilon such as `1e-5`
- inspect how row width changes runtime
