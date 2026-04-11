# CUDA Lab 01: Vector Add

## Goal

Map one element to one thread and validate the simplest useful kernel.

## Build

```bash
nvcc -O2 -std=c++17 vector_add.cu -o vector_add
./vector_add
```

## Follow-up questions

- Why is this kernel easy to parallelize?
- What happens if the block size changes from `256` to `128` or `512`?

