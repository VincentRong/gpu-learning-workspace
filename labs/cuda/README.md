# CUDA Labs

Run these in order. Each lab builds on the previous one.

## Suggested compile pattern

```bash
nvcc -O2 -std=c++17 <file>.cu -o <binary>
./<binary>
```

## Progression

1. `00_hello_cuda`
2. `01_vector_add`
3. `02_saxpy`
4. `03_reduction`
5. `04_matmul_naive`
6. `05_matmul_tiled`
7. `06_softmax_rowwise`
8. `07_layernorm_rowwise`

