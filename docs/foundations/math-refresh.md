# Math Refresh For GPU Work

You do not need to re-learn all of linear algebra before starting CUDA. Focus on
the pieces that show up in GPU kernels and LLM math.

## Priority topics

1. Dot product
2. Matrix multiplication
3. Reduction patterns: `sum`, `max`, `mean`
4. Broadcasting intuition
5. Numerical stability for `exp`, `softmax`, and variance

## Minimal formulas worth remembering

### Dot product

```text
dot(x, y) = sum_i x_i * y_i
```

### Matrix multiplication

```text
C[m, n] = sum_k A[m, k] * B[k, n]
```

### Stable softmax

```text
softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
```

### Layernorm per row

```text
mean = sum(x_i) / N
var = sum((x_i - mean)^2) / N
y_i = (x_i - mean) / sqrt(var + eps)
```

## Why this matters for GPU programming

- `sum` and `max` usually become reductions
- reductions need synchronization or staged aggregation
- softmax combines `max`, `exp`, and `sum`, so it stresses memory traffic and
  numerical stability
- matmul is the canonical example for shared memory tiling and data reuse

## Good enough mastery test

Before Week 4, you should be able to explain:

- why matmul has a reduction dimension `K`
- why softmax subtracts the row max
- why layernorm needs at least two logical passes per row

