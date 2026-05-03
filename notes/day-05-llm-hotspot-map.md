# Day 5: LLM Hotspot Map

Date: 2026-05-02

## Core mental model

LLM kernels are not magic shapes. They are repeated combinations of a few GPU
patterns:

- `GEMM`: lots of multiply-add work, high data reuse, benefits from tiling.
- `Softmax`: per-row reductions plus numerical stability.
- `Layernorm`: per-row reductions plus memory traffic.
- `Attention`: a pipeline that combines GEMM-style tiling with softmax-style
  reductions.

## Hotspot map

| Operation | Thread ownership | Reused data | Reduction needed | Stressed resource |
| --- | --- | --- | --- | --- |
| `GEMM` | Usually one output element or tile fragment | Tiles of `A` and `B` | Dot-product accumulation | Compute throughput and shared-memory reuse |
| `Softmax` | One row per block in the starter lab | Row values reused across max, sum, and normalize passes | Row max and row sum | Memory bandwidth, synchronization, `exp` cost |
| `Layernorm` | One row per block in the starter lab | Row values reused for mean, variance, and normalize | Row sum and squared-difference sum | Memory bandwidth and reduction efficiency |
| `Attention` | Blocks of query rows and key/value columns | Blocks of `Q`, `K`, `V`, and partial scores | Softmax max and sum over scores | Shared-memory reuse, memory traffic, numerical stability |

## Attention decomposition

Scaled dot-product attention is:

`softmax(QK^T / sqrt(d_k)) V`

Read it as three familiar pieces:

1. `QK^T`: a GEMM-like score matrix. Each score is a dot product between one
   query vector and one key vector.
2. `softmax`: a row-wise normalization over the scores for each query. Stable
   softmax subtracts the row max before exponentiation.
3. `P V`: another GEMM-like weighted sum. Each output vector is a mix of value
   vectors weighted by the softmax probabilities.

The performance trap is the intermediate score/probability matrix. If sequence
length is large, materializing all of `QK^T` and all of `P` creates a lot of
memory traffic. FlashAttention-style kernels improve this by tiling attention
and keeping partial state close to the compute instead of writing every
intermediate to global memory.

## Stable softmax from memory

For a row `x`, compute:

`m = max(x)`

`softmax(x_i) = exp(x_i - m) / sum_j exp(x_j - m)`

Subtracting `m` does not change the answer because every numerator is scaled by
the same factor. It does prevent large positive values from overflowing `exp`.

## Capstone leaning

Best next capstone candidate: `Triton fused softmax`.

Reason: it connects directly to LLM attention, builds on the completed CUDA
softmax/layernorm work, and prepares for practical framework-facing kernel
work. CUDA tiled matmul is still the cleanest pure CUDA capstone, but Triton
fused softmax gives a stronger bridge into real LLM optimization workflows.

## Checkpoint answer

Attention is not one kernel pattern. It starts with GEMM-like dot products,
then needs softmax reductions and numerical stability, then returns to a
GEMM-like weighted sum. Good implementations care about both tiling/reuse and
the reduction math.
