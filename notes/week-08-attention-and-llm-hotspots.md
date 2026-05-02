# Week 8: Attention and LLM Hotspots

## Goal

Stop seeing LLM kernels as magic and start seeing them as combinations of
patterns you already know.

## Theory block

- Re-read [docs/llm/kernel-hotspots.md](../docs/llm/kernel-hotspots.md)
- Map attention to matmul plus softmax plus another matmul

## Hands-on block

- Compare GEMM, softmax, and layernorm bottlenecks
- Pick your capstone track from [docs/capstone-options.md](../docs/capstone-options.md)

## Hotspot map draft

- `GEMM`: compute-heavy and reuse-heavy. The main optimization idea is tiling so
  global-memory loads feed many multiply-adds.
- `Softmax`: reduction-heavy and numerically sensitive. Each row needs a max
  reduction, exponentiation, a sum reduction, then normalization.
- `Layernorm`: reduction-heavy and memory-sensitive. Each row needs mean and
  variance, then a normalized write-back.
- `Attention`: a composition, not one pattern. It looks like
  `QK^T -> mask/scale -> softmax -> P V`, so it combines GEMM-style tiling with
  softmax-style numerical stability and reduction.

## Deliverables

- One page mapping `GEMM / softmax / layernorm / attention` to bottlenecks
- Capstone choice with a reason

## Exit criteria

You can explain why attention performance depends on both tiling and numerical
stability.
