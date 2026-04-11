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

## Deliverables

- One page mapping `GEMM / softmax / layernorm / attention` to bottlenecks
- Capstone choice with a reason

## Exit criteria

You can explain why attention performance depends on both tiling and numerical
stability.

