# LLM Kernel Hotspots

This note connects CUDA concepts to operations you will keep seeing in LLM
systems.

## GEMM

- Appears in QKV projections, feed-forward layers, and output projections
- Usually compute heavy
- Key idea: tile data so each load is reused many times

## Softmax

- Appears in attention score normalization
- Needs row max and row sum
- Usually limited by memory movement and reduction efficiency

## Layernorm

- Appears before or after major blocks, depending on architecture
- Needs per-row mean and variance
- Often bottlenecked by reading and writing large vectors

## Attention

- Combines matmul, masking, softmax, and another matmul
- Performance depends on tiling, SRAM/shared-memory reuse, and avoiding large
  intermediate tensors

## How to study them

For each hotspot, answer the same four questions:

1. What does each thread own?
2. What data is reused?
3. What reduction is needed?
4. What hardware resource is most stressed?

