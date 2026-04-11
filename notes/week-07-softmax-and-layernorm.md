# Week 7: Softmax and Layernorm

## Goal

Translate reduction-heavy math into kernels that look more like LLM building
blocks.

## Theory block

- Read [docs/llm/kernel-hotspots.md](../docs/llm/kernel-hotspots.md)
- Re-derive stable softmax and row-wise layernorm formulas

## Hands-on block

- Run or finish the row-wise softmax lab
- Run or finish the row-wise layernorm lab
- Test different row widths

## Deliverables

- Correct outputs against CPU reference checks
- Notes on why these kernels feel different from matmul

## Exit criteria

You can explain why stable softmax subtracts the row max before exponentiation.

