# Week 4: Memory Hierarchy and Coalescing

## Goal

Understand why memory access patterns often dominate CUDA performance.

## Theory block

- Re-read [GPU mental model](../docs/foundations/gpu-mental-model.md)
- Write your own explanation of global vs shared vs register memory

## Hands-on block

- Revisit reduction and inspect global memory access patterns
- Start the naive matmul lab
- Record one example of coalesced vs scattered access

## Deliverables

- One benchmark note showing a memory-sensitive kernel
- One paragraph explaining coalescing in your own words

## Exit criteria

You can point to a line in a kernel and explain whether it causes friendly or
unfriendly memory access.

