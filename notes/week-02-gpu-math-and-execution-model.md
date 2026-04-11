# Week 2: GPU Math and Execution Model

## Goal

Rebuild just enough math and execution-model intuition to make the next kernels
feel concrete.

## Theory block

- Read [docs/foundations/math-refresh.md](../docs/foundations/math-refresh.md)
- Read [docs/foundations/gpu-mental-model.md](../docs/foundations/gpu-mental-model.md)

## Hands-on block

- Run the CUDA hello world lab
- Sketch how `vector_add` will map indices to threads
- Write down how `threadIdx`, `blockIdx`, and `blockDim` interact

## Deliverables

- One page of notes on thread/block/grid
- A short explanation of why warp divergence hurts throughput

## Exit criteria

You can explain `thread`, `block`, `grid`, and `warp` without looking them up.

