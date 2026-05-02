# Week 5: Matmul and Tiling

## Goal

Use matrix multiplication to build intuition for data reuse and shared-memory
tiling.

## Theory block

- Review the matmul formula
- Predict why naive matmul reloads too much data

## Hands-on block

- Run the naive matmul lab
- Run the tiled matmul lab
- Capture timings in `benchmarks/templates/benchmark-template.csv`

## Benchmark snapshot

Recorded in `benchmarks/matmul-naive-vs-tiled.csv`.

- Shape: `1024x1024x1024`
- Naive kernel: `2.2060 ms`
- Tiled kernel: `1.3906 ms`
- Speedup: about `1.59x`
- Correctness: both runs reported `max_error=0.000000`

The tiled kernel wins because each block stages a small tile of `A` and `B` in
shared memory. Threads in the block reuse those shared values for multiple
multiply-adds instead of repeatedly fetching the same inputs from global memory.
Global memory is large but slow; shared memory is much smaller, block-local, and
faster, so tiling turns repeated global loads into cheaper shared-memory reads.

## Deliverables

- Naive vs tiled benchmark results: done
- A note explaining why tiling helps: done

## Exit criteria

You can explain what each shared-memory tile is saving you from reloading.
