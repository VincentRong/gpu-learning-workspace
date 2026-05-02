# Week 6: Profiling and Kernel Optimization

## Goal

Learn to use evidence, not intuition, when talking about performance.

## Theory block

- Read [benchmarks/nsight/README.md](../benchmarks/nsight/README.md)
- Learn the difference between occupancy, throughput, and achieved bandwidth

## Hands-on block

- Profile naive matmul
- Profile tiled matmul
- Compare one reduction variant against another

## Matmul profiling snapshot

Recorded in `benchmarks/nsight/matmul-profiling-note.md`.

- Nsight Compute was blocked by `ERR_NVGPUCTRPERM`, so occupancy and throughput
  counters were not available in this environment.
- Nsight Systems reports were generated for naive and tiled matmul.
- Under Nsight Systems profiling, the program's CUDA-event averages were:
  - naive: `1.7989 ms`
  - tiled: `1.3332 ms`
- The Nsight CUDA API summary showed `cudaEventSynchronize` waiting
  `35.891877 ms` for naive and `26.556983 ms` for tiled across the 20 measured
  launches.

This supports the same conclusion as the standalone benchmark: tiled matmul
finishes faster because shared-memory tiles let threads reuse values that the
naive kernel repeatedly fetches from global memory.

## Deliverables

- One Nsight report or screenshot: done
- A short note on where the optimized version wins: done

## Exit criteria

You can explain at least one profiler metric you used to justify an optimization.
