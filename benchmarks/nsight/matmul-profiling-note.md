# Matmul Profiling Note

Date: 2026-04-29

## Commands

```bash
nsys profile --trace=cuda,nvtx --stats=true --force-overwrite=true \
  -o /home/rwenxiao/dev-learning/GPU/benchmarks/nsight/matmul_naive_nsys \
  ./matmul_naive

nsys profile --trace=cuda,nvtx --stats=true --force-overwrite=true \
  -o /home/rwenxiao/dev-learning/GPU/benchmarks/nsight/matmul_tiled_nsys \
  ./matmul_tiled
```

Generated reports:

- `benchmarks/nsight/matmul_naive_nsys.nsys-rep`
- `benchmarks/nsight/matmul_tiled_nsys.nsys-rep`

## Profiler Output

Nsight Compute was attempted first with `ncu --set basic`, but the run was
blocked by `ERR_NVGPUCTRPERM`, so low-level performance counters such as
occupancy, SM throughput, and memory throughput were not available in this
environment.

Nsight Systems generated reports and CUDA API summaries. The CUDA kernel summary
tables were unavailable in the exported stats, but the CUDA API summary still
shows the host waiting for GPU work through `cudaEventSynchronize`.

| Kernel | Shape | Program CUDA-event avg | Nsight `cudaEventSynchronize` total |
| --- | --- | ---: | ---: |
| `matmul_naive` | `1024x1024x1024` | `1.7989 ms` | `35.891877 ms` |
| `matmul_tiled` | `1024x1024x1024` | `1.3332 ms` | `26.556983 ms` |

The `cudaEventSynchronize` call waits for the 20 measured benchmark launches to
finish. Dividing the profiler's wait time by 20 gives roughly `1.7946 ms` for
naive and `1.3278 ms` for tiled, which matches the program's own CUDA-event
averages closely.

## Interpretation

The tiled kernel wins on this run because it reduces repeated global memory
loads. In the naive kernel, each output element independently walks across one
row of `A` and one column of `B`, so neighboring threads reload many of the same
values from global memory. In the tiled kernel, each block first copies a
`16x16` tile of `A` and `B` into shared memory, then reuses those values across
the block while computing partial dot products.

The profiler evidence lines up with the benchmark: the tiled run spends less
time waiting for the measured GPU work to complete, and the direct CUDA-event
timing shows about a `1.35x` speedup under Nsight Systems profiling.
