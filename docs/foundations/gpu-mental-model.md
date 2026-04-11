# GPU Mental Model

Think of GPU programming as throughput programming, not low-latency programming.

## Core objects

- thread: one execution lane for one small slice of work
- block: a group of threads that can cooperate through shared memory
- grid: the full launch shape
- warp: the hardware scheduling unit, typically 32 threads on NVIDIA GPUs

## Memory hierarchy

- global memory: large, slowest, visible to all threads
- shared memory: block-local, much faster, manually managed
- registers: thread-local, fastest, most limited

## Three questions to ask for every kernel

1. What is the mapping from data element to thread?
2. Where is the bottleneck: compute, memory bandwidth, or synchronization?
3. Can the data be reused from shared memory instead of rereading global memory?

## Typical beginner mistakes

- launch too few threads and underuse the GPU
- ignore memory coalescing
- use one thread for too much sequential work
- skip error checks after kernel launches
- optimize before validating correctness

## LLM mapping

- matmul: high arithmetic intensity, often compute bound
- softmax: reduction heavy, often bandwidth bound
- layernorm: mean/variance reductions plus normalization writeback
- attention: mixes tiled matmul, masking, softmax, and memory reuse

