# Accelerated 6-Day GPU Sprint Plan

This is a compressed version of the 12-week plan for building strong practical
GPU intuition fast. It does not replace long-term repetition, but it does
optimize for reaching a solid project-capable beginner level in under a week.

## Target outcome

By the end of Day 6, you should be able to:

- explain `thread`, `block`, `grid`, `warp`, shared memory, and coalescing
- write and validate basic CUDA kernels
- explain why reduction is harder than elementwise kernels
- compare naive vs tiled matmul and describe why tiling helps
- map `GEMM`, `softmax`, `layernorm`, and `attention` to bottlenecks
- run at least one Triton kernel and compare it with raw CUDA
- discuss GPU kernel work in a way that sounds grounded, not superficial

## What this sprint is and is not

- This sprint is enough to become a strong learner and credible hands-on
  beginner.
- This sprint is not enough by itself to become a true mid-level GPU engineer.
- The goal is fast compression of the highest-value concepts, labs, and mental
  models.

## Priority rule

If time is tight, protect this chain:

`environment -> indexing -> reduction -> memory -> matmul -> profiling -> LLM hotspots -> Triton`

## Day 1: Environment and execution model

Primary sources:

- `notes/week-01-environment-and-workflow.md`
- `notes/week-02-gpu-math-and-execution-model.md`
- `docs/foundations/math-refresh.md`
- `docs/foundations/gpu-mental-model.md`
- `labs/cuda/00_hello_cuda`
- `labs/cuda/01_vector_add`

Must finish:

- verify CUDA, `nvcc`, and Python environment
- run `hello.cu`
- run `vector_add`
- write short notes explaining:
  - `<<<blocks, threads>>>`
  - `threadIdx`, `blockIdx`, `blockDim`
  - why one element per thread is easy to parallelize

End-of-day test:

- You can derive `index = blockIdx.x * blockDim.x + threadIdx.x` without looking
  it up.

## Day 2: First kernels and reduction

Primary sources:

- `notes/week-03-first-kernels-and-reduction.md`
- `labs/cuda/02_saxpy`
- `labs/cuda/03_reduction`

Must finish:

- run `saxpy`
- run `reduction`
- compare elementwise kernels vs reduction kernels
- write down what race conditions are and why reduction needs coordination

End-of-day test:

- You can explain why reduction is not just "vector add but shorter".

## Day 3: Memory hierarchy and matmul

Primary sources:

- `notes/week-04-memory-hierarchy-and-coalescing.md`
- `notes/week-05-matmul-and-tiling.md`
- `labs/cuda/04_matmul_naive`
- `labs/cuda/05_matmul_tiled`

Must finish:

- run naive matmul
- run tiled matmul
- explain `global memory`, `shared memory`, and `registers`
- record one example of coalesced vs unfriendly memory access

End-of-day test:

- You can explain what shared-memory tiling saves you from reloading.

## Day 4: Profiling and optimization evidence

Primary sources:

- `notes/week-06-profiling-and-kernel-optimization.md`
- `benchmarks/README.md`
- `benchmarks/nsight/README.md`

Must finish:

- capture timings for naive vs tiled matmul
- inspect at least one profiler output, report, or screenshot
- write one paragraph explaining an optimization with evidence

End-of-day test:

- You can point to one metric and explain why the optimized kernel wins.

## Day 5: LLM-style kernels

Primary sources:

- `notes/week-07-softmax-and-layernorm.md`
- `notes/week-08-attention-and-llm-hotspots.md`
- `docs/llm/kernel-hotspots.md`
- `labs/cuda/06_softmax_rowwise`
- `labs/cuda/07_layernorm_rowwise`

Must finish:

- run softmax lab
- run layernorm lab
- explain stable softmax
- map attention to `matmul + softmax + matmul`
- write one page on the bottlenecks of `GEMM / softmax / layernorm / attention`

End-of-day test:

- You can explain why attention is not one kernel pattern but a composition of
  several patterns.

## Day 6: Framework bridge and consolidation

Primary sources:

- `notes/week-09-pytorch-custom-cuda-op.md`
- `notes/week-10-triton-kernels.md`
- `labs/framework/01_torch_custom_op`
- `labs/triton/01_vector_add`
- `labs/triton/02_fused_softmax`

Must finish:

- trace Python -> C++ -> CUDA at a high level
- run Triton vector add
- run Triton fused softmax
- write down what Triton abstracts compared with raw CUDA
- choose one mini capstone direction

End-of-day test:

- You can explain when you would choose raw CUDA vs Triton.

## Deprioritized during the sprint

These are still useful, but they should not steal time from the core chain:

- polishing benchmark write-ups
- broad reading outside the linked notes
- trying to master every PyTorch build detail
- chasing advanced optimization tricks before basic correctness and mapping are stable

## Daily deliverables

Each day, write down:

- one thing you finished
- one thing that clicked
- one thing still fuzzy
- one concrete code or benchmark artifact

Record those in `notes/progress-tracker.md`.

## Honest expectation

If you finish this sprint seriously, you will likely be:

- clearly beyond "I just started learning CUDA"
- able to discuss GPU kernel basics with confidence
- able to run and modify foundational labs
- near the boundary between strong beginner and early project-capable engineer

You will likely not yet be:

- a true mid-level GPU performance engineer
- fluent in production-grade kernel optimization across many workloads
- ready to design advanced fused kernels from scratch without more repetition
