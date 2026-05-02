# GPU Learning Progress Tracker

Last updated: 2026-04-30

## Current status

- Week 1 environment setup: completed
- Week 2 math and execution model: completed at sprint level
- Week 3 first kernels and reduction: completed at sprint level
- Week 4-5 memory hierarchy and matmul: benchmarked
- Current focus: LLM-style kernels: softmax, layernorm, and attention patterns
- Active study mode: accelerated 6-day sprint
- Current sprint position: Day 5 started

## Completed checkpoints

- Read `docs/setup/wsl-ubuntu-cuda.md`
- Ran `scripts/verify_wsl_gpu.sh`
- Installed CUDA Toolkit in WSL and fixed `nvcc` availability
- Resolved CUDA runtime mismatch by switching from CUDA `13.2` to `12.9`
- Verified `nvcc --version` reports CUDA `12.9`
- Finished `docs/setup/pytorch-triton.md`
- Ran `labs/cuda/00_hello_cuda/hello.cu`
- Confirmed CUDA device is visible and a GPU kernel launches successfully
- Read the accelerated sprint plan and switched from the 12-week pace to a 6-day core-path sprint
- Ran `labs/cuda/01_vector_add/vector_add.cu`
- Verified `vector_add max_error=0.00000000`
- Ran `labs/cuda/02_saxpy/saxpy.cu`
- Verified `saxpy max_error=0.00000000`
- Ran `labs/cuda/03_reduction/reduction_sum.cu`
- Verified reduction output matches CPU reference
- Ran `labs/cuda/04_matmul_naive/matmul_naive.cu`
- Verified `matmul_naive max_error=0.000000 expected=256.0`
- Ran `labs/cuda/05_matmul_tiled/matmul_tiled.cu`
- Verified `matmul_tiled max_error=0.000000 expected=512.0 tile=16`
- Added CUDA event timing to naive and tiled matmul labs
- Captured `1024x1024x1024` matmul benchmark results in `benchmarks/matmul-naive-vs-tiled.csv`
- Measured `matmul_naive` at `2.2060 ms` and `matmul_tiled` at `1.3906 ms`
- Confirmed tiled matmul was about `1.59x` faster for this run
- Generated Nsight Systems reports for naive and tiled matmul
- Documented the profiling result in `benchmarks/nsight/matmul-profiling-note.md`
- Noted that Nsight Compute low-level counters were blocked by `ERR_NVGPUCTRPERM`
- Added CUDA event timing and shape arguments to row-wise softmax and layernorm labs
- Ran softmax and layernorm row-wise labs against CPU references
- Captured row-width benchmark results in `benchmarks/softmax-layernorm-rowwise.csv`
- Built a first-pass understanding of:
  - `<<<blocks, threads>>>`
  - `threadIdx`, `blockIdx`, `blockDim`
  - `index = blockIdx.x * blockDim.x + threadIdx.x`
  - why elementwise kernels are easier than reduction kernels

## Current environment snapshot

- Host OS: Windows
- WSL distro: Ubuntu on WSL2
- GPU visible in WSL: yes
- CUDA compiler: working
- Active CUDA version: `12.9`
- PyTorch/Triton setup doc: completed

## Next steps

- Continue Day 5:
  - explain stable softmax from memory
  - map attention to `matmul + softmax + matmul`
  - expand the hotspot map for `GEMM / softmax / layernorm / attention`
  - choose a capstone direction soon

## Sprint plan

- Active plan: `docs/accelerated-6-day-plan.md`
- Day 1 status: completed
- Day 2 status: completed
- Day 3 status: completed
- Day 4 status: completed at sprint level
- Day 4 target: benchmark/profiling evidence for naive vs tiled matmul
- Day 5 status: started
- Day 5 target: softmax, layernorm, and attention bottlenecks

## Weekly review seed

### Week 1

- What I finished:
  - WSL CUDA environment setup
  - toolkit/version debugging
  - CUDA hello world
- What clicked:
  - Windows driver and WSL toolkit are separate layers
  - toolkit version must not exceed driver capability
- What still feels fuzzy:
  - how CUDA launch dimensions map to real execution on the GPU

### Week 2 seed

- What I finished:
  - execution model basics
  - `vector_add`
  - indexing intuition
- What clicked:
  - global indexing in 1D kernels
  - why one-output-per-thread kernels are easy to parallelize
- What still feels fuzzy:
  - how warp scheduling relates to the code-level launch shape

### Week 3 seed

- What I finished:
  - `saxpy`
  - `reduction`
- What clicked:
  - reduction is harder because threads must cooperate to combine results
  - race conditions appear when multiple threads want to update shared results
- What still feels fuzzy:
  - the exact reduction pattern inside the kernel and how synchronization works step by step

### Week 4-5 seed

- What I finished:
  - `matmul_naive`
  - `matmul_tiled`
- What clicked:
  - correctness can be right even before performance is good
- What still feels fuzzy:
  - where the redundant global memory loads happen in naive matmul
  - exactly how the tile is staged through shared memory
