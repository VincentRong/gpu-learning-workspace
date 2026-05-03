# GPU Learning Progress Tracker

Last updated: 2026-05-02

## Current status

- Week 1 environment setup: completed
- Week 2 math and execution model: completed at sprint level
- Week 3 first kernels and reduction: completed at sprint level
- Week 4-5 memory hierarchy and matmul: benchmarked
- Current focus: capstone polish and review
- Active study mode: accelerated 6-day sprint
- Current sprint position: sprint completed; capstone first pass completed

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
- Added `notes/day-05-llm-hotspot-map.md` mapping `GEMM / softmax / layernorm / attention`
- Ran `labs/framework/01_torch_custom_op/test_op.py`
- Verified PyTorch custom CUDA SAXPY op with `max_error=0.00000095`
- Added `notes/day-06-framework-bridge.md` explaining the Python -> C++ -> CUDA path
- Ran `labs/triton/01_vector_add/vector_add.py`
- Verified Triton vector add with `max_error=0.00000000`
- Ran `labs/triton/02_fused_softmax/fused_softmax.py`
- Verified Triton fused softmax with `max_error=0.00000001`
- Added `notes/day-06-triton-kernels.md` describing what Triton abstracts compared with raw CUDA
- Chose capstone direction: Triton fused softmax
- Added `labs/triton/02_fused_softmax/benchmark_softmax.py`
- Added `notes/capstone-triton-fused-softmax.md`
- Ran Triton fused softmax capstone benchmark across seven shapes
- Captured best Triton result so far: `4096x1024`, `0.0863 ms`, `1.21x` vs PyTorch
- Added `sweep_num_warps.py` to test Triton softmax with `num_warps` in `1/2/4/8`
- Ran `sweep_num_warps.py`
- Captured best sweep result: `4096x1024`, `num_warps=4`, `0.0856 ms`, `1.01x` vs PyTorch
- Reviewed the Triton fused softmax capstone summary and explained the main conclusions
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

- Start capstone:
  - optionally rerun clean benchmark once for final numbers

## Sprint plan

- Active plan: `docs/accelerated-6-day-plan.md`
- Day 1 status: completed
- Day 2 status: completed
- Day 3 status: completed
- Day 4 status: completed at sprint level
- Day 4 target: benchmark/profiling evidence for naive vs tiled matmul
- Day 5 status: completed at sprint level
- Day 5 target: softmax, layernorm, and attention bottlenecks
- Day 6 status: completed at sprint level
- Day 6 target: framework bridge, Triton kernels, and capstone choice
- Capstone status: first pass completed

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
