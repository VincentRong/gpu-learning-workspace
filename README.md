# GPU Programming Learning Workspace

This workspace turns the 12-week `CUDA + LLM` study plan into something you can
use immediately: weekly notes, setup guides, starter labs, benchmark templates,
and validation scripts.

## Current machine snapshot

As of `2026-04-11`, this plan is tailored to the machine we inspected:

- Host OS: Windows
- GPU: NVIDIA GeForce RTX 3070 8GB
- Host Python: 3.12.5
- WSL: Ubuntu 24.04.1 LTS on WSL2
- WSL Python: 3.12.3
- GPU visible inside WSL: yes (`nvidia-smi` works)
- CUDA compiler in WSL: not installed yet (`nvcc` missing)

## How to use this repo

1. Read [docs/setup/wsl-ubuntu-cuda.md](docs/setup/wsl-ubuntu-cuda.md).
2. Run `scripts/verify_windows_gpu.ps1` on Windows.
3. Run `scripts/verify_wsl_gpu.sh` inside Ubuntu WSL.
4. Start with [notes/week-01-environment-and-workflow.md](notes/week-01-environment-and-workflow.md).
5. Each week, finish the note, run the linked lab, and record benchmark results.

## Workspace layout

- `curriculum/plan.json`: machine-tailored 12-week curriculum metadata
- `notes/`: weekly study guides and a review template
- `docs/`: setup, theory refreshers, LLM kernel mapping, and capstone guidance
- `labs/cuda/`: CUDA C++ labs from hello world to softmax/layernorm
- `labs/triton/`: Triton labs for framework-facing kernel work
- `labs/framework/`: PyTorch custom op learning path
- `benchmarks/`: benchmark templates and Nsight profiling notes
- `scripts/`: environment verification scripts
- `tests/`: lightweight checks for the workspace structure

## Execution model

Use the same weekly cadence every week:

- `2h` theory: read the linked docs and take notes
- `3-4h` hands-on: run the labs, then tweak parameters or kernel structure
- `1h` review: write down what you learned and what still feels fuzzy

## Practical advice

- Do your real CUDA and Triton builds inside the Linux filesystem, not the
  mounted Windows path. Once Week 1 passes, copy or clone this workspace to a
  path like `~/dev/gpu-learning` inside WSL for faster builds and fewer path
  surprises.
- Keep a `benchmarks/results.csv` file from Week 5 onward. Your goal is not
  just "it works", but "I can explain why this version is faster".
- When a concept feels abstract, relate it to an LLM hotspot:
  `GEMM -> compute bound`, `softmax -> reduction + bandwidth`,
  `layernorm -> reduction + memory traffic`, `attention -> tiling + reuse`.

## Suggested weekly ritual

- Monday or Tuesday: read the note and linked docs
- Mid-week: run the lab and capture one screenshot or benchmark
- Weekend: fill the review template and choose next week's stretch goal

