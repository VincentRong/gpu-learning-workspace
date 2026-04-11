# Week 1: Environment and Workflow

## Goal

Finish the CUDA-capable WSL environment and establish a repeatable workflow.

## Theory block

- Read [docs/setup/wsl-ubuntu-cuda.md](../docs/setup/wsl-ubuntu-cuda.md)
- Understand the difference between Windows host, WSL guest, and GPU driver

## Hands-on block

- Run `.\scripts\verify_windows_gpu.ps1`
- Run `bash scripts/verify_wsl_gpu.sh` inside Ubuntu
- Install missing Ubuntu packages
- Install CUDA toolkit until `nvcc --version` works
- Run the CUDA hello world lab

## Deliverables

- `nvidia-smi` works on Windows and inside Ubuntu
- `nvcc --version` works in Ubuntu
- You can explain why real builds should happen under `~/dev/...`

## Exit criteria

Do not move on until the full toolchain is ready.

