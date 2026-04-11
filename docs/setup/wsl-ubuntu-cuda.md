# WSL Ubuntu CUDA Setup

This setup guide is tailored to the machine we inspected on `2026-04-11`.

## What is already true on this machine

- WSL is installed and working
- Default distro: Ubuntu
- Ubuntu version: `24.04.1 LTS`
- WSL mode: `2`
- GPU is visible inside Ubuntu through `nvidia-smi`
- `nvcc` is still missing

## Week 1 target

By the end of Week 1, these commands should work inside Ubuntu:

```bash
python3 --version
nvidia-smi
nvcc --version
```

You should also be able to run:

- a CUDA hello world program
- a minimal PyTorch CUDA tensor script

## Recommended workspace location

Use this Windows path for reading and planning, but do real builds in the Linux
filesystem:

```bash
mkdir -p ~/dev
cp -r "/mnt/c/Users/rongw/OneDrive/文档/LLM/GPU" ~/dev/gpu-learning
cd ~/dev/gpu-learning
```

Why: compiling CUDA from `/mnt/c/...` is usually slower and more fragile than
building from `~/dev/...`.

## Host-side checks

From Windows PowerShell:

```powershell
.\scripts\verify_windows_gpu.ps1
```

## Ubuntu-side baseline packages

Inside Ubuntu:

```bash
sudo apt update
sudo apt install -y build-essential gdb git cmake ninja-build pkg-config python3-pip python3-venv
```

## CUDA toolkit installation

Toolkit packaging changes over time, so use the official NVIDIA Linux install
guide for the current Ubuntu 24.04 instructions and repository commands:

- CUDA on WSL guide: `docs.nvidia.com/cuda/.../wsl-user-guide`
- CUDA Linux install guide: `docs.nvidia.com/cuda/.../linux-installation-guide`

After the toolkit install, verify:

```bash
nvcc --version
```

## Python environment

Inside Ubuntu:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install PyTorch and Triton using the official commands for your current CUDA
stack. See [docs/setup/pytorch-triton.md](pytorch-triton.md).

## Week 1 verification

Run:

```bash
bash scripts/verify_wsl_gpu.sh
```

If `nvcc` is still missing, do not move to Week 2 CUDA labs yet. Finish the
toolchain first.

