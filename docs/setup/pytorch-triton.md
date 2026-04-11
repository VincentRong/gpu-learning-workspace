# PyTorch and Triton Setup

Use this after your CUDA toolkit and Python virtual environment are ready inside
Ubuntu WSL.

## PyTorch

Use the current official install selector on `pytorch.org/get-started/locally`
and choose:

- Linux
- Pip
- Python
- CUDA 12.x

After installation, verify:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
PY
```

## Triton

Triton evolves quickly, so keep the install simple:

```bash
python -m pip install triton
```

Verify:

```bash
python - <<'PY'
import triton
print(triton.__version__)
PY
```

## When to install what

- Week 1: PyTorch is enough if you only want a minimal CUDA tensor smoke test
- Weeks 2-8: focus on raw CUDA labs first
- Weeks 9-10: install Triton and run the two Triton labs in this repo

## Suggested package order

1. Create and activate `.venv`
2. Install PyTorch
3. Verify `torch.cuda.is_available()`
4. Install Triton
5. Run a very small Triton example before trying fused softmax

