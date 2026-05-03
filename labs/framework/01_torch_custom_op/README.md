# Framework Lab 01: PyTorch Custom CUDA Op

## Goal

Understand the moving parts involved in exposing a CUDA kernel to Python.

## Suggested mini scope

- Pick a tiny operation such as `vector_add` or `saxpy`
- Keep the Python API thin
- Focus on the call path, not on fancy packaging

## Files you would normally create in a standalone extension project

- `setup.py` or `pyproject.toml`
- `binding.cpp`
- `kernel.cu`
- `test_op.py`

This lab uses `torch.utils.cpp_extension.load` in `test_op.py`, so you do not
need a separate `setup.py` yet. PyTorch JIT-compiles the extension the first
time you run the test.

## Experiment

Activate the PyTorch environment:

```bash
source ~/pytorch-cuda/bin/activate
```

Check the environment:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

Run the lab:

```bash
cd GPU/labs/framework/01_torch_custom_op
python test_op.py
```

Expected result:

```text
max_error=0.00000000
call path: Python test_op.py -> binding.cpp -> kernel.cu -> CUDA kernel
```

If the run fails before compiling, install the build helpers in the active
virtual environment:

```bash
python -m pip install wheel ninja
```

## What to learn from this lab

- how Python calls into compiled C++ code
- where tensor shape and dtype checks belong
- why custom ops are still useful even when PyTorch has many built-ins
