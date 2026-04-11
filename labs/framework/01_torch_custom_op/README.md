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

## What to learn from this lab

- how Python calls into compiled C++ code
- where tensor shape and dtype checks belong
- why custom ops are still useful even when PyTorch has many built-ins

