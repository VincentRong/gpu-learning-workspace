# Week 9: PyTorch Custom CUDA Op

## Goal

Understand how a low-level CUDA kernel gets exposed to framework-level Python
code.

## Theory block

- Read [docs/setup/pytorch-triton.md](../docs/setup/pytorch-triton.md)
- Read the custom-op lab README

## Hands-on block

- Set up PyTorch in your WSL virtual environment
- Follow the custom-op lab path
- Trace how Python calls into compiled code

## Deliverables

- A diagram or note showing the Python -> C++ -> CUDA path
- One short explanation of why frameworks still need custom kernels

## Exit criteria

You can explain the integration path even if you are not yet fluent in all
build details.

