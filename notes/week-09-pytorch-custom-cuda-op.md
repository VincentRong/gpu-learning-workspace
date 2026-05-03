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

Explanation:
因为 PyTorch built-ins 很多，但它们是通用算子；custom ops 的价值在于：针对某个模型、shape、硬件、数据布局，把“通用”变成“专用”。
1. 减少中间张量和显存读写
2. 支持 PyTorch 还没有的算子或新算法
3. 针对特定 shape 优化
4. 利用硬件特性
5. 减少 Python 调度开销
6. 控制反向传播
7. 支持特殊数据格式

