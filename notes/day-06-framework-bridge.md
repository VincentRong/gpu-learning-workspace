# Day 6: Framework Bridge

Date: 2026-05-02

## What ran

Lab:

`labs/framework/01_torch_custom_op/test_op.py`

Result:

`max_error=0.00000095`

This is normal `float32` rounding error. The custom CUDA op matched the PyTorch
reference closely enough.

## Call path

```text
Python
  test_op.py
    calls torch.utils.cpp_extension.load(...)
    builds saxpy_extension.so
    calls module.saxpy(x, y, a)

C++ binding
  binding.cpp
    receives torch.Tensor objects
    checks device, dtype, shape, and contiguity
    calls saxpy_cuda(x, y, a)

CUDA wrapper and kernel
  kernel.cu
    allocates output tensor
    computes grid/block launch shape
    launches saxpy_kernel<<<blocks, threads>>>

GPU device code
  saxpy_kernel
    each thread owns one index
    computes out[idx] = a * x[idx] + y[idx]
```

## Why the checks live in C++

The C++ binding is the boundary between Python/PyTorch and raw CUDA. It still
understands `torch.Tensor` metadata, so it is the right place to reject invalid
inputs with clear errors.

Examples:

- CPU tensor instead of CUDA tensor
- `float64` tensor instead of `float32`
- mismatched shapes
- non-contiguous tensor layout

The CUDA kernel should focus on parallel computation. Once it receives raw
pointers and a length, each GPU thread should do the smallest useful unit of
work.

## Why custom ops still matter

PyTorch already has many operations, but custom kernels are still useful when:

- several operations can be fused to avoid extra memory reads/writes
- a workload has a special shape that generic PyTorch kernels do not optimize
  well
- you need to expose a CUDA implementation to Python while keeping the user API
  simple

## Next bridge

Move from raw CUDA extension work to Triton:

1. Run `labs/triton/01_vector_add/vector_add.py`
2. Run `labs/triton/02_fused_softmax/fused_softmax.py`
3. Compare what Triton hides versus what raw CUDA made explicit
