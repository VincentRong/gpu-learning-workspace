# Day 6: Triton Kernels

Date: 2026-05-02

## What ran

- `labs/triton/01_vector_add/vector_add.py`
  - result: `triton vector_add max_error=0.00000000`
- `labs/triton/02_fused_softmax/fused_softmax.py`
  - result: `triton fused_softmax max_error=0.00000001`

Both Triton kernels matched their PyTorch references.

## Programming model shift

Raw CUDA often starts from individual threads:

```cpp
idx = blockIdx.x * blockDim.x + threadIdx.x;
```

Triton starts from a program instance that owns a block of elements:

```python
pid = tl.program_id(axis=0)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
```

The mental model changes from "what does one thread do?" to "what vector of
elements does one program instance process?"

## Vector add mapping

In CUDA, each thread usually owns one scalar index.

In Triton, one program owns a vector of offsets:

```python
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
tl.store(out_ptr + offsets, x + y, mask=mask)
```

Triton still makes boundary handling explicit through `mask`, but it removes
much of the launch and per-thread boilerplate.

## Fused softmax mapping

The Triton softmax kernel expresses the stable softmax formula directly:

```python
row = tl.load(row_start + col_offsets, mask=mask, other=-float("inf"))
row_minus_max = row - tl.max(row, axis=0)
numerators = tl.exp(row_minus_max)
denominator = tl.sum(numerators, axis=0)
softmax = numerators / denominator
tl.store(output_start + col_offsets, softmax, mask=mask)
```

Compared with the CUDA row-wise softmax lab, Triton hides the manual shared
memory reduction pattern. `tl.max` and `tl.sum` describe reductions over the
program's vector.

## What Triton abstracts away

Triton abstracts:

- many details of CUDA thread indexing
- some shared-memory/reduction boilerplate
- C++ extension and binding code for Python-facing kernels
- low-level launch syntax

Triton does not abstract:

- choosing a good block size
- understanding memory access patterns
- masking out-of-bounds elements
- knowing where reductions happen
- reasoning about numerical stability

## Why Triton is attractive for LLM kernels

Many LLM kernels are small, shape-sensitive, and memory-sensitive. Triton lets
you write GPU kernels close to Python/PyTorch while still controlling tiling,
masking, and reductions. That makes it useful for experimenting with fused
softmax, layernorm, and attention-like kernels without writing full C++/CUDA
extension scaffolding every time.
