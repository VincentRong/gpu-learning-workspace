# Triton Lab 02: Fused Softmax

## Goal

Use Triton to express a row-wise softmax in a style that feels closer to
framework kernel work.

## Run

```bash
python fused_softmax.py
```

## Watch for

- block size assumptions
- row-wise reductions
- where Triton removes CUDA boilerplate and where it does not

