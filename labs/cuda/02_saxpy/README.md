# CUDA Lab 02: SAXPY

## Goal

Practice another elementwise kernel and get comfortable with launch geometry.

## Build

```bash
nvcc -O2 -std=c++17 saxpy.cu -o saxpy
./saxpy
```

## Formula

```text
y = a * x + y
```

