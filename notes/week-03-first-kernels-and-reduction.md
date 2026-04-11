# Week 3: First Kernels and Reduction

## Goal

Write your first three "real" CUDA kernels and validate them against CPU
results.

## Theory block

- Revisit the reduction formula in [math refresh](../docs/foundations/math-refresh.md)
- Learn what a race condition looks like in parallel code

## Hands-on block

- Complete `vector_add`
- Complete `saxpy`
- Complete `reduction`
- Add error checks after kernel launches

## Deliverables

- Three working kernels
- Notes on what changed between elementwise work and reduction work

## Exit criteria

You can explain host-device copy, kernel launch geometry, and why reduction is
harder than elementwise kernels.

