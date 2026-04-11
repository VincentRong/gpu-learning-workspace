# Week 5: Matmul and Tiling

## Goal

Use matrix multiplication to build intuition for data reuse and shared-memory
tiling.

## Theory block

- Review the matmul formula
- Predict why naive matmul reloads too much data

## Hands-on block

- Run the naive matmul lab
- Run the tiled matmul lab
- Capture timings in `benchmarks/templates/benchmark-template.csv`

## Deliverables

- Naive vs tiled benchmark results
- A note explaining why tiling helps

## Exit criteria

You can explain what each shared-memory tile is saving you from reloading.

