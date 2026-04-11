# Capstone Options

Choose one capstone in Weeks 11-12. Pick depth over breadth.

## Option 1: CUDA tiled matmul

- Build on Weeks 5-6
- Compare naive vs tiled
- Benchmark several matrix sizes
- Explain how shared memory improves reuse

## Option 2: CUDA softmax plus layernorm

- Build on Weeks 7-8
- Implement stable softmax and one row-wise layernorm kernel
- Measure how shape affects throughput
- Explain which parts are reduction bound

## Option 3: Triton fused softmax

- Build on Weeks 9-10
- Start from the Triton lab in this repo
- Benchmark across sequence lengths
- Explain what Triton abstracts away versus raw CUDA

## Selection heuristic

- Choose matmul if you want the cleanest path to tiling intuition
- Choose softmax/layernorm if you want LLM-flavored bandwidth intuition
- Choose Triton if your goal is practical LLM kernel work in framework code

