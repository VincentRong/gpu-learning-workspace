# Week 7: Softmax and Layernorm

## Goal

Translate reduction-heavy math into kernels that look more like LLM building
blocks.

## Theory block

- Read [docs/llm/kernel-hotspots.md](../docs/llm/kernel-hotspots.md)
- Re-derive stable softmax and row-wise layernorm formulas

## Hands-on block

- Run or finish the row-wise softmax lab
- Run or finish the row-wise layernorm lab
- Test different row widths

## Benchmark snapshot

Recorded in `benchmarks/softmax-layernorm-rowwise.csv`.

| Lab | Shape | Avg elapsed |
| --- | --- | ---: |
| `softmax_rowwise` | `4096x256` | `0.0640 ms` |
| `softmax_rowwise` | `4096x512` | `0.1755 ms` |
| `softmax_rowwise` | `4096x1024` | `0.2317 ms` |
| `layernorm_rowwise` | `4096x256` | `0.2294 ms` |
| `layernorm_rowwise` | `4096x512` | `0.2501 ms` |
| `layernorm_rowwise` | `4096x1024` | `0.2299 ms` |

Both kernels matched CPU references. The starter kernels use one block per row
and 256 threads per block.

## Key ideas

Stable softmax subtracts the row maximum before exponentiation:

`softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))`

This keeps large positive inputs from overflowing `exp`. Subtracting the same
constant from every element in the row does not change the final probabilities,
because the common scale factor cancels between the numerator and denominator.

Softmax and layernorm feel different from matmul because each row needs
reductions before it can write final outputs. Softmax needs a row max and a row
sum. Layernorm needs a row mean and variance. That means the threads in a block
must cooperate, use shared memory, synchronize, and then make another pass over
the row.

## Deliverables

- Correct outputs against CPU reference checks: done
- Notes on why these kernels feel different from matmul: done

## Exit criteria

You can explain why stable softmax subtracts the row max before exponentiation.
