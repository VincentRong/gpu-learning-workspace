# Capstone: Triton Fused Softmax

Date: 2026-05-02

## Goal

Compare `torch.softmax` with a Triton row-wise fused softmax kernel across
several matrix shapes.

## Question

When does a simple Triton fused softmax compete with PyTorch's built-in
softmax, and what still needs tuning?

## Benchmark matrix

Run:

```bash
source ~/pytorch-cuda/bin/activate
cd /home/rwenxiao/dev-learning/GPU/labs/triton/02_fused_softmax
python benchmark_softmax.py
```

Shapes:

- `1024x128`
- `1024x256`
- `1024x512`
- `1024x1024`
- `4096x256`
- `4096x512`
- `4096x1024`

Output:

`benchmarks/triton-fused-softmax.csv`

## Hypothesis

Triton should be competitive on row-wise softmax because the kernel fuses the
max, exponentiation, sum, division, and store into one program. Larger row
widths may benefit more from the fused structure, but performance will depend
on block size, memory traffic, and Triton's generated code.

## What to explain after running

- correctness: max error versus `torch.softmax`
- performance trend as row width grows
- what Triton made simpler than raw CUDA
- what still required GPU judgment

## Current capstone status

- Direction chosen: `Triton fused softmax`
- Runnable benchmark script: ready
- Benchmark CSV: captured in `benchmarks/triton-fused-softmax.csv`

## First benchmark results

| Shape | PyTorch softmax | Triton fused softmax | Speedup vs PyTorch | Max error |
| --- | ---: | ---: | ---: | ---: |
| `1024x128` | `0.0554 ms` | `0.1181 ms` | `0.47x` | `0.00000003` |
| `1024x256` | `0.0155 ms` | `0.0346 ms` | `0.45x` | `0.00000001` |
| `1024x512` | `0.0545 ms` | `0.1174 ms` | `0.46x` | `0.00000001` |
| `1024x1024` | `0.0230 ms` | `0.0327 ms` | `0.70x` | `0.00000001` |
| `4096x256` | `0.0322 ms` | `0.0315 ms` | `1.02x` | `0.00000001` |
| `4096x512` | `0.0451 ms` | `0.0459 ms` | `0.98x` | `0.00000001` |
| `4096x1024` | `0.1041 ms` | `0.0863 ms` | `1.21x` | `0.00000001` |

## First interpretation

The Triton kernel is correct across all tested shapes, with max error around
`1e-8` to `3e-8` versus `torch.softmax`.

Performance is shape-dependent:

- For `1024` rows, PyTorch is faster in this run.
- Around `4096x256` and `4096x512`, Triton is roughly tied with PyTorch.
- At `4096x1024`, Triton is faster, reaching `1.21x` speedup.

The likely lesson is that a simple Triton kernel has overhead and tuning limits
on smaller shapes, but can become competitive as there is more row-wise work to
amortize launch and program overhead. This supports the capstone hypothesis,
but only partially: fused Triton softmax is not automatically faster than
PyTorch; it needs the right shape and tuning.

## Next tuning questions

- Does changing `num_warps` improve `1024`-wide rows?
- Does benchmarking more row counts show a clearer crossover point?
- Does the result stay stable across repeated clean runs?

## Second benchmark note

A second run with `num_warps = 4` produced different timings:

- `4096x1024`: PyTorch `0.0863 ms`, Triton `0.1066 ms`, `0.81x`
- `4096x512`: PyTorch `0.0456 ms`, Triton `0.0457 ms`, `1.00x`
- `1024x1024`: PyTorch `0.0232 ms`, Triton `0.0318 ms`, `0.73x`

Since the original kernel already used `num_warps=4` for all tested column
sizes up to `1024`, this was not a real tuning change. Treat the difference as
benchmark variability until repeated clean runs or a true parameter sweep show
a stable trend.

Next script:

```bash
cd /home/rwenxiao/dev-learning/GPU/labs/triton/02_fused_softmax
python sweep_num_warps.py
```

## `num_warps` sweep

Output:

`benchmarks/triton-softmax-num-warps-sweep.csv`

Best result per shape:

| Shape | Best `num_warps` | PyTorch softmax | Best Triton softmax | Speedup vs PyTorch |
| --- | ---: | ---: | ---: | ---: |
| `1024x256` | `8` | `0.0131 ms` | `0.0297 ms` | `0.44x` |
| `1024x512` | `4` | `0.0181 ms` | `0.0280 ms` | `0.65x` |
| `1024x1024` | `1` | `0.0240 ms` | `0.0290 ms` | `0.83x` |
| `4096x256` | `1` | `0.0282 ms` | `0.0289 ms` | `0.98x` |
| `4096x512` | `4` | `0.0438 ms` | `0.0456 ms` | `0.96x` |
| `4096x1024` | `4` | `0.0866 ms` | `0.0856 ms` | `1.01x` |

All results matched `torch.softmax` with max error around `1e-8` to `3e-8`.

## Tuning interpretation

Changing `num_warps` matters, but there is no single best value for every
shape. The best choice depends on row width and row count:

- `1024x256`: best with `8` warps, but still much slower than PyTorch.
- `1024x512`: best with `4` warps.
- `1024x1024`: best with `1` warp in this run.
- `4096x256`: best with `1` warp and nearly tied with PyTorch.
- `4096x512`: best with `4` warps and nearly tied with PyTorch.
- `4096x1024`: best with `4` warps and slightly faster than PyTorch.

The strongest honest conclusion is that this simple Triton softmax is correct
and tunable, but PyTorch's built-in softmax is a strong baseline. Triton becomes
competitive only for some larger shapes in this experiment.

## Final capstone draft

This capstone compared PyTorch's built-in row-wise softmax with a Triton fused
softmax implementation. The Triton kernel expresses stable softmax directly:
load a row, subtract the row max, exponentiate, sum, divide, and store. Compared
with raw CUDA, Triton removes much of the manual thread indexing and reduction
boilerplate, but still requires choices about block size, masks, row layout, and
`num_warps`.

Correctness was strong across all tested shapes, with max error near `1e-8`.
Performance was shape-sensitive. PyTorch was faster on smaller `1024`-row test
cases. Triton nearly matched PyTorch on `4096x256` and `4096x512`, and slightly
beat PyTorch on `4096x1024` with `num_warps=4`.

The main lesson is that custom GPU kernels are not automatically faster than
framework kernels. Their value comes from controlling fusion, layout, and
specialized shapes. Triton is attractive because it makes that experimentation
much lighter than writing a full C++/CUDA PyTorch extension, while still keeping
the important GPU performance decisions visible.
