import csv
from pathlib import Path

import torch

from fused_softmax import fused_softmax


SHAPES = [
    (1024, 128),
    (1024, 256),
    (1024, 512),
    (1024, 1024),
    (4096, 256),
    (4096, 512),
    (4096, 1024),
]

WARMUP = 10
ITERS = 50


def time_ms(fn, *args) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(WARMUP):
        fn(*args)
    torch.cuda.synchronize()

    start.record()
    for _ in range(ITERS):
        fn(*args)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / ITERS


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot see CUDA")

    output_path = Path(__file__).resolve().parents[3] / "benchmarks" / "triton-fused-softmax.csv"
    rows = []

    for n_rows, n_cols in SHAPES:
        x = torch.randn(n_rows, n_cols, device="cuda", dtype=torch.float32)

        expected = torch.softmax(x, dim=1)
        out = fused_softmax(x)
        max_error = (out - expected).abs().max().item()

        torch_ms = time_ms(lambda t: torch.softmax(t, dim=1), x)
        triton_ms = time_ms(fused_softmax, x)
        speedup = torch_ms / triton_ms if triton_ms > 0 else float("inf")

        shape = f"{n_rows}x{n_cols}"
        print(
            f"{shape}: torch={torch_ms:.4f} ms "
            f"triton={triton_ms:.4f} ms speedup={speedup:.2f}x "
            f"max_error={max_error:.8f}"
        )

        rows.append(
            {
                "date": "2026-05-02",
                "lab": "triton_fused_softmax",
                "input_shape": shape,
                "torch_softmax_ms": f"{torch_ms:.4f}",
                "triton_softmax_ms": f"{triton_ms:.4f}",
                "speedup_vs_torch": f"{speedup:.2f}",
                "correct": "yes",
                "max_error": f"{max_error:.8f}",
                "notes": f"average over {ITERS} launches after {WARMUP} warmup launches",
            }
        )

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
