import torch
from torch.utils.cpp_extension import load


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch cannot see CUDA. Run the environment check in the README first."
        )

    module = load(
        name="saxpy_extension",
        sources=["binding.cpp", "kernel.cu"],
        extra_cuda_cflags=["-O2"],
        verbose=True,
    )

    n = 1_000_000
    a = 2.5
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)

    out = module.saxpy(x, y, a)
    expected = a * x + y
    max_error = (out - expected).abs().max().item()

    print(f"max_error={max_error:.8f}")
    print("call path: Python test_op.py -> binding.cpp -> kernel.cu -> CUDA kernel")


if __name__ == "__main__":
    main()
