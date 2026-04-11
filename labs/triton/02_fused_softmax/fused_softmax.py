import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row = tl.load(row_start + col_offsets, mask=mask, other=-float("inf"))
    row_minus_max = row - tl.max(row, axis=0)
    numerators = tl.exp(row_minus_max)
    denominator = tl.sum(numerators, axis=0)
    softmax = numerators / denominator

    output_start = output_ptr + row_idx * output_row_stride
    tl.store(output_start + col_offsets, softmax, mask=mask)


def fused_softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    assert x.ndim == 2

    rows, cols = x.shape
    output = torch.empty_like(x)
    block_size = triton.next_power_of_2(cols)
    num_warps = 4 if block_size <= 1024 else 8

    softmax_kernel[(rows,)](
        x,
        output,
        x.stride(0),
        output.stride(0),
        cols,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return output


def main() -> None:
    x = torch.randn(32, 256, device="cuda", dtype=torch.float32)
    out = fused_softmax(x)
    expected = torch.softmax(x, dim=1)
    max_error = (out - expected).abs().max().item()
    print(f"triton fused_softmax max_error={max_error:.8f}")


if __name__ == "__main__":
    main()

