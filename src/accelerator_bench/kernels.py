"""A deliberately small RMSNorm kernel: one Triton program per row."""
import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm(X, W, Y, WIDTH: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * WIDTH + col, col < WIDTH, other=0).to(tl.float32)
    w = tl.load(W + col, col < WIDTH, other=0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, axis=0) / WIDTH + EPS)
    tl.store(Y + row * WIDTH + col, x * inv * w, col < WIDTH)


def fused_rmsnorm(x, weight, eps=1e-6):
    if (x.ndim != 2 or weight.ndim != 1 or x.shape[1] != weight.numel()
            or not x.is_cuda or not weight.is_cuda or x.device != weight.device
            or not x.is_contiguous() or not weight.is_contiguous()
            or x.dtype not in (torch.float16, torch.float32) or weight.dtype != x.dtype
            or not 0 < x.shape[1] <= 8192 or eps <= 0):
        raise ValueError("Fused RMSNorm requires contiguous CUDA FP16/FP32 rows, matching weights, width 1..8192, eps > 0")
    y = torch.empty_like(x)
    block = triton.next_power_of_2(x.shape[1])
    _rmsnorm[(x.shape[0],)](x, weight, y, x.shape[1], eps, block,
                          num_warps=4 if block <= 2048 else 8)
    return y
