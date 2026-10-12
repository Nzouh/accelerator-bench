"""Shared, seeded workloads for local CUDA and remote AI Hub execution."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class Matmul(nn.Module):
    def __init__(self, k: int, n: int, generator: torch.Generator):
        super().__init__()
        self.register_buffer("weight", torch.randn(k, n, generator=generator) / math.sqrt(k))

    def forward(self, x):
        return x @ self.weight


class ConvRelu(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int, generator: torch.Generator):
        super().__init__()
        self.register_buffer("weight", torch.randn(cout, cin, kernel, kernel, generator=generator)
                             / math.sqrt(cin * kernel * kernel))
        self.padding = kernel // 2

    def forward(self, x):
        return F.relu(F.conv2d(x, self.weight, padding=self.padding))


class RMSNorm(nn.Module):
    def __init__(self, width: int, generator: torch.Generator, eps: float = 1e-6):
        super().__init__()
        self.register_buffer("weight", 1 + .1 * torch.randn(width, generator=generator))
        self.eps = eps

    def forward(self, x):
        # FP32 reduction for FP16 inputs; retain FP64 for the reference path.
        work = x if x.dtype == torch.float64 else x.float()
        normalized = work * torch.rsqrt(work.square().mean(dim=-1, keepdim=True) + self.eps)
        return (normalized * self.weight.to(work.dtype)).to(x.dtype)


def load_config(path: str | Path):
    config = json.loads(Path(path).read_text())
    ids = [c["id"] for c in config["cases"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Case IDs must be unique")
    for c in config["cases"]:
        expected = {"matmul": 3, "conv_relu": 6, "rmsnorm": 2}
        if c["op"] not in expected or len(c["shape"]) != expected[c["op"]]:
            raise ValueError(f"Invalid operation/shape: {c}")
        if any(not isinstance(v, int) or v <= 0 for v in c["shape"]):
            raise ValueError("Dimensions must be positive integers")
    if not set(config["dtypes"]) <= {"float16", "float32"}:
        raise ValueError("Supported dtypes: float16, float32")
    return config


def make_workload(case, seed=1729):
    """Independent per-case seed; input and weights do not depend on run order."""
    digest = hashlib.sha256(case["id"].encode()).digest()
    generator = torch.Generator().manual_seed((seed + int.from_bytes(digest[:4], "little")) % 2**32)
    shape = case["shape"]
    if case["op"] == "matmul":
        m, k, n = shape
        module, input_shape = Matmul(k, n, generator), (m, k)
    elif case["op"] == "conv_relu":
        b, cin, h, w, cout, kernel = shape
        module, input_shape = ConvRelu(cin, cout, kernel, generator), (b, cin, h, w)
    elif case["op"] == "rmsnorm":
        rows, width = shape
        module, input_shape = RMSNorm(width, generator), (rows, width)
    else:
        raise ValueError(case["op"])
    return module.eval(), torch.randn(*input_shape, generator=generator)


def fingerprint(module, x):
    digest = hashlib.sha256()
    for tensor in [x, *module.buffers()]:
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def reference_output(module, x):
    with torch.inference_mode():
        return copy.deepcopy(module).double()(x.double())


def correctness(actual, reference, dtype):
    actual = actual.detach().cpu().double()
    reference = reference.detach().cpu().double()
    atol, rtol = (0.01, 0.01) if dtype == "float16" else (2e-5, 2e-4)
    if actual.shape != reference.shape:
        return {"passed": False, "reason": "shape mismatch"}
    delta = actual - reference
    finite = bool(torch.isfinite(actual).all())
    return {
        "passed": finite and bool(torch.allclose(actual, reference, atol=atol, rtol=rtol)),
        "finite": finite, "atol": atol, "rtol": rtol,
        "max_abs_error": float(delta.abs().max()) if finite else None,
        "relative_l2_error": float(torch.linalg.vector_norm(delta) /
                                   torch.linalg.vector_norm(reference).clamp_min(1e-30)) if finite else None,
        "reference": "CPU float64 on original seeded float32 inputs and weights",
    }


def operation_count(case):
    """Conventional useful FLOPs; not a hardware instruction count."""
    s = case["shape"]
    if case["op"] == "matmul":
        m, k, n = s
        return 2 * m * k * n
    if case["op"] == "conv_relu":
        b, cin, h, w, cout, k = s
        out_h, out_w = h + 2 * (k // 2) - k + 1, w + 2 * (k // 2) - k + 1
        return b * cout * out_h * out_w * (2 * cin * k * k + 1)
    return None  # Avoid misleading TOPS for reductions and reciprocal square roots.
