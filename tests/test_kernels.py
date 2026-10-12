import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("width", [1, 17, 1024, 1537, 8192])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_fused_rmsnorm_tail_and_precision(width, dtype):
    pytest.importorskip("triton")
    from accelerator_bench.kernels import fused_rmsnorm
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(3, width, generator=generator).to("cuda", dtype)
    weight = torch.randn(width, generator=generator).to("cuda", dtype)
    ref = (x.double() * torch.rsqrt(x.double().square().mean(-1, keepdim=True) + 1e-6) * weight.double()).to(dtype)
    torch.testing.assert_close(fused_rmsnorm(x, weight), ref, atol=2e-3 if dtype == torch.float16 else 2e-6,
                               rtol=2e-3 if dtype == torch.float16 else 2e-5)
    assert torch.equal(fused_rmsnorm(torch.zeros_like(x), weight), torch.zeros_like(x))


def test_fused_rejects_noncontiguous_input():
    pytest.importorskip("triton")
    from accelerator_bench.kernels import fused_rmsnorm
    with pytest.raises(ValueError):
        fused_rmsnorm(torch.ones(5, 3, device="cuda").T, torch.ones(5, device="cuda"))
