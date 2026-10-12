import torch
import pytest

from accelerator_bench.workloads import make_workload, fingerprint, reference_output, correctness


@pytest.mark.parametrize("op,shape,out_shape", [
    ("matmul", [3, 7, 5], (3, 5)),
    ("conv_relu", [1, 3, 9, 7, 4, 3], (1, 4, 9, 7)),
    ("rmsnorm", [3, 37], (3, 37)),
])
def test_shared_workloads_are_reproducible_and_match_reference(op, shape, out_shape):
    case = {"id": "test_" + op, "op": op, "shape": shape}
    module, x = make_workload(case)
    module2, x2 = make_workload(case)
    assert fingerprint(module, x) == fingerprint(module2, x2)
    assert module(x).shape == out_shape
    assert correctness(module(x), reference_output(module, x), "float32")["passed"]


def test_correctness_rejects_nonfinite_and_wrong_results():
    reference = torch.ones(2, 3)
    assert not correctness(reference * 2, reference, "float32")["passed"]
    assert not correctness(reference * float("nan"), reference, "float32")["passed"]


def test_rmsnorm_zero_input():
    module, x = make_workload({"id": "zero", "op": "rmsnorm", "shape": [2, 17]})
    assert torch.equal(module(torch.zeros_like(x)), torch.zeros_like(x))
