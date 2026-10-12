"""Test orchestration with a fake service; no external jobs or benchmark claims."""
import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
from packaging.version import Version

from accelerator_bench.hub import execute


def test_remote_orchestration_persists_ids_and_leaves_placement_unverified(tmp_path, monkeypatch):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"test-only model placeholder")
    arrays = tmp_path / "arrays.npz"
    x = np.ones((1, 2), dtype=np.float32)
    np.savez(arrays, x=x, reference=x.astype(np.float64))
    manifest = {"cases": [{"case": {"id": "unit-test", "op": "matmul"},
        "model": model.name, "arrays": arrays.name, "input_weight_sha256": "test-only",
        "onnx_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "arrays_sha256": hashlib.sha256(arrays.read_bytes()).hexdigest()}]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    compile_job = SimpleNamespace(job_id="test-compile", get_target_model=lambda: object())
    inference = SimpleNamespace(job_id="test-inference", download_profile=lambda: {"execution_summary": {}},
                                download_output_data=lambda: {"output": [x]})
    fake_hub = SimpleNamespace(__version__=Version("1.2.3"),
        get_devices=lambda **kwargs: [SimpleNamespace(name="test-device")],
        submit_compile_job=lambda **kwargs: compile_job,
        submit_inference_job=lambda **kwargs: inference)
    monkeypatch.setitem(sys.modules, "qai_hub", fake_hub)
    result = execute(path, "test-device", "2.50", tmp_path / "results")
    assert result["client_version"] == "1.2.3"
    assert len(result["records"]) == 2
    for record in result["records"]:
        assert record["compile_job_id"] == "test-compile"
        assert record["inference_job_id"] == "test-inference"
        assert record["correctness"]["passed"]
        assert not record["placement_verified"]
        assert record["energy"]["status"] == "unavailable"
    assert json.loads((tmp_path / "results/results.json").read_text())["status"] == "requires_profile_review"
