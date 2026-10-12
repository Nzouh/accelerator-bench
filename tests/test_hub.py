import json
import pytest

from accelerator_bench.hub import compile_options, plan, export


def test_folding_ablation_changes_only_one_option():
    base = compile_options("optimized")
    assert compile_options("no_conv_relu_fold") == base + ";default_graph_htp_disable_fold_relu_activation_into_conv=true"
    assert "O=1" in compile_options("prepare_fast")


def test_hub_export_preserves_weights_and_checks_onnx(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    config = {"seed": 1729, "dtypes": ["float32"], "cases": [
        {"id": "conv", "op": "conv_relu", "shape": [1, 3, 5, 5, 4, 3]},
        {"id": "norm", "op": "rmsnorm", "shape": [2, 17]},
    ]}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    manifest = export(config_path, tmp_path / "export")
    assert len(manifest["cases"]) == 2
    assert all(item["onnx_cpu_correctness"]["passed"] for item in manifest["cases"])
    experiment = plan(tmp_path / "export/manifest.json", "test device", "2.42")
    assert len(experiment["jobs"]) == 5
    assert all("--compute_unit npu" in j["compile_options"] for j in experiment["jobs"])
    assert all("--qairt_version 2.42" in j["compile_options"] for j in experiment["jobs"])
