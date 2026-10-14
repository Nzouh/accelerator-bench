import json
import pytest

from accelerator_bench.hub import compile_options, plan, export


def test_folding_ablation_changes_only_one_option():
    base = compile_options("optimized")
    assert compile_options("no_conv_relu_fold") == base + ";default_graph_htp_disable_fold_relu_activation_into_conv=true"
    assert "O=1" in compile_options("prepare_fast")


def test_explicit_inference_preparation_flags(tmp_path):
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps({'cases': [{'case': {'id': 'conv', 'op': 'conv_relu'}}]}))
    original = plan(path, 'test', '2.50')
    explicit = plan(path, 'test', '2.50', prepare_at_inference=True)
    for before, after in zip(original['jobs'], explicit['jobs']):
        assert before['compile_options'] == after['compile_options']
        assert 'default_graph_htp_' not in before['inference_options']
        assert 'default_graph_htp_precision=FLOAT16' in after['inference_options']
        level = 'O=1' if after['variant'] == 'prepare_fast' else 'O=3'
        assert level in after['inference_options']
        assert ('disable_fold_relu_activation_into_conv=true' in after['inference_options']) == (after['variant'] == 'no_conv_relu_fold')


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


def test_nhwc_export_preserves_logical_values(tmp_path):
    import numpy as np
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    config = {"seed": 1729, "dtypes": ["float32"], "cases": [
        {"id": "conv", "op": "conv_relu", "shape": [1, 3, 5, 5, 4, 3]}]}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    nchw = export(path, tmp_path / "nchw")['cases'][0]
    nhwc = export(path, tmp_path / "nhwc", io_layout="nhwc")['cases'][0]
    assert nchw['input_weight_sha256'] == nhwc['input_weight_sha256']
    assert nhwc['input_shape'] == [1, 5, 5, 3]
    assert nhwc['onnx_cpu_correctness']['passed']
    with np.load(tmp_path / 'nchw' / nchw['arrays']) as a, np.load(tmp_path / 'nhwc' / nhwc['arrays']) as b:
        np.testing.assert_array_equal(a['x'].transpose(0, 2, 3, 1), b['x'])
        np.testing.assert_array_equal(a['reference'].transpose(0, 2, 3, 1), b['reference'])
