import json

import pytest

from accelerator_bench.comparison import aggregate_sweep, gpu_matches, check_layout_settings
from accelerator_bench.htp import summarize_htp
from accelerator_bench.hub_report import review_profile


def sweep_fixture(tmp_path):
    case = {'id': 'fixture', 'op': 'rmsnorm', 'shape': [1, 2]}
    (tmp_path / 'export-manifest.json').write_text(json.dumps({'cases': [
        {'case': case, 'input_weight_sha256': 'same-inputs'}]}))
    records = []
    for repeat, times in enumerate([[10] * 100, [20], [30]], 1):
        profile = f'{repeat}.json'
        (tmp_path / profile).write_text(json.dumps({'execution_detail': [{'compute_unit': 'NPU'}],
            'execution_summary': {'all_inference_times': times}}))
        records.append({'key': str(repeat), 'case_id': 'fixture', 'variant': 'optimized', 'repeat': repeat,
            'job_id': str(repeat), 'status': 'validated', 'profile_file': profile,
            'input_weight_sha256': 'same-inputs', 'correctness': {'passed': True, 'relative_l2_error': .001}})
    data = {'source': 'qualcomm_ai_hub_sweep', 'status': 'complete', 'repeats': 3, 'records': records}
    (tmp_path / 'results.json').write_text(json.dumps(data))
    return data, case


def test_independent_job_medians_are_not_pooled(tmp_path):
    sweep_fixture(tmp_path)
    result = aggregate_sweep(tmp_path)
    assert result['rows'][0]['median_us'] == 20
    assert result['rows'][0]['complete']
    assert result['rows'][0]['min_us'] == 10
    assert result['rows'][0]['max_us'] == 30


def test_fallback_cannot_enter_a_complete_npu_group(tmp_path):
    sweep_fixture(tmp_path)
    (tmp_path / '2.json').write_text(json.dumps({'execution_detail': [{'compute_unit': 'CPU'}],
        'execution_summary': {'all_inference_times': [20]}}))
    result = aggregate_sweep(tmp_path)
    assert not result['rows'][0]['complete']
    assert result['rows'][0]['n'] == 2
    assert len(result['failures']) == 1


def test_duplicate_job_is_not_an_independent_repeat(tmp_path):
    data, _ = sweep_fixture(tmp_path)
    data['records'][1]['job_id'] = data['records'][0]['job_id']
    (tmp_path / 'results.json').write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Duplicate'):
        aggregate_sweep(tmp_path)


def test_comparison_rejects_different_input_values(tmp_path):
    _, case = sweep_fixture(tmp_path)
    remote = aggregate_sweep(tmp_path)['rows']
    gpu = {'source': 'local_cuda_measured', 'status': 'complete', 'records': [
        {'case': case, 'dtype': 'float16', 'variant': 'compiled', 'status': 'ok',
         'correctness': {'passed': True}, 'graph_correctness': {'passed': True}, 'input_weight_sha256': 'different'}]}
    with pytest.raises(ValueError, match='workload mismatch'):
        gpu_matches(gpu, remote)


def test_layout_ablation_rejects_runtime_or_option_changes():
    import copy
    base = {'device': 'test', 'qairt_version': '2.50', 'jobs': [
        {'case_id': 'conv', 'variant': 'optimized', 'compile_options': 'O3', 'inference_options': 'NPU'}]}
    check_layout_settings(base, copy.deepcopy(base))
    changed = copy.deepcopy(base)
    changed['qairt_version'] = '2.51'
    with pytest.raises(ValueError, match='device or runtime'):
        check_layout_settings(base, changed)
    changed = copy.deepcopy(base)
    changed['jobs'][0]['inference_options'] = 'CPU'
    with pytest.raises(ValueError, match='compiler or inference'):
        check_layout_settings(base, changed)


def test_invalid_latency_samples_are_excluded():
    for times in ([0], [-1], [float('nan')], [float('inf')]):
        assert review_profile({'execution_summary': {'all_inference_times': times}})['sample_count'] == 0


def test_htp_signatures_ignore_timing_noise():
    raw = {'data': {'htp_overall_summary': {'data': [{}]}, 'htp_op_instances': {'data': [{'dtype': 'Float16'}]},
        'htp_op_types': {'data': [{'op': 'q::Transpose.2D', 'instances': 4, 'percent_dominant_path_cycles_htp_0': 50},
                                {'op': 'q::Conv', 'instances': 1, 'percent_dominant_path_cycles_htp_0': 20}]}}}
    result = summarize_htp(raw)
    assert result['transpose_instances'] == 4
    assert result['transpose_dominant_path_percent'] == 50
    assert result['implementation_signature'] == [['q::Conv', 1], ['q::Transpose.2D', 4]]


def test_htp_download_accepts_absolute_sdk_paths(tmp_path, monkeypatch):
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    from accelerator_bench.htp import collect_htp
    root = tmp_path / 'run'
    root.mkdir()
    (root / 'results.json').write_text(json.dumps({'records': [{'repeat': 1, 'status': 'validated',
        'case_id': 'test', 'variant': 'optimized', 'key': 'test_r1', 'job_id': 'test-job'}]}))
    raw = {'data': {'htp_overall_summary': {'data': [{}]}, 'htp_op_types': {'data': []},
                    'htp_op_instances': {'data': []}}}
    def download(folder, kind):
        path = Path(folder) / 'test_qhas_summary.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(raw))
        return [str(path.resolve())]
    fake = SimpleNamespace(set_verbose=lambda value: None, JobArtifactType=SimpleNamespace(QHAS_SUMMARY=8),
        get_job=lambda job_id: SimpleNamespace(download_artifacts_for_type=download))
    monkeypatch.setitem(sys.modules, 'qai_hub', fake)
    monkeypatch.chdir(tmp_path)
    records = collect_htp('run')
    assert records[0]['raw_file'] == 'htp/test_r1/test_qhas_summary.json'
