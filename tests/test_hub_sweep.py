import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from accelerator_bench.hub_sweep import run_sweep


def setup_service(tmp_path, monkeypatch):
    model = tmp_path / 'model.onnx'
    model.write_bytes(b'unit test fixture')
    arrays = tmp_path / 'arrays.npz'
    x = np.ones((1, 2), dtype=np.float32)
    np.savez(arrays, x=x, reference=x.astype(np.float64))
    manifest = {'seed': 4, 'cases': [{'case': {'id': 'fixture', 'op': 'matmul'},
        'model': model.name, 'arrays': arrays.name, 'input_weight_sha256': 'fixture',
        'onnx_sha256': hashlib.sha256(model.read_bytes()).hexdigest(),
        'arrays_sha256': hashlib.sha256(arrays.read_bytes()).hexdigest()}]}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    jobs, calls = {}, []

    def submit(kind, **kwargs):
        key = str(len(jobs))
        calls.append(kind)
        job = SimpleNamespace(job_id=key, get_target_model=lambda: object(),
            get_status=lambda: SimpleNamespace(finished=True, success=True),
            download_profile=lambda: {'execution_detail': [{'compute_unit': 'NPU'}],
                'execution_summary': {'all_inference_times': [10, 11, 12]}},
            download_output_data=lambda: {'output': [x]})
        jobs[key] = job
        return job

    fake = SimpleNamespace(__version__='test', set_verbose=lambda value: None,
        get_devices=lambda **kw: [SimpleNamespace(name='test', os='test', attributes=[])],
        get_job=lambda job_id: jobs[job_id],
        submit_compile_job=lambda **kw: submit('compile', **kw),
        submit_inference_job=lambda **kw: submit('inference', **kw))
    monkeypatch.setitem(sys.modules, 'qai_hub', fake)
    return path, fake, calls, jobs


def test_sweep_repeats_and_resume_do_not_resubmit(tmp_path, monkeypatch):
    path, fake, calls, jobs = setup_service(tmp_path, monkeypatch)
    output = tmp_path / 'results'
    original_get = fake.get_job
    fake.get_job = lambda job_id: (_ for _ in ()).throw(ConnectionError('simulated interruption'))
    with pytest.raises(ConnectionError):
        run_sweep(path, 'test', '2.50', output, repeats=2, max_active=1, poll_seconds=0)
    saved = json.loads((output / 'results.json').read_text())
    assert (output / 'export-manifest.json').read_bytes() == path.read_bytes()
    assert sum(c['status'] == 'submitted' for c in saved['compilations']) == 1
    fake.get_job = original_get
    result = run_sweep(path, 'test', '2.50', output, repeats=2, max_active=1, poll_seconds=0, resume=True)
    assert result['status'] == 'complete'
    assert calls == ['compile'] * 2 + ['inference'] * 4
    assert {r['repeat'] for r in result['records']} == {1, 2}
    assert all(r['correctness']['passed'] for r in result['records'])
    run_sweep(path, 'test', '2.50', output, repeats=2, poll_seconds=0, resume=True)
    assert len(calls) == 6
    with pytest.raises(ValueError, match='Resume configuration'):
        run_sweep(path, 'test', '2.51', output, repeats=2, resume=True)
    with pytest.raises(ValueError, match='Resume configuration'):
        run_sweep(path, 'test', '2.50', output, repeats=2, resume=True, prepare_at_inference=False)


def test_sweep_stops_after_uncertain_submission(tmp_path, monkeypatch):
    path, fake, calls, jobs = setup_service(tmp_path, monkeypatch)
    fake.submit_compile_job = lambda **kw: (_ for _ in ()).throw(ConnectionError())
    with pytest.raises(RuntimeError, match='Submission uncertain'):
        run_sweep(path, 'test', '2.50', tmp_path / 'out', poll_seconds=0)
    with pytest.raises(RuntimeError, match='Uncertain submission'):
        run_sweep(path, 'test', '2.50', tmp_path / 'out', poll_seconds=0, resume=True)
