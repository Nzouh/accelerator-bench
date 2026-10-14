import hashlib
from types import SimpleNamespace

import numpy as np

from accelerator_bench.hub_sweep import collect_inference


def test_top1_change_rejects_otherwise_close_logits(tmp_path):
    reference = np.array([[1., 1.001]], dtype=np.float64)
    actual = np.array([[1.001, 1.]], dtype=np.float32)
    np.savez(tmp_path / 'arrays.npz', reference=reference)
    job = SimpleNamespace(download_profile=lambda: {
        'execution_detail': [{'compute_unit': 'NPU'}],
        'execution_summary': {'all_inference_times': [10, 11]}},
        download_output_data=lambda: {'output': [actual]})
    record = {'key': 'case', 'profile_file': 'profile.json'}
    item = {'arrays': 'arrays.npz', 'output_kind': 'classification_logits',
            'reference_description': 'Pretrained CPU FP64 reference'}
    collect_inference(job, record, item, tmp_path / 'manifest.json', tmp_path)
    assert record['correctness']['max_abs_error'] < .01
    assert not record['classification']['top1_equal']
    assert record['status'] == 'rejected'
    assert record['correctness']['reference'] == item['reference_description']
    assert hashlib.sha256((tmp_path / record['output_file']).read_bytes()).hexdigest() == record['output_sha256']
