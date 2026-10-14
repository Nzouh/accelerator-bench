"""Audit rejected logits without changing the predeclared acceptance threshold."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from export_vision_layout import VisionLayout, state_hash
from accelerator_bench.runner import save_result
from accelerator_bench.workloads import correctness


def summarize(actual, reference):
    actual, reference = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    delta = np.abs(actual-reference)
    allowance = .01 + .01*np.abs(reference)
    result = {'max_tolerance_ratio': float((delta/allowance).max()),
            'logits_outside_tolerance': int((delta > allowance).sum()), 'logit_count': int(actual.size),
            **correctness(torch.from_numpy(actual), torch.from_numpy(reference), 'float16')}
    result['reference'] = 'Saved CPU FP64 evaluation of the case on original FP32 tensors and pretrained weights'
    return result


def review(manifest_path, source_dir):
    manifest_path, root = Path(manifest_path), Path(source_dir)
    manifest = json.loads(manifest_path.read_text())
    results = json.loads((root/'results.json').read_text())
    torch.set_num_threads(4)
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2).eval()
    assert state_hash(model) == manifest['state_sha256']
    baseline_root = root/'precision'
    baseline_root.mkdir(exist_ok=True)
    cases = []
    for item in manifest['cases']:
        case = item['case']['id']
        arrays_path = manifest_path.parent/item['arrays']
        assert hashlib.sha256(arrays_path.read_bytes()).hexdigest() == item['arrays_sha256']
        arrays = np.load(arrays_path)
        reference, x = arrays['reference'], torch.from_numpy(arrays['x'])
        module = VisionLayout(model, case).half().eval()
        with torch.inference_mode():
            half_output = module(x.half()).float().numpy()
        # Recreate FP32 state rather than keeping a model rounded by .half().
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2).eval()
        reference_file, half_file = baseline_root/f'{case}_fp64.npy', baseline_root/f'{case}_cpu_fp16.npy'
        np.save(reference_file, reference)
        np.save(half_file, half_output)
        records = []
        for record in results['records']:
            if record['case_id'] != case or record['status'] not in ('validated', 'rejected'):
                continue
            actual_path = root/record['output_file']
            assert hashlib.sha256(actual_path.read_bytes()).hexdigest() == record['output_sha256']
            actual = np.load(actual_path)
            records.append({'job_id': record['job_id'], 'repeat': record['repeat'],
                'status': record['status'], 'top1_equal': record['classification']['top1_equal'],
                'all_reported_entries_on_npu': record['review']['all_reported_entries_on_npu'],
                **summarize(actual, reference),
                'relative_l2_to_cpu_fp16': float(np.linalg.norm(actual.astype(np.float64)-half_output)/np.linalg.norm(half_output))})
        files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (reference_file, half_file)}
        cases.append({'case_id': case, 'cpu_fp16': summarize(half_output, reference),
                      'jobs': records, 'files_sha256': files})
    output = {'source': 'post_run_precision_diagnostic', 'run_status': results['status'],
              'torch_version': torch.__version__, 'device': 'CPU', 'cases': cases,
              'acceptance_changed': False,
              'scope': 'CPU FP16 is an independent rounding baseline, not an emulator of HTP arithmetic. No new remote jobs.'}
    save_result(root/'precision-review.json', output)
    for row in cases:
        print(row['case_id'], 'CPU FP16 max ratio', round(row['cpu_fp16']['max_tolerance_ratio'], 3),
              'NPU ratios', [round(r['max_tolerance_ratio'], 3) for r in row['jobs']], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest-path', required=True)
    parser.add_argument('--source-dir', required=True)
    review(**vars(parser.parse_args()))
