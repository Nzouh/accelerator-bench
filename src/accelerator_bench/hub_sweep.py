"""Bounded AI Hub sweeps with checkpointed job IDs and independent repeats."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .hub import plan
from .hub_report import review_profile
from .runner import save_result, shell_output
from .workloads import correctness


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def validate_artifacts(manifest_path, manifest):
    for item in manifest['cases']:
        for field in ('model', 'arrays'):
            key = 'onnx_sha256' if field == 'model' else 'arrays_sha256'
            path = manifest_path.parent / item[field]
            if hashlib.sha256(path.read_bytes()).hexdigest() != item[key]:
                raise ValueError(f'Artifact hash mismatch: {path.name}')


def collect_inference(job, record, item, manifest_path, root):
    profile = job.download_profile()
    save_result(root / record['profile_file'], profile)
    outputs = job.download_output_data()
    if not outputs or len(outputs) != 1 or len(next(iter(outputs.values()))) != 1:
        raise ValueError('Expected one output and one sample')
    with np.load(manifest_path.parent / item['arrays']) as arrays:
        record['correctness'] = correctness(torch.from_numpy(next(iter(outputs.values()))[0]),
                                            torch.from_numpy(arrays['reference']), 'float16')
    record['review'] = review_profile(profile)
    record['execution_summary'] = profile.get('execution_summary', {})
    record['status'] = ('validated' if record['correctness']['passed'] and
                        record['review']['all_reported_entries_on_npu'] and
                        record['review']['sample_count'] else 'rejected')
    record['completed_utc'] = utc_now()


def run_sweep(manifest_path, device, qairt_version, output_dir, repeats=3,
              max_active=4, poll_seconds=15, resume=False, variants=None, prepare_at_inference=True):
    import qai_hub as hub
    if repeats < 1 or max_active < 1 or poll_seconds < 0:
        raise ValueError('Invalid sweep counts')
    manifest_path, root = Path(manifest_path), Path(output_dir)
    manifest = json.loads(manifest_path.read_text())
    validate_artifacts(manifest_path, manifest)
    experiment = plan(manifest_path, device, qairt_version, prepare_at_inference=prepare_at_inference)
    if variants:
        allowed = {j['variant'] for j in experiment['jobs']}
        if not set(variants) <= allowed:
            raise ValueError('Unknown sweep variant')
        experiment['jobs'] = [j for j in experiment['jobs'] if j['variant'] in variants]
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    matches = [d for d in hub.get_devices(name=device) if d.name == device]
    if len(matches) != 1:
        raise ValueError('Exact device must resolve uniquely')
    target = matches[0]
    result_path = root / 'results.json'
    if root.exists():
        if not resume:
            raise FileExistsError('Use --resume for an existing sweep')
        result = json.loads(result_path.read_text())
        if (result['manifest_sha256'] != manifest_hash or result['plan'] != experiment or
                result['repeats'] != repeats):
            raise ValueError('Resume configuration differs from saved sweep')
        if any(r['status'] in ('submitting', 'submission_unknown')
               for r in result['compilations'] + result['records']):
            raise RuntimeError('Uncertain submission: reconcile saved job name before resuming')
    else:
        if resume:
            raise FileNotFoundError('No sweep to resume')
        source_commit = shell_output(['git', 'rev-parse', 'HEAD'])
        source_dirty = bool(shell_output(['git', 'status', '--porcelain']))
        root.mkdir(parents=True)
        save_result(root / 'export-manifest.json', manifest)
        rng = random.Random(manifest.get('seed', 1729))
        compilations = [{**spec, 'key': spec['case_id'] + '_' + spec['variant'], 'status': 'planned'}
                        for spec in experiment['jobs']]
        rng.shuffle(compilations)
        records = []
        for repeat in range(repeats):
            order = list(compilations)
            rng.shuffle(order)
            for spec in order:
                key = spec['key'] + f'_r{repeat + 1}'
                records.append({'key': key, 'compile_key': spec['key'], 'case_id': spec['case_id'],
                                'variant': spec['variant'], 'repeat': repeat + 1,
                                'status': 'planned', 'profile_file': key + '_profile.json',
                                'energy': {'status': 'unavailable'}})
        result = {'schema_version': 2, 'source': 'qualcomm_ai_hub_sweep', 'status': 'running',
                  'started_utc': utc_now(), 'client_version': str(hub.__version__),
                  'git_commit': source_commit, 'git_dirty': source_dirty,
                  'device': {'name': target.name, 'os': target.os, 'attributes': target.attributes},
                  'manifest_sha256': manifest_hash, 'plan': experiment, 'repeats': repeats,
                  'order_seed': manifest.get('seed', 1729),
                  'order': 'Seeded shuffled submission order within repeats; cloud scheduling is uncontrolled',
                  'compilations': compilations, 'records': records}
        save_result(result_path, result)
    lookup = {item['case']['id']: item for item in manifest['cases']}
    compiles = {r['key']: r for r in result['compilations']}
    hub.set_verbose(False)
    while True:
        for record in result['compilations'] + result['records']:
            if record['status'] != 'submitted':
                continue
            job = hub.get_job(record['job_id'])
            status = job.get_status()
            if not status.finished:
                continue
            if not status.success:
                record['status'] = 'failed'
                record['completed_utc'] = utc_now()
            elif 'compile_key' not in record:
                record['status'] = 'compiled'
                record['completed_utc'] = utc_now()
            else:
                # A interrupted download is safe to repeat: the remote job ID is retained.
                collect_inference(job, record, lookup[record['case_id']], manifest_path, root)
            save_result(result_path, result)
            print(f"{record['status']} {record['key']} {record['job_id']}", flush=True)
        for record in result['records']:
            if record['status'] == 'planned' and compiles[record['compile_key']]['status'] == 'failed':
                record['status'] = 'compile_failed'
        active = sum(r['status'] == 'submitted' for r in result['compilations'] + result['records'])
        # Finish compilation first so inference submission follows the shuffled
        # repeat schedule rather than the varying compile completion order.
        if any(r['status'] in ('planned', 'submitted') for r in result['compilations']):
            eligible = [r for r in result['compilations'] if r['status'] == 'planned']
        else:
            eligible = [r for r in result['records'] if r['status'] == 'planned' and
                        compiles[r['compile_key']]['status'] == 'compiled']
        for record in eligible[:max(0, max_active - active)]:
            item = lookup[record['case_id']]
            record.update(status='submitting', submitted_utc=utc_now(),
                          job_name=root.name + '_' + record['key'],
                          input_weight_sha256=item['input_weight_sha256'])
            save_result(result_path, result)
            try:
                if 'compile_key' not in record:
                    job = hub.submit_compile_job(model=str(manifest_path.parent / item['model']),
                          device=target, name=record['job_name'], options=record['compile_options'], retry=False)
                else:
                    compiled = compiles[record['compile_key']]
                    model = hub.get_job(compiled['job_id']).get_target_model()
                    if model is None:
                        raise RuntimeError('Successful compile has no target model')
                    with np.load(manifest_path.parent / item['arrays']) as arrays:
                        job = hub.submit_inference_job(model=model, device=target, inputs={'x': [arrays['x']]},
                              name=record['job_name'], options=compiled['inference_options'], profile=True, retry=False)
                    record['compile_job_id'] = compiled['job_id']
                record.update(status='submitted', job_id=job.job_id)
            except Exception as exc:
                record.update(status='submission_unknown', error_type=type(exc).__name__)
                save_result(result_path, result)
                raise RuntimeError('Submission uncertain; inspect saved job name before retrying') from None
            save_result(result_path, result)
            print(f"submitted {record['key']} {record['job_id']}", flush=True)
        pending = [r for r in result['compilations'] + result['records'] if r['status'] in ('planned', 'submitted')]
        if not pending:
            result.update(status='complete' if all(r['status'] == 'validated' for r in result['records'])
                          else 'complete_with_failures', completed_utc=utc_now())
            save_result(result_path, result)
            return result
        save_result(result_path, result)
        time.sleep(poll_seconds)
