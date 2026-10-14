"""Retain QHAS evidence and extract implementation-level diagnostics."""
import hashlib
import json
from pathlib import Path

from .runner import save_result


def summarize_htp(raw):
    data = raw['data']
    overall = data['htp_overall_summary']['data']
    if len(overall) != 1:
        raise ValueError('Expected one HTP core')
    core = overall[0]
    ops = data['htp_op_types']['data']
    instances = data['htp_op_instances']['data']
    return {
        **{key: core.get(key) for key in ('time_us', 'graph_execute_us', 'timeline_cycles',
            'total_dram_read', 'total_dram_write', 'peak_vtcm_alloc', 'htp_nodes', 'qnn_nodes')},
        'resources': core.get('htp_resources', {}).get('data', []),
        'implementation_signature': sorted([[r['op'], r['instances']] for r in ops]),
        'transpose_instances': sum(r['instances'] for r in ops if 'Transpose' in r['op']),
        'transpose_dominant_path_percent': sum(r.get('percent_dominant_path_cycles_htp_0', 0)
                                               for r in ops if 'Transpose' in r['op']),
        'cast_dominant_path_percent': sum(r.get('percent_dominant_path_cycles_htp_0', 0)
                                          for r in ops if 'Cast' in r['op']),
        'top_ops': sorted(ops, key=lambda r: r.get('percent_dominant_path_cycles_htp_0', 0), reverse=True)[:5],
        'reported_dtypes': sorted({r.get('dtype', 'unknown') for r in instances}),
        'scope': 'Separate optrace diagnostic; not steady-state benchmark latency',
    }


def collect_htp(source_dir, include_rejected=False):
    import qai_hub as hub
    hub.set_verbose(False)
    root = Path(source_dir).resolve()
    data = json.loads((root / 'results.json').read_text())
    diagnostics = []
    # First repeat only; selection is independent of measured performance.
    for record in data['records']:
        allowed = ('validated', 'rejected') if include_rejected else ('validated',)
        if record['repeat'] != 1 or record['status'] not in allowed:
            continue
        folder = root / 'htp' / record['key']
        paths = list(folder.glob('*_qhas_summary.json'))
        if not paths:
            job = hub.get_job(record['job_id'])
            paths = [Path(p) for p in job.download_artifacts_for_type(str(folder), hub.JobArtifactType.QHAS_SUMMARY)]
        if len(paths) != 1:
            diagnostics.append({'case_id': record['case_id'], 'variant': record['variant'], 'status': 'unavailable'})
            continue
        path = paths[0]
        diagnostics.append({'case_id': record['case_id'], 'variant': record['variant'],
            'job_id': record['job_id'], 'repeat': 1, 'status': 'available',
            'measurement_status': record['status'],
            'raw_file': path.relative_to(root).as_posix(),
            'raw_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            **summarize_htp(json.loads(path.read_text()))})
        print(f"HTP {record['case_id']} {record['variant']}", flush=True)
    save_result(root / 'htp-review.json', {'diagnostics': diagnostics})
    return diagnostics
