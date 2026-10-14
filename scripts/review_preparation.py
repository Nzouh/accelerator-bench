"""Compare preparation diagnostics; do not turn one-job audits into speed claims."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re

import qai_hub as hub

from accelerator_bench.htp import collect_htp
from accelerator_bench.runner import save_result


def review(original_source, audit_source, output_dir):
    original, audit, output = map(Path, (original_source, audit_source, output_dir))
    original_data = json.loads((original / 'results.json').read_text())
    audit_data = json.loads((audit / 'results.json').read_text())
    if audit_data['status'] != 'complete':
        raise ValueError('Audit must be complete')
    case_ids = {r['case_id'] for r in audit_data['records']}
    if len(case_ids) != 1:
        raise ValueError('Expected one diagnostic case')
    case_id = next(iter(case_ids))
    if any(original_data['plan'][k] != audit_data['plan'][k] for k in ('device', 'qairt_version')):
        raise ValueError('Device or runtime changed')
    specs = {(r['case_id'], r['variant']): r for r in original_data['plan']['jobs']}
    for spec in audit_data['plan']['jobs']:
        if spec['compile_options'] != specs[(case_id, spec['variant'])]['compile_options']:
            raise ValueError('Compile options changed')
    collect_htp(audit)
    hub.set_verbose(False)
    evidence = {}
    for label, root, data in [('legacy', original, original_data), ('explicit', audit, audit_data)]:
        htp = json.loads((root / 'htp-review.json').read_text())['diagnostics']
        evidence[label] = {}
        for record in data['records']:
            if record['case_id'] != case_id or record['repeat'] != 1:
                continue
            if record['status'] != 'validated':
                raise ValueError('Unvalidated diagnostic job')
            folder = (audit / 'logs' / label / record['variant']).resolve()
            paths = list(folder.glob('*.log'))
            if not paths:
                paths = [Path(p) for p in hub.get_job(record['job_id']).download_job_logs(str(folder))]
            excerpts, counts, logs = [], set(), []
            for path in paths:
                logs.append({'file': path.relative_to(audit.resolve()).as_posix(),
                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
                for number, line in enumerate(path.read_text().splitlines(), 1):
                    match = re.search(r'Applying (\d+) configurations to graph', line)
                    if match:
                        counts.add(int(match.group(1)))
                    if match or 'Loaded QNN backend from' in line:
                        excerpts.append({'file': logs[-1]['file'], 'line': number, 'text': line})
            diag = next(r for r in htp if r['case_id'] == case_id and r['variant'] == record['variant'])
            evidence[label][record['variant']] = {'job_id': record['job_id'], 'logs': logs,
                'input_weight_sha256': record['input_weight_sha256'],
                'graph_configuration_counts': sorted(counts), 'excerpts': excerpts,
                'htp_nodes': diag['htp_nodes'], 'implementation_signature': diag['implementation_signature']}
    output.mkdir(parents=True, exist_ok=True)
    lines = ['# HTP preparation audit', '',
        'Preparation flags are now explicit on inference jobs. The original sweep supplied them only at DLC conversion; effective HTP values were unverified.', '',
        f'Case: `{case_id}`. Same inputs, device and QAIRT. One job/request; no latency claim.', '',
        '| Request | Legacy graph configs | Explicit graph configs | Same HTP op counts |',
        '|---|---:|---:|---|']
    names = {'optimized': 'O3', 'prepare_fast': 'O1', 'no_conv_relu_fold': 'O3 / no fold'}
    for variant in names:
        before = evidence['legacy'][variant]
        after = evidence['explicit'][variant]
        if before['input_weight_sha256'] != after['input_weight_sha256']:
            raise ValueError('Input fingerprint changed')
        same = before['implementation_signature'] == after['implementation_signature']
        old = ', '.join(map(str, before['graph_configuration_counts'])) or 'unavailable'
        new = ', '.join(map(str, after['graph_configuration_counts'])) or 'unavailable'
        lines.append(f"| {names[variant]} | {old} | {new} | {'Yes' if same else 'No'} |")
    evidence_link = Path(os.path.relpath((audit / 'preparation-review.json').resolve(), output.resolve())).as_posix()
    lines += ['', 'Configuration counts show setup differences, not individual option values. HTP op counts do not establish binary identity.', '',
        '[Evidence and log excerpts](' + evidence_link + ') · '
        '[Preparation stages](https://workbench.aihub.qualcomm.com/docs/hub/api.html)']
    save_result(audit / 'preparation-review.json', evidence)
    (output / 'report.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('original_source')
    parser.add_argument('audit_source')
    parser.add_argument('--output-dir', required=True)
    review(**vars(parser.parse_args()))
