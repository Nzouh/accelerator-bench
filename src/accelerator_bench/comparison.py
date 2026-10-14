"""Compare recorded stacks while retaining their distinct timing boundaries."""
from collections import defaultdict
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np

from .hub_report import review_profile
from .report import save_figure
from .runner import save_result


OPS = [('matmul', 'Matmul', 'Rows M'), ('conv_relu', 'Conv + ReLU', 'H = W'),
       ('rmsnorm', 'RMSNorm', 'Rows')]
VARIANTS = {'optimized': ('O3', '#087e8b'), 'prepare_fast': ('O1', '#667085'),
            'no_conv_relu_fold': ('O3 / no fold', '#bd4b22')}


def case_size(case):
    return case['shape'][2 if case['op'] == 'conv_relu' else 0]


def case_label(case):
    names = {'matmul': 'Matmul M=', 'conv_relu': 'Conv H=W=', 'rmsnorm': 'RMSNorm rows='}
    return names[case['op']] + str(case_size(case))


def aggregate_sweep(source_dir):
    root = Path(source_dir)
    data = json.loads((root / 'results.json').read_text())
    if data['source'] != 'qualcomm_ai_hub_sweep' or data['status'] not in ('complete', 'complete_with_failures'):
        raise ValueError('Expected a completed NPU sweep')
    manifest = json.loads((root / 'export-manifest.json').read_text())
    cases = {item['case']['id']: item for item in manifest['cases']}
    groups, seen, job_ids, failures = defaultdict(list), set(), set(), []
    for record in data['records']:
        key = (record['case_id'], record['variant'], record['repeat'])
        if key in seen or (record.get('job_id') and record['job_id'] in job_ids):
            raise ValueError('Duplicate repeat or inference job')
        seen.add(key)
        if record.get('job_id'):
            job_ids.add(record['job_id'])
        item = cases[record['case_id']]
        group_key = key[:2]
        groups[group_key]  # Include failed-only groups in coverage.
        if record['status'] != 'validated' or not record.get('correctness', {}).get('passed'):
            failures.append({'key': record['key'], 'reason': record['status']})
            continue
        if record['input_weight_sha256'] != item['input_weight_sha256']:
            raise ValueError('NPU input fingerprint mismatch')
        profile_path = (root / record['profile_file']).resolve()
        if not profile_path.is_relative_to(root.resolve()):
            raise ValueError('Profile must remain inside the run directory')
        reviewed = review_profile(json.loads(profile_path.read_text()))
        if not reviewed['all_reported_entries_on_npu'] or not reviewed['sample_count']:
            failures.append({'key': record['key'], 'reason': 'invalid placement or timing samples'})
            continue
        groups[group_key].append({'repeat': record['repeat'], 'job_id': record['job_id'],
            'median_us': reviewed['all_samples_median_us'], 'sample_count': reviewed['sample_count'],
            'relative_l2_error': record['correctness']['relative_l2_error']})
    rows = []
    for (case_id, variant), jobs in groups.items():
        medians = [j['median_us'] for j in jobs]
        rows.append({'case': cases[case_id]['case'], 'variant': variant,
            'input_weight_sha256': cases[case_id]['input_weight_sha256'],
            'io_layout': manifest.get('io_layout', 'nchw'),
            'jobs': jobs, 'n': len(jobs), 'expected_n': data['repeats'],
            'complete': len(jobs) == data['repeats'],
            'median_us': float(np.median(medians)) if medians else None,
            'min_us': min(medians) if medians else None, 'max_us': max(medians) if medians else None})
    return {'metadata': data, 'rows': rows, 'failures': failures}


def gpu_matches(gpu, npu_rows):
    if gpu['source'] != 'local_cuda_measured' or gpu['status'] not in ('complete', 'complete_with_failures'):
        raise ValueError('Expected completed GPU measurements')
    rows = {r['case']['id']: r for r in gpu['records'] if r['dtype'] == 'float16' and r['variant'] == 'compiled'
            and r['status'] == 'ok' and r['correctness']['passed'] and r['graph_correctness']['passed']}
    for remote in npu_rows:
        local = rows.get(remote['case']['id'])
        if local is None or local['input_weight_sha256'] != remote['input_weight_sha256'] or local['case'] != remote['case']:
            raise ValueError('GPU/NPU workload mismatch')
    return rows


def check_layout_settings(base, layout):
    if (base['device'] != layout['device'] or base['qairt_version'] != layout['qairt_version']):
        raise ValueError('Layout ablation changed device or runtime')
    original = {(r['case_id'], r['variant']): r for r in base['jobs']}
    for spec in layout['jobs']:
        if spec != original.get((spec['case_id'], spec['variant'])):
            raise ValueError('Layout ablation changed compiler or inference options')


def axes_style(ax, cases, xlabel):
    ax.set_xscale('log', base=2)
    ticks = sorted({case_size(case) for case in cases})
    ax.set_xticks(ticks, [str(v) for v in ticks])
    ax.set_xlabel(xlabel)
    if ax.get_yscale() == 'log':
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(alpha=.16)
    ax.spines[['top', 'right']].set_visible(False)


def finish_plot(fig, axes, title, note, output):
    handles, labels = [], []
    for ax in np.asarray(axes).flat:
        h, labels_here = ax.get_legend_handles_labels()
        for handle, label in zip(h, labels_here):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.suptitle(title, x=.055, ha='left', fontsize=18, fontweight='bold')
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .065), ncol=3, frameon=False)
    fig.text(.055, .025, note, fontsize=9, color='#475467')
    fig.tight_layout(rect=[.025, .16, .99, .99])
    save_figure(fig, output)
    plt.close(fig)


def create_comparison(source_dir, gpu_source, output_dir, layout_source=None):
    root, output = Path(source_dir), Path(output_dir)
    remote = aggregate_sweep(root)
    gpu = json.loads(Path(gpu_source).read_text())
    local = gpu_matches(gpu, remote['rows'])
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titleweight': 'bold', 'figure.facecolor': 'white'})
    valid = [r for r in remote['rows'] if r['complete']]
    baseline = sorted([r for r in valid if r['variant'] == 'optimized'], key=lambda r: (r['case']['op'], case_size(r['case'])))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for ax, (op, name, xlabel) in zip(axes, OPS):
        rows = [r for r in baseline if r['case']['op'] == op]
        x = [case_size(r['case']) for r in rows]
        ax.plot(x, [local[r['case']['id']]['graph_cuda_ms']['median'] * 1000 for r in rows],
                'o--', color='#667085', label='RTX · CUDA graph')
        ax.plot(x, [local[r['case']['id']]['wall_ms']['median'] * 1000 for r in rows],
                's-', color='#bd4b22', label='RTX · Python loop')
        ax.plot(x, [r['median_us'] for r in rows], 'o-', color='#087e8b', label='Snapdragon · QNN call')
        ax.fill_between(x, [r['min_us'] for r in rows], [r['max_us'] for r in rows], color='#087e8b', alpha=.12)
        ax.set_title(name, loc='left')
        ax.set_yscale('log')
        axes_style(ax, [r['case'] for r in rows], xlabel)
    axes[0].set_ylabel('Latency (µs)')
    finish_plot(fig, axes, 'RTX 3060 Laptop vs Snapdragon 8 Gen 3',
                'FP16 requested; NPU FP32 I/O. Different timing boundaries. Band: range of NPU job medians.', output / 'latency')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for ax, (op, name, xlabel) in zip(axes, OPS):
        subset = [r for r in valid if r['case']['op'] == op]
        for variant, (label, color) in VARIANTS.items():
            rows = sorted([r for r in subset if r['variant'] == variant], key=lambda r: case_size(r['case']))
            if not rows:
                continue
            x = [case_size(r['case']) for r in rows]
            ax.plot(x, [r['median_us'] for r in rows], 'o-', color=color, label=label)
            ax.fill_between(x, [r['min_us'] for r in rows], [r['max_us'] for r in rows], color=color, alpha=.12)
        ax.set_title(name, loc='left')
        ax.set_yscale('log')
        axes_style(ax, [r['case'] for r in subset], xlabel)
    axes[0].set_ylabel('Latency (µs)')
    finish_plot(fig, axes, 'NPU compiler settings',
                f"Galaxy S24 · QAIRT {remote['metadata']['plan']['qairt_version']} · {remote['metadata']['repeats']} jobs/setting; band: job-median range.", output / 'compiler_settings')
    table = []
    for row in baseline:
        r = local[row['case']['id']]
        table.append({'case': row['case']['id'], 'gpu_graph_us': r['graph_cuda_ms']['median'] * 1000,
                      'gpu_python_us': r['wall_ms']['median'] * 1000, 'npu_qnn_us': row['median_us'],
                      'npu_min_job_median_us': row['min_us'], 'npu_max_job_median_us': row['max_us'], 'npu_jobs': row['n']})
    with (output / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['case', 'gpu_graph_us', 'gpu_python_us', 'npu_qnn_us',
            'npu_min_job_median_us', 'npu_max_job_median_us', 'npu_jobs'])
        writer.writeheader()
        writer.writerows(table)
    complete = sum(r['complete'] for r in remote['rows'])
    lines = ['# RTX 3060 Laptop vs Snapdragon 8 Gen 3', '',
        f"**{complete}/{len(remote['rows'])} NPU configurations validated**, {remote['metadata']['repeats']} jobs each. Same {len(local)} shapes and input/weight hashes.", '',
        f"GPU: RTX 3060 Laptop, PyTorch compiled FP16. NPU: {remote['metadata']['plan']['device']}, QAIRT {remote['metadata']['plan']['qairt_version']}, O3, requested FP16, FP32 I/O.", '',
        'Timing boundaries differ: CUDA graph, block-timed Python dispatch, and QNN framework call. No chip-only speedup ratio or NPU energy estimate.', '',
        '![Latency](latency.png)', '', '| Case | GPU graph (µs) | GPU Python loop (µs) | NPU QNN (µs) |', '|---|---:|---:|---:|']
    for r in table:
        lines.append(f"| {case_label(local[r['case']]['case'])} | {r['gpu_graph_us']:.2f} | {r['gpu_python_us']:.2f} | {r['npu_qnn_us']:.1f} |")
    lines += ['', '## Compiler settings', '', '![Compiler settings](compiler_settings.png)', '',
              'NPU points: median of job medians. Bands: minimum–maximum job medians, not confidence intervals. Internal fusion must be checked separately.', '']
    lines += ['## Numerical error', '', '| Operation | GPU max relative L2 | NPU max relative L2 |', '|---|---:|---:|']
    for op, name, _ in OPS:
        gpu_errors = [r[field]['relative_l2_error'] for r in local.values()
                      if r['case']['op'] == op for field in ('correctness', 'graph_correctness')]
        npu_errors = [j['relative_l2_error'] for r in baseline if r['case']['op'] == op for j in r['jobs']]
        if gpu_errors and npu_errors:
            lines.append(f"| {name} | {max(gpu_errors):.3e} | {max(npu_errors):.3e} |")
    lines += ['', 'CPU FP64 reference; NPU O3 and GPU compiled FP16.', '']
    layout_rows = []
    if layout_source:
        layout = aggregate_sweep(layout_source)
        check_layout_settings(remote['metadata']['plan'], layout['metadata']['plan'])
        originals = {r['case']['id']: r for r in baseline}
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
        for row in sorted(layout['rows'], key=lambda r: case_size(r['case'])):
            if not row['complete']:
                continue
            original = originals[row['case']['id']]
            if row['input_weight_sha256'] != original['input_weight_sha256'] or row['io_layout'] != 'nhwc':
                raise ValueError('Layout ablation changed the logical workload')
            layout_rows.append({'case_id': row['case']['id'], 'size': case_size(row['case']),
                'nchw': original, 'nhwc': row, 'speedup': original['median_us'] / row['median_us']})
        diag = {}
        for label, folder in [('NCHW', root), ('NHWC', Path(layout_source))]:
            path = folder / 'htp-review.json'
            diag[label] = {r['case_id']: r for r in json.loads(path.read_text())['diagnostics']
                           if r['variant'] == 'optimized' and r['status'] == 'available'} if path.exists() else {}
        for label, color in [('NCHW', '#667085'), ('NHWC', '#087e8b')]:
            x = [r['size'] for r in layout_rows]
            rows = [r[label.lower()] for r in layout_rows]
            axes[0].plot(x, [r['median_us'] for r in rows], 'o-', label=label, color=color)
            axes[0].fill_between(x, [r['min_us'] for r in rows], [r['max_us'] for r in rows], color=color, alpha=.12)
            profiled = [r for r in layout_rows if r['case_id'] in diag[label]]
            axes[1].plot([r['size'] for r in profiled],
                [diag[label][r['case_id']]['transpose_instances'] for r in profiled], 'o-', label=label, color=color)
        axes[0].set_title('QNN call', loc='left')
        axes[0].set_ylabel('Latency (µs)')
        axes[0].set_ylim(bottom=0)
        axes[1].set_title('HTP transposes', loc='left')
        axes[1].set_ylabel('Instances / profile')
        axes[1].set_ylim(bottom=0)
        for ax in axes:
            axes_style(ax, [r['nhwc']['case'] for r in layout_rows], 'H = W')
        finish_plot(fig, axes, 'NPU convolution layout',
                    f"Jobs/shape: NCHW {remote['metadata']['repeats']}, NHWC {layout['metadata']['repeats']}. "
                    'Band: job-median range. HTP: repeat 1.', output / 'layout')
        lines += ['## Layout ablation', '', '![Layout](layout.png)', '',
                  'Same convolution and weights; only the I/O layout changes. Input/output reordering is outside timing.', '',
                  'Separate sweeps; cloud device allocation and execution order were uncontrolled.', '',
                  '| H=W | NCHW (µs) | NHWC (µs) | Speedup |', '|---:|---:|---:|---:|']
        for r in layout_rows:
            lines.append(f"| {r['size']} | {r['nchw']['median_us']:.1f} | {r['nhwc']['median_us']:.1f} | {r['speedup']:.2f}× |")
    rel = lambda path: Path(os.path.relpath(Path(path).resolve(), output.resolve())).as_posix()
    htp_path = root / 'htp-review.json'
    if htp_path.exists():
        diagnostics = {(r['case_id'], r['variant']): r
                       for r in json.loads(htp_path.read_text())['diagnostics'] if r['status'] == 'available'}
        matches = []
        for other, label in [('prepare_fast', 'O3/O1'), ('no_conv_relu_fold', 'O3/no-fold')]:
            pairs = [(r, diagnostics[(case_id, other)])
                     for (case_id, variant), r in diagnostics.items() if variant == 'optimized'
                     and (case_id, other) in diagnostics]
            if pairs:
                unchanged = sum(a['implementation_signature'] == b['implementation_signature'] for a, b in pairs)
                matches.append(f'{label}: {unchanged}/{len(pairs)} shapes')
        if matches:
            lines += ['', '**Identical HTP op-type counts:** ' + '; '.join(matches) +
                      '. Counts alone do not establish identical binaries or fusion.']
    lines += ['', '## Evidence', '', '- [Comparison table](comparison.csv)',
              '- [Reviewed measurements](review.json)', f'- [Raw NPU jobs]({rel(root / "results.json")})',
              f'- [GPU measurements]({rel(gpu_source)})',
              '- [Timing definition](https://workbench.aihub.qualcomm.com/docs/hub/howitworks-inference.html)',
              '- [HTP diagnostics](https://docs.qualcomm.com/doc/80-63442-10/topic/htp_backend.html#qnn-htp-analysis-summary-qhas)']
    if htp_path.exists():
        lines.append(f'- [NCHW HTP evidence]({rel(htp_path)})')
    if layout_source:
        lines.append(f'- [NHWC jobs]({rel(Path(layout_source) / "results.json")})')
        if (Path(layout_source) / 'htp-review.json').exists():
            lines.append(f'- [NHWC HTP evidence]({rel(Path(layout_source) / "htp-review.json")})')
    if remote['failures']:
        lines += ['', '## Excluded jobs', ''] + [f"- {r['key']}: {r['reason']}" for r in remote['failures']]
    save_result(output / 'review.json', {'npu': remote['rows'], 'failures': remote['failures'], 'layout': layout_rows})
    (output / 'report.md').write_text('\n'.join(lines).rstrip() + '\n')
    return output / 'report.md'
