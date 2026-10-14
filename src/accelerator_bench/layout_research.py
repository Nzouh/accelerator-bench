"""Offline teaching experiments and figures built from retained measurements."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np
import torch
from torch.nn import functional as F

from .comparison import aggregate_sweep, check_layout_settings
from .report import save_figure

COLORS = ('#087e8b', '#bd4b22')
BASE = 'results/hub-s24-sweep-2026-10-13'
NHWC = 'results/hub-s24-nhwc-2026-10-13'
PREPARATION = 'results/hub-s24-preparation-2026-10-14'
VISION = 'results/hub-s24-vision-layout-2026-10-14'
VISION_CASES = ('model_nchw', 'model_nhwc', 'image_nchw_first', 'image_nhwc_first')


def conv_from_definition(x, weight):
    """NCHW, stride 1, valid padding, no bias; deliberately no tensor convolution."""
    x, weight = np.asarray(x, dtype=np.float64), np.asarray(weight, dtype=np.float64)
    n, c, h, w = x.shape
    out_c, in_c, kh, kw = weight.shape
    if c != in_c or h < kh or w < kw:
        raise ValueError('Incompatible convolution shapes')
    output = np.zeros((n, out_c, h-kh+1, w-kw+1), dtype=np.float64)
    for b in range(n):
        for o in range(out_c):
            for y in range(h-kh+1):
                for z in range(w-kw+1):
                    for i in range(c):
                        for ky in range(kh):
                            for kx in range(kw):
                                output[b, o, y, z] += x[b, i, y+ky, z+kx] * weight[o, i, ky, kx]
    return output


def hello_layout():
    x = torch.tensor([[[[0., 1., 2.], [3., 4., 5.]],
                       [[10., 11., 12.], [13., 14., 15.]]]], dtype=torch.float64)
    weight = torch.tensor([[[[2.]], [[1.]]]], dtype=torch.float64)
    view = x.permute(0, 2, 3, 1)
    packed = view.contiguous()
    channels_last = x.contiguous(memory_format=torch.channels_last)
    expected = conv_from_definition(x.numpy(), weight.numpy())
    normal = F.conv2d(x, weight)
    recovered = F.conv2d(packed.permute(0, 3, 1, 2), weight)
    wrong = F.conv2d(x.reshape(1, 2, 3, 2).permute(0, 3, 1, 2), weight)
    np.testing.assert_array_equal(normal.numpy(), expected)
    np.testing.assert_array_equal(recovered.numpy(), expected)
    assert not torch.equal(wrong, normal)
    assert view.data_ptr() == x.data_ptr() and packed.data_ptr() != x.data_ptr()
    assert channels_last.shape == x.shape
    assert torch.equal(F.conv2d(channels_last, weight), normal)
    return {'nchw_storage': x.flatten().tolist(), 'nhwc_storage': packed.flatten().tolist(),
        'output': expected[0, 0].tolist(), 'wrong_reshape_output': wrong[0, 0].tolist(),
        'max_error': float(np.max(np.abs(expected-recovered.numpy()))),
        'view_shares_storage': True, 'packing_copies_storage': True,
        'representations': [
            {'name': name, 'shape': list(t.shape), 'strides': list(t.stride())}
            for name, t in [('NCHW contiguous', x), ('NHWC view', view),
                            ('NHWC contiguous', packed), ('PyTorch channels_last', channels_last)]]}


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'axes.titleweight': 'bold'})


def finish(fig, output, name):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output / name)
    plt.close(fig)
    return output / f'{name}.png'


def plot_storage(output):
    style()
    data = hello_layout()
    fig, ax = plt.subplots(figsize=(10, 3.7))
    ax.set_xlim(-2.1, 12.1)
    ax.set_ylim(-.5, 3.5)
    ax.axis('off')
    for row, (label, values) in enumerate([('NCHW', data['nchw_storage']), ('NHWC', data['nhwc_storage'])]):
        y = 2.2 - row * 1.5
        ax.text(-.25, y+.4, label, ha='right', va='center', weight='bold', fontsize=12)
        for i, value in enumerate(values):
            ax.add_patch(Rectangle((i, y), .94, .8, facecolor=COLORS[int(value >= 10)], alpha=.85))
            ax.text(i+.47, y+.4, str(int(value)), ha='center', va='center', color='white', weight='bold')
    fig.suptitle('Same values, different storage', x=.04, ha='left', fontsize=18, weight='bold')
    fig.text(.14, .08, 'Channel 0', color=COLORS[0], weight='bold')
    fig.text(.31, .08, 'Channel 1', color=COLORS[1], weight='bold')
    fig.text(.57, .08, 'Memory order →', color='#475467')
    fig.tight_layout(rect=[0, .10, 1, .92])
    return finish(fig, output, 'storage_order')


def diagnostics(repo, source):
    root = Path(repo) / source
    data = json.loads((root/'htp-review.json').read_text())['diagnostics']
    for row in data:
        if row['status'] == 'available':
            assert hashlib.sha256((root/row['raw_file']).read_bytes()).hexdigest() == row['raw_sha256']
    return {(r['case_id'], r['variant']): r for r in data if r['status'] == 'available'}


def convolution_evidence(repo):
    result = []
    base = aggregate_sweep(Path(repo)/BASE)
    nhwc = aggregate_sweep(Path(repo)/NHWC)
    check_layout_settings(base['metadata']['plan'], nhwc['metadata']['plan'])
    for label, source, data in [('NCHW', BASE, base), ('NHWC', NHWC, nhwc)]:
        htp = diagnostics(repo, source)
        for row in data['rows']:
            if row['case']['op'] != 'conv_relu' or row['variant'] != 'optimized':
                continue
            if not row['complete']:
                raise ValueError('Incomplete convolution result')
            d = htp[(row['case']['id'], row['variant'])]
            result.append({**row, 'layout': label, 'size': row['case']['shape'][2],
                           'transpose_instances': d['transpose_instances'],
                           'htp_nodes': d['htp_nodes'], 'signature': d['implementation_signature']})
    for size in (16, 32, 64, 128):
        pair = [r for r in result if r['size'] == size]
        assert len(pair) == 2 and pair[0]['input_weight_sha256'] == pair[1]['input_weight_sha256']
    return result


def plot_convolution(repo, output):
    style()
    rows = convolution_evidence(repo)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    sizes = [16, 32, 64, 128]
    for j, label in enumerate(('NCHW', 'NHWC')):
        group = sorted([r for r in rows if r['layout'] == label], key=lambda r: r['size'])
        x = np.arange(4) + (j-.5)*.25
        med = np.array([r['median_us'] for r in group])
        err = [med-np.array([r['min_us'] for r in group]), np.array([r['max_us'] for r in group])-med]
        axes[0].errorbar(x, med, yerr=err, color=COLORS[j], marker='_', markersize=14,
                         capsize=4, linestyle='none', label=label, linewidth=1.5)
        for pos, row in zip(x, group):
            axes[0].scatter(pos+np.linspace(-.035, .035, row['n']),
                            [r['median_us'] for r in row['jobs']], color=COLORS[j], s=24, zorder=3)
        axes[1].bar(x, [r['transpose_instances'] for r in group], width=.22, color=COLORS[j])
    for ax in axes:
        ax.set_xticks(range(4), [str(s) for s in sizes])
        ax.set_xlabel('H = W')
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Latency (µs)')
    axes[1].set_ylabel('Transpose instances')
    axes[1].set_yticks([0, 4, 18])
    axes[0].set_title('QNN call', loc='left')
    axes[1].set_title('HTP profile', loc='left')
    fig.suptitle('Convolution: layout and execution', x=.055, ha='left', fontsize=18, weight='bold')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.5, .075),
               ncol=2, frameon=False)
    fig.text(.055, .025, 'Galaxy S24 · dots: 3 job medians; whiskers: range · HTP: first repeat', fontsize=9)
    fig.tight_layout(rect=[0, .16, 1, .94])
    return finish(fig, output, 'convolution_layout')


def transpose_details(repo, size=64):
    rows = [r for r in convolution_evidence(repo) if r['size'] == size]
    signatures = {r['layout']: dict(r['signature']) for r in rows}
    ops = sorted({op for sig in signatures.values() for op in sig
                  if 'Transpose' in op or 'ForceFormat' in op})
    return [{'operation': op, **{k: v.get(op, 0) for k, v in signatures.items()}} for op in ops]


def preparation_evidence(repo):
    data = aggregate_sweep(Path(repo)/PREPARATION)
    htp = diagnostics(repo, PREPARATION)
    assert len(data['rows']) == 3 and all(r['complete'] for r in data['rows'])
    return [{**r, 'htp_nodes': htp[(r['case']['id'], r['variant'])]['htp_nodes']} for r in data['rows']]


def plot_preparation(repo, output):
    style()
    rows = {r['variant']: r for r in preparation_evidence(repo)}
    order = ['optimized', 'prepare_fast', 'no_conv_relu_fold']
    labels = ['O3', 'O1', 'O3 / no fold']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True)
    for i, variant in enumerate(order):
        row = rows[variant]
        color = [COLORS[0], '#667085', COLORS[1]][i]
        axes[0].barh(i, row['median_us'], color=color)
        axes[0].errorbar(row['median_us'], i,
            xerr=[[row['median_us']-row['min_us']], [row['max_us']-row['median_us']]], color='black', capsize=4)
        axes[1].barh(i, row['htp_nodes'], color=color)
    axes[0].set_yticks(range(3), labels)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.set_xlim(left=0)
        ax.grid(axis='x', alpha=.15)
        ax.set_axisbelow(True)
    axes[0].set_xlabel('QNN latency (µs)')
    axes[1].set_xlabel('HTP nodes')
    fig.suptitle('Preparation: fewer nodes, more time', x=.055, ha='left', fontsize=18, weight='bold')
    fig.text(.055, .035, 'Galaxy S24 · 16×16 convolution · 3 jobs/setting · HTP: first repeat', fontsize=9)
    fig.tight_layout(rect=[0, .09, 1, .94])
    return finish(fig, output, 'preparation')


def plot_boundary(output):
    style()
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')
    boxes = [(0.1, 2.0, 'Decode + crop', '#f2f4f7'),
             (2.5, 2.0, 'Normalize + pack', '#f2f4f7'),
             (5.1, 2.0, 'Classifier', '#d7eeef'),
             (.1, .5, 'Decode + crop', '#f2f4f7'),
             (2.5, .5, 'Normalize + layout', '#d7eeef'),
             (5.1, .5, 'Classifier', '#d7eeef')]
    for x, y, label, color in boxes:
        ax.add_patch(Rectangle((x, y), 2.1, .6, facecolor=color, edgecolor='#667085', linewidth=.7))
        ax.text(x+1.05, y+.3, label, ha='center', va='center', fontsize=10)
    for y in (2.3, .8):
        for left, right in ((2.2, 2.5), (4.6, 5.1)):
            ax.add_patch(FancyArrowPatch((left, y), (right, y), arrowstyle='->', mutation_scale=12))
    ax.text(7.5, 2.3, 'Model-only pair', va='center', weight='bold')
    ax.text(7.5, .8, 'Image-graph pair', va='center', weight='bold')
    fig.suptitle('Where timing starts', x=.055, ha='left', fontsize=18, weight='bold')
    fig.text(.055, .06, 'Teal: inside QNN call · gray: excluded · both start from resident FP32 tensors', fontsize=10)
    fig.tight_layout(rect=[0, .13, 1, .90])
    return finish(fig, output, 'timing_boundary')


def vision_evidence(repo):
    root = Path(repo)/VISION
    data = aggregate_sweep(root)
    manifest = json.loads((root/'export-manifest.json').read_text())
    assert len({r['input_weight_sha256'] for r in data['rows']}) == 1
    assert len({(j['compile_options'], j['inference_options']) for j in data['metadata']['plan']['jobs']}) == 1
    htp = diagnostics(repo, VISION)
    records = data['metadata']['records']
    for record in records:
        if record.get('output_file'):
            assert hashlib.sha256((root/record['output_file']).read_bytes()).hexdigest() == record['output_sha256']
    rows = {r['case']['id']: r for r in data['rows']}
    for case, row in rows.items():
        d = htp.get((case, 'optimized'))
        row.update(transpose_instances=d['transpose_instances'] if d else None,
                   htp_nodes=d['htp_nodes'] if d else None)
    return {'rows': rows, 'metadata': data['metadata'], 'manifest': manifest, 'failures': data['failures']}


def plot_vision(repo, output):
    style()
    data = vision_evidence(repo)
    if any(not r['complete'] for r in data['rows'].values()):
        raise ValueError('Incomplete or rejected model measurements; inspect raw results')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), sharex=True)
    titles = ['Normalized model input', 'Cropped RGB input']
    labels = [('NCHW', 'NHWC'), ('Permute first', 'Normalize first')]
    for p, ax in enumerate(axes):
        for j in range(2):
            row = data['rows'][VISION_CASES[p*2+j]]
            ax.barh(j, row['median_us']/1000, color=COLORS[j], alpha=.85)
            ax.errorbar(row['median_us']/1000, j,
                xerr=[[(row['median_us']-row['min_us'])/1000], [(row['max_us']-row['median_us'])/1000]],
                color='#101828', capsize=4)
            ax.scatter([r['median_us']/1000 for r in row['jobs']], j+np.linspace(-.12, .12, row['n']),
                       c='white', edgecolor='#101828', s=24, zorder=3)
        ax.set_yticks([0, 1], labels[p])
        ax.invert_yaxis()
        ax.set_xlim(left=0)
        ax.set_xlabel('QNN latency (ms)')
        ax.set_title(titles[p], loc='left')
        ax.grid(axis='x', alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle('MobileNetV2: layout at two boundaries', x=.055, ha='left', fontsize=17, weight='bold')
    fig.text(.055, .035, 'Galaxy S24 · batch 1 · 224×224 · dots: 3 job medians; whiskers: range', fontsize=9)
    fig.tight_layout(rect=[0, .09, 1, .94])
    return finish(fig, output, 'vision_layout')


def vision_validation(repo):
    root = Path(repo)/VISION
    data = json.loads((root/'precision-review.json').read_text())
    assert hashlib.sha256((root/'results.json').read_bytes()).hexdigest() == data['results_sha256']
    assert hashlib.sha256((root/'export-manifest.json').read_bytes()).hexdigest() == data['export_manifest_sha256']
    records = {r['job_id']: r for r in json.loads((root/'results.json').read_text())['records']}
    for case in data['cases']:
        for path, expected in case['files_sha256'].items():
            assert hashlib.sha256((root/path).read_bytes()).hexdigest() == expected
        reference = np.load(root/'precision'/f"{case['case_id']}_fp64.npy")
        for job in case['jobs']:
            record = records[job['job_id']]
            output = root/record['output_file']
            assert hashlib.sha256(output.read_bytes()).hexdigest() == record['output_sha256']
            error = np.abs(np.load(output).astype(np.float64)-reference)
            allowance = .01 + .01*np.abs(reference)
            assert int((error > allowance).sum()) == job['logits_outside_tolerance']
            assert np.isclose(float((error/allowance).max()), job['max_tolerance_ratio'])
            assert record['status'] == job['status']
    if data['run_status'] not in ('complete', 'complete_with_failures'):
        raise ValueError('Model precision review is not final')
    if any(len(c['jobs']) != 3 for c in data['cases']):
        raise ValueError('Expected three checked jobs per model case')
    return data


def plot_vision_validation(repo, output):
    style()
    data = vision_validation(repo)
    cases = {c['case_id']: c for c in data['cases']}
    labels = ['NCHW', 'NHWC', 'Permute\nfirst', 'Normalize\nfirst']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, field, title, ylabel in zip(axes,
            ('max_tolerance_ratio', 'logits_outside_tolerance'),
            ('Largest error relative to allowance', 'Logits outside tolerance'),
            ('Error / allowance', 'Count / 1,000')):
        for i, case in enumerate(VISION_CASES):
            row = cases[case]
            values = np.array([r[field] for r in row['jobs']])
            med = float(np.median(values))
            ax.errorbar(i-.10, med, yerr=[[med-values.min()], [values.max()-med]],
                         fmt='o', capsize=4, color=COLORS[1], label='NPU' if i == 0 else None)
            ax.scatter(i+.10, row['cpu_fp16'][field], color='#667085', marker='D', s=40,
                       label='CPU FP16' if i == 0 else None)
        ax.set_title(title, loc='left', fontsize=11)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(4), labels)
        ax.set_ylim(bottom=0)
        ax.axvline(1.5, color='#98a2b3', linewidth=.8, linestyle=':')
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    axes[0].axhline(1, color='#101828', linewidth=1, linestyle='--')
    axes[0].text(2.95, 1.02, 'Acceptance limit', fontsize=8, ha='right')
    fig.suptitle('MobileNetV2: checking numerical agreement', x=.055, ha='left', fontsize=17, weight='bold')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.5, .075),
               ncol=2, frameon=False)
    fig.text(.055, .025, 'CPU FP64 reference · fixed tolerance · NPU: 3 jobs/case · CPU FP16 is a separate diagnostic', fontsize=9)
    fig.tight_layout(rect=[0, .17, 1, .94])
    return finish(fig, output, 'vision_validation')
