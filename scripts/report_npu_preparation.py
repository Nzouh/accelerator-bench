"""Report a repeated one-case experiment with explicit HTP preparation options."""
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from accelerator_bench.comparison import aggregate_sweep, VARIANTS
from accelerator_bench.report import save_figure


def report(source, output_dir):
    source, output = Path(source), Path(output_dir)
    data = aggregate_sweep(source)
    plan = data['metadata']['plan']
    rows = data['rows']
    if len({r['case']['id'] for r in rows}) != 1 or len(rows) != 3 or not all(r['complete'] for r in rows):
        raise ValueError('Expected three complete settings for one case')
    if not all('default_graph_htp_optimizations=' in j['inference_options'] for j in data['metadata']['plan']['jobs']):
        raise ValueError('Preparation options must be explicit at inference')
    diagnostics = {r['variant']: r for r in json.loads((source / 'htp-review.json').read_text())['diagnostics']
                   if r['status'] == 'available'}
    indexed = {r['variant']: r for r in rows}
    ordered = [indexed[v] for v in VARIANTS]
    labels = [VARIANTS[r['variant']][0] for r in ordered]
    colors = [VARIANTS[r['variant']][1] for r in ordered]
    med = np.array([r['median_us'] for r in ordered])
    lo, hi = np.array([r['min_us'] for r in ordered]), np.array([r['max_us'] for r in ordered])
    counts = [diagnostics[r['variant']]['htp_nodes'] for r in ordered]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    axes[0].barh(labels, med, xerr=[med-lo, hi-med], color=colors, capsize=4)
    axes[1].barh(labels, counts, color=colors)
    axes[0].set_xlabel('Latency (µs)')
    axes[1].set_xlabel('Nodes / profile')
    axes[0].set_title('QNN call', loc='left', fontweight='bold')
    axes[1].set_title('HTP graph', loc='left', fontweight='bold')
    axes[0].invert_yaxis()
    for ax in axes:
        ax.set_xlim(left=0)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle('NPU preparation settings', x=.04, ha='left', fontsize=18, fontweight='bold')
    fig.text(.04, .035, f"{plan['device']} · QAIRT {plan['qairt_version']} · {data['metadata']['repeats']} jobs/setting; whiskers: job-median range. HTP: repeat 1.", fontsize=9)
    fig.tight_layout(rect=[0, .10, 1, .98])
    output.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output / 'preparation')
    plt.close(fig)
    lines = ['# NPU preparation settings', '',
        f"**{sum(r['n'] for r in rows)} jobs validated.** `{rows[0]['case']['id']}`; FP32 I/O, requested FP16, QAIRT {plan['qairt_version']}.", '',
        f"HTP options are explicitly set on inference jobs. {data['metadata']['repeats']} separate jobs per setting; each passes correctness and NPU-placement checks.", '',
        '![Preparation settings](preparation.png)', '',
        '| Setting | Median (µs) | Job-median range (µs) | HTP nodes |', '|---|---:|---:|---:|']
    for row in ordered:
        lines.append(f"| {VARIANTS[row['variant']][0]} | {row['median_us']:.1f} | {row['min_us']:.1f}–{row['max_us']:.1f} | {diagnostics[row['variant']]['htp_nodes']} |")
    lines += ['', 'Single shape; uncontrolled cloud device assignment. HTP counts are diagnostic, not a predictor of latency.', '',
        '[Flag-placement audit](../preparation-audit/report.md) · '
        f"[Raw jobs]({Path(os.path.relpath((source/'results.json').resolve(), output.resolve())).as_posix()}) · "
        f"[HTP evidence]({Path(os.path.relpath((source/'htp-review.json').resolve(), output.resolve())).as_posix()})"]
    (output / 'report.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('--output-dir', required=True)
    report(**vars(parser.parse_args()))
