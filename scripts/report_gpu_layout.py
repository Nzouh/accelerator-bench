import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from accelerator_bench.comparison import axes_style, finish_plot


def report(source, output_dir):
    source, output = Path(source), Path(output_dir)
    data = json.loads(source.read_text())
    if data['status'] != 'complete' or data['source'] != 'local_cuda_layout_ablation':
        raise ValueError('Expected complete layout measurements')
    rows = data['records']
    assert len(rows) == 16 and all(r['correctness']['passed'] and r['graph_correctness']['passed'] for r in rows)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titleweight': 'bold'})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, dtype in zip(axes, ['float32', 'float16']):
        subset = [r for r in rows if r['dtype'] == dtype]
        for variant, label, color in [('layout_on', 'On', '#667085'), ('layout_off', 'Off', '#087e8b')]:
            selected = sorted([r for r in subset if r['variant'] == variant], key=lambda r: r['case']['shape'][2])
            x = [r['case']['shape'][2] for r in selected]
            ax.plot(x, [r['graph_cuda_ms']['median']*1000 for r in selected], 'o-', label=label, color=color)
            ax.fill_between(x, [r['graph_cuda_ms']['p10']*1000 for r in selected],
                            [r['graph_cuda_ms']['p90']*1000 for r in selected], color=color, alpha=.12)
        ax.set_title(dtype.replace('float', 'FP'), loc='left')
        ax.set_yscale('log')
        axes_style(ax, [r['case'] for r in subset], 'H = W')
    axes[0].set_ylabel('Latency (µs)')
    finish_plot(fig, axes, 'GPU layout optimization',
                'RTX 3060 · CUDA graph replay. Median; band: P10–P90 block means.', output / 'layout')
    indexed = {(r['case']['id'], r['dtype'], r['variant']): r for r in rows}
    ratios = {dtype: [r['graph_cuda_ms']['median']/indexed[r['case']['id'], dtype, 'layout_off']['graph_cuda_ms']['median']
                     for r in rows if r['dtype'] == dtype and r['variant'] == 'layout_on'] for dtype in ['float32', 'float16']}
    lines = ['# GPU layout regression', '',
        f"**FP32: {min(ratios['float32']):.2f}–{max(ratios['float32']):.2f}× faster with layout optimization off.** "
        f"FP16: {1/max(ratios['float16']):.2f}–{1/min(ratios['float16']):.2f}× slower.", '',
        '![Layout](layout.png)', '',
        'Only `torch.compile(..., options={"layout_optimization": False})` changes. All 16 configurations pass ordinary and captured-graph checks.', '',
        '| Precision | H=W | On (µs) | Off (µs) | On/off time |', '|---|---:|---:|---:|---:|']
    for row in rows:
        if row['variant'] != 'layout_on':
            continue
        off = indexed[row['case']['id'], row['dtype'], 'layout_off']
        on_us, off_us = row['graph_cuda_ms']['median']*1000, off['graph_cuda_ms']['median']*1000
        lines.append(f"| {row['dtype'].replace('float', 'FP')} | {row['case']['shape'][2]} | {on_us:.2f} | {off_us:.2f} | {on_us/off_us:.2f}× |")
    lines += ['', 'At FP32 16×16, the separate profile drops from **8 to 3 kernels/call**; the layout conversions disappear. At FP16 16×16, it rises from **4 to 5**.', '',
              'One AC desktop session; clocks unlocked. 25 blocks/configuration; five profiled calls. Reordering and kernel selection change together.', '',
              f"[Raw measurements and kernel names]({Path(os.path.relpath(source.resolve(), output.resolve())).as_posix()})"]
    (output / 'report.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('--output-dir', required=True)
    report(**vars(parser.parse_args()))
