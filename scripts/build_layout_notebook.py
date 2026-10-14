"""Build and execute the research notebook without CUDA or remote submissions."""
import argparse
import json
from pathlib import Path
import sys

import nbformat as nbf
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager


def build(with_vision=False):
    root = Path(__file__).resolve().parents[1]
    destination = root / 'research/layout-study'
    cells = []
    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text.strip()))
    def code(text):
        cells.append(nbf.v4.new_code_cell(text.strip()))

    md('''# Tensor layout on Snapdragon
**Research notebook · Nabil Zouhari**

Question: does changing storage order reduce the work a neural-network compiler
generates, and does the benefit survive beyond a single convolution?

This notebook follows the weather study's sequence: a small example from the
definitions, an independent correctness check, measured execution, and a model
application. It reuses saved device results. Running it submits no AI Hub jobs.
The [experiment plan](PLAN.md) was committed before the new model runs.''')
    code('''from pathlib import Path
import sys, json, importlib.metadata
import numpy as np
from IPython.display import Image, Markdown, display
from accelerator_bench.layout_research import (
    hello_layout, plot_storage, convolution_evidence, plot_convolution,
    transpose_details, preparation_evidence, plot_preparation, plot_boundary,
    vision_evidence, plot_vision, vision_validation, plot_vision_validation,
    diagnostics, VISION, VISION_CASES,
)

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / 'configs/rtx3060.json').exists())
FIGURES = ROOT / 'research/layout-study/figures'
def table(headers, rows):
    text = '| ' + ' | '.join(headers) + ' |\\n'
    text += '|' + '|'.join(['---'] * len(headers)) + '|\\n'
    text += '\\n'.join('| ' + ' | '.join(str(x) for x in row) + ' |' for row in rows)
    display(Markdown(text))

print('Python', sys.version.split()[0])
for package in ('torch', 'numpy', 'matplotlib'):
    print(package, importlib.metadata.version(package))''')
    md('''## 1. One operation, two storage orders

N, C, H and W mean batch, channels, height and width. For one 2-channel, 2×3
image, NCHW stores each channel's six values together. NHWC puts the two channels
of each pixel next to one another. The colors below identify channels; reading
left to right gives physical memory order. The numbers and their logical pixel
positions are unchanged.''')
    code('''toy = hello_layout()
display(Image(filename=str(plot_storage(FIGURES))))''')
    md('''A 1×1 convolution with weights `[2, 1]` computes
`output[h,w] = 2 × channel0[h,w] + channel1[h,w]`.
The top-left pixel therefore gives `2 × 0 + 10 = 10` in either representation.
An explicit-loop implementation is the reference; PyTorch is checked against it.
This is a correctness exercise, not a speed measurement.''')
    code('''print('Convolution output:', toy['output'])
print('Maximum layout-path error:', toy['max_error'])
table(['Representation', 'Shape', 'Strides (elements)'],
      [[r['name'], str(r['shape']), str(r['strides'])] for r in toy['representations']])
print('Permuted view shares storage:', toy['view_shares_storage'])
print('Contiguous NHWC requires a new allocation here:', toy['packing_copies_storage'])''')
    md('''Strides tell us how far to move in memory when an index increases by one.
`permute` can change shape and strides while sharing storage. Materializing the
new order with `contiguous()` copies values in this example. A compiler may fuse
or eliminate such a conversion; the source expression alone cannot establish its
runtime cost.

PyTorch's `channels_last` retains the **logical NCHW shape** and changes strides.
An NHWC interface changes the axis order too. These are related but distinct
representations. [PyTorch memory-format explanation](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html).

Simply calling `reshape` would assign the existing values to the wrong axes.
The deliberately incorrect path below demonstrates why output validation matters.''')
    code("print('Wrong reshape output:', toy['wrong_reshape_output'])")
    md('''## 2. What changed on the NPU?

The measured convolution uses batch 1, 32 input channels, 64 output channels,
a 3×3 kernel and ReLU. Only the external layout changes. Logical inputs, weights,
device type, QAIRT version and requested options match.

**Left:** each dot is one AI Hub job's median QNN-call latency. The short central
mark is the median across three jobs; whiskers show their range.
**Right:** transpose instances in the separate first-repeat HTP diagnostic.
Both panels vary image size. Diagnostic counts are not timing samples.''')
    code('''conv = convolution_evidence(ROOT)
display(Image(filename=str(plot_convolution(ROOT, FIGURES))))
by_size = {s: {r['layout']: r for r in conv if r['size'] == s} for s in (16, 32, 64, 128)}
table(['H = W', 'NCHW (µs)', 'NHWC (µs)', 'Latency change'],
      [[s, p['NCHW']['median_us'], p['NHWC']['median_us'],
        f"{100*(p['NHWC']['median_us']/p['NCHW']['median_us']-1):+.1f}%"] for s,p in by_size.items()])''')
    md('''At 64×64, median latency falls from **207 to 142 µs (31.4%)**, while
transpose instances fall from **18 to zero**. At 32×32, the medians are 103.5 and
109 µs and the ranges overlap. Removing transposes did not guarantee a measurable
speedup at every size.

The trace supports a change in layout handling, but it does not assign the whole
65 µs difference to transposes. Other generated operations change too:''')
    code('''table(['HTP operation at 64×64', 'NCHW instances', 'NHWC instances'],
      [[r['operation'], r['NCHW'], r['NHWC']] for r in transpose_details(ROOT)])''')
    md('''`ForceFormat_Crouton` instances remain and increase in this trace. Therefore
“no transpose operations” does not mean “no data-format work.” Instrumented HTP
times cannot be added to or subtracted from the separately measured QNN latency.

These were separate sweeps with uncontrolled cloud allocation and clocks. Input
and output reordering were outside timing. The result describes a pipeline that
can supply and consume the chosen layout, not a complete phone application.
The [raw comparison](../../reports/snapdragon-comparison/report.md) retains all cases.''')
    md('''## 3. Verify the setting before interpreting the result

The first sweep supplied HTP flags only during DLC conversion. Their application
when the device prepared the graph was unverified. The correction explicitly sets
them on inference jobs; [runtime logs and graph evidence](../../reports/preparation-audit/report.md)
record the audit. Qualcomm documents [DLC preparation and the relevant options](https://workbench.aihub.qualcomm.com/docs/hub/api.html).

The following corrected experiment keeps the 16×16 convolution fixed.
The left panel measures QNN-call latency; the right counts HTP nodes in the
first repeat. O1 and O3 are preparation settings, not graph-size guarantees.''')
    code('''display(Image(filename=str(plot_preparation(ROOT, FIGURES))))''')
    md('''O1 produced **24 nodes instead of 47**, yet median latency increased from
**80 to 91 µs**. Node count treats very different operations as equal, so it cannot
stand in for runtime. The folding-disabled request produced 47 nodes and a timing
range overlapping O3. These measurements do not establish a folding benefit.

This is the same kind of check used in the weather research: validate what the
implementation actually does before interpreting an option or a default.''')
    md('''## 4. A pretrained model and a shared image boundary

The follow-up uses **MobileNetV2 with IMAGENET1K_V2 weights**, batch 1, 224×224.
The image below is the actual center crop used for every case: NASA's photograph
of Eileen Collins, distributed as the public-domain scikit-image astronaut image.
It provides a reproducible natural-image input. One image cannot measure model accuracy.
[Image credit](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut) ·
[Weight transforms](https://docs.pytorch.org/vision/0.26/models/generated/torchvision.models.mobilenet_v2.html).''')
    code("display(Image(filename=str(ROOT/'research/layout-study/assets/input-crop.png'), width=224))")
    md('''The crop above fixes which pixels enter the experiment. It is resized to 232
and center-cropped to 224 before any timing. The two comparisons below answer
different questions; compare within each pair.''')
    code("display(Image(filename=str(plot_boundary(FIGURES))))")
    md('''**Top row:** normalized inputs are supplied as NCHW or NHWC. The complete
classifier is timed, including layout operations that survive compilation.
External normalization and packing are excluded.

**Bottom row:** both variants receive the same cropped FP32 RGB NHWC array,
with values in 0–255. One permutes then normalizes; the other normalizes then
permutes. Both expressions are inside the timed graph. The compiler may make the
two implementations equivalent.

Teal boxes are timed; gray boxes are excluded. Decode, resize, host casting,
transfers and camera capture remain excluded. This measures an image-to-logits
graph, **not phone end-to-end latency**. The image graph may use different
arithmetic after FP16 lowering, so its outputs are checked independently too.''')
    vision_passed = False
    if with_vision:
        recorded = json.loads((root/'results/hub-s24-vision-layout-2026-10-14/results.json').read_text())
        if recorded['status'] not in ('complete', 'complete_with_failures'):
            raise ValueError('Wait for the bounded model experiment to finish')
        vision_passed = recorded['status'] == 'complete'
    if with_vision and vision_passed:
        code('''vision = vision_evidence(ROOT)
display(Image(filename=str(plot_vision(ROOT, FIGURES))))
table(['Case', 'Median (ms)', 'Job range (ms)', 'HTP transposes'],
      [[c, f"{vision['rows'][c]['median_us']/1000:.3f}",
        f"{vision['rows'][c]['min_us']/1000:.3f}–{vision['rows'][c]['max_us']/1000:.3f}",
        vision['rows'][c]['transpose_instances']] for c in VISION_CASES])''')
        code('''for title, (a, b) in zip(('Model interface', 'Shared image interface'),
                                  (VISION_CASES[:2], VISION_CASES[2:])):
    left, right = vision['rows'][a], vision['rows'][b]
    change = 100 * (right['median_us']/left['median_us'] - 1)
    overlap = max(left['min_us'], right['min_us']) <= min(left['max_us'], right['max_us'])
    display(Markdown(f"**{title}:** the second case changes median latency by {change:+.1f}%. "
                     + ("The job ranges overlap; these repeats do not establish a clear advantage."
                        if overlap else "The observed job ranges do not overlap. This remains a three-job, one-image measurement.")))
errors = [r['correctness']['relative_l2_error'] for r in vision['metadata']['records']]
display(Markdown(f"All **{len(errors)} jobs** pass the numerical gate and report NPU placement. "
                 f"Maximum relative L2 logit error: **{max(errors):.3g}**. "
                 "Each job preserves the CPU reference's top prediction. This is numerical agreement on this image, not ImageNet accuracy."))''')
        md('''The whole-model figure uses the same three-job summary as the convolution
figure. Read each panel separately: its bars begin from a different input boundary.
Small layout effects can be hidden by the larger network and remote-run variation.
Any surviving difference here applies to this model, runtime and input; it does
not transfer the convolution's 31% result to arbitrary models.

[Raw model jobs and checks](../../results/hub-s24-vision-layout-2026-10-14/results.json) ·
[Model exports and hashes](../../results/hub-s24-vision-layout-2026-10-14/export-manifest.json) ·
[HTP evidence](../../results/hub-s24-vision-layout-2026-10-14/htp-review.json).''')
    elif with_vision:
        md('''### The numerical gate failed

The model test exposed a limit before it could support a speedup claim. The
predeclared rule checks every logit against the CPU FP64 reference:

`abs(output − reference) ≤ 0.01 + 0.01 × abs(reference)`.

**Left:** the largest error divided by its allowed error. A value above the dashed
line at 1 fails. **Right:** how many of the 1,000 logits exceed their allowance.
Orange shows the NPU's three-job median and range. Gray is an additional CPU
FP16 calculation made after the rejection to investigate reduced-precision error.
It is a diagnostic comparison, not a replacement acceptance rule.''')
        code('''validation = vision_validation(ROOT)
display(Image(filename=str(plot_vision_validation(ROOT, FIGURES))))
table(['Case', 'Accepted jobs', 'Max relative L2', 'Top-1 agreement'],
      [[r['case_id'], f"{sum(j['status'] == 'validated' for j in r['jobs'])}/{len(r['jobs'])}",
        f"{max(j['relative_l2_error'] for j in r['jobs']):.3%}",
        f"{sum(j['top1_equal'] for j in r['jobs'])}/{len(r['jobs'])}"] for r in validation['cases']])''')
        md('''The local CPU FP16 path also crosses the fixed tolerance. This shows
that the threshold can reject ordinary reduced-precision execution of the complete
network; it does not identify a Qualcomm-specific defect. CPU FP16 is not an
emulator of the NPU's kernels or accumulation strategy.

Preserving the top prediction does not satisfy a rule about all logits, nor does
it establish dataset-level accuracy. The original acceptance rule stays unchanged.
These runs remain rejected, their raw timings remain available for audit, and
**no validated whole-model speedup is claimed**. The earlier convolution results
retain their separate, passing checks.

The application question therefore remains open under this numerical contract.
A follow-up should choose a task-level accuracy requirement on a representative
dataset before measuring again, and assess how FP16 error accumulates through the
network. It should not choose a new threshold merely to accept these outputs.

[Raw jobs and unchanged checks](../../results/hub-s24-vision-layout-2026-10-14/results.json) ·
[Precision diagnostic](../../results/hub-s24-vision-layout-2026-10-14/precision-review.json) ·
[Model and image provenance](../../results/hub-s24-vision-layout-2026-10-14/export-manifest.json).
HTP diagnostics for rejected runs are retained with their rejection status.
They still describe generated operations, even though the outputs failed the
acceptance rule:''')
        code('''htp = diagnostics(ROOT, VISION)
table(['Case', 'HTP nodes', 'Transpose instances', 'Output status'],
      [[c, htp[(c, 'optimized')]['htp_nodes'], htp[(c, 'optimized')]['transpose_instances'],
        htp[(c, 'optimized')]['measurement_status']] for c in VISION_CASES])
same_signature = (htp[('image_nchw_first', 'optimized')]['implementation_signature'] ==
                  htp[('image_nhwc_first', 'optimized')]['implementation_signature'])
print('Same operation-type/count signature for image-input pair:', same_signature)''')
        md('''The counts expose how the compiler handles each interface. Matching
operation types and counts can support a hypothesis of similar lowering; they
do not prove identical binaries or identical latency. Rejected outputs stay
rejected regardless of the operation counts.''')
    else:
        md('''**Remote results pending.** The model exports passed the CPU reference
check. No model latency is inferred from the convolution experiment. Rebuild with
`--with-vision` once all remote outputs have been checked and HTP artifacts saved.''')
    md('''## 5. What remains to test

- Repeat the layout pairs on a controlled physical device and alternate their order.
- Include actual host preprocessing and transfers on that same phone.
- Use multiple images and evaluate dataset-level accuracy before deployment claims.
- Investigate the 32×32 case through kernel choices and format conversion, without
  assuming transpose count alone predicts latency.

GPU timings use different execution boundaries and are kept in the
[separate comparison](../../reports/snapdragon-comparison/report.md). No NPU
energy measurement is available. This notebook's contribution is the checked
connection between representation, generated execution and measured latency.

## Reproduce

Install the project with `pip install -e '.[notebook]'`, then run
`python scripts/build_layout_notebook.py --with-vision` from the repository root.
The tiny convolution runs on CPU; all device plots use committed measurements.
Fresh model exports additionally require `.[hub,vision]`, downloaded pretrained
weights, and the separate AI Hub submission commands in the study README.''')
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}})
    path = destination / 'companion.ipynb'
    nbf.write(notebook, path)
    manager = KernelManager(kernel_name='python3')
    manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    NotebookClient(notebook, km=manager, timeout=180, resources={'metadata': {'path': str(root)}}).execute()
    nbf.write(notebook, path)
    exporter = HTMLExporter(template_name='lab')
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    body, _ = exporter.from_notebook_node(notebook)
    (destination/'companion.html').write_text(body, encoding='utf-8')
    print(f'Executed {sum(c.cell_type == "code" for c in notebook.cells)} cells: {path}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--with-vision', action='store_true')
    build(**vars(parser.parse_args()))
