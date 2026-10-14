# Tensor layout on Snapdragon

A research notebook following the weather study's progression: definitions,
independent checks, execution evidence, then a pretrained model. Each figure has
an adjacent explanation. This is research material; the website is unchanged.

The MobileNetV2 follow-up completed 12 jobs. All preserve the reference's top
prediction and report NPU execution, but all fail the fixed per-logit tolerance.
A separate CPU FP16 diagnostic also fails that tolerance. The notebook explains
the rejection without relaxing it or claiming a validated model speedup.

- [Read the notebook](companion.ipynb)
- [Open the executed HTML locally](companion.html)
- [Experiment plan, committed before model jobs](PLAN.md)
- [Figure and validation code](../../src/accelerator_bench/layout_research.py)
- [Image attribution](assets/ATTRIBUTION.md)
- [Recorded WSL packages](environment-lock.txt)

## Offline reproduction

```bash
pip install -e '.[notebook]'
python scripts/build_layout_notebook.py --with-vision
```

This runs the small CPU correctness example and regenerates plots from saved
measurements. It does not submit cloud jobs or require a GPU or API token.

## Fresh model measurements

The recorded run uses torch 2.11 and torchvision 0.26, MobileNetV2 IMAGENET1K_V2,
Samsung Galaxy S24, QAIRT 2.50, three jobs per case. See the plan for boundaries.

```bash
pip install -e '.[hub,vision]'
python scripts/export_vision_layout.py --output-dir artifacts/my-vision-layout
accelerator-bench hub sweep --device 'Samsung Galaxy S24' --qairt-version 2.50 \
  --manifest artifacts/my-vision-layout/manifest.json --variants optimized \
  --repeats 3 --max-active 4 --output-dir results/local/my-vision-layout
# Resume an interrupted run with exactly the same arguments and --resume.
accelerator-bench hub htp results/local/my-vision-layout
```

For rejected outputs, retain the rejection and inspect precision separately:

```bash
accelerator-bench hub htp results/local/my-vision-layout --include-rejected
python scripts/review_vision_precision.py \
  --manifest-path artifacts/my-vision-layout/manifest.json \
  --source-dir results/local/my-vision-layout
```

This adds CPU FP16 diagnostics and error counts without changing the numerical
gate or accepting rejected timings.

The notebook reads the recorded October 14 run by design. For a new experiment,
point `VISION` in the figure module to the new run and inspect all checks before
interpreting or publishing its measurements.
