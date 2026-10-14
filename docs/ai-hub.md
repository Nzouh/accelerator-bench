# Snapdragon NPU execution through AI Hub

## What is ready locally

- The same 12 seeded workloads used by the CUDA runner export to checked ONNX.
- The manifest stores input/weight and exported-artifact hashes.
- The run plan targets `qnn_dlc`, explicitly requests NPU and FP16 execution, and
  requires a selected device and QAIRT major.minor version.
- Matmul/RMSNorm plans compare optimization levels O=3 and O=1. Conv+ReLU also has
  a separate folding-disabled variant. All other requested settings stay fixed.
- Job IDs are saved as soon as available. Downloaded profiles and numerical checks
  are preserved. Results remain `measured_unverified_placement` until reviewed.

Remote execution was validated on 2026-10-12 for `conv_b1_c32_h32_o64` on
**Samsung Galaxy S24 / Snapdragon 8 Gen 3 / Hexagon v75 / Android 14**, with
QAIRT 2.50. All three jobs passed numerical checks; all listed profile entries
report NPU. See [the first measured report](../reports/snapdragon-s24-smoke/report.md).
This does not prove every exported operation compiles for every Snapdragon.
Physical fusion changes remain unverified. Compile failures must be investigated
rather than relabeled as successful NPU measurements.

## Account setup

Create/sign into [AI Hub Workbench](https://workbench.aihub.qualcomm.com/). Get the
token from **Account → Settings → API Token**, following the
[official setup guide](https://workbench.aihub.qualcomm.com/docs/hub/getting_started.html).

On this Windows laptop, run in PowerShell:

```powershell
wsl -d ubuntu1 -- /home/admin1/.venvs/accelerator-bench/bin/python /mnt/c/Users/nabil/accelerator-bench/scripts/configure_hub.py
```

The hidden prompt writes the token to `~/.qai_hub/client.ini` inside WSL with owner
read/write permissions. It never places the token in command arguments, shell
history, result files, or the repository. Do not paste it into an issue or chat.

For subsequent commands, enter WSL and activate the environment:

```bash
cd /mnt/c/Users/nabil/accelerator-bench
source ~/.venvs/accelerator-bench/bin/activate
accelerator-bench hub devices
qai-hub list-frameworks
```

Pick an **exact device name returned for your account** and a supported QAIRT
major.minor version. A named Snapdragon phone or reference laptop is preferable
to silently changing devices between runs. Availability, account limits, and any
applicable charges must be checked in the account before a large sweep.

## Start with one convolution case

```bash
accelerator-bench hub export

# Replace the two placeholders with values returned by the commands above.
accelerator-bench hub plan \
  --device 'EXACT DEVICE NAME' --qairt-version 'MAJOR.MINOR' \
  --case conv_b1_c32_h32

accelerator-bench hub run \
  --device 'EXACT DEVICE NAME' --qairt-version 'MAJOR.MINOR' \
  --case conv_b1_c32_h32 --output-dir results/local/hub-first-conv
```

`plan` is entirely local. The legacy `run` command uploads generated synthetic graphs/tensors and
submits three compile+inference pairs for this convolution case. Start there;
the whole default suite would submit 28 pairs. Use a new output directory for
each run. Failed/interrupted jobs retain IDs for inspection; automatic resume is
not implemented, so inspect existing jobs before resubmitting.

## What to inspect before making a chart

1. Compilation succeeded; outputs pass the numerical check.
2. The profile identifies HTP/NPU placement, with no unexpected CPU fallback.
3. Both folding variants have the same shapes, runtime, device and precision.
4. The profile or runtime analysis supports whether folding actually changed.
5. The timing definition is compatible with the claim. Keep AI Hub framework-call
   latency separate from local CUDA Graph replay latency.
6. Load/preparation time is separate from warmed inference, and remote queue time
   is never called compiler time.

AI Hub's published summary lists timing and memory; this project has not verified
a compatible power metric. No NPU energy results are supplied until such a source
is available. The requested `SUSTAINED_HIGH_PERFORMANCE` mode is a setting, not a
power measurement.

## Sources and API verification

- [AI Hub setup](https://workbench.aihub.qualcomm.com/docs/hub/getting_started.html)
- [Compile, compute-unit, folding and optimization options](https://workbench.aihub.qualcomm.com/docs/hub/api.html)
- [Profiling timing and memory boundaries](https://workbench.aihub.qualcomm.com/docs/hub/howitworks-inference.html)

The local adapter was checked against installed `qai-hub==0.56.0` signatures and
the documentation on 2026-10-12, followed by the three live device runs above.

To regenerate the remote report from committed evidence:

```bash
accelerator-bench hub report results/hub-s24-conv-2026-10-12 \
  --output-dir reports/snapdragon-s24-smoke
```

The review report evaluates returned placement fields without altering the raw
result records. Those retain their original `measured_unverified_placement` state;
the separate `profile-review.json` records the placement evidence examined later.

## Repeated sweep

Samsung Galaxy S24 / Snapdragon 8 Gen 3 / QAIRT 2.50. The default sweep compiles
28 configurations and submits three inference jobs each, in seeded shuffled order.
Job IDs are checkpointed. Each output must pass the FP64 reference check and all
profile entries must report NPU.

```bash
accelerator-bench hub export --output-dir artifacts/hub
accelerator-bench hub sweep --device 'Samsung Galaxy S24' --qairt-version 2.50 \
  --manifest artifacts/hub/manifest.json --repeats 3 --max-active 4 \
  --output-dir results/local/s24-sweep
# After an interruption: repeat the same command with --resume.
accelerator-bench hub htp results/local/s24-sweep
```

Resume reuses recorded jobs. An uncertain submission stops for reconciliation
by saved job name; it is never silently resubmitted.

## Layout ablation

NHWC export preserves the original convolution, weights and logical inputs.
Only the input/output layout changes; reference tensors are permuted accordingly.
Layout reordering occurs outside measured inference.

```bash
accelerator-bench hub export --case conv_ --io-layout nhwc --output-dir artifacts/hub-nhwc
accelerator-bench hub sweep --device 'Samsung Galaxy S24' --qairt-version 2.50 \
  --manifest artifacts/hub-nhwc/manifest.json --variants optimized --repeats 3 \
  --output-dir results/local/s24-nhwc
accelerator-bench hub htp results/local/s24-nhwc
accelerator-bench hub compare results/local/s24-sweep \
  --gpu-source results/rtx3060-ac-2026-10-13.json \
  --layout-source results/local/s24-nhwc --output-dir reports/s24-comparison
```

Charts use the median of three job medians, with their minimum–maximum range.
QHAS diagnostics come from repeat 1; their instrumented times are not benchmark
latencies. Raw op types, instance counts, precision and memory fields are retained.

GPU/NPU charts label their different timing boundaries. No cross-device energy
ratio is calculated. Workbench was listed as free in the
[official FAQ](https://workbench.aihub.qualcomm.com/docs/hub/faq.html) when this sweep was launched.
