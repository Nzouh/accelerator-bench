# Accelerator Bench

Reproducible experiments on neural-network operations, compiler optimizations,
latency, and energy. Initial target: NVIDIA RTX 3060 Laptop GPU (6 GB).
Qualcomm AI Hub integration targets measured Snapdragon Hexagon NPU execution.

The question: **when do compiler optimizations help, and what limits them?**

## First measured report

[RTX 3060 charts and analysis](reports/rtx3060-exploratory/report.md) ·
[Raw measurements](results/rtx3060-exploratory-2026-10-12.json) ·
[Profiler evidence](results/rtx3060-rmsnorm-profile.json)

All **56 configurations passed numerical checks**. This first run is exploratory
because concurrent GPU applications were left running. Its timings and gross
energy are not isolated hardware measurements.

A separate FP16 RMSNorm diagnostic records **9 CUDA kernel activities per call
in eager execution versus 1 with `torch.compile` and 1 with the explicit fused
kernel**, over five profiled calls at shape 32×1024. This directly demonstrates
the change in execution structure; it does not turn the exploratory speedups
into controlled performance claims.

![Kernel count evidence](reports/rtx3060-exploratory/kernel_counts.png)

## First real Snapdragon NPU run

[Galaxy S24 NPU report](reports/snapdragon-s24-smoke/report.md) ·
[Raw remote results and profiles](results/hub-s24-conv-2026-10-12)

Three actual AI Hub jobs ran the shared convolution + ReLU case on the Galaxy S24
(Snapdragon 8 Gen 3), using QAIRT 2.50. All three passed numerical checks, and all
reported profile entries were assigned to NPU. The tested settings were O=3,
O=1, and O=3 with convolution–ReLU folding disabled.

Returned sample medians were 94.0, 100.5, and 99.0 microseconds respectively, with
overlapping distributions. One job per setting is a working pipeline demonstration,
not evidence of a reliable compiler-setting advantage. The profile does not prove
that the requested folding change altered physical fusion. NPU energy is unavailable.

![NPU settings experiment](reports/snapdragon-s24-smoke/npu-ablation.png)

## Experiments

- Matrix multiplication: shape and precision sensitivity.
- Convolution + ReLU: eager versus compiled execution; NPU folding ablation.
- RMSNorm: eager, compiled, and an explicit fused implementation.
- Sustained GPU energy telemetry, separate from timing measurements.

Results must include numerical checks, environment information, raw measurements,
and exact timing boundaries. No estimated NPU measurements or fabricated results.
Cross-device measurements compare hardware **and** software stacks; controlled
ablations on one device isolate optimization effects more directly.

## Run locally

This RTX 3060 laptop uses Windows with CUDA available through WSL2. The project
directory is `C:\Users\nabil\accelerator-bench`, visible in WSL at
`/mnt/c/Users/nabil/accelerator-bench`. The isolated Python environment lives at
`~/.venvs/accelerator-bench` in WSL, outside the repository.

```powershell
wsl -d ubuntu1 -- bash /mnt/c/Users/nabil/accelerator-bench/scripts/setup_wsl.sh
```

Then in WSL:

```bash
cd /mnt/c/Users/nabil/accelerator-bench
source ~/.venvs/accelerator-bench/bin/activate
python -m pytest -q
accelerator-bench doctor
accelerator-bench run --output results/local/my-run.json \
  --context 'Describe AC power, power mode, background applications and controls here'
accelerator-bench report results/local/my-run.json --output-dir reports/my-run
```

For a short check, add `--case rmsnorm_r32 --dtypes float16 --rounds 2 --blocks 2
--iterations 20 --energy-seconds 2`. Full defaults cover 56 configurations across
12 shapes, two precisions, eager/compiled execution, and explicit fused RMSNorm.
Existing run files are never overwritten.

To collect separate profiler evidence and regenerate the committed report:

```bash
accelerator-bench profile --case rmsnorm_r32_d1024 --dtype float16 \
  --output results/local/my-profile.json
accelerator-bench report results/rtx3060-exploratory-2026-10-12.json \
  --profile results/rtx3060-rmsnorm-profile.json --output-dir reports/rtx3060-exploratory
```

On another Linux CUDA machine, clone the repo and run `bash scripts/setup_wsl.sh`
from its root. The script name reflects this project's initial environment; the
setup also works in a suitable Linux environment with Python venv and GCC.

## Read the measurements correctly

- **Graph timing:** repeated CUDA Graph replay amortizes host launch overhead.
- **Dispatch timing:** synchronized wall time includes Python dispatch.
- **Energy:** integrated gross NVML device power during sustained graph replay.
- **Correctness:** every timed output is compared against an FP64 reference.
- **Evidence:** JSON contains raw blocks, raw power samples, settings and source
  revision; generated charts and CSVs can be regenerated from it.

The first run is **exploratory**, with other GPU applications present by explicit
choice. It is not an isolated hardware comparison. No NPU numbers are fabricated.

Validation: **23 local tests passed**, including CUDA kernel checks; GitHub Actions
also runs the CPU/export/orchestration checks (CUDA-only tests skip on CPU runners).

Read [the measurement contract](docs/methodology.md) before interpreting speedups
or energy. See [AI Hub setup and experiment instructions](docs/ai-hub.md) for the
Snapdragon path. Generated large model artifacts and credentials stay out of Git.
