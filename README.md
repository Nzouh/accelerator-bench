# Accelerator Bench

Operation and compiler experiments on an **RTX 3060 Laptop GPU** and a
**Snapdragon 8 Gen 3 NPU**: matmul, convolution + ReLU, and RMSNorm.

[Research notebook](research/layout-study/companion.ipynb): a convolution from
the definitions, explained figures, and a pretrained MobileNetV2 follow-up.
Its 12 model jobs preserve top-1 but fail the predeclared logit tolerance;
their timings remain excluded from validated comparisons.

## Results

| Experiment | Finding | Evidence |
|---|---|---|
| GPU/NPU latency | RTX graph latency is lower across all 12 shapes; timing boundaries differ. | [Comparison](reports/snapdragon-comparison/report.md) |
| NPU layout | 64×64: 207 → 142 µs and 18 → 0 transposes. At 32×32, timing ranges overlap. | [Layout study](reports/snapdragon-comparison/report.md#layout-ablation) |
| NPU preparation | O1: 47 → 24 HTP nodes, but 80 → 91 µs. Folding-disabled timing overlaps O3. | [Compiler study](reports/npu-preparation/report.md) |
| GPU layout ablation | Disabling layout optimization makes FP32 convolution 1.54–1.66× faster, but FP16 1.22–1.27× slower. | [Report](reports/gpu-layout/report.md) |
| GPU operation sweep | 56 configurations validated. FP16 RMSNorm at 32×1024: 9 → 1 kernel/call. | [Report](reports/rtx3060-ac/report.md) · [Findings](reports/rtx3060-ac/findings.md) |

![GPU/NPU latency](reports/snapdragon-comparison/latency.png)

**108 NPU jobs validated:** 84 original, 12 NHWC, 9 preparation, 3 flag-audit jobs.
Timing studies use three jobs per setting. A [stage audit](reports/preparation-audit/report.md)
caught HTP flags applied only at conversion; the corrected compiler study sets
them explicitly during device preparation.

## Measurement boundaries

- **GPU:** resident inputs; CUDA Graph replay and Python dispatch measured separately.
- **NPU:** AI Hub QNN framework-call latency; FP16 requested, FP32 I/O.
- **Correctness:** CPU FP64 reference; identical logical inputs and weight hashes.
- **Energy:** gross GPU energy from NVML. NPU energy is unavailable.

GPU/NPU timings have different boundaries. Within-device ablations isolate
compiler choices more directly. Raw blocks, job IDs, profiles, errors, settings,
and source revisions are retained.

## Reproduce

Windows + WSL2 on this laptop; Linux + CUDA also supported.

```bash
git clone https://github.com/Nzouh/accelerator-bench.git
cd accelerator-bench
bash scripts/setup_wsl.sh
source ~/.venvs/accelerator-bench/bin/activate
python -m pytest -q

accelerator-bench run --output results/local/my-run.json \
  --context 'AC power; power mode; background workloads'
accelerator-bench report results/local/my-run.json --output-dir reports/my-run
```

For a short run, add `--case rmsnorm_r32 --dtypes float16 --rounds 2 --blocks 2
--iterations 20 --energy-seconds 2`.

[AI Hub setup and repeated sweeps](docs/ai-hub.md) ·
[Measurement contract](docs/methodology.md) ·
[Recorded dependencies](requirements-wsl-lock.txt) ·
[Tests](https://github.com/Nzouh/accelerator-bench/actions)

Archive: [GPU with background workloads](reports/rtx3060-exploratory/report.md) ·
[NPU compile requests](reports/snapdragon-s24-smoke/report.md).
