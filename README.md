# Accelerator Bench

Operation and compiler experiments on an **RTX 3060 Laptop GPU** and a
**Snapdragon 8 Gen 3 NPU**: matmul, convolution + ReLU, and RMSNorm.

## Results

| Experiment | Finding | Evidence |
|---|---|---|
| GPU/NPU latency | RTX graph latency is lower across all 12 shapes; timing boundaries differ. | [Comparison](reports/snapdragon-comparison/report.md) |
| NPU layout | 64×64: 207 → 142 µs and 18 → 0 transposes. At 32×32, timing ranges overlap. | [Layout study](reports/snapdragon-comparison/report.md#layout-ablation) |
| GPU layout ablation | Disabling layout optimization makes FP32 convolution 1.54–1.66× faster, but FP16 1.22–1.27× slower. | [Report](reports/gpu-layout/report.md) |
| GPU operation sweep | 56 configurations validated. FP16 RMSNorm at 32×1024: 9 → 1 kernel/call. | [Report](reports/rtx3060-ac/report.md) · [Findings](reports/rtx3060-ac/findings.md) |

![GPU/NPU latency](reports/snapdragon-comparison/latency.png)

**96 NPU jobs validated:** 84 in the original sweep and 12 in the NHWC study.
Each compile request has three independent jobs. The original sweep's HTP flag
application was unverified; see the [preparation audit](reports/preparation-audit/report.md).

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
