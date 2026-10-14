# Accelerator Bench

Operation and compiler experiments on an **RTX 3060 Laptop GPU** and a
**Snapdragon 8 Gen 3 NPU**: matmul, convolution + ReLU, and RMSNorm.

## Results

| Experiment | Finding | Evidence |
|---|---|---|
| GPU layout ablation | Disabling layout optimization makes FP32 convolution 1.54–1.66× faster, but FP16 1.22–1.27× slower. | [Report](reports/gpu-layout/report.md) |
| GPU operation sweep | 56 configurations validated; compiled FP16 RMSNorm uses 1 kernel/call versus 9 eager. | [Report](reports/rtx3060-ac/report.md) · [Findings](reports/rtx3060-ac/findings.md) |
| NPU pilot | Three Galaxy S24 jobs passed correctness and NPU-placement checks. | [Report](reports/snapdragon-s24-smoke/report.md) |

![GPU layout ablation](reports/gpu-layout/layout.png)

The repeated NPU sweep covers 28 compile requests, three jobs each.
A separate NHWC ablation tests four convolution shapes. Reports are generated
only after their jobs finish.

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

Earlier run: [GPU with background workloads](reports/rtx3060-exploratory/report.md).
