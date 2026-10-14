# GPU compiler findings

56/56 configurations passed. [Charts](report.md) · [Measurements](measurements.csv)

| Case | Precision | Eager (µs) | Compiled (µs) | Speedup |
|---|---|---:|---:|---:|
| Matmul M=1 | FP16 | 10.305 | 10.313 | 1.00× |
| Conv + ReLU, 16×16 | FP32 | 10.270 | 16.514 | 0.62× |
| Conv + ReLU, 128×128 | FP16 | 68.758 | 51.762 | 1.33× |
| RMSNorm 32×1024 | FP16 | 13.179 | 1.311 | 10.05× |
| RMSNorm 256×1024 | FP16 | 25.379 | 2.337 | 10.86× |

Median CUDA Graph time. One session; 25 blocks/configuration.

## Kernel evidence

| Diagnostic | Eager kernels/call | Compiled | Fused |
|---|---:|---:|---:|
| FP16 RMSNorm 32×1024 | 9 | 1 | 1 |
| FP32 Conv + ReLU, 16×16 | 3 | 8 | — |

[RMSNorm trace](../../results/rtx3060-ac-rmsnorm-profile.json): compiled execution uses one Triton reduction kernel.

[Convolution trace](../../results/rtx3060-ac-conv-profile.json): compiled execution adds two NHWC→NCHW and one NCHW→NHWC kernel per call. Both paths use Winograd. The conversions are a candidate cause of the regression.

## Dispatch cost

FP16 RMSNorm 32×1024: graph speedup **10.05×**; synchronized Python dispatch **1.94×** (153.247→78.948 µs). Transfers excluded.

## Energy

FP16 RMSNorm 32×1024, separate sustained replay:

| Variant | GPU power (W) | Energy (mJ/op) |
|---|---:|---:|
| Eager | 59.49 | 0.7564 |
| Compiled | 64.05 | 0.0833 |
| Fused | 61.12 | 0.0761 |

Gross NVML device energy; one window/configuration. Includes baseline and background consumption.

## Conditions

[Host snapshots](../../results/rtx3060-ac-2026-10-13-host.json): AC, Balanced mode, Rainbow Six closed, desktop apps present. Clocks unlocked. Energy-phase samples span 38–84°C and 435–2010 MHz; latency phases were not monitored. Independent session repeats remain necessary.
