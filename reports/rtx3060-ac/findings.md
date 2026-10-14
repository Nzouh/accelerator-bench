# What the AC-powered RTX 3060 run shows

All 56 configurations passed both ordinary and captured-graph checks against the
CPU FP64 reference. Each configuration has 25 timing blocks in five paired rounds
and a separate measured power window. [Full charts](report.md),
[measurement table](measurements.csv), and
[raw data](../../results/rtx3060-ac-2026-10-13.json) retain the evidence.

## Compilation has different effects on different operations

Times below are median CUDA Graph microseconds per operation on repeated resident
inputs. Speedup is eager / compiled; below 1 means a slowdown.

| Case | Precision | Eager (µs) | Compiled (µs) | Speedup |
|---|---|---:|---:|---:|
| Matmul M=1, K=N=1024 | FP16 | 10.305 | 10.313 | 1.00× |
| Conv + ReLU, 16×16, C=32→64 | FP32 | 10.270 | 16.514 | 0.62× |
| Conv + ReLU, 128×128, C=32→64 | FP16 | 68.758 | 51.762 | 1.33× |
| RMSNorm 32×1024 | FP16 | 13.179 | 1.311 | 10.05× |
| RMSNorm 256×1024 | FP16 | 25.379 | 2.337 | 10.86× |

Across all tested shapes, compiled matmul graph times stayed within roughly 3%
of eager. Compiled FP16 convolution improved by 1.26–1.33×, while FP32 convolution
took 1.39–1.61× as long. Compiled RMSNorm improved by 3.50–6.34× for FP32 and
10.05–10.86× for FP16. The explicit fused RMSNorm reached 1.956 µs at FP16
256×1024, versus 25.379 µs eager (12.98×) and 2.337 µs compiled.

These are within-session observations. The five rounds are not five independent
sessions, and P10–P90 block ranges are not confidence intervals.

## Profiler evidence explains where to investigate

Separate diagnostic captures profiled five calls per implementation after warmup.
Their timings include instrumentation overhead and are not benchmark latencies.

| Diagnostic | Eager activities/call | Compiled activities/call | Explicit fused |
|---|---:|---:|---:|
| FP16 RMSNorm 32×1024 | 9 | 1 | 1 |
| FP32 Conv + ReLU, 16×16 | 3 | 8 | — |

The [RMSNorm trace](../../results/rtx3060-ac-rmsnorm-profile.json) records one
generated Triton reduction kernel in the compiled path, replacing the eager
sequence of casts, elementwise operations and reduction. This supports fusion
as an explanation for the improvement, without claiming to isolate its share
from every other compiler change.

The [convolution trace](../../results/rtx3060-ac-conv-profile.json) shows both
paths using Winograd kernels. The compiled path additionally contains two
`nhwcToNchwKernel` activities and one `nchwToNhwcKernel` per call, plus generated
Triton kernels. Those conversions are a plausible contributor to the regression.
Kernel names and counts alone cannot attribute the entire slowdown; a layout
ablation would be the next causal experiment.

## Device execution and Python dispatch tell different stories

At FP16 RMSNorm 32×1024, compilation changes graph timing from 13.179 to 1.311 µs
(10.05×), but synchronized Python-dispatch wall time changes from 153.247 to
78.948 µs (1.94×). The explicit fused path measures 31.852 µs at that boundary.

At FP16 matmul M=1, graph times are effectively equal, while Python-dispatch wall
time is 25.973 µs eager versus 97.001 µs compiled. A faster device kernel does not
automatically imply an equivalent application gain. Neither benchmark includes
transfers or full model execution.

## GPU power and energy per operation are different quantities

For FP16 RMSNorm 32×1024, the separate sustained replay windows recorded:

| Variant | Mean gross GPU power (W) | Gross GPU energy/operation (mJ) |
|---|---:|---:|
| Eager | 59.49 | 0.7564 |
| Compiled | 64.05 | 0.0833 |
| Explicit fused | 61.12 | 0.0761 |

The compiled window has higher average power but much lower energy per completed
operation. These are sampled NVML device measurements, including baseline and
background consumption; they are not isolated workload energy or laptop wall
power. There is one energy window per configuration, so these differences have
not been tested across independent repeats. Power was measured separately from
latency; energy is not calculated by multiplying the timing median by power.

## Conditions and limits

[Windows host snapshots](../../results/rtx3060-ac-2026-10-13-host.json) confirm AC
power before and after, and the Balanced power plan. Rainbow Six was absent from
the reported GPU process list. Preflight utilization was mostly 0%, with brief
desktop activity; background apps remained. GPU clocks and power limits were not
changed. The vendor fan mode was not recorded.

Across the recorded energy samples, GPU temperature ranged from 38 to 84°C and
graphics clocks from 435 to 2010 MHz. Those readings demonstrate that the session
was not thermally or clock controlled; they were not collected during the latency
blocks. Case order was fixed, so drift remains a limitation. Independent sessions
are required before treating these ratios as stable hardware characteristics.

The [Galaxy S24 NPU experiment](../snapdragon-s24-smoke/report.md) remains separate:
AI Hub framework-call timing and CUDA Graph replay have different boundaries,
and NPU energy is unavailable. These results support a compiler investigation;
they do not establish an RTX-versus-Snapdragon efficiency ranking.
