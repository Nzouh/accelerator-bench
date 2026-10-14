# RTX 3060 operation benchmark report

> AC-powered desktop run: Windows confirmed AC power and Balanced power plan before starting; idle GPU preflight recorded separately. Rainbow Six was closed. Resident desktop apps remain; background activity is not isolated. GPU clocks and power limits were not changed. One full session, not an independent-session repeatability study. See results/rtx3060-ac-2026-10-13-host.json for before/after context.

This report is generated from real measurements. No Snapdragon NPU numbers have been inferred or added.

- Validated configurations: **56** / 56 attempted.
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU; 6 GiB VRAM.
- PyTorch 2.11.0+cu128; CUDA 12.8; driver 616.64; 6.6.87.2-microsoft-standard-WSL2.
- Measurement source commit: `399b19c7b4750360dc917fdc7f4d9a71db579350`; dirty at start: `False`.
- Run time (UTC): 2026-10-14T00:43:30.933841+00:00.
- TF32 disabled; FP16 matmul reduced-precision reductions disabled; cuDNN benchmarking disabled.

## How to read these results

CUDA Graph replay amortizes host dispatch across repeated operations. The separate dispatch chart includes Python and synchronization overhead. Neither includes input transfers or full application work. Graph replay repeatedly uses the same resident inputs and can favor cache reuse.

Shaded ranges are P10–P90 across **block-average** timings, not per-request tail latencies or confidence intervals. Variants are shuffled within paired rounds. Input shapes and precision run in a fixed order, so thermal drift and background activity remain possible confounders.

Energy is gross GPU-device telemetry integrated during a separate sustained replay window. It includes baseline consumption and any unrelated GPU activity. It is not isolated workload-only energy or whole-laptop power, and cannot support an NPU efficiency comparison. Sensor averaging limits the meaning of short intervals.

The compiled variant can change layout, dispatch, and kernels. A speedup alone does not establish which compiler transformation caused it. The explicit fused implementation only applies to RMSNorm.

## Execution with host dispatch amortized

![Execution with host dispatch amortized](graph_latency.png)

## Execution including Python dispatch

![Execution including Python dispatch](dispatch_latency.png)

## Optimization benefit depends on workload size

![Optimization benefit depends on workload size](speedup.png)

## Gross GPU energy during sustained replay

![Gross GPU energy during sustained replay](energy.png)

## Compiler evidence

![Profiler-observed CUDA activities](kernel_counts.png)

The separate profiler capture records fewer CUDA activities for the compiled and explicit fused RMSNorm implementations. Kernel names and counts are retained in the diagnostic JSON. This is evidence about execution structure; timings come from the separate benchmark run.

[Raw profiler evidence](../../results/rtx3060-ac-rmsnorm-profile.json)

## Numerical validation

All timed configurations passed both ordinary and captured-graph output checks against a CPU FP64 reference computed from the original seeded FP32 inputs and weights. FP16 error includes input and weight rounding. This is numerical agreement on synthetic tensors, not task accuracy on a dataset.

| Operation | Precision | Maximum relative L2 error across shapes/variants |
|---|---|---:|
| Matrix multiplication | float16 | 3.727e-04 |
| Matrix multiplication | float32 | 3.340e-07 |
| Convolution + ReLU | float16 | 3.607e-04 |
| Convolution + ReLU | float32 | 2.037e-07 |
| RMSNorm | float16 | 3.729e-04 |
| RMSNorm | float32 | 7.509e-08 |

## Raw evidence

- [Flat measurement table](measurements.csv)
- [Raw measurement JSON](../../results/rtx3060-ac-2026-10-13.json) (raw timing blocks, power samples, environment, checks).
- `first_call_s` includes initialization, possible cache hits, compilation and execution; it is **not** isolated compiler duration.

## Before drawing hardware conclusions

Repeat whole sessions on AC power with GPU-heavy applications closed and documented power settings to quantify variation. Before any cross-device comparison, verify NPU placement for the same checked workload and reconcile precision and timing boundaries. Keep graph-replay timing separate from AI Hub framework-call timing.
