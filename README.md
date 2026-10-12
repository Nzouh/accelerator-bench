# Accelerator Bench

Reproducible experiments on neural-network operations, compiler optimizations,
latency, and energy. Initial target: NVIDIA RTX 3060 Laptop GPU (6 GB).
Qualcomm AI Hub integration targets measured Snapdragon Hexagon NPU execution.

The question: **when do compiler optimizations help, and what limits them?**

## Planned experiments

- Matrix multiplication: shape and precision sensitivity.
- Convolution + ReLU: eager versus compiled execution; NPU folding ablation.
- RMSNorm: eager, compiled, and an explicit fused implementation.
- Sustained GPU energy telemetry, separate from timing measurements.

Results must include numerical checks, environment information, raw measurements,
and exact timing boundaries. No estimated NPU measurements or fabricated results.
Cross-device measurements compare hardware **and** software stacks; controlled
ablations on one device isolate optimization effects more directly.

## Status

Implementation in progress. No benchmark claims yet.
