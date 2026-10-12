# Measurement contract

## Scope

These are operation microbenchmarks, not a model-quality leaderboard. All workloads
are inference-only, with static shapes, resident inputs, fixed seeded weights, and
no gradients. They do not represent training, end-to-end application latency,
general LLM performance, or every capability of either accelerator.

Each case is independently seeded from its ID and the config seed. A SHA-256
fingerprint covers the original FP32 input and weight bytes. Local and remote
runners share those inputs and modules. Matrix weights are fixed buffers, matching
an inference layer. Convolution uses stride 1, same padding for odd kernels, no
bias, then ReLU. RMSNorm uses epsilon 1e-6 and FP32 reduction on FP16 inputs.

## Correctness and precision

Before measuring, both ordinary and CUDA Graph outputs must pass against a CPU
FP64 reference computed using the original FP32 tensors. FP16 tolerances are
atol=0.01/rtol=0.01; FP32 tolerances are atol=2e-5/rtol=2e-4. Maximum absolute error
and relative L2 error are also saved. These tolerances are gates, not task-accuracy
claims. Synthetic Gaussian inputs do not establish robustness on all inputs;
separate kernel tests cover zero inputs and dimensions with masked tails.

TF32 is explicitly disabled for matmul and cuDNN. FP16 matmul reduced-precision
reductions are disabled. Thus FP32 is a full-precision baseline, not the fastest
possible NVIDIA configuration. cuDNN benchmark/autotuning is disabled. The compiled
baseline is `torch.compile(fullgraph=True, dynamic=False, mode="default")` using
Inductor; failing variants are recorded as failures, never substituted with eager.

## Two timing boundaries

1. **Dispatch timing:** invoke the callable repeatedly from Python, with CUDA events
   around the block and synchronization at the end. The CUDA stream span can
   include gaps while the host issues work; it is not the sum of kernel durations.
   Wall time also includes Python dispatch and the final wait.
2. **Graph replay timing:** capture 100 consecutive calls in a CUDA Graph, then
   time multiple graph replays. Divide by both the replay count and calls per
   graph. This substantially amortizes host launch overhead, but is a repeated,
   cache-friendly workload. It is not single-request latency or an application
   measurement. All implementations get the same capture treatment.

Both exclude transfers, compilation, correctness checks, and power sampling.
Each variant is warmed before its block group. Variant order is shuffled within
five paired rounds using the config seed. Each round has five blocks by default.
Median and P10/P90 describe **block averages**, not individual request percentiles
or confidence intervals. Case/dtype ordering remains fixed; separate whole-run
repetitions are needed to assess drift and run-to-run uncertainty.

`first_call_s` includes initialization, possible compilation cache hits, and first
execution. It is not pure compiler time. Comparing compiler preparation costs
requires explicitly controlled cache state and additional instrumentation.

## Power and energy

Latency and energy use separate phases. The energy phase warms the exact workload
for one second, then replays graphs for at least five seconds. NVML is requested
every 50 ms, but hardware sampling/averaging may be slower. Samples bracket the
measurement interval; a trapezoidal integral with interpolated endpoints gives
joules. Dividing by the number of original operations yields millijoules per
operation. Sampling does not change device clocks or power limits.

This is **gross NVML GPU-device energy**. It includes baseline and unrelated GPU
activity. It is not whole-system energy, and no idle-baseline subtraction is made.
The first run was explicitly authorized as exploratory with other GPU applications
present; its energy cannot be attributed solely to the benchmark. Unsupported
telemetry fields are null, not zero. Power telemetry failures are marked unavailable.

For a controlled repeat, close GPU-heavy applications, plug into AC, document the
laptop power/fan mode, allow temperatures to stabilize, and repeat whole sessions.
Do not silently kill other applications or change hardware power settings.

## Cross-device comparisons

AI Hub runs real remote devices and reports steady-state inference framework-call
timing, with separate profiling passes for layer metrics. That is not equivalent
to our CUDA Graph timing. Its process memory and HTP memory are also distinct.
See [Qualcomm's measurement description](https://workbench.aihub.qualcomm.com/docs/hub/howitworks-inference.html).

Compare only after recording the exact device, runtime version, performance mode,
tensor shapes, precision and numerical error. Check actual compute placement and
fallbacks. FP32 IO around FP16 execution can incur conversion cost unlike resident
FP16 local tensors; annotate that boundary. A difference between RTX and Snapdragon
is a hardware-and-software-stack result, not an isolated architectural conclusion.

Use the controlled NPU ablations to study compiler choices: change one setting,
hold the graph and device fixed, verify correctness and inspect the generated
profile. Requested flags do not prove that an optimization occurred. An unchanged
graph or performance result is a valid outcome.

Do not estimate NPU energy from TOPS, chip TDP, or a speed ratio. Do not mix GPU
device power with phone wall power on an unlabeled chart. Leave NPU power absent
until compatible measured data is available. Published third-party results belong
in a separately sourced comparison, not the measured-data table.
