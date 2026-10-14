# NPU pilot run

Device: **Samsung Galaxy S24**. Requested QAIRT: **2.50**.

Case: `conv_b1_c32_h32_o64`. FP32 I/O, requested FP16 execution, sustained high performance.

One job per setting. Power unavailable; fusion changes unverified.

| Requested setting | AI Hub estimate (µs) | Median of returned samples (µs) | P10–P90 (µs) | Numerical check | Reported placement |
|---|---:|---:|---:|---|---|
| optimized | 82.0 | 94.0 | 90.0–122.0 | pass | NPU |
| prepare_fast | 86.0 | 100.5 | 90.9–128.1 | pass | NPU |
| no_conv_relu_fold | 92.0 | 99.0 | 94.0–141.1 | pass | NPU |

![NPU ablation](npu-ablation.png)

All returned samples included. Timing includes the QNN framework call. Chart requires numerical pass and reported NPU placement.

## Evidence

- [Machine-readable profile review](profile-review.json)
- [optimized inference job](https://workbench.aihub.qualcomm.com/jobs/jprxn1q7p/) (account access may be required)
- [prepare_fast inference job](https://workbench.aihub.qualcomm.com/jobs/jpx0dm98p/) (account access may be required)
- [no_conv_relu_fold inference job](https://workbench.aihub.qualcomm.com/jobs/j56odven5/) (account access may be required)
