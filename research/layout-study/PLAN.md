# Layout study: experiment plan

Written before the MobileNetV2 jobs. Existing convolution results are exploratory
evidence used to choose this follow-up, not a prediction registered in advance.

## Question

Does the NHWC benefit seen in one convolution persist through a pretrained vision
model, and when normalization and layout expressions are inside the timed graph?

## Learning and validation

1. Show a 2-channel, 2×3 tensor in NCHW and NHWC storage order.
2. Calculate a tiny convolution with explicit loops and compare both layouts with
   PyTorch. Distinguish a strided view from a materialized copy.
3. Reconstruct all four measured convolution sizes, with each remote job visible.
4. Explain the preparation-stage audit and the corrected O3/O1 experiment.
5. Run the following model experiment, then create an offline companion notebook.

## Model experiment

Model: torchvision 0.26 MobileNetV2, IMAGENET1K_V2 weights, evaluation mode.
Input: NASA's public-domain astronaut photograph distributed by scikit-image;
official weight transforms (resize 232, center crop 224, channel normalization).
One image, batch 1. This checks numerical agreement, not classification accuracy.

| Case | Input | Computation inside QNN timing |
|---|---|---|
| model_nchw | Normalized FP32 NCHW | Complete classifier |
| model_nhwc | Normalized FP32 NHWC | Layout expression + complete classifier |
| image_nchw_first | Cropped RGB FP32 NHWC, 0–255 | Permute, normalize, classifier |
| image_nhwc_first | Same cropped RGB FP32 NHWC | Normalize, permute, classifier |

Compare cases 1–2 and 3–4 separately. The first pair changes the model interface;
the second pair keeps the same public image interface and changes the placement
of normalization relative to the permutation. Optimizers may make cases 3–4
equivalent. A source-level permutation need not survive as a runtime copy.

Resize, decode, upload, host casting to FP32 and camera capture are excluded in
every case. The second pair includes in-graph normalization and layout handling;
it is not phone end-to-end latency. No laptop CPU timing is added to phone timing.

Device: Samsung Galaxy S24, QAIRT 2.50, QNN DLC, NPU only, requested FP16, FP32 I/O,
explicit O3 at inference preparation, sustained-high-performance mode.
Four compilations, three inference jobs each, shuffled submission order, maximum
four active jobs. Remote scheduling and physical device allocation are uncontrolled.

Checks: ONNX CPU vs CPU FP64 reference before submission; device logits vs CPU
FP64 with the existing FP16 tolerances (atol=0.01, rtol=0.01); unchanged top-1;
all reported execution entries on NPU. Rejections stay visible. No tolerance
changes after viewing results. Store image, weights/input hashes, profiles,
logits, job IDs, export metadata and first-repeat HTP summaries.

## Expectations and interpretation

- Model NHWC may remove boundary transposes, but the relative gain may be small
  because the complete network has much more work than one convolution.
- The shared-image-interface pair may compile to equivalent execution.
- Overlapping job-median ranges will not support a clear timing advantage.
- Three jobs are descriptive evidence; ranges are not confidence intervals.
- HTP traces can support a mechanism but cannot assign all saved time to one op.
- Keep every case. No NPU power estimate or claim of universal NHWC superiority.

## Figures and explanations

Use a storage-order diagram, a four-shape latency/transpose plot, a compact
preparation-settings figure, and one model figure with the input crop. Each gets
adjacent prose explaining the axes, observation, and limit of the inference.
No decorative imagery. The notebook regenerates figures from saved results;
remote submission remains a separate, explicit command.

Sources: [MobileNetV2 weights and transforms](https://docs.pytorch.org/vision/0.26/models/generated/torchvision.models.mobilenet_v2.html),
[QNN options](https://workbench.aihub.qualcomm.com/docs/hub/api.html),
[image attribution](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut).
