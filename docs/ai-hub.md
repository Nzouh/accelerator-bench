# Snapdragon NPU execution through AI Hub

## What is ready locally

- The same 12 seeded workloads used by the CUDA runner export to checked ONNX.
- The manifest stores input/weight and exported-artifact hashes.
- The run plan targets `qnn_dlc`, explicitly requests NPU and FP16 execution, and
  requires a selected device and QAIRT major.minor version.
- Matmul/RMSNorm plans compare optimization levels O=3 and O=1. Conv+ReLU also has
  a separate folding-disabled variant. All other requested settings stay fixed.
- Job IDs are saved as soon as available. Downloaded profiles and numerical checks
  are preserved. Results remain `measured_unverified_placement` until reviewed.

Remote execution is pending account access. Export validation does not prove that
every operation will compile for every Snapdragon. A compile failure is recorded
and must be investigated rather than relabeled as a successful NPU measurement.

## Account setup

Create/sign into [AI Hub Workbench](https://workbench.aihub.qualcomm.com/). Get the
token from **Account → Settings → API Token**, following the
[official setup guide](https://workbench.aihub.qualcomm.com/docs/hub/getting_started.html).

On this Windows laptop, run in PowerShell:

```powershell
wsl -d ubuntu1 -- /home/admin1/.venvs/accelerator-bench/bin/python /mnt/c/Users/nabil/accelerator-bench/scripts/configure_hub.py
```

The hidden prompt writes the token to `~/.qai_hub/client.ini` inside WSL with owner
read/write permissions. It never places the token in command arguments, shell
history, result files, or the repository. Do not paste it into an issue or chat.

For subsequent commands, enter WSL and activate the environment:

```bash
cd /mnt/c/Users/nabil/accelerator-bench
source ~/.venvs/accelerator-bench/bin/activate
accelerator-bench hub devices
qai-hub list-frameworks
```

Pick an **exact device name returned for your account** and a supported QAIRT
major.minor version. A named Snapdragon phone or reference laptop is preferable
to silently changing devices between runs. Availability, account limits, and any
applicable charges must be checked in the account before a large sweep.

## Start with one convolution case

```bash
accelerator-bench hub export

# Replace the two placeholders with values returned by the commands above.
accelerator-bench hub plan \
  --device 'EXACT DEVICE NAME' --qairt-version 'MAJOR.MINOR' \
  --case conv_b1_c32_h32

accelerator-bench hub run \
  --device 'EXACT DEVICE NAME' --qairt-version 'MAJOR.MINOR' \
  --case conv_b1_c32_h32 --output-dir results/local/hub-first-conv
```

`plan` is entirely local. `run` uploads generated synthetic graphs/tensors and
submits three compile+inference pairs for this convolution case. Start there;
the whole default suite would submit 28 pairs. Use a new output directory for
each run. Failed/interrupted jobs retain IDs for inspection; automatic resume is
not implemented, so inspect existing jobs before resubmitting.

## What to inspect before making a chart

1. Compilation succeeded; outputs pass the numerical check.
2. The profile identifies HTP/NPU placement, with no unexpected CPU fallback.
3. Both folding variants have the same shapes, runtime, device and precision.
4. The profile or runtime analysis supports whether folding actually changed.
5. The timing definition is compatible with the claim. Keep AI Hub framework-call
   latency separate from local CUDA Graph replay latency.
6. Load/preparation time is separate from warmed inference, and remote queue time
   is never called compiler time.

AI Hub's published summary lists timing and memory; this project has not verified
a compatible power metric. No NPU energy results are supplied until such a source
is available. The requested `SUSTAINED_HIGH_PERFORMANCE` mode is a setting, not a
power measurement.

## Sources and API verification

- [AI Hub setup](https://workbench.aihub.qualcomm.com/docs/hub/getting_started.html)
- [Compile, compute-unit, folding and optimization options](https://workbench.aihub.qualcomm.com/docs/hub/api.html)
- [Profiling timing and memory boundaries](https://workbench.aihub.qualcomm.com/docs/hub/howitworks-inference.html)

The local adapter was checked against installed `qai-hub==0.56.0` signatures and
the documentation on 2026-10-12. Live device compilation is a separate validation.
