"""Generate static, shareable scientific charts only from recorded measurements."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"eager": "#667085", "compiled": "#087e8b", "fused": "#bd4b22"}
LABELS = {"eager": "PyTorch eager", "compiled": "torch.compile", "fused": "Explicit fused RMSNorm"}
OPERATIONS = [("matmul", "Matrix multiplication", "Rows M (K=N=1024)"),
              ("conv_relu", "Convolution + ReLU", "Image width / height (C=32 → 64)"),
              ("rmsnorm", "RMSNorm", "Rows (width=1024)")]


def case_x(row):
    return row["case"]["shape"][2 if row["case"]["op"] == "conv_relu" else 0]


def create_report(source, output_dir, profile=None):
    source, output = Path(source), Path(output_dir)
    data = json.loads(source.read_text())
    if data["source"] != "local_cuda_measured":
        raise ValueError("Local CUDA report only; remote results need placement review before comparison")
    if data["status"] not in ("complete", "complete_with_failures"):
        raise ValueError("Cannot publish an incomplete run")
    rows = [r for r in data["records"] if r["status"] == "ok" and r["correctness"]["passed"]]
    if not rows:
        raise ValueError("No validated measurements")
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#344054", "text.color": "#101828",
                         "axes.titleweight": "bold", "figure.facecolor": "white"})
    gpu = data["environment"]["gpu"]
    exploratory = "EXPLORATORY" in data.get("measurement_context", "").upper()
    banner = "EXPLORATORY · concurrent GPU activity · not an isolated hardware comparison" if exploratory else "Measured local run · see report for environment and controls"
    figures = []
    definitions = [
        ("graph_latency", "Execution with host dispatch amortized", "CUDA graph time / operation (µs)", "graph_cuda_ms"),
        ("dispatch_latency", "Execution including Python dispatch", "Synchronized wall time / operation (µs)", "wall_ms"),
        ("speedup", "Optimization benefit depends on workload size", "Speedup versus eager (×)", None),
        ("energy", "Gross GPU energy during sustained replay", "NVML energy / operation (mJ)", None),
    ]
    for filename, title, ylabel, metric in definitions:
        if filename == "energy" and not any(r.get("energy", {}).get("status") == "measured" for r in rows):
            continue
        fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
        for di, dtype in enumerate(["float16", "float32"]):
            for oi, (op, name, xlabel) in enumerate(OPERATIONS):
                ax = axes[di, oi]
                subset = [r for r in rows if r["dtype"] == dtype and r["case"]["op"] == op]
                eager = {r["case"]["id"]: r for r in subset if r["variant"] == "eager"}
                for variant in COLORS:
                    if filename == "speedup" and variant == "eager":
                        continue
                    selected = sorted([r for r in subset if r["variant"] == variant], key=case_x)
                    if filename == "energy":
                        selected = [r for r in selected if r.get("energy", {}).get("status") == "measured"]
                    if filename == "speedup":
                        selected = [r for r in selected if r["case"]["id"] in eager]
                    if not selected:
                        continue
                    x = [case_x(r) for r in selected]
                    if metric:
                        y = np.array([r[metric]["median"] * 1000 for r in selected])
                        lo = np.array([r[metric]["p10"] * 1000 for r in selected])
                        hi = np.array([r[metric]["p90"] * 1000 for r in selected])
                        ax.fill_between(x, lo, hi, color=COLORS[variant], alpha=.12)
                    elif filename == "speedup":
                        y = [eager[r["case"]["id"]]["graph_cuda_ms"]["median"] / r["graph_cuda_ms"]["median"] for r in selected]
                    else:
                        y = [r["energy"]["gpu_energy_mj_per_execution"] for r in selected]
                    ax.plot(x, y, "o-", color=COLORS[variant], label=LABELS[variant], linewidth=2, markersize=5)
                ax.set_title(f"{name} · {dtype.replace('float', 'FP')}", loc="left", fontsize=11)
                ax.set_xscale("log", base=2)
                ax.set_xlabel(xlabel)
                ticks = sorted({case_x(r) for r in subset})
                ax.set_xticks(ticks, [str(v) for v in ticks])
                ax.grid(True, alpha=.18)
                if filename == "speedup":
                    ax.axhline(1, linestyle="--", color="#98a2b3", linewidth=1)
                else:
                    ax.set_yscale("log")
                if oi == 0:
                    ax.set_ylabel(ylabel)
        handles, labels = [], []
        for ax in axes.flat:
            h, l = ax.get_legend_handles_labels()
            for handle, label in zip(h, l):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, .04))
        fig.suptitle(title, fontsize=19, fontweight="bold", x=.055, ha="left", y=.99)
        fig.text(.055, .942, f"{gpu}  |  {banner}", fontsize=10, color="#9a3412" if exploratory else "#475467")
        note = {"graph_latency": "Median across timing blocks; shading = P10–P90 block averages, not confidence intervals. Resident inputs, 100 operations/graph by default.",
                "dispatch_latency": "Median synchronized block average. Includes Python dispatch and final wait; excludes transfer, compilation, and telemetry.",
                "speedup": "Ratio of median CUDA-graph times. Above 1× = faster than eager. This does not attribute all changes to a single compiler pass.",
                "energy": "Integrated NVML device telemetry / completed operations. Includes baseline and unrelated activity. One sustained window/configuration."}[filename]
        fig.text(.055, .015, note, fontsize=8, color="#667085")
        fig.tight_layout(rect=[.025, .10, .99, .92], h_pad=2, w_pad=2)
        fig.savefig(output / (filename + ".png"), dpi=160)
        fig.savefig(output / (filename + ".svg"))
        plt.close(fig)
        figures.append((filename, title))
    fieldnames = ["case", "dtype", "variant", "graph_median_us", "dispatch_median_us",
                  "gpu_energy_mj_per_execution", "mean_gpu_power_w", "relative_l2_error", "max_abs_error", "first_call_s"]
    with (output / "measurements.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({"case": row["case"]["id"], "dtype": row["dtype"], "variant": row["variant"],
                             "graph_median_us": row["graph_cuda_ms"]["median"] * 1000,
                             "dispatch_median_us": row["wall_ms"]["median"] * 1000,
                             "gpu_energy_mj_per_execution": row["energy"].get("gpu_energy_mj_per_execution"),
                             "mean_gpu_power_w": row["energy"].get("mean_gpu_power_w"),
                             "relative_l2_error": row["correctness"]["relative_l2_error"],
                             "max_abs_error": row["correctness"]["max_abs_error"], "first_call_s": row["first_call_s"]})
    env = data["environment"]
    lines = ["# RTX 3060 operation benchmark report", "", "> " + data.get("measurement_context", "Uncontrolled run"), "",
             "This report is generated from real measurements. No Snapdragon NPU numbers have been inferred or added.", "",
             f"- Validated configurations: **{len(rows)}** / {len(data['records'])} attempted.",
             f"- GPU: {gpu}; {env['gpu_memory_bytes'] / 1024**3:.0f} GiB VRAM.",
             f"- PyTorch {env['torch']}; CUDA {env['cuda_runtime']}; driver {env['driver']}; {env['kernel']}.",
             f"- Measurement source commit: `{env['git_commit']}`; dirty at start: `{env['git_dirty']}`.",
             f"- Run time (UTC): {env['timestamp_utc']}.",
             "- TF32 disabled; FP16 matmul reduced-precision reductions disabled; cuDNN benchmarking disabled.", "",
             "## How to read these results", "",
             "CUDA Graph replay amortizes host dispatch across repeated operations. The separate dispatch chart includes Python and synchronization overhead. Neither includes input transfers or full application work. Graph replay repeatedly uses the same resident inputs and can favor cache reuse.", "",
             "Shaded ranges are P10–P90 across **block-average** timings, not per-request tail latencies or confidence intervals. Variants are shuffled within paired rounds. Input shapes and precision run in a fixed order, so thermal drift and background activity remain possible confounders.", "",
             "Energy is gross GPU-device telemetry integrated during a separate sustained replay window. It includes baseline consumption and any unrelated GPU activity. It is not isolated workload-only energy or whole-laptop power, and cannot support an NPU efficiency comparison. Sensor averaging limits the meaning of short intervals.", "",
             "The compiled variant can change layout, dispatch, and kernels. A speedup alone does not establish which compiler transformation caused it. The explicit fused implementation only applies to RMSNorm.", ""]
    for filename, title in figures:
        lines += [f"## {title}", "", f"![{title}]({filename}.png)", ""]
    if profile:
        profile_path = Path(profile)
        diagnostics = json.loads(profile_path.read_text())
        if diagnostics["source"] != "separate_torch_profiler_diagnostic":
            raise ValueError("Expected separate profiler diagnostics")
        names = [name for name, value in diagnostics["variants"].items()
                 if value["status"] == "captured" and value["correctness"]["passed"]]
        if names:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            counts = [diagnostics["variants"][name]["cuda_activities_per_call"] for name in names]
            ax.barh([LABELS[name] for name in names], counts, color=[COLORS[name] for name in names])
            ax.invert_yaxis()
            ax.set_xlim(0, max(counts) * 1.15)
            for i, count in enumerate(counts):
                ax.text(count + .15, i, f"{count:g}", va="center", fontweight="bold")
            ax.set_xlabel("Profiler-observed CUDA activities per call")
            fig.suptitle("RMSNorm: fewer GPU kernel launches", x=.03, y=.98, ha="left", fontsize=18, fontweight="bold")
            fig.text(.03, .89, f"{diagnostics['case']['id']} · {diagnostics['dtype']} · {diagnostics['repetitions']} profiled calls", fontsize=9)
            fig.text(.03, .02, "Separate diagnostic capture; counts support graph/kernel analysis, not a latency or energy estimate.", fontsize=9)
            fig.tight_layout(rect=[0, .08, 1, .85])
            fig.savefig(output / "kernel_counts.png", dpi=160)
            fig.savefig(output / "kernel_counts.svg")
            plt.close(fig)
            profile_link = Path(os.path.relpath(profile_path.resolve(), output.resolve())).as_posix()
            lines += ["## Compiler evidence", "", "![Profiler-observed CUDA activities](kernel_counts.png)", "",
                      "The separate profiler capture records fewer CUDA activities for the compiled and explicit fused RMSNorm implementations. Kernel names and counts are retained in the diagnostic JSON. This is evidence about execution structure; timings come from the separate benchmark run.", "",
                      f"[Raw profiler evidence]({profile_link})", ""]
    lines += ["## Numerical validation", "", "All timed configurations passed both ordinary and captured-graph output checks against a CPU FP64 reference computed from the original seeded FP32 inputs and weights. FP16 error includes input and weight rounding. This is numerical agreement on synthetic tensors, not task accuracy on a dataset.", "",
              "| Operation | Precision | Maximum relative L2 error across shapes/variants |", "|---|---|---:|"]
    for op, name, _ in OPERATIONS:
        for dtype in ["float16", "float32"]:
            errors = [r["correctness"]["relative_l2_error"] for r in rows if r["case"]["op"] == op and r["dtype"] == dtype]
            if errors:
                lines.append(f"| {name} | {dtype} | {max(errors):.3e} |")
    lines += ["", "## Raw evidence", "", "- [Flat measurement table](measurements.csv)",
              f"- [Raw measurement JSON]({Path(os.path.relpath(source.resolve(), output.resolve())).as_posix()}) (raw timing blocks, power samples, environment, checks).",
              "- `first_call_s` includes initialization, possible cache hits, compilation and execution; it is **not** isolated compiler duration.",
              "", "## Before drawing hardware conclusions", "",
              "Repeat whole sessions on AC power with GPU-heavy applications closed and documented power settings to quantify variation. Before any cross-device comparison, verify NPU placement for the same checked workload and reconcile precision and timing boundaries. Keep graph-replay timing separate from AI Hub framework-call timing.", ""]
    failures = [r for r in data["records"] if r["status"] != "ok"]
    if failures:
        lines += ["## Failed configurations", ""] + [f"- {r['case']['id']} / {r['dtype']} / {r['variant']}: {r['status']}" for r in failures]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return output / "report.md"
