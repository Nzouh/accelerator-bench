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
LABELS = {"eager": "Eager", "compiled": "Compiled", "fused": "Fused"}
OPERATIONS = [("matmul", "Matmul", "Rows M"),
              ("conv_relu", "Conv + ReLU", "H = W"),
              ("rmsnorm", "RMSNorm", "Rows")]


def case_x(row):
    return row["case"]["shape"][2 if row["case"]["op"] == "conv_relu" else 0]


def save_figure(fig, stem):
    fig.savefig(stem.with_suffix(".png"), dpi=160)
    svg = stem.with_suffix(".svg")
    fig.savefig(svg)
    # Matplotlib emits trailing spaces in SVG paths; normalize generated text.
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")


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
    banner = " · concurrent GPU workloads" if exploratory else ""
    figures = []
    definitions = [
        ("graph_latency", "CUDA graph latency", "Latency (µs)", "graph_cuda_ms"),
        ("dispatch_latency", "Python dispatch latency", "Latency (µs)", "wall_ms"),
        ("speedup", "Speedup vs eager", "Speedup (×)", None),
        ("energy", "GPU energy", "Energy (mJ/op)", None),
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
        if filename == "speedup":
            # Use a zero baseline and matching precision scales per operation;
            # autoscaling around 1x would visually inflate tiny matmul changes.
            for oi in range(len(OPERATIONS)):
                upper = max(1.5, max(axes[di, oi].get_ylim()[1] for di in range(2)))
                for di in range(2):
                    axes[di, oi].set_ylim(0, upper)
        handles, labels = [], []
        for ax in axes.flat:
            h, l = ax.get_legend_handles_labels()
            for handle, label in zip(h, l):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, .04))
        fig.suptitle(title, fontsize=19, fontweight="bold", x=.055, ha="left", y=.99)
        fig.text(.055, .942, f"{gpu}{banner}", fontsize=10, color="#9a3412" if exploratory else "#475467")
        note = {"graph_latency": f"{data['settings']['graph_operations']} ops/graph. Median; band: P10–P90 block means.",
                "dispatch_latency": "Median; band: P10–P90 block means. Includes Python and synchronization.",
                "speedup": "CUDA graph timing.",
                "energy": "Gross NVML energy. One window per configuration."}[filename]
        fig.text(.055, .015, note, fontsize=8, color="#667085")
        fig.tight_layout(rect=[.025, .10, .99, .92], h_pad=2, w_pad=2)
        save_figure(fig, output / filename)
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
    lines = ["# RTX 3060", "",
             f"**{len(rows)}/{len(data['records'])} configurations passed.** {gpu}, {env['gpu_memory_bytes'] / 1024**3:.0f} GiB.", "",
             f"PyTorch {env['torch']} · CUDA {env['cuda_runtime']} · driver {env['driver']}.", "",
             "| Metric | Boundary |", "|---|---|",
             f"| Graph latency | Resident inputs; {data['settings']['graph_operations']} calls per graph |",
             "| Dispatch latency | Python calls + synchronization; no transfers |",
             "| Energy | Separate sustained replay; gross NVML device energy |", "",
             "Bands: P10–P90 block means, not confidence intervals. One session; clocks unlocked.", "",
             "Matmul K=N=1024; convolution C=32→64, kernel=3; RMSNorm width=1024. TF32 off.", ""]
    if exploratory:
        lines += ["**Exploratory: concurrent GPU workloads.**", ""]
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
            ax.set_xlabel("Kernels / call")
            fig.suptitle("RMSNorm kernel count", x=.03, y=.98, ha="left", fontsize=18, fontweight="bold")
            shape_text = '×'.join(str(d) for d in diagnostics['case']['shape'])
            fig.text(.03, .89, f"{diagnostics['dtype'].replace('float', 'FP')} · {shape_text}", fontsize=9)
            fig.text(.03, .02, f"{diagnostics['repetitions']} profiled calls.", fontsize=9)
            fig.tight_layout(rect=[0, .08, 1, .85])
            save_figure(fig, output / "kernel_counts")
            plt.close(fig)
            profile_link = Path(os.path.relpath(profile_path.resolve(), output.resolve())).as_posix()
            lines += ["## Compiler evidence", "", "![Profiler-observed CUDA activities](kernel_counts.png)", "",
                      f"[Raw profiler evidence]({profile_link})", ""]
    lines += ["## Numerical error", "", "CPU FP64 reference; ordinary and captured-graph outputs checked.", "",
              "| Operation | Precision | Max relative L2 error |", "|---|---|---:|"]
    for op, name, _ in OPERATIONS:
        for dtype in ["float16", "float32"]:
            errors = [r["correctness"]["relative_l2_error"] for r in rows if r["case"]["op"] == op and r["dtype"] == dtype]
            if errors:
                lines.append(f"| {name} | {dtype} | {max(errors):.3e} |")
    lines += ["", "## Raw evidence", "", "- [Flat measurement table](measurements.csv)",
              f"- [Raw data and run conditions]({Path(os.path.relpath(source.resolve(), output.resolve())).as_posix()})",
              f"- Source: `{env['git_commit']}`; dirty: `{env['git_dirty']}`; UTC: {env['timestamp_utc']}.",
              "- `first_call_s` includes initialization and possible cache hits; not isolated compile time.",
              "- NPU timing has a different boundary; NPU power is unavailable.", ""]
    failures = [r for r in data["records"] if r["status"] != "ok"]
    if failures:
        lines += ["## Failed configurations", ""] + [f"- {r['case']['id']} / {r['dtype']} / {r['variant']}: {r['status']}" for r in failures]
    (output / "report.md").write_text("\n".join(lines).rstrip() + "\n")
    return output / "report.md"
