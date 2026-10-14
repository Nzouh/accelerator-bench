"""Review remote evidence without silently merging incompatible benchmark metrics."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .runner import save_result


def review_profile(profile):
    details = profile.get("execution_detail", [])
    units = sorted({str(item.get("compute_unit", "unknown")) for item in details})
    summary = profile.get("execution_summary", {})
    times = summary.get("all_inference_times", [])
    valid_times = bool(times) and bool(np.isfinite(times).all()) and min(times) > 0
    if not valid_times:
        times = []
    return {
        "reported_compute_units": units,
        "all_reported_entries_on_npu": bool(details) and units == ["NPU"],
        "reported_entry_count": len(details),
        "reported_entries": [{key: row.get(key) for key in ["name", "compute_unit", "execution_cycles"]} for row in details],
        "hub_estimated_inference_us": summary.get("estimated_inference_time"),
        "all_samples_median_us": float(np.median(times)) if times else None,
        "all_samples_p10_us": float(np.percentile(times, 10)) if times else None,
        "all_samples_p90_us": float(np.percentile(times, 90)) if times else None,
        "sample_count": len(times),
        "fusion_verified": False,
        "qualification": "Reported layer placement only. Logical layer names/cycle attribution do not establish physical fusion. Timings include the inference framework boundary.",
    }


def create_hub_report(source_dir, output_dir):
    root, output = Path(source_dir), Path(output_dir)
    data = json.loads((root / "results.json").read_text())
    if data["status"] == "running":
        raise ValueError("Remote run is still in progress")
    case_ids = sorted({r["case_id"] for r in data["records"]})
    if len(case_ids) != 1:
        raise ValueError("This report compares settings for one case; provide a single-case run")
    output.mkdir(parents=True, exist_ok=True)
    reviews = []
    for record in data["records"]:
        profile_path = root / (record["case_id"] + "_" + record["variant"] + "_profile.json")
        if not record.get("correctness", {}).get("passed") or not profile_path.exists():
            continue
        reviewed = review_profile(json.loads(profile_path.read_text()))
        reviews.append({"variant": record["variant"], "case_id": record["case_id"],
                        "inference_job_id": record["inference_job_id"], "correctness": record["correctness"], **reviewed})
    save_result(output / "profile-review.json", {"device": data["plan"]["device"],
                "qairt_version": data["plan"]["qairt_version"], "reviews": reviews,
                "scope": "One job per requested setting; smoke experiment, not a repeated performance study"})
    valid = [r for r in reviews if r["all_reported_entries_on_npu"] and r["sample_count"]]
    lines = ["# NPU pilot run", "",
             f"Device: **{data['plan']['device']}**. Requested QAIRT: **{data['plan']['qairt_version']}**.", "",
             f"Case: `{case_ids[0]}`. FP32 I/O, requested FP16 execution, sustained high performance.", "",
             "One job per setting. Power unavailable; fusion changes unverified.", "",
             "| Requested setting | AI Hub estimate (µs) | Median of returned samples (µs) | P10–P90 (µs) | Numerical check | Reported placement |",
             "|---|---:|---:|---:|---|---|"]
    for r in reviews:
        fmt = lambda value: "unavailable" if value is None else f"{value:.1f}"
        lines.append(f"| {r['variant']} | {fmt(r['hub_estimated_inference_us'])} | {fmt(r['all_samples_median_us'])} | {fmt(r['all_samples_p10_us'])}–{fmt(r['all_samples_p90_us'])} | pass | {', '.join(r['reported_compute_units'])} |")
    if valid:
        fig, ax = plt.subplots(figsize=(10, 5))
        med = np.array([r["all_samples_median_us"] for r in valid])
        lo = np.array([r["all_samples_p10_us"] for r in valid])
        hi = np.array([r["all_samples_p90_us"] for r in valid])
        names = {"optimized": "O3", "prepare_fast": "O1", "no_conv_relu_fold": "O3 / no fold"}
        ax.barh([names[r["variant"]] for r in valid], med, xerr=[med - lo, hi - med],
                color=["#087e8b", "#667085", "#bd4b22"][:len(valid)], capsize=4)
        ax.set_xlabel("Latency (µs)")
        ax.set_xlim(left=0)
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle("NPU compiler settings", x=.03, y=.98, ha="left", fontsize=16, fontweight="bold")
        fig.text(.03, .865, f"{data['plan']['device']} · QAIRT {data['plan']['qairt_version']}", fontsize=10)
        fig.text(.03, .025, "Median; whiskers: P10–P90. One job/setting.", fontsize=9)
        fig.tight_layout(rect=[0, .08, 1, .83])
        from .report import save_figure
        save_figure(fig, output / "npu-ablation")
        plt.close(fig)
        lines += ["", "![NPU ablation](npu-ablation.png)"]
    lines += ["", "All returned samples included. Timing includes the QNN framework call. Chart requires numerical pass and reported NPU placement.", "",
              "## Evidence", "", "- [Machine-readable profile review](profile-review.json)"]
    for r in reviews:
        lines.append(f"- [{r['variant']} inference job](https://workbench.aihub.qualcomm.com/jobs/{r['inference_job_id']}/) (account access may be required)")
    (output / "report.md").write_text("\n".join(lines).rstrip() + "\n")
    return output / "report.md"
