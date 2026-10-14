"""Export identical workloads and submit explicit, auditable AI Hub experiments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .runner import save_result
from .workloads import correctness, fingerprint, load_config, make_workload, reference_output


class ConvNHWC(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).contiguous()


def export(config_path, output_dir, case_filter=None, io_layout="nchw"):
    import onnx
    import onnxruntime as ort
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    if io_layout not in ("nchw", "nhwc"):
        raise ValueError("Unknown IO layout")
    manifest = {"schema_version": 1, "seed": config["seed"], "cases": [], "io_layout": io_layout,
                "source_precision": "float32", "requested_execution_precision": "float16",
                "note": "FP32 IO with explicitly requested NPU FP16 execution; inspect device profile for actual placement and precision"}
    for case in config["cases"]:
        if case_filter and case_filter not in case["id"]:
            continue
        module, x = make_workload(case, config["seed"])
        input_hash = fingerprint(module, x)
        reference = reference_output(module, x)
        if io_layout == "nhwc":
            if case["op"] != "conv_relu":
                raise ValueError("NHWC export requires convolution-only selection")
            module = ConvNHWC(module).eval()
            x = x.permute(0, 2, 3, 1).contiguous()
            reference = reference.permute(0, 2, 3, 1).contiguous()
        model_path = root / (case["id"] + ".onnx")
        with torch.inference_mode():
            torch.onnx.export(module, (x,), model_path, input_names=["x"], output_names=["output"],
                              opset_version=17, dynamo=False)
        onnx.checker.check_model(onnx.load(str(model_path)))
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(model_path), sess_options=session_options,
                                       providers=["CPUExecutionProvider"])
        export_check = correctness(torch.from_numpy(session.run(None, {"x": x.numpy()})[0]), reference, "float32")
        if not export_check["passed"]:
            raise RuntimeError(f"ONNX export differs from reference for {case['id']}")
        arrays_path = root / (case["id"] + ".npz")
        np.savez_compressed(arrays_path, x=x.numpy(), reference=reference.numpy())
        manifest["cases"].append({"case": case, "model": model_path.name, "arrays": arrays_path.name,
                                  "input_weight_sha256": input_hash,
                                  "input_shape": list(x.shape), "io_layout": io_layout,
                                  "onnx_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                                  "arrays_sha256": hashlib.sha256(arrays_path.read_bytes()).hexdigest(),
                                  "onnx_cpu_correctness": export_check})
    if not manifest["cases"]:
        raise ValueError("No matching cases")
    save_result(root / "manifest.json", manifest)
    return manifest


def compile_options(variant):
    common = "--target_runtime qnn_dlc --compute_unit npu --qnn_options "
    options = ["default_graph_htp_precision=FLOAT16", "default_graph_htp_optimizations=O=3"]
    if variant == "no_conv_relu_fold":
        options.append("default_graph_htp_disable_fold_relu_activation_into_conv=true")
    elif variant == "prepare_fast":
        options[1] = "default_graph_htp_optimizations=O=1"
    elif variant != "optimized":
        raise ValueError(variant)
    return common + ";".join(options)


def plan(manifest_path, device, qairt_version, case_filter=None):
    manifest = json.loads(Path(manifest_path).read_text())
    jobs = []
    for item in manifest["cases"]:
        if case_filter and case_filter not in item["case"]["id"]:
            continue
        variants = ["optimized", "prepare_fast"]
        if item["case"]["op"] == "conv_relu":
            variants.append("no_conv_relu_fold")
        for variant in variants:
            jobs.append({"case_id": item["case"]["id"], "variant": variant,
                         "compile_options": compile_options(variant) + f" --qairt_version {qairt_version}",
                         "inference_options": f"--compute_unit npu --qairt_version {qairt_version} --qnn_options context_htp_performance_mode=SUSTAINED_HIGH_PERFORMANCE"})
    if not jobs:
        raise ValueError("No matching cases")
    return {"device": device, "qairt_version": qairt_version, "jobs": jobs,
            "power": "unavailable until independently verified; never inferred from TOPS or TDP",
            "qualification": "These are requested compiler settings, not proof that a pass changed the graph. Inspect profiles and verify NPU placement."}


def execute(manifest_path, device, qairt_version, output_dir, case_filter=None):
    import qai_hub as hub
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError(f"Choose a new output directory: {root}")
    # Check access and resolve an exact available device before uploading anything.
    matches = [d for d in hub.get_devices(name=device) if d.name == device]
    if not matches:
        raise ValueError("Exact device not available; run `hub devices` and use a returned name")
    target = matches[0]
    root.mkdir(parents=True)
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    experiment = plan(manifest_path, device, qairt_version, case_filter)
    lookup = {i["case"]["id"]: i for i in manifest["cases"]}
    result = {"schema_version": 1, "source": "qualcomm_ai_hub", "status": "running",
              "client_version": str(hub.__version__), "plan": experiment, "records": [],
              "comparison_status": "not approved for cross-device aggregation; inspect actual placement, precision, and timing boundaries"}
    save_result(root / "results.json", result)
    for job_spec in experiment["jobs"]:
        item = lookup[job_spec["case_id"]]
        prefix = job_spec["case_id"] + "_" + job_spec["variant"]
        record = {**job_spec, "input_weight_sha256": item["input_weight_sha256"], "status": "running",
                  "placement_verified": False, "energy": {"status": "unavailable"}}
        result["records"].append(record)
        try:
            model_path = manifest_path.parent / item["model"]
            arrays_path = manifest_path.parent / item["arrays"]
            for path, expected in [(model_path, item["onnx_sha256"]), (arrays_path, item["arrays_sha256"])]:
                if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    raise ValueError(f"Artifact hash mismatch: {path.name}")
            arrays = np.load(arrays_path)
            compile_job = hub.submit_compile_job(model=str(model_path), device=target, name=prefix,
                                                  options=job_spec["compile_options"])
            record["compile_job_id"] = compile_job.job_id
            save_result(root / "results.json", result)
            model = compile_job.get_target_model()
            if model is None:
                raise RuntimeError("Compile job failed; inspect saved job ID")
            inference = hub.submit_inference_job(model=model, device=target, inputs={"x": [arrays["x"]]},
                                                  name=prefix, options=job_spec["inference_options"], profile=True)
            record["inference_job_id"] = inference.job_id
            save_result(root / "results.json", result)
            profile = inference.download_profile()
            save_result(root / (prefix + "_profile.json"), profile)
            outputs = inference.download_output_data()
            if not outputs or len(outputs) != 1:
                raise RuntimeError("Expected exactly one output tensor")
            values = next(iter(outputs.values()))
            if len(values) != 1:
                raise RuntimeError("Expected one inference sample")
            record["correctness"] = correctness(torch.from_numpy(values[0]), torch.from_numpy(arrays["reference"]), "float16")
            record["execution_summary"] = profile.get("execution_summary", {})
            record["status"] = "measured_unverified_placement" if record["correctness"]["passed"] else "failed_correctness"
        except Exception as exc:
            record["status"] = "failed"
            # Don't copy exception strings that could contain URLs with credentials.
            record["error_type"] = type(exc).__name__
            print(f"Failed {prefix}: {type(exc).__name__}; inspect saved job IDs", flush=True)
        save_result(root / "results.json", result)
    result["status"] = "requires_profile_review"
    save_result(root / "results.json", result)
    return result
