"""CUDA timing with raw blocks, paired rounds, and checked outputs."""
from __future__ import annotations

import copy
import datetime as dt
import gc
import json
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .telemetry import energy_measurement
from .workloads import correctness, fingerprint, load_config, make_workload, operation_count, reference_output


def shell_output(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def environment():
    props = torch.cuda.get_device_properties(0)
    return {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": platform.python_version(), "os": platform.system(), "kernel": platform.release(),
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": props.name, "gpu_memory_bytes": props.total_memory,
        "compute_capability": f"{props.major}.{props.minor}",
        "driver": shell_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "git_commit": shell_output(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(shell_output(["git", "status", "--porcelain"])),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "torch_cpu_threads": torch.get_num_threads(),
        "power_limit_or_clock_changes": "none requested by this tool",
    }


def summarize(samples):
    values = np.asarray(samples, dtype=float)
    return {"median": float(np.median(values)), "p10": float(np.percentile(values, 10)),
            "p90": float(np.percentile(values, 90)), "min": float(values.min()),
            "max": float(values.max()), "n": len(samples)}


def timed_block(fn, iterations):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000 / iterations
    return {"cuda_stream_ms": start.elapsed_time(end) / iterations, "wall_ms": wall_ms}


def capture_replay(fn, operations=100):
    """A single captured graph contains many calls, amortizing host launch cost."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(operations):
            output = fn()
    torch.cuda.synchronize()
    # Retain output and graph pool together for the entire replay lifetime.
    return graph, output


def save_result(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def run(config_path, output, *, rounds=5, blocks=5, iterations=100, warmup=30,
        energy_seconds=5.0, case_filter=None, dtypes=None, variants=None, graph_operations=100,
        context="Uncontrolled desktop session; background activity not verified"):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch and a supported NVIDIA GPU are required; CPU timing is never substituted")
    if min(rounds, blocks, iterations, warmup, graph_operations) < 1 or energy_seconds < 0:
        raise ValueError("Timing counts must be positive and energy duration nonnegative")
    if Path(output).exists():
        raise FileExistsError(f"Refusing to overwrite {output}; choose a new run path")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False
    config = load_config(config_path)
    selected = [c for c in config["cases"] if not case_filter or case_filter in c["id"]]
    if not selected:
        raise ValueError("No matching cases")
    selected_dtypes = dtypes or config["dtypes"]
    selected_variants = variants or config["variants"]
    if not set(selected_dtypes) <= {"float16", "float32"}:
        raise ValueError("Unknown dtype")
    if not set(selected_variants) <= {"eager", "compiled", "fused"}:
        raise ValueError("Unknown variant")
    result = {
        "schema_version": 1, "source": "local_cuda_measured", "environment": environment(),
        "measurement_context": context,
        "config": config, "settings": {"rounds": rounds, "blocks": blocks, "iterations": iterations,
        "warmup": warmup, "energy_seconds": energy_seconds, "case_filter": case_filter,
        "dtypes": selected_dtypes, "variants": selected_variants, "graph_operations": graph_operations,
        "timing_boundary": "Repeated Python dispatch on resident CUDA inputs; CUDA event span includes device idle gaps; synchronized wall includes host dispatch and final wait; no transfer, compilation, or power sampling in timing",
        "graph_timing_boundary": "100 calls captured per CUDA graph by default; event span per replay divided by calls, reducing host dispatch gaps. Same repeated resident inputs; not application end-to-end latency.",
        "execution_order": "Cases and dtype order fixed; variant order shuffled per paired round using seed"},
        "records": [], "status": "running",
    }
    save_result(output, result)
    rng = random.Random(config["seed"])
    for case in selected:
        source, original_x = make_workload(case, config["seed"])
        reference = reference_output(source, original_x)
        for dtype_name in selected_dtypes:
            torch.compiler.reset()
            module = copy.deepcopy(source).to("cuda", getattr(torch, dtype_name))
            x = original_x.to("cuda", getattr(torch, dtype_name))
            prepared = {}
            for variant in selected_variants:
                if variant == "fused" and case["op"] != "rmsnorm":
                    continue
                row = {"case": case, "dtype": dtype_name, "variant": variant,
                       "input_weight_sha256": fingerprint(source, original_x),
                       "useful_flops": operation_count(case), "raw_blocks": [], "raw_graph_blocks": [], "status": "preparing"}
                result["records"].append(row)
                try:
                    if variant == "eager":
                        fn = lambda: module(x)
                    elif variant == "compiled":
                        compiled = torch.compile(module, fullgraph=True, dynamic=False, mode="default")
                        fn = lambda compiled=compiled: compiled(x)
                    else:
                        from .kernels import fused_rmsnorm
                        fn = lambda: fused_rmsnorm(x, module.weight, module.eps)
                    with torch.inference_mode():
                        torch.cuda.synchronize()
                        first = time.perf_counter()
                        actual = fn()
                        torch.cuda.synchronize()
                        row["first_call_s"] = time.perf_counter() - first
                        row["correctness"] = correctness(actual, reference, dtype_name)
                        if not row["correctness"]["passed"]:
                            raise RuntimeError("Numerical validation failed; excluded from benchmarks")
                        del actual
                        for _ in range(warmup):
                            fn()
                        torch.cuda.synchronize()
                    with torch.inference_mode():
                        graph, graph_output = capture_replay(fn, graph_operations)
                        graph.replay()
                        torch.cuda.synchronize()
                        row["graph_correctness"] = correctness(graph_output, reference, dtype_name)
                        if not row["graph_correctness"]["passed"]:
                            raise RuntimeError("CUDA graph output validation failed")
                    prepared[variant] = (fn, row, graph, graph_output)
                    row["status"] = "ready"
                    print(f"ready {case['id']} {dtype_name} {variant}", flush=True)
                except Exception as exc:
                    row["status"] = "failed"
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    print(f"FAILED {case['id']} {dtype_name} {variant}: {row['error'][:200]}", flush=True)
                save_result(output, result)
            with torch.inference_mode():
                for round_index in range(rounds):
                    order = list(prepared)
                    rng.shuffle(order)
                    for position, variant in enumerate(order):
                        fn, row, graph, graph_output = prepared[variant]
                        for _ in range(warmup):
                            fn()
                        for _ in range(blocks):
                            measurement = timed_block(fn, iterations)
                            measurement.update(round=round_index, order=position)
                            row["raw_blocks"].append(measurement)
                            replay = timed_block(graph.replay, max(3, iterations // 10))
                            replay = {key: value / graph_operations for key, value in replay.items()}
                            replay.update(round=round_index, order=position)
                            row["raw_graph_blocks"].append(replay)
                energy_order = list(prepared)
                rng.shuffle(energy_order)
                for variant in energy_order:
                    fn, row, graph, graph_output = prepared[variant]
                    row["cuda_stream_ms"] = summarize([b["cuda_stream_ms"] for b in row["raw_blocks"]])
                    row["wall_ms"] = summarize([b["wall_ms"] for b in row["raw_blocks"]])
                    row["graph_cuda_ms"] = summarize([b["cuda_stream_ms"] for b in row["raw_graph_blocks"]])
                    row["graph_wall_ms"] = summarize([b["wall_ms"] for b in row["raw_graph_blocks"]])
                    row["energy"] = energy_measurement(graph.replay, energy_seconds)
                    if row["energy"]["status"] == "measured":
                        row["energy"]["executions"] *= graph_operations
                        row["energy"]["gpu_energy_mj_per_execution"] /= graph_operations
                        row["energy"]["wall_ms_per_execution"] /= graph_operations
                        row["energy"]["workload"] = "CUDA graph replay; executions count original workload calls, not graph launches"
                    row["status"] = "ok"
                    print(f"done {case['id']} {dtype_name} {variant}: {row['graph_cuda_ms']['median']:.5f} ms graph / {row['wall_ms']['median']:.5f} ms wall", flush=True)
                    save_result(output, result)
            prepared.clear()
            del module, x
            gc.collect()
            torch.cuda.empty_cache()
    result["status"] = "complete_with_failures" if any(r["status"] == "failed" for r in result["records"]) else "complete"
    save_result(output, result)
    return result
