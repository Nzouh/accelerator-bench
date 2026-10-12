"""Separate diagnostic profiling; never used as benchmark latency evidence."""
from collections import defaultdict
from pathlib import Path

import torch

from .runner import environment, save_result
from .workloads import correctness, load_config, make_workload, reference_output


def profile_case(config_path, case_id, dtype, output, repetitions=5):
    if Path(output).exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False
    config = load_config(config_path)
    case = next((c for c in config["cases"] if c["id"] == case_id), None)
    if case is None:
        raise ValueError("Use an exact case ID from the config")
    source, original_x = make_workload(case, config["seed"])
    reference = reference_output(source, original_x)
    module = source.to("cuda", getattr(torch, dtype))
    x = original_x.to("cuda", getattr(torch, dtype))
    compiled = torch.compile(module, fullgraph=True, dynamic=False, mode="default")
    functions = {"eager": lambda: module(x), "compiled": lambda: compiled(x)}
    if case["op"] == "rmsnorm":
        from .kernels import fused_rmsnorm
        functions["fused"] = lambda: fused_rmsnorm(x, module.weight, module.eps)
    result = {"source": "separate_torch_profiler_diagnostic", "environment": environment(),
              "case": case, "dtype": dtype, "repetitions": repetitions,
              "note": "Counts are profiler-observed CUDA activities. Profile timings have instrumentation overhead and are not benchmark results.", "variants": {}}
    for name, fn in functions.items():
        with torch.inference_mode():
            actual = fn()
            check = correctness(actual, reference, dtype)
            if not check["passed"]:
                raise RuntimeError(f"Incorrect {name} output")
            for _ in range(20):
                fn()
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(repetitions):
                    fn()
                torch.cuda.synchronize()
        groups = defaultdict(lambda: {"count": 0, "profile_total_us": 0.0})
        for event in prof.events():
            if str(event.device_type) == "DeviceType.CUDA":
                group = groups[event.name]
                group["count"] += 1
                group["profile_total_us"] += event.device_time_total
        result["variants"][name] = {
            "correctness": check,
            "cuda_activities_per_call": sum(v["count"] for v in groups.values()) / repetitions,
            "activities": [{"name": key, **value} for key, value in groups.items()],
            "status": "captured" if groups else "unavailable_no_cuda_events",
        }
        print(f"{name}: {result['variants'][name]['cuda_activities_per_call']} CUDA activities/call", flush=True)
    save_result(output, result)
    return result
