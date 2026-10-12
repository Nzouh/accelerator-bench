import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Measured accelerator experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    bench = sub.add_parser("run", help="Run seeded CUDA benchmarks")
    bench.add_argument("--config", default="configs/rtx3060.json")
    bench.add_argument("--output", required=True)
    bench.add_argument("--rounds", type=int, default=5)
    bench.add_argument("--blocks", type=int, default=5)
    bench.add_argument("--iterations", type=int, default=100)
    bench.add_argument("--warmup", type=int, default=30)
    bench.add_argument("--energy-seconds", type=float, default=5)
    bench.add_argument("--graph-operations", type=int, default=100)
    bench.add_argument("--context", default="Uncontrolled desktop session; background activity not verified")
    bench.add_argument("--case", dest="case_filter")
    bench.add_argument("--dtypes", nargs="+", choices=["float16", "float32"])
    bench.add_argument("--variants", nargs="+", choices=["eager", "compiled", "fused"])
    sub.add_parser("doctor", help="Report CUDA environment without running a benchmark")
    hub = sub.add_parser("hub", help="Export and run Qualcomm AI Hub experiments")
    hub_sub = hub.add_subparsers(dest="hub_command", required=True)
    export = hub_sub.add_parser("export", help="Export and check ONNX artifacts locally")
    export.add_argument("--config", default="configs/rtx3060.json")
    export.add_argument("--output-dir", default="artifacts/hub")
    export.add_argument("--case", dest="case_filter")
    hub_sub.add_parser("devices", help="List account-accessible devices (requires configured API token)")
    for name in ("plan", "run"):
        p = hub_sub.add_parser(name)
        p.add_argument("--manifest", default="artifacts/hub/manifest.json")
        p.add_argument("--device", required=True)
        p.add_argument("--qairt-version", required=True, help="Pin a supported major.minor QAIRT version")
        p.add_argument("--case", dest="case_filter")
        if name == "run":
            p.add_argument("--output-dir", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "doctor":
        import torch
        from .runner import environment
        print(json.dumps(environment() if torch.cuda.is_available() else {"cuda_available": False, "torch": torch.__version__}, indent=2))
    elif command == "run":
        from .runner import run
        config_path = args.pop("config")
        result = run(config_path, **args)
        if result["status"] != "complete":
            raise SystemExit(1)
    elif command == "hub":
        from . import hub
        action = args.pop("hub_command")
        if action == "devices":
            import qai_hub
            print(json.dumps([{"name": d.name, "os": d.os, "attributes": d.attributes}
                              for d in qai_hub.get_devices()], indent=2))
        elif action == "export":
            config_path = args.pop("config")
            result = hub.export(config_path, **args)
            print(f"Exported and checked {len(result['cases'])} workloads")
        else:
            manifest_path = args.pop("manifest")
            print(json.dumps((hub.plan if action == "plan" else hub.execute)(manifest_path, **args), indent=2))


if __name__ == "__main__":
    main()
