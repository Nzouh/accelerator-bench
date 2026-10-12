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
    bench.add_argument("--case", dest="case_filter")
    bench.add_argument("--dtypes", nargs="+", choices=["float16", "float32"])
    bench.add_argument("--variants", nargs="+", choices=["eager", "compiled", "fused"])
    sub.add_parser("doctor", help="Report CUDA environment without running a benchmark")
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


if __name__ == "__main__":
    main()
