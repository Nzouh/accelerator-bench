"""Change only Inductor's layout_optimization flag; verify latency and kernels."""
import argparse
import copy
import gc
import random
from collections import Counter
from pathlib import Path

import torch

from accelerator_bench.runner import capture_replay, environment, save_result, summarize, timed_block
from accelerator_bench.workloads import correctness, fingerprint, load_config, make_workload, reference_output


def run(output):
    if Path(output).exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False
    config = load_config('configs/rtx3060.json')
    result = {'source': 'local_cuda_layout_ablation', 'environment': environment(), 'status': 'running',
        'context': 'AC verified; GPU idle before run. Desktop apps and AI Hub orchestration clients present. Clocks unlocked.',
        'settings': {'rounds': 5, 'blocks': 5, 'replays_per_block': 10, 'calls_per_graph': 100,
                     'variants': {'layout_on': True, 'layout_off': False},
                     'timing': 'CUDA graph replay; resident tensors; no transfers or telemetry'}, 'records': []}
    rng = random.Random(1729)
    with torch.inference_mode():
        for case in [c for c in config['cases'] if c['op'] == 'conv_relu']:
            source, x_original = make_workload(case, config['seed'])
            reference = reference_output(source, x_original)
            for dtype in ['float32', 'float16']:
                torch.compiler.reset()
                module = copy.deepcopy(source).to('cuda', getattr(torch, dtype))
                x = x_original.to('cuda', getattr(torch, dtype))
                prepared = {}
                for variant, flag in result['settings']['variants'].items():
                    compiled = torch.compile(module, fullgraph=True, dynamic=False, options={'layout_optimization': flag})
                    fn = lambda compiled=compiled: compiled(x)
                    check = correctness(fn(), reference, dtype)
                    assert check['passed'], (case['id'], dtype, variant)
                    for _ in range(30):
                        fn()
                    graph, out = capture_replay(fn, 100)
                    graph.replay()
                    torch.cuda.synchronize()
                    graph_check = correctness(out, reference, dtype)
                    assert graph_check['passed']
                    row = {'case': case, 'dtype': dtype, 'variant': variant,
                        'input_weight_sha256': fingerprint(source, x_original),
                        'correctness': check, 'graph_correctness': graph_check, 'raw_blocks': []}
                    prepared[variant] = (fn, graph, out, row)
                for repeat in range(5):
                    order = list(prepared)
                    rng.shuffle(order)
                    for variant in order:
                        fn, graph, out, row = prepared[variant]
                        for _ in range(30):
                            fn()
                        for block in range(5):
                            measured = timed_block(graph.replay, 10)
                            row['raw_blocks'].append({'round': repeat, 'block': block,
                                'cuda_ms': measured['cuda_stream_ms'] / 100})
                for fn, graph, out, row in prepared.values():
                    row['graph_cuda_ms'] = summarize([b['cuda_ms'] for b in row['raw_blocks']])
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
                        for _ in range(5):
                            fn()
                        torch.cuda.synchronize()
                    counts = Counter(e.name for e in prof.events() if str(e.device_type) == 'DeviceType.CUDA')
                    row['diagnostic'] = {'profiled_calls': 5, 'kernels_per_call': sum(counts.values()) / 5,
                                         'kernel_counts': dict(counts)}
                    row['status'] = 'ok'
                    result['records'].append(row)
                    save_result(output, result)
                    print(f"{case['id']} {dtype} {row['variant']}: {row['graph_cuda_ms']['median']*1000:.3f} us; {sum(counts.values())/5:g} kernels", flush=True)
                prepared.clear()
                del module, x, graph, out, fn
                gc.collect()
                torch.cuda.empty_cache()
    result['status'] = 'complete'
    save_result(output, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    run(parser.parse_args().output)
