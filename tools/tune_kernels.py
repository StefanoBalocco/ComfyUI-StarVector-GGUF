#!/usr/bin/env python3
"""Tune the production SoA GEMV launch parameters (quantization.SOA_GEMV_CONFIG).

Grid-search (bn, blocks, num_warps) for Q4_0/Q5_0/Q8_0 and 1 or 2 decode rows
on random SoA payloads with the StarVector 1B/8B matrix shapes. Timing uses
CUDA-graph replays, as production decoding does. Configurations rank by the
geometric mean of their slowdown against the fastest configuration of each
shape. The search takes about an hour on an RTX 3060, mostly Triton compilation.

Run it from the repository root.
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import quantization as quant  # noqa: E402

# Output x input shape of the quantized decoder linears and output head.
MODEL_SHAPES = {
    '1b': ((2304, 2048), (2048, 2048), (8192, 2048), (2048, 8192), (49156, 2048)),
    '8b': ((4608, 4608), (512, 4608), (18432, 4608), (4608, 18432), (49157, 4608)),
}
GRID = tuple(itertools.product((1, 2, 4, 8), (16, 32, 64), (2, 4, 8)))


def graph_ms(fn, warmups, repeats):
    """CUDA event time per replay of a captured call."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmups):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) / repeats


def sweep(qtype, rows, shapes, warmups, repeats):
    width = quant.BLOCK_BYTES[qtype] - 2
    cases = []
    for n, k in shapes:
        blocks = n * k // 32
        q = torch.randint(0, 256, (blocks * width,), dtype=torch.uint8, device='cuda')
        s = (torch.rand(blocks, device='cuda') * 0.01).half()
        x = torch.randn(rows, k, device='cuda', dtype=torch.float16)
        out = torch.empty(rows, n, device='cuda', dtype=torch.float16)
        times = {}
        for config in GRID:
            bn, chunks, warps = config
            try:
                times[config] = graph_ms(lambda: quant.soa_gemv(x, q, s, out, qtype, bn=bn, blocks=chunks, num_warps=warps), warmups, repeats)
            except Exception as exc:  # Triton rejects some tiles (e.g. register limits)
                print(f'   SKIP {qtype} M={rows} N={n} K={k} {config}: {type(exc).__name__}: {exc}', flush=True)
        reference = F.linear(x.float(), quant.dequantize_soa(q, s.view(torch.uint8), qtype, (n, k), torch.float32))
        cases.append(((n, k), blocks * (width + 2), times, x, q, s, reference))

    def slowdown(config):
        return math.exp(sum(math.log(times[config] / min(times.values())) for _, _, times, *_ in cases) / len(cases))

    best = min(set.intersection(*(set(times) for _, _, times, *_ in cases)), key=slowdown)
    production = quant.SOA_GEMV_CONFIG[qtype, rows]
    print(f'{qtype} M={rows}: best (bn, blocks, num_warps)={best} {slowdown(best):.3f}x of per-shape best; '
          f'production {production} {slowdown(production):.3f}x', flush=True)
    for (n, k), size, times, x, q, s, reference in cases:
        out = torch.empty(rows, n, device='cuda', dtype=torch.float16)
        quant.soa_gemv(x, q, s, out, qtype, bn=best[0], blocks=best[1], num_warps=best[2])
        error = ((out.float() - reference).abs().max() / reference.abs().max()).item()
        print(f'   N={n:>5} K={k:>5}: best {times[best] * 1000:7.1f} us ({size / times[best] / 1e6:4.0f} GB/s), '
              f'production {times[production] * 1000:7.1f} us, relative error {error:.1e}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('1b', '8b', 'all'), default='all', help='Model shapes to tune (default: all)')
    parser.add_argument('--quant', nargs='+', choices=('Q4_0', 'Q5_0', 'Q8_0'), default=['Q4_0', 'Q5_0', 'Q8_0'], help='Quantizations to tune (default: all three)')
    parser.add_argument('--rows', nargs='+', type=int, choices=(1, 2), default=[1, 2], help='Decode row counts to tune (default: 1 and 2)')
    parser.add_argument('--warmups', type=int, default=20, help='Warm-up launches per configuration')
    parser.add_argument('--repeats', type=int, default=200, help='Timed CUDA-graph replays per configuration')
    args = parser.parse_args()
    if not torch.cuda.is_available() or quant.triton is None:
        parser.error('CUDA with Triton is required')
    models = ('1b', '8b') if args.model == 'all' else (args.model,)
    shapes = sorted({shape for model in models for shape in MODEL_SHAPES[model]})
    print(f'GPU: {torch.cuda.get_device_name()} | {len(shapes)} shapes x {len(GRID)} configs', flush=True)
    for qtype in args.quant:
        for rows in args.rows:
            sweep(qtype, rows, shapes, args.warmups, args.repeats)


if __name__ == '__main__':
    main()
