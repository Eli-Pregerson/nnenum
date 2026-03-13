"""
Collect timing data for conv-layer instances.

For each decided (sat/unsat) conv instance found in all_results_batching.csv, this script
runs the instance three ways and records total time + conv time for each:
  - CONV_METHOD='batching'  (greedy spatial grouping + im2col batched matmul)
  - CONV_METHOD='dense'     (unbatched im2col matmul on all generators at once)
  - CONV_METHOD='sparse'    (prebuilt Toeplitz sparse matrix W; scipy sparse @ gen_mat)

Output CSV (conv_timing_data.csv):
  category, onnx_path, vnnlib_path, settings_str,
  batching_total_secs, batching_conv_secs, batching_result,
  dense_total_secs,    dense_conv_secs,    dense_result,
  sparse_total_secs,   sparse_conv_secs,   sparse_result

Usage:
  PYTHONPATH=src python3 run_conv_timing.py [--top N] [--timeout-mult X]

  --top N          Only run the N slowest instances (default 20)
  --timeout-mult X Multiply baseline runtime by X for the timeout of each run (default 3.0)
"""

import sys
import csv
import os
import argparse
import numpy as np

# Allow running from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from nnenum.enumerate import enumerate_network
from nnenum.settings import Settings
from nnenum.onnx_network import load_onnx_network_optimized, load_onnx_network
from nnenum.nnenum import make_spec, set_image_settings

# ── Timer names we want to capture ────────────────────────────────────────────
CONV_TIMERS = [
    'transform_star_batched_conv',
    'transform_star_unbatched',
    'transform_star_sparse_conv',
    'transform_zono_batched_conv',
    'transform_zono_unbatched',
    'transform_zono_sparse_conv',
    'batch_generators_for_conv',
]


RESULTS_CSV = 'vnncomp2025_benchmarks/all_results_batching.csv'
OUTPUT_CSV  = 'conv_timing_data.csv'

# Benchmarks known to have conv layers
CONV_CATS = {'malbeware', 'metaroom_2023', 'relusplitter', 'cgan_2023', 'collins_rul_cnn_2022'}

# Map from benchmark category to settings string used in competition
SETTINGS_MAP = {
    'malbeware':          'malbeware',
    'metaroom_2023':      'metaroom',
    'relusplitter':       'relusplitter',
    'cgan_2023':          'cgan',
    'collins_rul_cnn_2022': 'collins_rul_cnn',
}


def configure_settings(settings_str: str, timeout: float, method: str):
    """Apply image settings for the given benchmark, then set the conv method.

    method: 'batching' | 'dense' | 'sparse'
    """
    set_image_settings()
    Settings.TIMING_STATS = True
    Settings.RESULT_SAVE_TIMERS = CONV_TIMERS
    Settings.TIMEOUT = timeout
    Settings.PRINT_OUTPUT = False
    Settings.PRINT_PROGRESS = False
    Settings.CONV_METHOD = method
    # 'batching' requires CONV_BATCHING_ENABLED=True; other methods ignore it.
    Settings.CONV_BATCHING_ENABLED = (method == 'batching')


def run_one(onnx_path: str, vnnlib_path: str, settings_str: str,
            timeout: float, method: str) -> dict:
    """Run a single instance and return timing dict."""
    configure_settings(settings_str, timeout, method)

    try:
        network = load_onnx_network_optimized(onnx_path)
    except Exception:
        network = load_onnx_network(onnx_path)

    spec_list, input_dtype = make_spec(vnnlib_path, onnx_path)

    result_str = 'none'
    total_secs = 0.0
    conv_secs = 0.0

    remaining = timeout
    for init_box, spec in spec_list:
        if remaining <= 0:
            result_str = 'timeout'
            break

        Settings.TIMEOUT = remaining
        init_box_np = np.array(init_box, dtype=input_dtype)
        init_box_np = network.chw_to_hwc_init_box(init_box_np)

        res = enumerate_network(init_box_np, network, spec)

        # enumerate_network now merges both worker timers (shared.timer_secs) and
        # main-process timers (Timers.top_level_timer) into res.timers, so the full
        # conv cost is captured regardless of NUM_PROCESSES.
        for name in CONV_TIMERS:
            if name in res.timers:
                _, secs = res.timers[name]
                conv_secs += secs

        result_str = res.result_str
        total_secs += res.total_secs
        remaining -= res.total_secs

        if result_str != 'safe':
            break

    # Normalise result string to sat/unsat/timeout/error
    if result_str == 'safe':
        result_str = 'unsat'
    elif 'unsafe' in result_str:
        result_str = 'sat'

    return {
        'result': result_str,
        'total_secs': total_secs,
        'conv_secs': conv_secs,
    }


def has_conv_transpose(onnx_path: str) -> bool:
    """Return True if the ONNX file contains any ConvTranspose nodes."""
    import onnx as onnx_mod
    try:
        model = onnx_mod.load(onnx_path)
        return any(node.op_type == 'ConvTranspose' for node in model.graph.node)
    except Exception:
        return False


# Cache ConvTranspose check results to avoid re-loading the same ONNX file repeatedly
_conv_transpose_cache: dict = {}

def load_candidates(top_n: int, max_baseline: float = np.inf) -> list:
    """Load the top_n slowest decided conv instances from the batching CSV.

    Skips instances whose ONNX file contains ConvTranspose layers (not yet supported
    in the optimized path — see GOALS.md item 2).

    max_baseline: skip instances whose baseline runtime exceeds this threshold.
    """
    rows = []
    with open(RESULTS_CSV, newline='') as f:
        for row in csv.reader(f):
            if len(row) < 6:
                continue
            cat, onnx, vnnlib, overhead, result, runtime = row
            if cat not in CONV_CATS:
                continue
            if result not in ('sat', 'unsat'):
                continue
            try:
                rt = float(runtime)
            except ValueError:
                continue
            if rt > max_baseline:
                continue
            if onnx not in _conv_transpose_cache:
                _conv_transpose_cache[onnx] = has_conv_transpose(onnx)
            if _conv_transpose_cache[onnx]:
                continue
            rows.append((rt, cat, onnx, vnnlib))

    rows.sort(reverse=True)
    return rows[:top_n]


def _fmt(r: dict) -> str:
    """Format a run_one result for console output."""
    if r['total_secs'] < 0:
        return f"ERROR"
    pct = 100 * r['conv_secs'] / max(r['total_secs'], 1e-6)
    return (f"{r['result']}  total={r['total_secs']:.2f}s  "
            f"conv={r['conv_secs']:.2f}s ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Conv timing analysis')
    parser.add_argument('--top', type=int, default=20,
                        help='Number of slowest instances to analyse (default 20)')
    parser.add_argument('--timeout-mult', type=float, default=3.0,
                        help='Multiply baseline runtime by this for per-run timeout (default 3.0)')
    parser.add_argument('--max-baseline', type=float, default=np.inf,
                        help='Skip instances whose baseline runtime exceeds this (default: no limit)')
    args = parser.parse_args()

    candidates = load_candidates(args.top, max_baseline=args.max_baseline)
    print(f"Collected {len(candidates)} candidate instances (top {args.top} by batching runtime).")

    # Write header immediately so the file exists even if we crash partway
    with open(OUTPUT_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'category', 'onnx_path', 'vnnlib_path', 'settings_str',
            'batching_total_secs', 'batching_conv_secs', 'batching_result',
            'dense_total_secs',    'dense_conv_secs',    'dense_result',
            'sparse_total_secs',   'sparse_conv_secs',   'sparse_result',
        ])

    for i, (baseline_rt, cat, onnx_path, vnnlib_path) in enumerate(candidates):
        settings_str = SETTINGS_MAP.get(cat, 'image')
        timeout = max(baseline_rt * args.timeout_mult, 60.0)

        print(f"\n[{i+1}/{len(candidates)}] {cat} | {os.path.basename(onnx_path)} | "
              f"{os.path.basename(vnnlib_path)}")
        print(f"  Baseline runtime: {baseline_rt:.1f}s  |  Per-run timeout: {timeout:.0f}s")

        err_sentinel = {'result': 'error', 'total_secs': -1.0, 'conv_secs': -1.0}

        # ── batching ──────────────────────────────────────────────────────────
        print("  [batching] ", end='', flush=True)
        try:
            bat = run_one(onnx_path, vnnlib_path, settings_str, timeout=timeout, method='batching')
            print(_fmt(bat))
        except Exception as e:
            print(f"ERROR: {e}")
            bat = dict(err_sentinel, result=f'error:{e}')

        # ── dense ─────────────────────────────────────────────────────────────
        print("  [dense]    ", end='', flush=True)
        try:
            den = run_one(onnx_path, vnnlib_path, settings_str, timeout=timeout, method='dense')
            print(_fmt(den))
        except Exception as e:
            print(f"ERROR: {e}")
            den = dict(err_sentinel, result=f'error:{e}')

        # ── sparse ────────────────────────────────────────────────────────────
        print("  [sparse]   ", end='', flush=True)
        try:
            spa = run_one(onnx_path, vnnlib_path, settings_str, timeout=timeout, method='sparse')
            print(_fmt(spa))
        except Exception as e:
            print(f"ERROR: {e}")
            spa = dict(err_sentinel, result=f'error:{e}')

        # ── Append row to CSV ─────────────────────────────────────────────────
        with open(OUTPUT_CSV, 'a', newline='') as f:
            csv.writer(f).writerow([
                cat, onnx_path, vnnlib_path, settings_str,
                bat['total_secs'], bat['conv_secs'], bat['result'],
                den['total_secs'], den['conv_secs'], den['result'],
                spa['total_secs'], spa['conv_secs'], spa['result'],
            ])

        print(f"  -> Row written to {OUTPUT_CSV}")

    print(f"\nFinished. Results in {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
