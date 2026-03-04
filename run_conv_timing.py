"""
Collect timing data for conv-layer instances.

For each decided (sat/unsat) conv instance found in all_results_batching.csv, this script:
  - Runs the instance with conv batching ENABLED and records total time + conv time
  - Runs the instance with conv batching DISABLED (with extended timeout) and records the same

Output CSV (conv_timing_data.csv):
  category, onnx_path, vnnlib_path, settings_str,
  batching_total_secs, batching_conv_secs,
  nobatch_total_secs, nobatch_conv_secs,
  batching_result, nobatch_result

Usage:
  PYTHONPATH=src python3 run_conv_timing.py [--top N] [--nobatch-timeout-mult X]

  --top N             Only run the N slowest instances (default 20)
  --nobatch-timeout-mult X  Multiply timeout by X for the no-batching run (default 3.0)
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
# With NUM_PROCESSES=1 (forced in configure_settings), the worker runs in the main
# process and shares the same Timers global.  The worker's _save_timers_to_shared()
# scans Timers.top_level_timer recursively, picking up BOTH the initial
# propagate_up_to_split star pass and all subsequent worker zono passes.
# res.timers therefore contains the full conv cost — no separate main-process read needed.
CONV_TIMERS = [
    'transform_star_batched_conv',
    'transform_star_unbatched',
    'transform_zono_batched_conv',
    'transform_zono_unbatched',
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


def configure_settings(settings_str: str, timeout: float, batching: bool):
    """Apply image settings for the given benchmark, then override batching flag."""
    set_image_settings()
    Settings.TIMING_STATS = True
    Settings.RESULT_SAVE_TIMERS = CONV_TIMERS
    Settings.TIMEOUT = timeout
    Settings.PRINT_OUTPUT = False
    Settings.PRINT_PROGRESS = False
    Settings.CONV_BATCHING_ENABLED = batching
    # Force single-process so main-process conv timers (transform_star in
    # propagate_up_to_split) are visible.  With multiple workers, RESULT_SAVE_TIMERS
    # only captures worker-side zono passes and misses the star-init conv cost entirely
    # (e.g. malbeware shows 0% conv with >1 process even though conv is ~48% of work).
    Settings.NUM_PROCESSES = 1


def run_one(onnx_path: str, vnnlib_path: str, settings_str: str,
            timeout: float, batching: bool) -> dict:
    """Run a single instance and return timing dict."""
    configure_settings(settings_str, timeout, batching)

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

        # With NUM_PROCESSES=1 (forced in configure_settings), the worker runs in-process
        # and shares the same Timers tree as the main process.  The worker's
        # _save_timers_to_shared() call reads Timers.top_level_timer recursively, which
        # includes BOTH the initial propagate_up_to_split star pass AND all worker zono
        # passes.  So res.timers already captures the full conv cost.
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

def load_candidates(top_n: int) -> list:
    """Load the top_n slowest decided conv instances from the batching CSV.

    Skips instances whose ONNX file contains ConvTranspose layers (not yet supported
    in the optimized path — see GOALS.md item 2).
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
            if onnx not in _conv_transpose_cache:
                _conv_transpose_cache[onnx] = has_conv_transpose(onnx)
            if _conv_transpose_cache[onnx]:
                continue
            rows.append((rt, cat, onnx, vnnlib))

    rows.sort(reverse=True)
    return rows[:top_n]


def main():
    parser = argparse.ArgumentParser(description='Conv timing analysis')
    parser.add_argument('--top', type=int, default=20,
                        help='Number of slowest instances to analyse (default 20)')
    parser.add_argument('--nobatch-timeout-mult', type=float, default=3.0,
                        help='Multiply baseline runtime by this for no-batching timeout (default 3.0)')
    args = parser.parse_args()

    candidates = load_candidates(args.top)
    print(f"Collected {len(candidates)} candidate instances (top {args.top} by batching runtime).")

    # Write header immediately so the file exists even if we crash partway
    with open(OUTPUT_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'category', 'onnx_path', 'vnnlib_path', 'settings_str',
            'batching_total_secs', 'batching_conv_secs',
            'nobatch_total_secs', 'nobatch_conv_secs',
            'batching_result', 'nobatch_result',
        ])

    for i, (baseline_rt, cat, onnx_path, vnnlib_path) in enumerate(candidates):
        settings_str = SETTINGS_MAP.get(cat, 'image')
        nobatch_timeout = max(baseline_rt * args.nobatch_timeout_mult, 60.0)

        print(f"\n[{i+1}/{len(candidates)}] {cat} | {os.path.basename(onnx_path)} | "
              f"{os.path.basename(vnnlib_path)}")
        print(f"  Baseline runtime: {baseline_rt:.1f}s  |  No-batch timeout: {nobatch_timeout:.0f}s")

        # ── Run WITH batching ──────────────────────────────────────────────────
        print("  Running WITH batching...", end=' ', flush=True)
        try:
            bat = run_one(onnx_path, vnnlib_path, settings_str,
                          timeout=nobatch_timeout, batching=True)
            print(f"done: {bat['result']}  total={bat['total_secs']:.2f}s  "
                  f"conv={bat['conv_secs']:.2f}s "
                  f"({100*bat['conv_secs']/max(bat['total_secs'],1e-6):.1f}%)")
        except Exception as e:
            print(f"ERROR: {e}")
            bat = {'result': f'error:{e}', 'total_secs': -1.0, 'conv_secs': -1.0}

        # ── Run WITHOUT batching ───────────────────────────────────────────────
        print("  Running WITHOUT batching...", end=' ', flush=True)
        try:
            nob = run_one(onnx_path, vnnlib_path, settings_str,
                          timeout=nobatch_timeout, batching=False)
            print(f"done: {nob['result']}  total={nob['total_secs']:.2f}s  "
                  f"conv={nob['conv_secs']:.2f}s "
                  f"({100*nob['conv_secs']/max(nob['total_secs'],1e-6):.1f}%)")
        except Exception as e:
            print(f"ERROR: {e}")
            nob = {'result': f'error:{e}', 'total_secs': -1.0, 'conv_secs': -1.0}

        # ── Append row to CSV ─────────────────────────────────────────────────
        with open(OUTPUT_CSV, 'a', newline='') as f:
            csv.writer(f).writerow([
                cat, onnx_path, vnnlib_path, settings_str,
                bat['total_secs'], bat['conv_secs'],
                nob['total_secs'], nob['conv_secs'],
                bat['result'], nob['result'],
            ])

        print(f"  -> Row written to {OUTPUT_CSV}")

    print(f"\nFinished. Results in {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
