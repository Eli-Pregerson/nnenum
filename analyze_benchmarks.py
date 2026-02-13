#!/usr/bin/env python3
"""
Analyze VNN-COMP benchmark results to identify optimization priorities

Compares alpha-beta-CROWN (abc) vs nnenum performance

Only compares cases where BOTH tools got valid results (sat/unsat/timeout).
Bug fixes changed nnenum behavior, so invalid results (errors) are not comparable.

Usage:
    python analyze_benchmarks.py                        # Analyze all benchmarks (pre-changes)
    python analyze_benchmarks.py --post                 # Analyze post-changes results
    python analyze_benchmarks.py cifar relusplitter     # Focus on specific suites
    python analyze_benchmarks.py --post cifar100_2024   # Post-changes, specific suite
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict, Counter

def normalize_path(path):
    """Normalize path for comparison (extract relative benchmark path)"""
    # Extract from "benchmarks/" onwards
    if 'benchmarks/' in path:
        return path.split('benchmarks/', 1)[1]
    return path

def is_valid_result(result):
    """Check if result is valid (sat, unsat, or timeout)

    Note: timeout is valid - the tool ran but didn't finish.
    Invalid results are errors like error_exit_code_1, no_result_in_file, etc.
    """
    return result.lower() in ['sat', 'unsat', 'timeout']

def get_benchmark_suite(benchmark_name):
    """Extract benchmark suite name (e.g., 'cifar', 'relusplitter')"""
    # Benchmark names are typically like "cifar_spec_1" or "relusplitter_0"
    parts = benchmark_name.split('_')
    return parts[0] if parts else benchmark_name

def parse_csv(filename):
    """Parse benchmark CSV file"""
    results = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 6:
                benchmark, onnx_path, vnnlib_path, overhead, result, time = row
                results.append({
                    'benchmark': benchmark,
                    'onnx': onnx_path,
                    'onnx_norm': normalize_path(onnx_path),
                    'vnnlib': vnnlib_path,
                    'vnnlib_norm': normalize_path(vnnlib_path),
                    'result': result,
                    'time': float(time) if time != 'timeout' else 999999.0
                })
    return results

def compare_results(abc_results, nnenum_results, filter_suites=None):
    """Compare ABC vs nnenum performance

    Args:
        abc_results: ABC benchmark results
        nnenum_results: nnenum benchmark results
        filter_suites: Optional list of benchmark suites to focus on (e.g., ['cifar', 'relusplitter'])
    """

    # Create lookup by normalized (onnx, vnnlib) for matching
    abc_map = {(r['onnx_norm'], r['vnnlib_norm']): r for r in abc_results}
    nnenum_map = {(r['onnx_norm'], r['vnnlib_norm']): r for r in nnenum_results}

    # Analysis
    gaps = []  # (gap_ratio, benchmark_name, abc_time, nnenum_time, onnx_path)
    abc_wins_nnenum_invalid = []  # ABC valid, nnenum invalid/timeout
    both_timeout = []
    nnenum_faster = []

    # Track filtering stats
    skipped_invalid_abc = 0
    skipped_invalid_nnenum = 0
    skipped_suite_filter = 0

    for key, abc in abc_map.items():
        if key not in nnenum_map:
            continue

        nnenum = nnenum_map[key]

        # CRITICAL: Only compare if BOTH tools got valid results (sat/unsat)
        if not is_valid_result(abc['result']):
            skipped_invalid_abc += 1
            continue

        if not is_valid_result(nnenum['result']):
            skipped_invalid_nnenum += 1
            # Still track if ABC succeeded but nnenum didn't
            if abc['time'] < 500:
                abc_wins_nnenum_invalid.append((abc['benchmark'], abc['time'], abc['onnx'], nnenum['result']))
            continue

        # Optional: Filter by benchmark suite
        if filter_suites:
            suite = get_benchmark_suite(abc['benchmark'])
            if suite not in filter_suites:
                skipped_suite_filter += 1
                continue

        abc_time = abc['time']
        nnenum_time = nnenum['time']

        # Skip if ABC timed out (shouldn't happen if result is valid, but just in case)
        if abc_time > 500:
            if nnenum_time > 500:
                both_timeout.append((abc['benchmark'], abc['onnx']))
            continue

        # ABC succeeded with valid result
        if nnenum_time > 500:
            # nnenum timed out but ABC didn't - HIGH PRIORITY
            abc_wins_nnenum_invalid.append((abc['benchmark'], abc_time, abc['onnx'], 'timeout'))
        else:
            # Both succeeded with valid results - calculate gap
            if nnenum_time > abc_time * 1.5:  # nnenum at least 50% slower
                gap_ratio = nnenum_time / abc_time
                gaps.append((gap_ratio, abc['benchmark'], abc_time, nnenum_time, abc['onnx']))
            elif abc_time > nnenum_time * 1.5:  # nnenum faster!
                speedup = abc_time / nnenum_time
                nnenum_faster.append((speedup, abc['benchmark'], abc_time, nnenum_time, abc['onnx']))

    stats = {
        'skipped_invalid_abc': skipped_invalid_abc,
        'skipped_invalid_nnenum': skipped_invalid_nnenum,
        'skipped_suite_filter': skipped_suite_filter
    }

    return gaps, abc_wins_nnenum_invalid, both_timeout, nnenum_faster, stats

def is_conv_benchmark(onnx_path):
    """Check if benchmark likely uses Conv layers"""
    path_lower = onnx_path.lower()
    conv_indicators = ['conv', 'cnn', 'resnet', 'vgg', 'cgan', 'traffic']
    return any(indicator in path_lower for indicator in conv_indicators)

def main():
    base_dir = Path('vnncomp2025_benchmarks')

    # Parse command-line arguments for suite filtering and version
    args = sys.argv[1:]
    use_post = '--post' in args
    if use_post:
        args.remove('--post')
    filter_suites = args if args else None

    # Load results
    print("Loading benchmark results...")
    abc_results = parse_csv(base_dir / 'abc_results.csv')
    nnenum_file = 'all_results.csv' if use_post else 'all_results_pre.csv'
    nnenum_results = parse_csv(base_dir / nnenum_file)
    nnenum_label = 'post-changes' if use_post else 'pre-changes'

    print(f"ABC results: {len(abc_results)} benchmarks")
    print(f"nnenum {nnenum_label}: {len(nnenum_results)} benchmarks")

    # Show distribution of benchmark suites in dataset
    suite_counts = Counter(get_benchmark_suite(r['benchmark']) for r in abc_results)
    print(f"\nBenchmark suites in dataset:")
    for suite, count in sorted(suite_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        mark = " 🎯" if filter_suites and suite in filter_suites else ""
        print(f"  {suite:30} {count:4} benchmarks{mark}")
    if len(suite_counts) > 10:
        print(f"  ... and {len(suite_counts) - 10} more suites")

    # Compare
    print("\n" + "="*80)
    if filter_suites:
        print(f"PERFORMANCE ANALYSIS: Focusing on {', '.join(filter_suites)}")
    else:
        print("PERFORMANCE ANALYSIS: All benchmarks")
    print(f"alpha-beta-CROWN vs nnenum ({nnenum_label})")
    print("NOTE: Only comparing valid results (sat/unsat/timeout)")
    print("="*80)

    gaps, abc_wins, both_timeout, nnenum_faster, stats = compare_results(
        abc_results, nnenum_results, filter_suites=filter_suites
    )

    # Show filtering stats
    print(f"\nFiltering statistics:")
    print(f"  Skipped (ABC invalid result): {stats['skipped_invalid_abc']}")
    print(f"  Skipped (nnenum invalid result): {stats['skipped_invalid_nnenum']}")
    if filter_suites:
        print(f"  Skipped (suite filter): {stats['skipped_suite_filter']}")
    total_compared = len(gaps) + len(abc_wins) + len(both_timeout) + len(nnenum_faster)
    print(f"  Total compared (both valid): {total_compared}")

    # Sort by gap ratio (worst first)
    gaps.sort(reverse=True)
    abc_wins.sort(key=lambda x: x[1])  # Sort by ABC time
    nnenum_faster.sort(reverse=True)

    # Filter abc_wins by suite if requested
    abc_wins_filtered = abc_wins
    if filter_suites:
        abc_wins_filtered = [(b, t, o, r) for b, t, o, r in abc_wins
                             if get_benchmark_suite(b) in filter_suites]

    # Print high-priority issues (ABC valid, nnenum invalid/timeout)
    print(f"\n🔴 HIGH PRIORITY: ABC valid result, nnenum invalid/timeout")
    if filter_suites:
        print(f"   Total (all suites): {len(abc_wins)} cases")
        print(f"   In target suites: {len(abc_wins_filtered)} cases")
    else:
        print(f"   Total: {len(abc_wins)} cases")
    print("-" * 80)

    for i, (benchmark, abc_time, onnx_path, nnenum_result) in enumerate(abc_wins_filtered[:20], 1):
        conv_mark = "🔧 CONV" if is_conv_benchmark(onnx_path) else ""
        print(f"{i:3}. {benchmark:40} | ABC: {abc_time:7.2f}s | nnenum: {nnenum_result} {conv_mark}")
        if i <= 5:
            print(f"     {Path(onnx_path).name}")

    if len(abc_wins_filtered) > 20:
        print(f"     ... and {len(abc_wins_filtered) - 20} more")

    # Print large performance gaps
    print(f"\n⚠️  LARGE GAPS: Both succeed, nnenum >1.5x slower ({len(gaps)} cases)")
    print("-" * 80)
    print("Top 20 slowest (sorted by gap ratio):")
    for i, (ratio, benchmark, abc_time, nnenum_time, onnx_path) in enumerate(gaps[:20], 1):
        conv_mark = "🔧 CONV" if is_conv_benchmark(onnx_path) else ""
        print(f"{i:3}. {benchmark:40} | {ratio:5.1f}x slower | ABC: {abc_time:6.2f}s nnenum: {nnenum_time:7.2f}s {conv_mark}")

    if len(gaps) > 20:
        print(f"     ... and {len(gaps) - 20} more")

    # Print where nnenum is faster (for encouragement!)
    if nnenum_faster:
        print(f"\n✅ nnenum FASTER than ABC ({len(nnenum_faster)} cases)")
        print("-" * 80)
        for i, (speedup, benchmark, abc_time, nnenum_time, onnx_path) in enumerate(nnenum_faster[:10], 1):
            print(f"{i:3}. {benchmark:40} | {speedup:5.1f}x faster | ABC: {abc_time:6.2f}s nnenum: {nnenum_time:6.2f}s")

    # Conv-specific analysis
    print(f"\n🔧 CONV BENCHMARK ANALYSIS")
    print("-" * 80)
    conv_invalid = [x for x in abc_wins_filtered if is_conv_benchmark(x[2])]
    conv_gaps = [x for x in gaps if is_conv_benchmark(x[4])]

    print(f"Conv benchmarks where nnenum invalid/timeout: {len(conv_invalid)}")
    print(f"Conv benchmarks with large gaps (>1.5x): {len(conv_gaps)}")

    if conv_gaps:
        avg_gap = sum(x[0] for x in conv_gaps) / len(conv_gaps)
        print(f"Average slowdown on Conv benchmarks: {avg_gap:.1f}x")
        print(f"\nTop 10 Conv gaps:")
        for i, (ratio, benchmark, abc_time, nnenum_time, onnx_path) in enumerate(conv_gaps[:10], 1):
            print(f"  {i:2}. {benchmark:40} {ratio:5.1f}x | ABC: {abc_time:6.2f}s nnenum: {nnenum_time:7.2f}s")

    # Summary
    print(f"\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total benchmarks in dataset: {len(abc_results)}")
    print(f"Benchmarks with both tools valid: {total_compared}")
    print(f"  ABC valid, nnenum invalid/timeout: {len(abc_wins)} (HIGH PRIORITY)")
    print(f"    - Conv benchmarks: {len(conv_invalid)}")
    print(f"  Both valid, nnenum >1.5x slower: {len(gaps)}")
    print(f"    - Conv benchmarks: {len(conv_gaps)}")
    print(f"  Both valid, nnenum faster: {len(nnenum_faster)}")
    print(f"  Both timeout: {len(both_timeout)}")

    print(f"\n💡 OPTIMIZATION PRIORITY:")
    if abc_wins:
        print(f"   1. Fix cases where ABC gets valid result but nnenum doesn't ({len(abc_wins)} cases)")
        if abc_wins[0][1] < 100:
            print(f"      - ABC solves these quickly (fastest: {abc_wins[0][1]:.1f}s)")
    if conv_invalid or conv_gaps:
        print(f"   2. Conv layer optimization (affects {len(conv_invalid) + len(conv_gaps)} benchmarks)")
    if gaps:
        avg_top_gap = sum(x[0] for x in gaps[:20]) / min(20, len(gaps))
        print(f"   3. Reduce performance gap (top cases are {avg_top_gap:.1f}x slower)")

if __name__ == '__main__':
    main()
