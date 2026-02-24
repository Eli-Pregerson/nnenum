#!/usr/bin/env python3
"""Analyze the pattern of mismatches between nnenum and ABC"""

import csv
from collections import defaultdict, Counter

def normalize_path(path):
    """Normalize path by extracting just the relative path from benchmarks/ onward"""
    if '/benchmarks/' in path:
        return path.split('/benchmarks/', 1)[1]
    return path

def load_results(filepath):
    """Load results CSV"""
    results = {}
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 5:
                continue
            benchmark = row[0]
            onnx = normalize_path(row[1])
            vnnlib = normalize_path(row[2])
            result = row[4]

            key = (onnx, vnnlib)
            results[key] = (benchmark, result)
    return results

def main():
    abc_results = load_results('vnncomp2025_benchmarks/abc_results.csv')
    nnenum_results = load_results('vnncomp2025_benchmarks/all_results.csv')

    # Find mismatches
    nnenum_definitive = {k: v for k, v in nnenum_results.items()
                         if v[1] in ('sat', 'unsat')}

    mismatch_patterns = Counter()
    mismatch_by_benchmark = defaultdict(int)

    for key, (benchmark, nnenum_result) in nnenum_definitive.items():
        if key in abc_results:
            abc_benchmark, abc_result = abc_results[key]
            if nnenum_result != abc_result:
                pattern = f"nnenum={nnenum_result}, abc={abc_result}"
                mismatch_patterns[pattern] += 1
                mismatch_by_benchmark[benchmark] += 1

    print("=== Mismatch Patterns ===")
    for pattern, count in mismatch_patterns.most_common():
        print(f"{pattern}: {count}")

    print("\n=== Mismatches by Benchmark ===")
    for benchmark, count in sorted(mismatch_by_benchmark.items(), key=lambda x: -x[1]):
        print(f"{benchmark}: {count}")

if __name__ == '__main__':
    main()
