#!/usr/bin/env python3
"""
Validate that nnenum sat/unsat results match alpha-beta-CROWN (abc) results.

Compares all_results.csv (nnenum) against abc_results.csv (alpha-beta-CROWN)
for instances where nnenum returned a definitive result (sat/unsat).
"""

import csv
from collections import defaultdict

def normalize_path(path):
    """Normalize path by extracting just the relative path from benchmarks/ onward"""
    if '/benchmarks/' in path:
        return path.split('/benchmarks/', 1)[1]
    return path

def load_results(filepath):
    """Load results CSV into a dict keyed by (onnx_rel, vnnlib_rel)"""
    results = {}
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 5:
                continue
            onnx = normalize_path(row[1])
            vnnlib = normalize_path(row[2])
            result = row[4]  # Column 4 is the result

            key = (onnx, vnnlib)
            results[key] = result
    return results

def main():
    print("Loading results...")
    abc_results = load_results('vnncomp2025_benchmarks/abc_results.csv')
    nnenum_results = load_results('vnncomp2025_benchmarks/all_results.csv')

    print(f"ABC results: {len(abc_results)} instances")
    print(f"nnenum results: {len(nnenum_results)} instances")

    # Find instances where nnenum has definitive results (sat or unsat)
    nnenum_definitive = {k: v for k, v in nnenum_results.items()
                         if v in ('sat', 'unsat')}

    print(f"\nnnenum definitive results (sat/unsat): {len(nnenum_definitive)}")

    # Count result types in nnenum
    result_counts = defaultdict(int)
    for v in nnenum_results.values():
        result_counts[v] += 1

    print("\nnnenum result breakdown:")
    for result, count in sorted(result_counts.items(), key=lambda x: -x[1]):
        print(f"  {result}: {count}")

    # Validate: check that definitive results match ABC
    matches = 0
    mismatches = 0
    abc_missing = 0

    print("\n=== Validation ===")

    for key, nnenum_result in nnenum_definitive.items():
        onnx, vnnlib = key

        if key not in abc_results:
            abc_missing += 1
            if abc_missing <= 10:  # Only print first 10 warnings
                print(f"WARN: ABC has no result for {onnx} / {vnnlib}")
            continue

        abc_result = abc_results[key]

        if nnenum_result == abc_result:
            matches += 1
        else:
            mismatches += 1
            print(f"MISMATCH: {onnx} / {vnnlib}")
            print(f"  nnenum: {nnenum_result}, ABC: {abc_result}")

    print(f"\n=== Summary ===")
    print(f"Definitive nnenum results: {len(nnenum_definitive)}")
    print(f"  Matches with ABC: {matches}")
    print(f"  Mismatches: {mismatches}")
    print(f"  ABC missing: {abc_missing}")

    if mismatches == 0 and abc_missing == 0:
        print("\n✓ ALL definitive results match ABC!")
        return True
    else:
        print(f"\n✗ Found {mismatches} mismatches")
        return False

if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
