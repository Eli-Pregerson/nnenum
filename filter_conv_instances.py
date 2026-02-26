#!/usr/bin/env python3
"""
Create filtered instances.csv files that only include networks with Conv layers.

This script scans all benchmark directories, identifies ONNX models with Conv layers,
and creates new instances_conv.csv files containing only those instances.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import onnx
from pathlib import Path

def has_conv_layers(onnx_path):
    """Check if an ONNX model contains Conv layers."""
    try:
        model = onnx.load(onnx_path)
        for node in model.graph.node:
            if node.op_type == 'Conv':
                return True
        return False
    except Exception as e:
        print(f"  Error loading {onnx_path}: {e}")
        return False

def get_model_info(onnx_path):
    """Get information about model layers."""
    try:
        model = onnx.load(onnx_path)
        layer_types = {}
        for node in model.graph.node:
            layer_types[node.op_type] = layer_types.get(node.op_type, 0) + 1
        return layer_types
    except Exception as e:
        return {"error": str(e)}

def filter_benchmark_instances(benchmark_dir):
    """Filter instances in a benchmark directory to only include Conv networks."""

    benchmark_path = Path(benchmark_dir)
    instances_file = benchmark_path / "instances.csv"

    if not instances_file.exists():
        print(f"  No instances.csv found")
        return None

    print(f"\nProcessing: {benchmark_path.name}")
    print("=" * 80)

    # Read instances
    with open(instances_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"  Total instances: {len(lines)}")

    # Group by ONNX file
    onnx_files = {}
    for line in lines:
        parts = line.split(',')
        if len(parts) >= 2:
            onnx_rel_path = parts[0].strip()
            if onnx_rel_path not in onnx_files:
                onnx_files[onnx_rel_path] = []
            onnx_files[onnx_rel_path].append(line)

    print(f"  Unique ONNX files: {len(onnx_files)}")

    # Check each unique ONNX file for Conv layers
    conv_instances = []
    conv_onnx_files = []

    for onnx_rel_path, instance_lines in onnx_files.items():
        onnx_abs_path = benchmark_path / onnx_rel_path

        if not onnx_abs_path.exists():
            print(f"  ⚠️  File not found: {onnx_rel_path}")
            continue

        # Check for Conv layers
        has_conv = has_conv_layers(onnx_abs_path)

        if has_conv:
            conv_instances.extend(instance_lines)
            conv_onnx_files.append(onnx_rel_path)

            # Get model info
            info = get_model_info(onnx_abs_path)
            layer_info = ", ".join([f"{k}={v}" for k, v in sorted(info.items())])
            print(f"  ✓ {onnx_rel_path}: {layer_info}")
        else:
            info = get_model_info(onnx_abs_path)
            layer_info = ", ".join([f"{k}={v}" for k, v in sorted(info.items())])
            print(f"  ✗ {onnx_rel_path}: {layer_info} (no Conv)")

    print(f"\n  Conv networks: {len(conv_onnx_files)}/{len(onnx_files)}")
    print(f"  Conv instances: {len(conv_instances)}/{len(lines)}")

    if conv_instances:
        # Write filtered instances
        output_file = benchmark_path / "instances_conv.csv"
        with open(output_file, 'w') as f:
            f.write('\n'.join(conv_instances) + '\n')

        print(f"  ✓ Written to: {output_file}")
        return len(conv_instances)
    else:
        print(f"  ⚠️  No Conv instances found")
        return 0

def main():
    benchmarks_dir = Path("vnncomp2025_benchmarks/benchmarks")

    if not benchmarks_dir.exists():
        print(f"Error: {benchmarks_dir} not found")
        sys.exit(1)

    print("=" * 80)
    print("FILTERING INSTANCES WITH CONV LAYERS")
    print("=" * 80)

    # Get all benchmark directories
    benchmark_dirs = sorted([d for d in benchmarks_dir.iterdir() if d.is_dir()])

    print(f"\nFound {len(benchmark_dirs)} benchmark directories")

    total_conv_instances = 0
    benchmarks_with_conv = []

    for benchmark_dir in benchmark_dirs:
        conv_count = filter_benchmark_instances(benchmark_dir)
        if conv_count and conv_count > 0:
            total_conv_instances += conv_count
            benchmarks_with_conv.append((benchmark_dir.name, conv_count))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nBenchmarks with Conv layers: {len(benchmarks_with_conv)}")
    print(f"Total Conv instances: {total_conv_instances}")

    if benchmarks_with_conv:
        print("\nBenchmarks with Conv instances:")
        for name, count in benchmarks_with_conv:
            print(f"  {name}: {count} instances")

    print("\n" + "=" * 80)
    print("RECOMMENDED TEST INSTANCES")
    print("=" * 80)

    if benchmarks_with_conv:
        print("\nTo test the flatten fix, run these benchmarks:")
        for name, count in benchmarks_with_conv[:5]:  # Show first 5
            instances_file = f"vnncomp2025_benchmarks/benchmarks/{name}/instances_conv.csv"
            print(f"\n  {name} ({count} instances):")
            print(f"    Use: {instances_file}")
    else:
        print("\n⚠️  No Conv instances found in any benchmark!")

if __name__ == "__main__":
    main()
