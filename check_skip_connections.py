#!/usr/bin/env python3
"""
Check ONNX benchmarks for skip connections (residual Add nodes where both inputs
are computed activations, not initializers/bias vectors).
"""

import os
import sys
import onnx
import numpy as np
from collections import defaultdict

BENCHMARKS_DIR = "/home/elipregerson/nnenum/vnncomp2025_benchmarks/benchmarks"

TARGETS = [
    "acasxu_2023",
    "cersyve",
    "cgan_2023",
    "cifar100_2024",
    "collins_aerospace_benchmark",
    "collins_rul_cnn_2022",
    "cora_2024",
    "dist_shift_2023",
    "lsnc_relu",
    "malbeware",
    "metaroom_2023",
    "relusplitter",
    "soundnessbench",
    "tinyimagenet_2024",
    "vit_2023",
]


def find_onnx_files(bench_dir):
    """Find all ONNX files in a benchmark directory."""
    onnx_files = []
    for root, dirs, files in os.walk(bench_dir):
        for f in files:
            if f.endswith(".onnx"):
                onnx_files.append(os.path.join(root, f))
    return sorted(onnx_files)


def get_input_shape(model):
    """Get input shape from ONNX model."""
    graph = model.graph
    inputs = []
    for inp in graph.input:
        shape = []
        for dim in inp.type.tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(dim.dim_value)
            elif dim.HasField("dim_param"):
                shape.append(dim.dim_param)
            else:
                shape.append("?")
        inputs.append((inp.name, shape))
    return inputs


def count_nodes_by_op(model):
    """Count nodes by op type."""
    counts = defaultdict(int)
    for node in model.graph.node:
        counts[node.op_type] += 1
    return dict(counts)


def find_skip_connections(model):
    """
    Find Add nodes where both inputs are computed activations (not initializers).

    Returns list of (node_index, node_name, input1, input2) for skip connections.
    """
    graph = model.graph

    # Build set of initializer names (weights, biases)
    initializer_names = set(init.name for init in graph.initializer)

    # Also include graph inputs that are not the main data input
    # (Some models have constant inputs in graph.input)
    # We need to be careful: the first graph input is usually the data
    graph_input_names = set(inp.name for inp in graph.input)

    # Build set of all computed (non-initializer, non-constant) tensors
    # These are outputs of nodes, plus the main graph inputs (data inputs)
    computed_tensors = set()

    # Add main graph inputs (actual data inputs, not initializers)
    for inp in graph.input:
        if inp.name not in initializer_names:
            computed_tensors.add(inp.name)

    # Collect Constant node outputs (these are effectively initializers)
    constant_outputs = set()
    for node in graph.node:
        if node.op_type == "Constant":
            for out in node.output:
                constant_outputs.add(out)

    # Walk through nodes, building up set of computed tensors
    skip_connections = []

    for i, node in enumerate(graph.node):
        if node.op_type == "Add":
            inputs = list(node.input)
            if len(inputs) == 2:
                inp1, inp2 = inputs[0], inputs[1]

                # Check if both inputs are computed activations (not initializers/constants)
                inp1_is_activation = (inp1 in computed_tensors and
                                       inp1 not in initializer_names and
                                       inp1 not in constant_outputs)
                inp2_is_activation = (inp2 in computed_tensors and
                                       inp2 not in initializer_names and
                                       inp2 not in constant_outputs)

                if inp1_is_activation and inp2_is_activation:
                    skip_connections.append({
                        "node_idx": i,
                        "node_name": node.name,
                        "input1": inp1,
                        "input2": inp2,
                    })

        # Add this node's outputs to computed tensors
        for out in node.output:
            computed_tensors.add(out)

    return skip_connections


def analyze_onnx_file(onnx_path):
    """Analyze a single ONNX file."""
    try:
        model = onnx.load(onnx_path)
    except Exception as e:
        return {"error": str(e), "path": onnx_path}

    input_shapes = get_input_shape(model)
    node_counts = count_nodes_by_op(model)
    skip_connections = find_skip_connections(model)

    total_nodes = sum(node_counts.values())

    # Compute total input dimension
    total_input_dim = None
    if input_shapes:
        # Try to compute product of all dimensions (skip batch dim if string)
        shape = input_shapes[0][1]
        if all(isinstance(d, int) for d in shape):
            total_input_dim = 1
            for d in shape:
                total_input_dim *= d
        elif all(isinstance(d, int) for d in shape[1:]):
            # Batch dim is dynamic
            total_input_dim = 1
            for d in shape[1:]:
                total_input_dim *= d

    return {
        "path": onnx_path,
        "input_shapes": input_shapes,
        "total_input_dim": total_input_dim,
        "node_counts": node_counts,
        "total_nodes": total_nodes,
        "skip_connections": skip_connections,
        "has_skip": len(skip_connections) > 0,
    }


def pick_representative_file(onnx_files):
    """Pick one representative ONNX file from the list."""
    # Prefer smaller files; pick first one by default
    if not onnx_files:
        return None
    # Sort by file size, pick the first (smallest) for speed
    sized = [(os.path.getsize(f), f) for f in onnx_files]
    sized.sort()
    return sized[0][1]


def main():
    results = {}

    for bench in TARGETS:
        bench_dir = os.path.join(BENCHMARKS_DIR, bench)
        if not os.path.isdir(bench_dir):
            results[bench] = {"error": "Directory not found"}
            continue

        onnx_files = find_onnx_files(bench_dir)
        if not onnx_files:
            results[bench] = {"error": "No ONNX files found"}
            continue

        # Analyze representative file
        rep_file = pick_representative_file(onnx_files)
        info = analyze_onnx_file(rep_file)
        info["total_onnx_files"] = len(onnx_files)
        info["representative_file"] = os.path.relpath(rep_file, BENCHMARKS_DIR)
        results[bench] = info

        # If no skip connections in representative, check a few more
        if not info.get("has_skip") and len(onnx_files) > 1:
            # Check up to 3 more files
            for f in onnx_files[1:4]:
                extra_info = analyze_onnx_file(f)
                if extra_info.get("has_skip"):
                    results[bench]["also_checked"] = os.path.relpath(f, BENCHMARKS_DIR)
                    results[bench]["found_skip_in_other"] = True
                    results[bench]["skip_connections_other"] = extra_info["skip_connections"]
                    break

    # Print report
    print("=" * 80)
    print("SKIP CONNECTION ANALYSIS REPORT")
    print("=" * 80)
    print()

    skip_benchmarks = []
    no_skip_benchmarks = []

    for bench, info in results.items():
        if info.get("error"):
            print(f"{bench}: ERROR - {info['error']}")
            continue

        has_skip = info.get("has_skip") or info.get("found_skip_in_other", False)
        if has_skip:
            skip_benchmarks.append(bench)
        else:
            no_skip_benchmarks.append(bench)

    print("BENCHMARKS WITH SKIP CONNECTIONS:")
    print("-" * 40)
    for bench in skip_benchmarks:
        info = results[bench]
        print(f"\n  {bench}:")
        print(f"    File: {info['representative_file']}")
        print(f"    Total ONNX files: {info['total_onnx_files']}")
        print(f"    Input shapes: {info['input_shapes']}")
        print(f"    Input dim: {info['total_input_dim']}")
        print(f"    Total nodes: {info['total_nodes']}")
        print(f"    Node counts: {info['node_counts']}")
        skips = info.get("skip_connections") or info.get("skip_connections_other", [])
        print(f"    Skip connections ({len(skips)} found):")
        for sc in skips[:5]:  # Show first 5
            print(f"      Add node '{sc['node_name']}': {sc['input1']} + {sc['input2']}")
        if len(skips) > 5:
            print(f"      ... and {len(skips)-5} more")
        tractable = info['total_input_dim'] is not None and info['total_input_dim'] < 5000
        print(f"    Tractable (dim < 5000): {tractable}")

    print()
    print("BENCHMARKS WITHOUT SKIP CONNECTIONS:")
    print("-" * 40)
    for bench in no_skip_benchmarks:
        info = results[bench]
        print(f"\n  {bench}:")
        print(f"    File: {info['representative_file']}")
        print(f"    Total ONNX files: {info['total_onnx_files']}")
        print(f"    Input shapes: {info['input_shapes']}")
        print(f"    Input dim: {info['total_input_dim']}")
        print(f"    Total nodes: {info['total_nodes']}")
        print(f"    Node counts: {info['node_counts']}")
        tractable = info['total_input_dim'] is not None and info['total_input_dim'] < 5000
        print(f"    Tractable (dim < 5000): {tractable}")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Benchmarks WITH skip connections: {skip_benchmarks}")
    print(f"Benchmarks WITHOUT skip connections: {no_skip_benchmarks}")

    return results


if __name__ == "__main__":
    main()
