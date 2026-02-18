#!/usr/bin/env python3
"""
Debug script to understand generator structure when reaching Conv layer
"""

import numpy as np
from nnenum.onnx_network import load_onnx_network_optimized
from nnenum.lp_star import LpStar
from nnenum.network import nn_unflatten, nn_flatten
import sys

def analyze_generator_structure():
    """Analyze how generators look when they reach the first Conv layer"""

    model_path = "vnncomp2025_benchmarks/benchmarks/soundnessbench/onnx/model.onnx"

    print("Loading model...")
    network = load_onnx_network_optimized(model_path)

    print(f"\nNetwork has {len(network.layers)} layers:")
    for i, layer in enumerate(network.layers):
        print(f"  {i}: {layer}")

    # Create a simple initial star with identity basis
    # Soundnessbench starts with Gemm input of size 25
    dims = 25
    init_box = [(-1.0, 1.0)] * dims

    # Create identity basis matrix
    init_bm = np.identity(dims)
    init_bias = np.zeros(dims)

    star = LpStar(init_bm, init_bias, init_box)

    print(f"\nInitial star:")
    print(f"  a_mat shape: {star.a_mat.shape}")
    print(f"  bias shape: {star.bias.shape}")

    # Process through layers until we hit the first Conv
    for layer_idx, layer in enumerate(network.layers):
        print(f"\nProcessing layer {layer_idx}: {layer}")

        if hasattr(layer, '__class__') and 'Convolutional2d' in layer.__class__.__name__:
            print(f"\n=== CONV LAYER REACHED ===")
            print(f"Input shape: {layer.get_input_shape()}")
            print(f"Output shape: {layer.get_output_shape()}")
            print(f"Star a_mat shape: {star.a_mat.shape}")
            print(f"Star bias shape: {star.bias.shape}")

            # Analyze generator structure
            print(f"\nAnalyzing generator sparsity:")
            for col_idx in range(min(5, star.a_mat.shape[1])):
                column = star.a_mat[:, col_idx]
                num_nonzero = np.count_nonzero(np.abs(column) > 1e-10)
                max_val = np.max(np.abs(column))

                # Unflatten to image shape to see spatial structure
                img = nn_unflatten(column, layer.get_input_shape())

                print(f"  Generator {col_idx}:")
                print(f"    Nonzero elements: {num_nonzero}/{len(column)}")
                print(f"    Max abs value: {max_val:.6f}")
                print(f"    Image shape: {img.shape}")

                # Find which pixels are non-zero
                nonzero_positions = np.argwhere(np.abs(img) > 1e-10)
                if len(nonzero_positions) > 0:
                    print(f"    Non-zero positions (first 5): {nonzero_positions[:5]}")

            break

        # Apply transformation
        if hasattr(layer, '__class__') and 'Relu' in layer.__class__.__name__:
            print("  Skipping ReLU layer for this analysis")
            continue

        layer.transform_star(star)
        print(f"  After transform: a_mat shape {star.a_mat.shape}, bias shape {star.bias.shape}")

if __name__ == '__main__':
    analyze_generator_structure()
