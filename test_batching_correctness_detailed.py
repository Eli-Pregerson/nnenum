#!/usr/bin/env python3
"""
Detailed correctness test to find batching bugs
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from nnenum.network import Convolutional2dLayer, nn_unflatten, nn_flatten

# Test various configurations
test_cases = [
    {
        'name': 'Small 8x8, kernel=3',
        'input_shape': (8, 8, 3),
        'kernel_size': 3,
        'num_filters': 4,
        'mode': 'same',
        'stride': (1, 1),
        'num_gens': 20,
    },
    {
        'name': 'CIFAR 32x32, kernel=3',
        'input_shape': (32, 32, 3),
        'kernel_size': 3,
        'num_filters': 64,
        'mode': 'same',
        'stride': (1, 1),
        'num_gens': 100,
    },
    {
        'name': 'Valid padding, kernel=5',
        'input_shape': (32, 32, 3),
        'kernel_size': 5,
        'num_filters': 32,
        'mode': 'valid',
        'stride': (1, 1),
        'num_gens': 50,
    },
    {
        'name': 'Stride=2, kernel=3',
        'input_shape': (32, 32, 3),
        'kernel_size': 3,
        'num_filters': 64,
        'mode': 'same',
        'stride': (2, 2),
        'num_gens': 50,
    },
]

print("="*80)
print("DETAILED CORRECTNESS TEST FOR CONVOLUTION BATCHING")
print("="*80)

all_pass = True

for test_case in test_cases:
    print(f"\n{'='*80}")
    print(f"Test: {test_case['name']}")
    print('='*80)

    input_shape = test_case['input_shape']
    kernel_size = test_case['kernel_size']
    num_filters = test_case['num_filters']
    mode = test_case['mode']
    stride = test_case['stride']
    num_gens = test_case['num_gens']
    num_channels = input_shape[2]

    # Create layer
    np.random.seed(42)
    kernels = np.random.randn(num_filters, num_channels, kernel_size, kernel_size).astype(np.float32) * 0.1
    biases = np.random.randn(num_filters).astype(np.float32) * 0.1

    layer = Convolutional2dLayer(0, kernels, biases, input_shape, mode=mode, strides=stride)

    # Create sparse one-hot generators
    input_size = np.prod(input_shape)
    a_mat = np.eye(input_size, num_gens, dtype=np.float32)

    print(f"Config: {input_shape}, {num_filters} filters, {kernel_size}x{kernel_size}, mode={mode}, stride={stride}")
    print(f"Output shape: {layer.get_output_shape()}")
    print(f"Generators: {num_gens}")

    # BATCHED version
    class MockStar:
        def __init__(self, a_mat, bias):
            self.a_mat = a_mat
            self.bias = bias

    bias = np.zeros(input_size, dtype=np.float32)
    star_batched = MockStar(a_mat.copy(), bias.copy())

    from nnenum.settings import Settings
    Settings.CONV_BATCHING_FIRST_LAYER_ONLY = False
    layer.layer_num = 0

    layer.transform_star(star_batched)
    result_batched = star_batched.a_mat

    # UNBATCHED version (force skip batching)
    star_unbatched = MockStar(a_mat.copy(), bias.copy())

    Settings.CONV_BATCHING_FIRST_LAYER_ONLY = True
    layer.layer_num = 1

    layer.transform_star(star_unbatched)
    result_unbatched = star_unbatched.a_mat

    # Compare
    max_diff = np.max(np.abs(result_batched - result_unbatched))
    mean_diff = np.mean(np.abs(result_batched - result_unbatched))

    # Check which columns differ
    mismatched_cols = []
    for i in range(result_batched.shape[1]):
        col_diff = np.max(np.abs(result_batched[:, i] - result_unbatched[:, i]))
        if col_diff > 1e-6:
            mismatched_cols.append((i, col_diff))

    # Detailed comparison
    if len(mismatched_cols) == 0:
        print(f"✓ PASS: All columns match (max diff: {max_diff:.2e})")
    else:
        print(f"✗ FAIL: {len(mismatched_cols)}/{result_batched.shape[1]} columns differ")
        print(f"  Max diff: {max_diff:.2e}")
        print(f"  Mean diff: {mean_diff:.2e}")
        print(f"  Mismatched columns (showing first 5):")
        for i, (col_idx, diff) in enumerate(mismatched_cols[:5]):
            print(f"    Column {col_idx}: max diff = {diff:.2e}")

            # Show where differences are
            col_batched = result_batched[:, col_idx]
            col_unbatched = result_unbatched[:, col_idx]
            diff_mask = np.abs(col_batched - col_unbatched) > 1e-6

            if np.any(diff_mask):
                diff_indices = np.where(diff_mask)[0]
                print(f"      Differences at indices: {diff_indices[:10].tolist()}{'...' if len(diff_indices) > 10 else ''}")
                print(f"      Example: batched={col_batched[diff_indices[0]]:.6f}, unbatched={col_unbatched[diff_indices[0]]:.6f}")

        all_pass = False

print(f"\n{'='*80}")
print("FINAL RESULT")
print('='*80)

if all_pass:
    print("✓ ALL TESTS PASSED")
else:
    print("✗ SOME TESTS FAILED - Batching optimization has bugs!")
    print("\nPossible causes:")
    print("  1. Output region calculation error in _compute_output_region()")
    print("  2. Masking logic error when extracting individual generators")
    print("  3. Overlap detection allowing conflicting generators in same batch")
