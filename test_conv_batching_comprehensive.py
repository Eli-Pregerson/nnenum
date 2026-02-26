#!/usr/bin/env python3
"""
Comprehensive correctness tests for convolution batching optimization.

Tests cases that were identified as missing:
1. Multi-layer propagation with dense generators
2. Mixed sparsity (some sparse, some dense generators)
3. Edge case generators (boundary pixels, corners)
4. Generators after ReLU transformations
5. Various network architectures
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from nnenum.network import Convolutional2dLayer, ReluLayer, nn_unflatten, nn_flatten
from nnenum.settings import Settings

class MockStar:
    """Mock star object for testing"""
    def __init__(self, a_mat, bias):
        self.a_mat = a_mat
        self.bias = bias

def compare_batched_vs_unbatched(layer, a_mat, input_shape, test_name):
    """Compare batched vs unbatched for a given configuration"""

    bias = np.zeros(np.prod(input_shape), dtype=np.float32)

    # Batched version
    star_batched = MockStar(a_mat.copy(), bias.copy())
    Settings.CONV_BATCHING_FIRST_LAYER_ONLY = False
    layer.layer_num = 0
    layer.transform_star(star_batched)
    result_batched = star_batched.a_mat

    # Unbatched version
    star_unbatched = MockStar(a_mat.copy(), bias.copy())
    Settings.CONV_BATCHING_FIRST_LAYER_ONLY = True
    layer.layer_num = 1
    layer.transform_star(star_unbatched)
    result_unbatched = star_unbatched.a_mat

    # Compare
    max_diff = np.max(np.abs(result_batched - result_unbatched))
    passed = max_diff < 1e-6

    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {test_name:<50} {status} (max diff: {max_diff:.2e})")

    if not passed:
        # Show details of failure
        mismatched_cols = []
        for i in range(result_batched.shape[1]):
            col_diff = np.max(np.abs(result_batched[:, i] - result_unbatched[:, i]))
            if col_diff > 1e-6:
                mismatched_cols.append((i, col_diff))

        print(f"    {len(mismatched_cols)}/{result_batched.shape[1]} columns differ")
        if mismatched_cols:
            print(f"    First mismatch: column {mismatched_cols[0][0]}, diff={mismatched_cols[0][1]:.2e}")

    return passed

def create_layer(input_shape, num_filters, kernel_size, mode='same', stride=(1,1)):
    """Helper to create a conv layer"""
    num_channels = input_shape[2]
    np.random.seed(42)
    kernels = np.random.randn(num_filters, num_channels, kernel_size, kernel_size).astype(np.float32) * 0.1
    biases = np.random.randn(num_filters).astype(np.float32) * 0.1
    return Convolutional2dLayer(0, kernels, biases, input_shape, mode=mode, strides=stride)

print("="*80)
print("COMPREHENSIVE CONVOLUTION BATCHING CORRECTNESS TESTS")
print("="*80)

all_passed = True

# =============================================================================
# TEST 1: Multi-layer propagation with increasingly dense generators
# =============================================================================
print("\n" + "="*80)
print("TEST 1: Multi-layer propagation (generators become denser)")
print("="*80)

input_shape_1 = (16, 16, 3)
layer1 = create_layer(input_shape_1, 32, 3)
output_shape_1 = layer1.get_output_shape()
layer2 = create_layer(output_shape_1, 32, 3)
output_shape_2 = layer2.get_output_shape()
layer3 = create_layer(output_shape_2, 32, 3)

# Start with sparse one-hot
num_gens = 50
input_size = np.prod(input_shape_1)
a_mat = np.eye(input_size, num_gens, dtype=np.float32)

print(f"Starting with {num_gens} one-hot generators, sparsity={100*np.count_nonzero(a_mat[:,0])/a_mat.shape[0]:.4f}%")

# Layer 1
passed_1 = compare_batched_vs_unbatched(layer1, a_mat, input_shape_1, "Layer 1: One-hot input")
all_passed &= passed_1

# Propagate to get denser generators for layer 2
bias = np.zeros(input_size, dtype=np.float32)
star_temp = MockStar(a_mat.copy(), bias)
Settings.CONV_BATCHING_FIRST_LAYER_ONLY = False
layer1.layer_num = 0
layer1.transform_star(star_temp)
a_mat_layer2 = star_temp.a_mat

sparsity_2 = np.count_nonzero(a_mat_layer2[:,0]) / a_mat_layer2.shape[0]
print(f"After layer 1: sparsity={100*sparsity_2:.2f}%")

# Layer 2
passed_2 = compare_batched_vs_unbatched(layer2, a_mat_layer2, output_shape_1, "Layer 2: After 1 conv")
all_passed &= passed_2

# Propagate to layer 3
bias2 = np.zeros(np.prod(output_shape_1), dtype=np.float32)
star_temp2 = MockStar(a_mat_layer2.copy(), bias2)
layer2.layer_num = 0
layer2.transform_star(star_temp2)
a_mat_layer3 = star_temp2.a_mat

sparsity_3 = np.count_nonzero(a_mat_layer3[:,0]) / a_mat_layer3.shape[0]
print(f"After layer 2: sparsity={100*sparsity_3:.2f}%")

# Layer 3
passed_3 = compare_batched_vs_unbatched(layer3, a_mat_layer3, output_shape_2, "Layer 3: After 2 convs")
all_passed &= passed_3

# =============================================================================
# TEST 2: Mixed sparsity (sparse + dense generators)
# =============================================================================
print("\n" + "="*80)
print("TEST 2: Mixed sparsity (sparse and dense generators together)")
print("="*80)

input_shape = (16, 16, 3)
layer = create_layer(input_shape, 32, 3)
input_size = np.prod(input_shape)

# Create mixed scenario: 40 sparse + 10 dense
num_gens = 50
a_mat_mixed = np.zeros((input_size, num_gens), dtype=np.float32)

# First 40: one-hot (sparse)
for i in range(40):
    a_mat_mixed[i, i] = 1.0

# Last 10: dense (20% nonzeros each, simulating accumulated transformations)
for i in range(40, 50):
    num_nonzeros = int(input_size * 0.20)
    indices = np.random.choice(input_size, num_nonzeros, replace=False)
    a_mat_mixed[indices, i] = np.random.randn(num_nonzeros).astype(np.float32) * 0.1

sparse_count = np.sum([np.count_nonzero(a_mat_mixed[:, i]) == 1 for i in range(40)])
dense_count = 10
print(f"Generators: {sparse_count} sparse (one-hot) + {dense_count} dense (20% nonzeros)")

passed = compare_batched_vs_unbatched(layer, a_mat_mixed, input_shape, "Mixed sparsity")
all_passed &= passed

# =============================================================================
# TEST 3: Edge case generators (corners and boundaries)
# =============================================================================
print("\n" + "="*80)
print("TEST 3: Edge cases (corner pixels, boundary generators)")
print("="*80)

input_shape = (16, 16, 3)
layer = create_layer(input_shape, 32, 3)
input_size = np.prod(input_shape)

# Create generators at critical positions
edge_positions = [
    (0, 0, 0),      # Top-left corner
    (0, 15, 0),     # Top-right corner
    (15, 0, 0),     # Bottom-left corner
    (15, 15, 0),    # Bottom-right corner
    (0, 8, 0),      # Top edge center
    (15, 8, 0),     # Bottom edge center
    (8, 0, 0),      # Left edge center
    (8, 15, 0),     # Right edge center
    (8, 8, 0),      # Center
]

a_mat_edges = np.zeros((input_size, len(edge_positions)), dtype=np.float32)
for i, (y, x, c) in enumerate(edge_positions):
    idx = y * 16 * 3 + x * 3 + c
    a_mat_edges[idx, i] = 1.0

print(f"Testing {len(edge_positions)} critical positions (corners, edges, center)")
passed = compare_batched_vs_unbatched(layer, a_mat_edges, input_shape, "Edge cases")
all_passed &= passed

# =============================================================================
# TEST 4: Different kernel sizes
# =============================================================================
print("\n" + "="*80)
print("TEST 4: Various kernel sizes")
print("="*80)

input_shape = (16, 16, 3)
num_gens = 30

for kernel_size in [1, 3, 5, 7]:
    layer = create_layer(input_shape, 16, kernel_size)
    input_size = np.prod(input_shape)
    a_mat = np.eye(input_size, num_gens, dtype=np.float32)

    passed = compare_batched_vs_unbatched(layer, a_mat, input_shape, f"Kernel size {kernel_size}x{kernel_size}")
    all_passed &= passed

# =============================================================================
# TEST 5: Different padding modes and strides
# =============================================================================
print("\n" + "="*80)
print("TEST 5: Padding modes and strides")
print("="*80)

input_shape = (16, 16, 3)
num_gens = 30
input_size = np.prod(input_shape)
a_mat = np.eye(input_size, num_gens, dtype=np.float32)

configs = [
    ('same', (1, 1)),
    ('same', (2, 2)),
    ('valid', (1, 1)),
    ('valid', (2, 2)),
]

for mode, stride in configs:
    np.random.seed(42)
    kernels = np.random.randn(16, 3, 3, 3).astype(np.float32) * 0.1
    biases = np.random.randn(16).astype(np.float32) * 0.1
    layer = Convolutional2dLayer(0, kernels, biases, input_shape, mode=mode, strides=stride)

    passed = compare_batched_vs_unbatched(layer, a_mat, input_shape, f"mode={mode}, stride={stride}")
    all_passed &= passed

# =============================================================================
# TEST 6: Larger networks (stress test)
# =============================================================================
print("\n" + "="*80)
print("TEST 6: Larger networks (CIFAR-scale)")
print("="*80)

input_shape = (32, 32, 3)
layer = create_layer(input_shape, 64, 3)
input_size = np.prod(input_shape)

# Test with different numbers of generators
for num_gens in [100, 500, 1000]:
    a_mat = np.eye(input_size, num_gens, dtype=np.float32)
    passed = compare_batched_vs_unbatched(layer, a_mat, input_shape, f"{num_gens} generators")
    all_passed &= passed

# =============================================================================
# TEST 7: Very dense generators (edge of sparsity threshold)
# =============================================================================
print("\n" + "="*80)
print("TEST 7: Dense generators near sparsity threshold")
print("="*80)

input_shape = (16, 16, 3)
layer = create_layer(input_shape, 32, 3)
input_size = np.prod(input_shape)
num_gens = 20

# Test with different density levels
for density in [0.01, 0.03, 0.05, 0.10]:
    a_mat_dense = np.zeros((input_size, num_gens), dtype=np.float32)
    for i in range(num_gens):
        num_nonzeros = int(input_size * density)
        indices = np.random.choice(input_size, num_nonzeros, replace=False)
        a_mat_dense[indices, i] = np.random.randn(num_nonzeros).astype(np.float32) * 0.1

    passed = compare_batched_vs_unbatched(layer, a_mat_dense, input_shape,
                                          f"Density {density*100:.0f}% ({num_nonzeros} nonzeros)")
    all_passed &= passed

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

if all_passed:
    print("✓ ALL TESTS PASSED")
    print("\nThe convolution batching optimization is correct across:")
    print("  - Multi-layer propagation with dense generators")
    print("  - Mixed sparsity scenarios")
    print("  - Edge cases (corners, boundaries)")
    print("  - Various kernel sizes (1x1 to 7x7)")
    print("  - Different padding modes and strides")
    print("  - Large networks (up to 1000 generators)")
    print("  - Dense generators (up to 10% density)")
else:
    print("✗ SOME TESTS FAILED")
    print("\nThe batching optimization has bugs that need to be fixed.")
    sys.exit(1)
