#!/usr/bin/env python3
"""
Test batching through 3 conv layers
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from nnenum.network import Convolutional2dLayer, nn_unflatten, nn_flatten

def analyze_batching(layer, a_mat, input_shape, layer_name):
    """Analyze batching for a layer"""
    batches, gen_info = layer._batch_generators_for_conv(a_mat, input_shape)
    batch_sizes = [len(b['indices']) for b in batches]
    num_gens = a_mat.shape[1]
    num_batches = len(batches)
    compression = num_gens / num_batches if num_batches > 0 else 0
    single_batches = sum(1 for b in batches if len(b['indices']) == 1)

    print(f"\n{layer_name}:")
    print(f"  Generators: {num_gens}")
    print(f"  Batches: {num_batches}")
    print(f"  Compression: {compression:.2f}x")
    print(f"  Batch sizes: min={min(batch_sizes)}, max={max(batch_sizes)}, mean={np.mean(batch_sizes):.1f}")
    print(f"  Single-gen batches: {single_batches}/{num_batches} ({100*single_batches/num_batches:.1f}%)")

    # Sparsity info
    nonzeros_per_gen = [np.count_nonzero(a_mat[:, i]) for i in range(a_mat.shape[1])]
    print(f"  Nonzeros/gen: min={min(nonzeros_per_gen)}, max={max(nonzeros_per_gen)}, mean={np.mean(nonzeros_per_gen):.0f}")
    print(f"  Overall sparsity: {100 * np.count_nonzero(a_mat) / a_mat.size:.2f}%")

    return compression

# Build 3-layer conv network
input_shape = (32, 32, 3)
kernel_size = 3
num_filters = 64

np.random.seed(42)

# Layer 1
kernels_1 = np.random.randn(num_filters, 3, kernel_size, kernel_size).astype(np.float32) * 0.1
biases_1 = np.random.randn(num_filters).astype(np.float32) * 0.1
layer_1 = Convolutional2dLayer(0, kernels_1, biases_1, input_shape, mode='same', strides=(1, 1))

# Layer 2
shape_2 = layer_1.get_output_shape()
kernels_2 = np.random.randn(num_filters, num_filters, kernel_size, kernel_size).astype(np.float32) * 0.1
biases_2 = np.random.randn(num_filters).astype(np.float32) * 0.1
layer_2 = Convolutional2dLayer(1, kernels_2, biases_2, shape_2, mode='same', strides=(1, 1))

# Layer 3
shape_3 = layer_2.get_output_shape()
kernels_3 = np.random.randn(num_filters, num_filters, kernel_size, kernel_size).astype(np.float32) * 0.1
biases_3 = np.random.randn(num_filters).astype(np.float32) * 0.1
layer_3 = Convolutional2dLayer(2, kernels_3, biases_3, shape_3, mode='same', strides=(1, 1))

print("Network architecture:")
print(f"  Layer 1: {input_shape} -> {layer_1.get_output_shape()}")
print(f"  Layer 2: {shape_2} -> {layer_2.get_output_shape()}")
print(f"  Layer 3: {shape_3} -> {layer_3.get_output_shape()}")

# Start with 100 one-hot generators
num_test_gens = 100
input_size = np.prod(input_shape)
a_mat_1 = np.eye(input_size, num_test_gens, dtype=np.float32)

print(f"\n{'='*70}")
print("BATCHING ANALYSIS THROUGH 3 CONV LAYERS")
print('='*70)

# Layer 1
compression_1 = analyze_batching(layer_1, a_mat_1, input_shape, "Layer 1 (input)")

# Propagate through Layer 1
print("\nPropagating through Layer 1...")
result_cols = []
for i in range(a_mat_1.shape[1]):
    col = a_mat_1[:, i]
    state = nn_unflatten(col, input_shape)
    output = layer_1.execute(state, zero_bias=True)
    flat = nn_flatten(output)
    flat.shape = (flat.size, 1)
    result_cols.append(flat)
a_mat_2 = np.hstack(result_cols)

compression_2 = analyze_batching(layer_2, a_mat_2, shape_2, "Layer 2")

# Propagate through Layer 2
print("\nPropagating through Layer 2...")
result_cols = []
for i in range(a_mat_2.shape[1]):
    col = a_mat_2[:, i]
    state = nn_unflatten(col, shape_2)
    output = layer_2.execute(state, zero_bias=True)
    flat = nn_flatten(output)
    flat.shape = (flat.size, 1)
    result_cols.append(flat)
a_mat_3 = np.hstack(result_cols)

compression_3 = analyze_batching(layer_3, a_mat_3, shape_3, "Layer 3")

print(f"\n{'='*70}")
print("SUMMARY")
print('='*70)
print(f"Layer 1 compression: {compression_1:.2f}x")
print(f"Layer 2 compression: {compression_2:.2f}x ({compression_2/compression_1*100:.0f}% of Layer 1)")
print(f"Layer 3 compression: {compression_3:.2f}x ({compression_3/compression_1*100:.0f}% of Layer 1)")

print(f"\nRecommendation:")
if compression_3 < 1.5:
    print("  ✓ Apply batching to Layer 1 only")
    print("  ✗ Skip batching for Layer 2+ (overhead exceeds benefit)")
elif compression_2 > 3 and compression_3 > 2:
    print("  ✓ Apply batching to all layers")
else:
    print("  ✓ Apply batching to Layers 1-2")
    print("  ✗ Skip batching for Layer 3+")
