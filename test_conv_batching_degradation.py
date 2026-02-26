#!/usr/bin/env python3
"""
Test how convolution batching effectiveness degrades through layers
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from nnenum.network import Convolutional2dLayer, nn_unflatten, nn_flatten

# Simulate a 2-layer convolutional network
input_shape_1 = (32, 32, 3)
kernel_size = 3
num_filters_1 = 64

# Layer 1
np.random.seed(42)
kernels_1 = np.random.randn(num_filters_1, 3, kernel_size, kernel_size).astype(np.float32) * 0.1
biases_1 = np.random.randn(num_filters_1).astype(np.float32) * 0.1

conv_layer_1 = Convolutional2dLayer(
    layer_num=0,
    kernels=kernels_1,
    biases=biases_1,
    prev_layer_output_shape=input_shape_1,
    mode='same',
    boundary='fill',
    strides=(1, 1)
)

# Layer 2
output_shape_1 = conv_layer_1.get_output_shape()
num_filters_2 = 64

kernels_2 = np.random.randn(num_filters_2, num_filters_1, kernel_size, kernel_size).astype(np.float32) * 0.1
biases_2 = np.random.randn(num_filters_2).astype(np.float32) * 0.1

conv_layer_2 = Convolutional2dLayer(
    layer_num=1,
    kernels=kernels_2,
    biases=biases_2,
    prev_layer_output_shape=output_shape_1,
    mode='same',
    boundary='fill',
    strides=(1, 1)
)

print("Layer 1:")
print(f"  Input: {conv_layer_1.get_input_shape()}")
print(f"  Output: {conv_layer_1.get_output_shape()}")
print("\nLayer 2:")
print(f"  Input: {conv_layer_2.get_input_shape()}")
print(f"  Output: {conv_layer_2.get_output_shape()}")

# Test with first 100 one-hot generators (typical for image verification)
num_test_gens = 100
input_size_1 = np.prod(input_shape_1)
a_mat_1 = np.eye(input_size_1, num_test_gens, dtype=np.float32)

print(f"\n{'='*60}")
print("LAYER 1: One-hot input generators")
print('='*60)

# Analyze Layer 1 batching
batches_1, gen_info_1 = conv_layer_1._batch_generators_for_conv(a_mat_1, input_shape_1)

batch_sizes_1 = [len(b['indices']) for b in batches_1]
print(f"Generators: {num_test_gens}")
print(f"Batches: {len(batches_1)}")
print(f"Compression ratio: {num_test_gens / len(batches_1):.2f}x")
print(f"Batch sizes: min={min(batch_sizes_1)}, max={max(batch_sizes_1)}, mean={np.mean(batch_sizes_1):.1f}")

# Now propagate through Layer 1 to get generators for Layer 2
print(f"\nPropagating {num_test_gens} generators through Layer 1...")
result_columns_1 = []

for cindex in range(a_mat_1.shape[1]):
    column = a_mat_1[:, cindex]
    multichannel_state = nn_unflatten(column, input_shape_1)
    output = conv_layer_1.execute(multichannel_state, zero_bias=True)
    flat = nn_flatten(output)
    flat.shape = (flat.size, 1)
    result_columns_1.append(flat)

a_mat_2 = np.hstack(result_columns_1)

# Analyze sparsity
print(f"\nLayer 2 input (Layer 1 output):")
print(f"  Shape: {a_mat_2.shape}")
print(f"  Sparsity: {100 * np.count_nonzero(a_mat_2) / a_mat_2.size:.2f}%")

# Check how "spread out" each generator is
print(f"  Nonzero entries per generator:")
nonzeros_per_gen = [np.count_nonzero(a_mat_2[:, i]) for i in range(a_mat_2.shape[1])]
print(f"    Min: {min(nonzeros_per_gen)}")
print(f"    Max: {max(nonzeros_per_gen)}")
print(f"    Mean: {np.mean(nonzeros_per_gen):.1f}")

print(f"\n{'='*60}")
print("LAYER 2: Conv-transformed generators from Layer 1")
print('='*60)

# Analyze Layer 2 batching
batches_2, gen_info_2 = conv_layer_2._batch_generators_for_conv(a_mat_2, output_shape_1)

batch_sizes_2 = [len(b['indices']) for b in batches_2]
print(f"Generators: {a_mat_2.shape[1]}")
print(f"Batches: {len(batches_2)}")
print(f"Compression ratio: {a_mat_2.shape[1] / len(batches_2):.2f}x")
print(f"Batch sizes: min={min(batch_sizes_2)}, max={max(batch_sizes_2)}, mean={np.mean(batch_sizes_2):.1f}")

# Count single-generator batches (no benefit)
single_batches_2 = sum(1 for b in batches_2 if len(b['indices']) == 1)
print(f"Single-generator batches: {single_batches_2}/{len(batches_2)} ({100*single_batches_2/len(batches_2):.1f}%)")

print(f"\n{'='*60}")
print("SUMMARY")
print('='*60)
print(f"Layer 1 compression: {num_test_gens / len(batches_1):.2f}x (EXCELLENT)")
print(f"Layer 2 compression: {a_mat_2.shape[1] / len(batches_2):.2f}x", end="")

if a_mat_2.shape[1] / len(batches_2) < 1.5:
    print(" (POOR - mostly overhead)")
elif a_mat_2.shape[1] / len(batches_2) < 3:
    print(" (MARGINAL)")
else:
    print(" (GOOD)")

print(f"\nRecommendation: ", end="")
if a_mat_2.shape[1] / len(batches_2) < 1.5:
    print("Skip batching optimization for Layer 2+ (adds overhead without benefit)")
else:
    print("Batching still beneficial for Layer 2")
