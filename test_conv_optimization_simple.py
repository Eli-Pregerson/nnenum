#!/usr/bin/env python3
"""
Test the convolution batching optimization (simplified, without LP dependencies)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from nnenum.network import Convolutional2dLayer, nn_unflatten, nn_flatten

# Test with a simple convolutional layer
# Input: 8x8x3 image
# Kernel: 3x3, 2 output channels

input_shape = (8, 8, 3)
kernel_size = 3
num_output_channels = 2
num_input_channels = 3

# Create random kernels and biases
np.random.seed(42)
kernels = np.random.randn(num_output_channels, num_input_channels, kernel_size, kernel_size).astype(np.float32) * 0.1
biases = np.random.randn(num_output_channels).astype(np.float32) * 0.1

# Create convolutional layer
conv_layer = Convolutional2dLayer(
    layer_num=0,
    kernels=kernels,
    biases=biases,
    prev_layer_output_shape=input_shape,
    mode='same',
    boundary='fill',
    strides=(1, 1)
)

print(f"Input shape: {conv_layer.get_input_shape()}")
print(f"Output shape: {conv_layer.get_output_shape()}")

# Create a generator matrix with sparse generators
# Most generators have a single nonzero entry (typical for image verification)
num_generators = 10
input_size = np.prod(input_shape)

# Create generator matrix (each column is a generator)
a_mat = np.zeros((input_size, num_generators), dtype=np.float32)

# Generator 0: nonzero at position (0, 0, 0)
a_mat[0, 0] = 1.0

# Generator 1: nonzero at position (0, 0, 1)
a_mat[1, 1] = 1.0

# Generator 2: nonzero at position (5, 5, 0) - far from others
idx_2 = 5 * 8 * 3 + 5 * 3 + 0
a_mat[idx_2, 2] = 1.0

# Generator 3: nonzero at position (5, 5, 1) - overlaps with generator 2 in output
idx_3 = 5 * 8 * 3 + 5 * 3 + 1
a_mat[idx_3, 3] = 1.0

# Generator 4-9: other sparse generators at different locations
for i in range(4, 10):
    y = (i - 4) % 4
    x = (i - 4) // 4
    channel = i % 3
    idx = y * 8 * 3 + x * 3 + channel
    a_mat[idx, i] = 1.0

print(f"\nOriginal a_mat:")
print(f"  shape: {a_mat.shape}")
print(f"  sparsity: {np.count_nonzero(a_mat)} / {a_mat.size} = {100 * np.count_nonzero(a_mat) / a_mat.size:.2f}%")

# Test batching algorithm
print(f"\n=== Testing batching algorithm ===")
batches, generator_info = conv_layer._batch_generators_for_conv(a_mat, input_shape)

print(f"Number of batches: {len(batches)}")
for i, batch in enumerate(batches):
    print(f"  Batch {i}: {len(batch['indices'])} generators - indices {batch['indices']}")

# Test actual convolution
print(f"\n=== Testing convolution ===")

# OPTIMIZED version - with batching
import time
a_mat_opt = a_mat.copy()

# Simulate transform_star logic
start = time.time()
result_columns_opt = [None] * a_mat_opt.shape[1]

for batch in batches:
    if len(batch['indices']) == 1:
        idx = batch['indices'][0]
        column = generator_info[idx]['column']
        multichannel_state = nn_unflatten(column, input_shape)
        multichannel_state = conv_layer.execute(multichannel_state, zero_bias=True)
        flat = nn_flatten(multichannel_state)
        flat.shape = (flat.size, 1)
        result_columns_opt[idx] = flat
    else:
        # Combine generators
        combined = np.zeros(input_shape, dtype=a_mat_opt.dtype)
        for idx in batch['indices']:
            column = generator_info[idx]['column']
            multichannel_state = nn_unflatten(column, input_shape)
            combined += multichannel_state

        # Single convolution
        combined_result = conv_layer.execute(combined, zero_bias=True)
        combined_result_2d = combined_result

        output_shape = conv_layer.get_output_shape()

        for i, idx in enumerate(batch['indices']):
            output_region = batch['output_regions'][i]

            if output_region is None:
                flat = np.zeros((np.prod(output_shape), 1), dtype=a_mat_opt.dtype)
                result_columns_opt[idx] = flat
            else:
                out_min_y, out_max_y, out_min_x, out_max_x = output_region
                masked_output = np.zeros(output_shape, dtype=a_mat_opt.dtype)
                masked_output[out_min_y:out_max_y+1, out_min_x:out_max_x+1, :] = \
                    combined_result_2d[out_min_y:out_max_y+1, out_min_x:out_max_x+1, :]
                flat = nn_flatten(masked_output)
                flat.shape = (flat.size, 1)
                result_columns_opt[idx] = flat

a_mat_result_opt = np.hstack(result_columns_opt)
elapsed_opt = time.time() - start

print(f"Optimized time: {elapsed_opt:.4f}s")
print(f"Optimized result shape: {a_mat_result_opt.shape}")

# UNOPTIMIZED version - naive loop
a_mat_unopt = a_mat.copy()

start = time.time()
result_columns_unopt = []

for cindex in range(a_mat_unopt.shape[1]):
    column = a_mat_unopt[:, cindex]
    multichannel_state = nn_unflatten(column, input_shape)
    multichannel_state = conv_layer.execute(multichannel_state, zero_bias=True)
    flat = nn_flatten(multichannel_state)
    flat.shape = (flat.size, 1)
    result_columns_unopt.append(flat)

a_mat_result_unopt = np.hstack(result_columns_unopt)
elapsed_unopt = time.time() - start

print(f"\nUnoptimized time: {elapsed_unopt:.4f}s")
print(f"Unoptimized result shape: {a_mat_result_unopt.shape}")

# Compare results
print(f"\n=== COMPARISON ===")
print(f"Speedup: {elapsed_unopt / elapsed_opt:.2f}x")
print(f"Max difference: {np.max(np.abs(a_mat_result_opt - a_mat_result_unopt))}")
print(f"Mean absolute difference: {np.mean(np.abs(a_mat_result_opt - a_mat_result_unopt))}")

if np.allclose(a_mat_result_opt, a_mat_result_unopt, rtol=1e-5, atol=1e-7):
    print("\n✓ CORRECTNESS: Results match!")
else:
    print("\n✗ CORRECTNESS: Results DO NOT match!")
    print("Checking which columns differ...")
    for i in range(a_mat_result_opt.shape[1]):
        col_diff = np.max(np.abs(a_mat_result_opt[:, i] - a_mat_result_unopt[:, i]))
        if col_diff > 1e-6:
            print(f"  Column {i}: max diff = {col_diff}")
