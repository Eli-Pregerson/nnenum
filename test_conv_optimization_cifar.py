#!/usr/bin/env python3
"""
Test the convolution batching optimization with CIFAR-scale inputs
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from nnenum.network import Convolutional2dLayer, nn_unflatten, nn_flatten

# Test with CIFAR-10 scale: 32x32x3 image
input_shape = (32, 32, 3)
kernel_size = 3
num_output_channels = 64  # First conv layer in ResNet typically has 64 filters
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
# In CIFAR verification, we typically have 3072 generators (one per input pixel-channel)
# Each generator is a one-hot vector (extremely sparse)
num_generators = 3072  # 32 * 32 * 3
input_size = np.prod(input_shape)

# Create generator matrix (each column is a generator)
# Standard image star: each generator is a one-hot vector
a_mat = np.eye(input_size, num_generators, dtype=np.float32)

print(f"\nOriginal a_mat:")
print(f"  shape: {a_mat.shape}")
print(f"  sparsity: {np.count_nonzero(a_mat)} / {a_mat.size} = {100 * np.count_nonzero(a_mat) / a_mat.size:.4f}%")

# Test batching algorithm
print(f"\n=== Testing batching algorithm ===")
import time
start = time.time()
batches, generator_info = conv_layer._batch_generators_for_conv(a_mat, input_shape)
batching_time = time.time() - start

print(f"Number of batches: {len(batches)}")
print(f"Batching time: {batching_time:.4f}s")

batch_sizes = [len(batch['indices']) for batch in batches]
print(f"Batch size statistics:")
print(f"  Min: {min(batch_sizes)}")
print(f"  Max: {max(batch_sizes)}")
print(f"  Mean: {np.mean(batch_sizes):.2f}")
print(f"  Median: {np.median(batch_sizes):.2f}")

# Show first few batches
for i in range(min(5, len(batches))):
    batch = batches[i]
    print(f"  Batch {i}: {len(batch['indices'])} generators")

# Test actual convolution on a subset to measure speedup
print(f"\n=== Testing convolution (first 100 generators) ===")

# Use subset for faster testing
num_test_generators = 100
a_mat_test = a_mat[:, :num_test_generators].copy()

# OPTIMIZED version - with batching
start = time.time()
batches_test, generator_info_test = conv_layer._batch_generators_for_conv(a_mat_test, input_shape)
result_columns_opt = [None] * a_mat_test.shape[1]

for batch in batches_test:
    if len(batch['indices']) == 1:
        idx = batch['indices'][0]
        column = generator_info_test[idx]['column']
        multichannel_state = nn_unflatten(column, input_shape)
        multichannel_state = conv_layer.execute(multichannel_state, zero_bias=True)
        flat = nn_flatten(multichannel_state)
        flat.shape = (flat.size, 1)
        result_columns_opt[idx] = flat
    else:
        # Combine generators
        combined = np.zeros(input_shape, dtype=a_mat_test.dtype)
        for idx in batch['indices']:
            column = generator_info_test[idx]['column']
            multichannel_state = nn_unflatten(column, input_shape)
            combined += multichannel_state

        # Single convolution
        combined_result = conv_layer.execute(combined, zero_bias=True)
        output_shape = conv_layer.get_output_shape()

        for i, idx in enumerate(batch['indices']):
            output_region = batch['output_regions'][i]

            if output_region is None:
                flat = np.zeros((np.prod(output_shape), 1), dtype=a_mat_test.dtype)
                result_columns_opt[idx] = flat
            else:
                out_min_y, out_max_y, out_min_x, out_max_x = output_region
                masked_output = np.zeros(output_shape, dtype=a_mat_test.dtype)
                masked_output[out_min_y:out_max_y+1, out_min_x:out_max_x+1, :] = \
                    combined_result[out_min_y:out_max_y+1, out_min_x:out_max_x+1, :]
                flat = nn_flatten(masked_output)
                flat.shape = (flat.size, 1)
                result_columns_opt[idx] = flat

a_mat_result_opt = np.hstack(result_columns_opt)
elapsed_opt = time.time() - start

print(f"Optimized time: {elapsed_opt:.4f}s")
print(f"  Batches used: {len(batches_test)}")

# UNOPTIMIZED version - naive loop
a_mat_unopt = a_mat_test.copy()

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

# Compare results
print(f"\n=== COMPARISON ===")
print(f"Speedup: {elapsed_unopt / elapsed_opt:.2f}x")
print(f"Theoretical max speedup: {num_test_generators / len(batches_test):.2f}x")
print(f"Max difference: {np.max(np.abs(a_mat_result_opt - a_mat_result_unopt))}")
print(f"Mean absolute difference: {np.mean(np.abs(a_mat_result_opt - a_mat_result_unopt))}")

if np.allclose(a_mat_result_opt, a_mat_result_unopt, rtol=1e-5, atol=1e-6):
    print("\n✓ CORRECTNESS: Results match!")
else:
    print("\n✗ CORRECTNESS: Results DO NOT match!")
    # Find which columns differ
    mismatches = 0
    for i in range(a_mat_result_opt.shape[1]):
        col_diff = np.max(np.abs(a_mat_result_opt[:, i] - a_mat_result_unopt[:, i]))
        if col_diff > 1e-6:
            mismatches += 1
            if mismatches <= 5:  # Show first 5
                print(f"  Column {i}: max diff = {col_diff}")
    print(f"  Total mismatches: {mismatches}/{a_mat_result_opt.shape[1]}")
