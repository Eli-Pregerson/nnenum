#!/usr/bin/env python3
"""
Test the convolution batching optimization
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from nnenum.network import Convolutional2dLayer
from nnenum.lp_star import LpStar
from nnenum.zonotope import Zonotope

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

# Create a star set with sparse generators
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

# Create bias
bias = np.random.randn(input_size).astype(np.float32) * 0.01

# Create LpStar
star = LpStar(a_mat, bias)

print(f"\nOriginal star:")
print(f"  a_mat shape: {star.a_mat.shape}")
print(f"  bias shape: {star.bias.shape}")
print(f"  a_mat sparsity: {np.count_nonzero(star.a_mat)} / {star.a_mat.size} = {100 * np.count_nonzero(star.a_mat) / star.a_mat.size:.2f}%")

# Transform using OPTIMIZED method (with batching)
import time
start = time.time()
conv_layer.transform_star(star)
elapsed_optimized = time.time() - start

print(f"\nTransformed star (OPTIMIZED):")
print(f"  a_mat shape: {star.a_mat.shape}")
print(f"  bias shape: {star.bias.shape}")
print(f"  Time: {elapsed_optimized:.4f}s")

# Now test with UNOPTIMIZED method (original naive loop)
# Reset star
a_mat_orig = np.zeros((input_size, num_generators), dtype=np.float32)
a_mat_orig[0, 0] = 1.0
a_mat_orig[1, 1] = 1.0
a_mat_orig[idx_2, 2] = 1.0
a_mat_orig[idx_3, 3] = 1.0
for i in range(4, 10):
    y = (i - 4) % 4
    x = (i - 4) // 4
    channel = i % 3
    idx = y * 8 * 3 + x * 3 + channel
    a_mat_orig[idx, i] = 1.0

star_unopt = LpStar(a_mat_orig.copy(), bias.copy())

# Temporarily disable batching by modifying the method
def transform_star_unoptimized(self, star):
    """Original unoptimized version"""
    from nnenum.network import nn_unflatten, nn_flatten
    shape = self.get_input_shape()
    result_columns = []

    for cindex in range(star.a_mat.shape[1]):
        column = star.a_mat[:, cindex]
        multichannel_state = nn_unflatten(column, shape)
        multichannel_state = self.execute(multichannel_state, zero_bias=True)
        flat = nn_flatten(multichannel_state)
        flat.shape = (flat.size, 1)
        result_columns.append(flat)

    star.a_mat = np.hstack(result_columns)

    multichannel_state = nn_unflatten(star.bias, shape)
    multichannel_state = self.execute(multichannel_state)
    flat = nn_flatten(multichannel_state)
    star.bias = flat

# Create fresh layer for unoptimized test
conv_layer_unopt = Convolutional2dLayer(
    layer_num=0,
    kernels=kernels,
    biases=biases,
    prev_layer_output_shape=input_shape,
    mode='same',
    boundary='fill',
    strides=(1, 1)
)

start = time.time()
transform_star_unoptimized(conv_layer_unopt, star_unopt)
elapsed_unopt = time.time() - start

print(f"\nTransformed star (UNOPTIMIZED):")
print(f"  a_mat shape: {star_unopt.a_mat.shape}")
print(f"  bias shape: {star_unopt.bias.shape}")
print(f"  Time: {elapsed_unopt:.4f}s")

# Compare results
print(f"\n=== COMPARISON ===")
print(f"Speedup: {elapsed_unopt / elapsed_optimized:.2f}x")
print(f"Max difference in a_mat: {np.max(np.abs(star.a_mat - star_unopt.a_mat))}")
print(f"Max difference in bias: {np.max(np.abs(star.bias - star_unopt.bias))}")

if np.allclose(star.a_mat, star_unopt.a_mat, rtol=1e-5, atol=1e-7) and \
   np.allclose(star.bias, star_unopt.bias, rtol=1e-5, atol=1e-7):
    print("\n✓ CORRECTNESS: Results match!")
else:
    print("\n✗ CORRECTNESS: Results DO NOT match!")
    print("This suggests a bug in the batching implementation.")
