#!/usr/bin/env python3
"""
Test adaptive batching that skips dense generators
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from nnenum.network import Convolutional2dLayer, nn_unflatten, nn_flatten
from nnenum.settings import Settings

# Enable logging
Settings.LOG_CONV_BATCHING = True

# Test with different sparsity levels
input_shape = (32, 32, 3)
kernel_size = 3
num_filters = 64

np.random.seed(42)
kernels = np.random.randn(num_filters, 3, kernel_size, kernel_size).astype(np.float32) * 0.1
biases = np.random.randn(num_filters).astype(np.float32) * 0.1

layer = Convolutional2dLayer(0, kernels, biases, input_shape, mode='same', strides=(1, 1))

print(f"Default MIN_SPARSITY threshold: {Settings.CONV_BATCHING_MIN_SPARSITY}")
print()

# Test 1: Sparse (one-hot) - should batch
print("="*70)
print("TEST 1: Sparse one-hot generators (sparsity < 0.05%)")
print("="*70)
num_gens = 100
input_size = np.prod(input_shape)
a_mat_sparse = np.eye(input_size, num_gens, dtype=np.float32)
sparsity_sparse = np.count_nonzero(a_mat_sparse) / a_mat_sparse.size

print(f"Generator matrix: {a_mat_sparse.shape}")
print(f"Sparsity: {sparsity_sparse*100:.4f}%")
print(f"Should batch: {sparsity_sparse <= Settings.CONV_BATCHING_MIN_SPARSITY}")

# Create mock star object
class MockStar:
    def __init__(self, a_mat, bias):
        self.a_mat = a_mat
        self.bias = bias

bias = np.random.randn(input_size).astype(np.float32) * 0.01
star_sparse = MockStar(a_mat_sparse, bias)

import time
start = time.time()
layer.transform_star(star_sparse)
elapsed_sparse = time.time() - start

print(f"Time: {elapsed_sparse:.4f}s")
print()

# Test 2: Medium density (2% sparse) - should batch
print("="*70)
print("TEST 2: Medium density generators (sparsity ~2%)")
print("="*70)

# Create generators with ~2% nonzeros
a_mat_medium = np.zeros((input_size, num_gens), dtype=np.float32)
for i in range(num_gens):
    # Add ~2% nonzeros randomly
    num_nonzeros = int(input_size * 0.02)
    indices = np.random.choice(input_size, num_nonzeros, replace=False)
    a_mat_medium[indices, i] = np.random.randn(num_nonzeros).astype(np.float32)

sparsity_medium = np.count_nonzero(a_mat_medium) / a_mat_medium.size

print(f"Generator matrix: {a_mat_medium.shape}")
print(f"Sparsity: {sparsity_medium*100:.4f}%")
print(f"Should batch: {sparsity_medium <= Settings.CONV_BATCHING_MIN_SPARSITY}")

star_medium = MockStar(a_mat_medium, bias.copy())

start = time.time()
layer.transform_star(star_medium)
elapsed_medium = time.time() - start

print(f"Time: {elapsed_medium:.4f}s")
print()

# Test 3: Dense (10% sparse) - should NOT batch
print("="*70)
print("TEST 3: Dense generators (sparsity ~10%)")
print("="*70)

# Create generators with ~10% nonzeros
a_mat_dense = np.zeros((input_size, num_gens), dtype=np.float32)
for i in range(num_gens):
    num_nonzeros = int(input_size * 0.10)
    indices = np.random.choice(input_size, num_nonzeros, replace=False)
    a_mat_dense[indices, i] = np.random.randn(num_nonzeros).astype(np.float32)

sparsity_dense = np.count_nonzero(a_mat_dense) / a_mat_dense.size

print(f"Generator matrix: {a_mat_dense.shape}")
print(f"Sparsity: {sparsity_dense*100:.4f}%")
print(f"Should batch: {sparsity_dense <= Settings.CONV_BATCHING_MIN_SPARSITY}")

star_dense = MockStar(a_mat_dense, bias.copy())

start = time.time()
layer.transform_star(star_dense)
elapsed_dense = time.time() - start

print(f"Time: {elapsed_dense:.4f}s")
print()

print("="*70)
print("SUMMARY")
print("="*70)
print(f"Sparse (0.03%):  {elapsed_sparse:.4f}s - BATCHED")
print(f"Medium (2%):     {elapsed_medium:.4f}s - {'BATCHED' if sparsity_medium <= Settings.CONV_BATCHING_MIN_SPARSITY else 'UNBATCHED'}")
print(f"Dense (10%):     {elapsed_dense:.4f}s - UNBATCHED (skipped overhead)")
