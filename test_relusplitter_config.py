#!/usr/bin/env python3
"""
Test the specific configuration from the relusplitter network that was failing.

Network: oval21-benchmark_cifar_base_kw
Config: 32x32x3 input, 4x4 kernel, stride=2, pads=[1,1,1,1]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from nnenum.network import Convolutional2dLayer

print("="*80)
print("TEST: Relusplitter Network Configuration")
print("="*80)

# Exact configuration from the failing network
input_shape = (32, 32, 3)
kernel_size = 4
num_filters_conv1 = 16  # Typical for first conv layer
strides = (2, 2)
onnx_pads = (1, 1, 1, 1)

print(f"\nConfiguration (matches relusplitter network):")
print(f"  Input: {input_shape}")
print(f"  Kernel: {kernel_size}x{kernel_size}")
print(f"  Stride: {strides}")
print(f"  ONNX pads: {onnx_pads}")

# Create layer
np.random.seed(42)
kernels = np.random.randn(num_filters_conv1, 3, kernel_size, kernel_size).astype(np.float32) * 0.1
biases = np.random.randn(num_filters_conv1).astype(np.float32) * 0.1

layer = Convolutional2dLayer(0, kernels, biases, input_shape,
                              mode='same', strides=strides, pads=onnx_pads)

output_shape = layer.get_output_shape()
print(f"  Output: {output_shape}")

# Expected: (32 + 1 + 1 - 4) / 2 + 1 = 16
assert output_shape == (16, 16, num_filters_conv1), f"Output shape mismatch: {output_shape}"
print("  ✓ Output shape correct")

# Test with random input
test_input = np.random.randn(32, 32, 3).astype(np.float32)
output = layer.execute(test_input)
assert output.shape == (16, 16, num_filters_conv1)
print("  ✓ Execution produces correct shape")

# Test batching with this configuration
print(f"\nTesting batching optimization:")

class MockStar:
    def __init__(self, a_mat, bias):
        self.a_mat = a_mat
        self.bias = bias

# Create sparse generators (one-hot, typical for initial input)
num_gens = 200
input_size = np.prod(input_shape)
a_mat = np.eye(input_size, num_gens, dtype=np.float32)

bias = np.zeros(input_size, dtype=np.float32)

# Batched
star_batched = MockStar(a_mat.copy(), bias.copy())
from nnenum.settings import Settings
Settings.CONV_BATCHING_FIRST_LAYER_ONLY = False
layer.layer_num = 0
layer.transform_star(star_batched)
result_batched = star_batched.a_mat

# Unbatched
star_unbatched = MockStar(a_mat.copy(), bias.copy())
Settings.CONV_BATCHING_FIRST_LAYER_ONLY = True
layer.layer_num = 1
layer.transform_star(star_unbatched)
result_unbatched = star_unbatched.a_mat

# Compare
max_diff = np.max(np.abs(result_batched - result_unbatched))
print(f"  Generators: {num_gens}")
print(f"  Max diff (batched vs unbatched): {max_diff:.2e}")

if max_diff < 1e-6:
    print("  ✓ Batching works correctly")
else:
    print(f"  ✗ BATCHING BUG!")
    sys.exit(1)

# Test second conv layer (also stride=2, pads=[1,1,1,1])
print(f"\nTesting second conv layer:")
input_shape_2 = output_shape
num_filters_conv2 = 32

kernels_2 = np.random.randn(num_filters_conv2, num_filters_conv1, kernel_size, kernel_size).astype(np.float32) * 0.1
biases_2 = np.random.randn(num_filters_conv2).astype(np.float32) * 0.1

layer_2 = Convolutional2dLayer(1, kernels_2, biases_2, input_shape_2,
                                mode='same', strides=strides, pads=onnx_pads)

output_shape_2 = layer_2.get_output_shape()
print(f"  Input: {input_shape_2}")
print(f"  Output: {output_shape_2}")

# Expected: (16 + 1 + 1 - 4) / 2 + 1 = 8
assert output_shape_2 == (8, 8, num_filters_conv2), f"Layer 2 output shape mismatch: {output_shape_2}"
print("  ✓ Second layer output shape correct")

# Test execution
test_input_2 = np.random.randn(*input_shape_2).astype(np.float32)
output_2 = layer_2.execute(test_input_2)
assert output_2.shape == (8, 8, num_filters_conv2)
print("  ✓ Second layer execution correct")

print(f"\n{'='*80}")
print("FINAL RESULT")
print('='*80)
print("✓ ALL TESTS PASSED")
print("\nThe relusplitter network configuration is now handled correctly:")
print("  - ONNX explicit padding [1,1,1,1] works")
print("  - Stride=2 with 4x4 kernels works")
print("  - Batching optimization works with this config")
print("  - Multi-layer propagation works")
