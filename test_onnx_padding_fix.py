#!/usr/bin/env python3
"""
Test that ONNX explicit padding is handled correctly.

This tests the fix for the soundness bug where ONNX pads=[1,1,1,1] was
incorrectly treated as scipy mode='same' with different padding.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from nnenum.network import Convolutional2dLayer, nn_unflatten, nn_flatten
from scipy.signal import convolve2d

print("="*80)
print("TEST: ONNX Explicit Padding Correctness")
print("="*80)

# Test case matching the relusplitter network configuration
input_shape = (32, 32, 3)
kernel_size = 4
num_filters = 16
strides = (2, 2)
onnx_pads = (1, 1, 1, 1)  # [top, left, bottom, right]

# Create layer with ONNX explicit padding
np.random.seed(42)
kernels = np.random.randn(num_filters, 3, kernel_size, kernel_size).astype(np.float32) * 0.1
biases = np.random.randn(num_filters).astype(np.float32) * 0.1

layer_onnx = Convolutional2dLayer(0, kernels, biases, input_shape,
                                   mode='same', strides=strides, pads=onnx_pads)

# Create layer with scipy 'same' padding (the old buggy behavior)
layer_same = Convolutional2dLayer(0, kernels, biases, input_shape,
                                  mode='same', strides=strides, pads=None)

print(f"\nConfiguration:")
print(f"  Input: {input_shape}")
print(f"  Kernel: {kernel_size}x{kernel_size}")
print(f"  Filters: {num_filters}")
print(f"  Stride: {strides}")
print(f"  ONNX pads: {onnx_pads}")

# Check output shapes
output_shape_onnx = layer_onnx.get_output_shape()
output_shape_same = layer_same.get_output_shape()

print(f"\nOutput shapes:")
print(f"  ONNX explicit pads: {output_shape_onnx}")
print(f"  scipy 'same':       {output_shape_same}")

# Expected ONNX output shape:
# Input: 32x32
# After padding [1,1,1,1]: 34x34
# After 4x4 conv: 34 - 4 + 1 = 31x31
# After stride 2: (31 - 1) / 2 + 1 = 16x16
expected_shape = (16, 16, num_filters)

print(f"  Expected (ONNX):    {expected_shape}")

assert output_shape_onnx == expected_shape, \
    f"ONNX output shape {output_shape_onnx} doesn't match expected {expected_shape}"
print("  ✓ ONNX output shape correct")

# Test execution with a random input
test_input = np.random.randn(32, 32, 3).astype(np.float32)

output_onnx = layer_onnx.execute(test_input)
output_same = layer_same.execute(test_input)

print(f"\nActual output shapes after execution:")
print(f"  ONNX explicit pads: {output_onnx.shape}")
print(f"  scipy 'same':       {output_same.shape}")

assert output_onnx.shape == expected_shape, \
    f"ONNX execution output shape {output_onnx.shape} doesn't match expected {expected_shape}"
print("  ✓ ONNX execution shape correct")

# Verify they produce DIFFERENT results (confirming the bug was real)
if np.allclose(output_onnx, output_same):
    print("\n  ⚠️  WARNING: ONNX and 'same' padding produce identical results!")
    print("  This suggests the fix may not be working correctly.")
else:
    max_diff = np.max(np.abs(output_onnx - output_same))
    mean_diff = np.mean(np.abs(output_onnx - output_same))
    print(f"\n  ✓ ONNX and 'same' produce different results (confirming bug was real)")
    print(f"    Max diff: {max_diff:.4f}")
    print(f"    Mean diff: {mean_diff:.4f}")

# Manually verify ONNX execution matches expected behavior
print(f"\n{'='*80}")
print("MANUAL VERIFICATION: ONNX padding correctness")
print('='*80)

# Manual computation for first output filter across all input channels
# Output filter 0 sums contributions from all 3 input channels

result_manual = np.zeros((16, 16), dtype=np.float32)

for in_channel in range(3):
    test_channel = test_input[:, :, in_channel]  # 32x32
    kernel_channel = layer_onnx.kernels[0][in_channel]  # Already flipped

    # Step 1: Pad input with [1,1,1,1]
    padded = np.pad(test_channel, ((1, 1), (1, 1)), mode='constant', constant_values=0)

    # Step 2: Convolve with valid mode
    conv_result = convolve2d(padded, kernel_channel, mode='valid')

    # Step 3: Subsample by stride 2
    strided_result = conv_result[::2, ::2]

    # Accumulate
    result_manual += strided_result

# Add bias
result_manual += biases[0]

# Compare with layer's output for first filter
layer_output_filter0 = output_onnx[:, :, 0]

manual_diff = np.max(np.abs(layer_output_filter0 - result_manual))
print(f"  Manual computation (all 3 input channels)")
print(f"  Max difference between layer and manual: {manual_diff:.2e}")

if manual_diff < 1e-5:
    print("  ✓ Layer execution matches manual ONNX-style computation")
else:
    print(f"  ✗ MISMATCH! Layer execution differs from expected")

# Test batching with explicit padding
print(f"\n{'='*80}")
print("TEST: Batching with explicit ONNX padding")
print('='*80)

# Create sparse generators
num_gens = 50
input_size = np.prod(input_shape)
a_mat = np.eye(input_size, num_gens, dtype=np.float32)

class MockStar:
    def __init__(self, a_mat, bias):
        self.a_mat = a_mat
        self.bias = bias

bias = np.zeros(input_size, dtype=np.float32)

# Test batched
star_batched = MockStar(a_mat.copy(), bias.copy())
from nnenum.settings import Settings
Settings.CONV_BATCHING_FIRST_LAYER_ONLY = False
layer_onnx.layer_num = 0
layer_onnx.transform_star(star_batched)
result_batched = star_batched.a_mat

# Test unbatched
star_unbatched = MockStar(a_mat.copy(), bias.copy())
Settings.CONV_BATCHING_FIRST_LAYER_ONLY = True
layer_onnx.layer_num = 1
layer_onnx.transform_star(star_unbatched)
result_unbatched = star_unbatched.a_mat

# Compare
max_diff = np.max(np.abs(result_batched - result_unbatched))
print(f"  Generators: {num_gens}")
print(f"  Max diff (batched vs unbatched): {max_diff:.2e}")

if max_diff < 1e-6:
    print("  ✓ Batching works correctly with ONNX explicit padding")
else:
    print(f"  ✗ BATCHING BUG with ONNX padding!")

print(f"\n{'='*80}")
print("FINAL RESULT")
print('='*80)

if manual_diff < 1e-5 and max_diff < 1e-6:
    print("✓ ALL TESTS PASSED")
    print("\nThe ONNX explicit padding bug is fixed:")
    print("  - Output shapes are correct")
    print("  - Execution matches expected ONNX behavior")
    print("  - Batching optimization works with explicit padding")
else:
    print("✗ SOME TESTS FAILED")
    sys.exit(1)
