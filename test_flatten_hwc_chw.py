#!/usr/bin/env python3
"""Test that FlattenLayer correctly converts HWC → CHW before flattening."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from nnenum.network import FlattenLayer, nn_flatten

print("="*80)
print("TEST: FlattenLayer HWC → CHW Conversion")
print("="*80)

# Test 1: Simple 2x2x2 tensor
print("\nTest 1: Simple 2x2x2 tensor")
print("-" * 40)

# HWC format: H=2, W=2, C=2
data_hwc = np.array([
    [[1, 2],   # h=0, w=0: [c=0, c=1]
     [3, 4]],  # h=0, w=1: [c=0, c=1]
    [[5, 6],   # h=1, w=0: [c=0, c=1]
     [7, 8]]   # h=1, w=1: [c=0, c=1]
], dtype=np.float32)

print(f"Input (HWC): shape={data_hwc.shape}")
print(f"  [[[1, 2], [3, 4]],")
print(f"   [[5, 6], [7, 8]]]")

# Create flatten layer
layer = FlattenLayer(0, input_shape=(2, 2, 2))

# Execute
output = layer.execute(data_hwc)

print(f"\nOutput (after HWC→CHW): {output}")

# Expected: CHW format would be C=2, H=2, W=2
# [[[1, 3],   # c=0, h=0: [w=0, w=1]
#   [5, 7]],  # c=0, h=1: [w=0, w=1]
#  [[2, 4],   # c=1, h=0: [w=0, w=1]
#   [6, 8]]]  # c=1, h=1: [w=0, w=1]
# Flattened (row-major): [1, 3, 5, 7, 2, 4, 6, 8]

expected = np.array([1, 3, 5, 7, 2, 4, 6, 8], dtype=np.float32)
print(f"Expected:                {expected}")

if np.allclose(output, expected):
    print("✓ PASS: Output matches expected CHW-flattened order")
else:
    print("✗ FAIL: Output does NOT match expected")
    print(f"  Difference: {output - expected}")
    sys.exit(1)

# Test 2: Match ONNX behavior
print("\n" + "="*80)
print("Test 2: Match ONNX Flatten behavior")
print("-" * 40)

import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

# Create simple ONNX model: Input → Flatten → Output
input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 2, 3, 4])  # NCHW
output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 24])

flatten_node = helper.make_node('Flatten', ['input'], ['output'], axis=1)

graph = helper.make_graph([flatten_node], 'test_flatten', [input_tensor], [output_tensor])
model = helper.make_model(graph)

# Run with onnxruntime
sess = ort.InferenceSession(model.SerializeToString())

# Create test input (NCHW format)
test_input_nchw = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)
print(f"ONNX input shape (NCHW): {test_input_nchw.shape}")
print(f"ONNX input (first batch): {test_input_nchw[0, :, :, :]}")

output_onnx = sess.run(['output'], {'input': test_input_nchw})[0]
output_onnx_flat = output_onnx.flatten()

print(f"\nONNX output: {output_onnx_flat}")

# Convert NCHW to HWC for nnenum
test_input_hwc = np.transpose(test_input_nchw[0], (1, 2, 0))  # CHW → HWC
print(f"\nnnenum input shape (HWC): {test_input_hwc.shape}")

# Run through nnenum FlattenLayer
layer_nnenum = FlattenLayer(0, input_shape=test_input_hwc.shape)
output_nnenum = layer_nnenum.execute(test_input_hwc)

print(f"nnenum output: {output_nnenum}")

# Compare
diff = np.abs(output_onnx_flat - output_nnenum)
max_diff = np.max(diff)

print(f"\nMax difference: {max_diff:.2e}")

if max_diff < 1e-6:
    print("✓ PASS: nnenum matches ONNX Flatten behavior")
else:
    print("✗ FAIL: nnenum does NOT match ONNX")
    sys.exit(1)

# Test 3: Verify 1D input passes through unchanged
print("\n" + "="*80)
print("Test 3: 1D input passes through unchanged")
print("-" * 40)

input_1d = np.array([1, 2, 3, 4, 5], dtype=np.float32)
layer_1d = FlattenLayer(0, input_shape=(5,))

output_1d = layer_1d.execute(input_1d)

print(f"Input (1D): {input_1d}")
print(f"Output:     {output_1d}")

if np.allclose(input_1d, output_1d):
    print("✓ PASS: 1D input unchanged")
else:
    print("✗ FAIL: 1D input was modified")
    sys.exit(1)

# Test 4: Real network shape (from relusplitter)
print("\n" + "="*80)
print("Test 4: Relusplitter network shape (16x16x16)")
print("-" * 40)

input_shape = (16, 16, 16)
test_data = np.random.randn(*input_shape).astype(np.float32)

layer_real = FlattenLayer(0, input_shape=input_shape)
output_real = layer_real.execute(test_data)

expected_size = 16 * 16 * 16
print(f"Input shape: {input_shape}")
print(f"Output shape: {output_real.shape}")
print(f"Expected size: {expected_size}")

if output_real.shape == (expected_size,):
    print("✓ PASS: Output shape correct")
else:
    print("✗ FAIL: Output shape incorrect")
    sys.exit(1)

# Verify CHW ordering by checking specific elements
# HWC[h, w, c] should map to CHW[c, h, w]
# After flattening CHW: index = c * (H*W) + h * W + w
test_positions = [
    (0, 0, 0),  # First element
    (0, 0, 15), # Last channel at h=0, w=0
    (0, 15, 0), # Last width at h=0, c=0
    (15, 15, 15) # Last element
]

print("\nVerifying element ordering:")
all_match = True
for h, w, c in test_positions:
    # Get value from original HWC data
    hwc_val = test_data[h, w, c]

    # Calculate expected index in CHW-flattened array
    chw_idx = c * (16 * 16) + h * 16 + w

    # Get value from flattened output
    output_val = output_real[chw_idx]

    match = np.isclose(hwc_val, output_val)
    symbol = "✓" if match else "✗"
    print(f"  {symbol} HWC[{h}, {w}, {c}] = {hwc_val:.4f} → CHW index {chw_idx} = {output_val:.4f}")

    if not match:
        all_match = False

if all_match:
    print("\n✓ PASS: CHW ordering verified")
else:
    print("\n✗ FAIL: CHW ordering incorrect")
    sys.exit(1)

print("\n" + "="*80)
print("FINAL RESULT")
print("="*80)
print("✓ ALL TESTS PASSED")
print("\nThe FlattenLayer correctly:")
print("  - Converts HWC → CHW before flattening")
print("  - Matches ONNX Flatten behavior")
print("  - Handles 1D inputs correctly")
print("  - Works with real network shapes")
