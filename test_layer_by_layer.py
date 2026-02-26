#!/usr/bin/env python3
"""Compare layer-by-layer execution between onnxruntime and nnenum."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import onnx
import onnxruntime as ort

print("="*80)
print("LAYER-BY-LAYER EXECUTION COMPARISON")
print("="*80)

onnx_path = "vnncomp2025_benchmarks/benchmarks/relusplitter/onnx/oval21-benchmark_cifar_base_kw.onnx"
vnnlib_path = "vnncomp2025_benchmarks/benchmarks/relusplitter/vnnlib/oval21-benchmark_cifar_base_kw-img2537-eps0.006666666666666667.vnnlib"

# Load spec to get input box
from nnenum.vnnlib import get_num_inputs_outputs, read_vnnlib_simple

num_inputs, num_outputs, dtype = get_num_inputs_outputs(onnx_path)
specs = read_vnnlib_simple(vnnlib_path, num_inputs, num_outputs)
box, spec_list = specs[0]

# Create test input (center of box)
test_input_flat = np.array([(low + high) / 2.0 for low, high in box], dtype=np.float32)

# Reshape to 32x32x3 (HWC format for nnenum)
test_input_hwc = test_input_flat.reshape(32, 32, 3)

print(f"\nTest input shape (HWC): {test_input_hwc.shape}")

# Execute with nnenum layer by layer
print(f"\n{'='*80}")
print("NNENUM LAYER-BY-LAYER")
print('='*80)

from nnenum.onnx_network import load_onnx_network_optimized
from nnenum.network import nn_flatten, Convolutional2dLayer, ReluLayer, FlattenLayer, FullyConnectedLayer

network = load_onnx_network_optimized(onnx_path)

state = test_input_hwc.copy()

for i, layer in enumerate(network.layers):
    print(f"\nLayer {i}: {layer.__class__.__name__}")
    if isinstance(layer, Convolutional2dLayer):
        print(f"  Input shape: {layer.get_input_shape()}")
        print(f"  Output shape: {layer.get_output_shape()}")
        print(f"  Num filters: {len(layer.kernels)}")
        print(f"  Kernel shape (first filter): {layer.kernels[0][0].shape}")
        print(f"  Strides: {layer.strides}")
        print(f"  Pads: {layer.pads}")
        print(f"  Mode: {layer.mode}")

    print(f"  Input: shape={state.shape}, mean={np.mean(state):.6f}, std={np.std(state):.6f}")

    state = layer.execute(state)

    print(f"  Output: shape={state.shape}, mean={np.mean(state):.6f}, std={np.std(state):.6f}")
    print(f"  Output (first 10 elements): {nn_flatten(state)[:10]}")

print(f"\n{'='*80}")
print("ONNXRUNTIME WITH INTERMEDIATE OUTPUTS")
print('='*80)

# Load ONNX model and add intermediate outputs
onnx_model = onnx.load(onnx_path)

# Get all node outputs
intermediate_layer_value_info = []
for node in onnx_model.graph.node:
    for output in node.output:
        intermediate_layer_value_info.append(output)

print(f"\nIntermediate outputs: {intermediate_layer_value_info}")

# Create modified model with all outputs
from onnx import helper

# Add intermediate outputs to the model
for output_name in intermediate_layer_value_info:
    # Check if already in outputs
    existing_outputs = [o.name for o in onnx_model.graph.output]
    if output_name not in existing_outputs:
        # Create a new output
        intermediate_output = helper.ValueInfoProto()
        intermediate_output.name = output_name
        onnx_model.graph.output.append(intermediate_output)

# Run modified model
sess = ort.InferenceSession(onnx_model.SerializeToString())

# Convert input to NCHW format
test_input_nchw = np.transpose(test_input_hwc, (2, 0, 1))  # CHW
test_input_nchw = test_input_nchw[np.newaxis, ...]         # BCHW

input_name = sess.get_inputs()[0].name
outputs = sess.run(None, {input_name: test_input_nchw})

print(f"\nONNXruntime produced {len(outputs)} outputs")

for i, (output, name) in enumerate(zip(outputs, [o.name for o in sess.get_outputs()])):
    print(f"\nOutput {i}: {name}")
    print(f"  Shape: {output.shape}")
    print(f"  Mean: {np.mean(output):.6f}, Std: {np.std(output):.6f}")
    flat = output.flatten()
    print(f"  First 10 elements: {flat[:10]}")

print(f"\n{'='*80}")
print("COMPARISON")
print('='*80)

# Final output comparison
final_output_ort = outputs[-1].flatten()
final_output_nnenum = nn_flatten(state)

diff = np.abs(final_output_ort - final_output_nnenum)
print(f"\nFinal output comparison:")
print(f"  Max diff: {np.max(diff):.2e}")
print(f"  Mean diff: {np.mean(diff):.2e}")
