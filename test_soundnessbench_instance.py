#!/usr/bin/env python3
"""
Test a soundnessbench instance to verify the flatten fix works correctly.

Soundnessbench has Conv layers with explicit padding and Flatten before FC,
making it a perfect test case for the HWC→CHW conversion fix.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import onnx
import onnxruntime as ort

print("="*80)
print("TESTING SOUNDNESSBENCH WITH FLATTEN FIX")
print("="*80)

benchmark_dir = "vnncomp2025_benchmarks/benchmarks/soundnessbench"
onnx_path = f"{benchmark_dir}/onnx/model.onnx"
vnnlib_path = f"{benchmark_dir}/vnnlib/model_0.vnnlib"

print(f"\nModel: {onnx_path}")
print(f"Spec: {vnnlib_path}")

# Load and check model structure
print(f"\n{'='*80}")
print("MODEL STRUCTURE")
print('='*80)

model = onnx.load(onnx_path)
layer_types = {}
for node in model.graph.node:
    layer_types[node.op_type] = layer_types.get(node.op_type, 0) + 1

print(f"\nLayers:")
for op, count in sorted(layer_types.items()):
    print(f"  {op}: {count}")

# Check Conv layers have explicit padding
conv_nodes = [n for n in model.graph.node if n.op_type == 'Conv']
print(f"\nConv layers: {len(conv_nodes)}")
for i, node in enumerate(conv_nodes):
    pads = None
    for attr in node.attribute:
        if attr.name == 'pads':
            pads = list(attr.ints)
    print(f"  Conv {i}: pads={pads}")

# Load with nnenum
print(f"\n{'='*80}")
print("LOADING WITH NNENUM")
print('='*80)

from nnenum.onnx_network import load_onnx_network_optimized
from nnenum.network import Convolutional2dLayer, FlattenLayer, FullyConnectedLayer

network = load_onnx_network_optimized(onnx_path)
print(f"✓ Loaded {len(network.layers)} layers")

# Check for Conv and Flatten layers
conv_layers = [i for i, l in enumerate(network.layers) if isinstance(l, Convolutional2dLayer)]
flatten_layers = [i for i, l in enumerate(network.layers) if isinstance(l, FlattenLayer)]
fc_layers = [i for i, l in enumerate(network.layers) if isinstance(l, FullyConnectedLayer)]

print(f"\nLayer types:")
print(f"  Convolutional: {len(conv_layers)} layers at indices {conv_layers}")
print(f"  Flatten: {len(flatten_layers)} layers at indices {flatten_layers}")
print(f"  FullyConnected: {len(fc_layers)} layers at indices {fc_layers}")

# Check Conv layers have explicit pads
print(f"\nConv layer padding:")
for i in conv_layers:
    layer = network.layers[i]
    print(f"  Layer {i}: pads={layer.pads}, mode={layer.mode}")

# Load spec
print(f"\n{'='*80}")
print("LOADING SPECIFICATION")
print('='*80)

from nnenum.vnnlib import get_num_inputs_outputs, read_vnnlib_simple

num_inputs, num_outputs, dtype = get_num_inputs_outputs(onnx_path)
print(f"  Inputs: {num_inputs}, Outputs: {num_outputs}")

specs = read_vnnlib_simple(vnnlib_path, num_inputs, num_outputs)
box, spec_list = specs[0]

print(f"  Spec clauses: {len(spec_list)}")

# Create test input (center of box)
test_input_flat = np.array([(low + high) / 2.0 for low, high in box], dtype=np.float32)

# Get input shape
input_shape = network.layers[0].get_input_shape()
print(f"  Network input shape: {input_shape}")

# Reshape to HWC format for nnenum
test_input_hwc = test_input_flat.reshape(input_shape)

# Execute with onnxruntime
print(f"\n{'='*80}")
print("ONNXRUNTIME EXECUTION")
print('='*80)

sess = ort.InferenceSession(model.SerializeToString())
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# Convert to ONNX format
if len(test_input_hwc.shape) == 1:
    # 1D input, just add batch dimension
    test_input_onnx = test_input_hwc[np.newaxis, ...]
else:
    # Multi-dimensional, convert HWC → CHW and add batch dimension
    test_input_nchw = np.transpose(test_input_hwc, (2, 0, 1))
    test_input_onnx = test_input_nchw[np.newaxis, ...]

output_ort = sess.run([output_name], {input_name: test_input_onnx})[0]
output_ort_flat = output_ort.flatten()

print(f"Output shape: {output_ort.shape}")
print(f"Output: {output_ort_flat}")

# Execute with nnenum
print(f"\n{'='*80}")
print("NNENUM EXECUTION")
print('='*80)

from nnenum.network import nn_flatten

output_nnenum = network.execute(test_input_hwc)
output_nnenum_flat = nn_flatten(output_nnenum)

print(f"Output shape: {output_nnenum.shape}")
print(f"Output: {output_nnenum_flat}")

# Compare
print(f"\n{'='*80}")
print("COMPARISON")
print('='*80)

diff = np.abs(output_ort_flat - output_nnenum_flat)
max_diff = np.max(diff)
mean_diff = np.mean(diff)

print(f"\nMax difference: {max_diff:.2e}")
print(f"Mean difference: {mean_diff:.2e}")

print(f"\nPer-output comparison:")
for i in range(len(output_ort_flat)):
    print(f"  Output {i}: ort={output_ort_flat[i]:10.6f}, nnenum={output_nnenum_flat[i]:10.6f}, diff={diff[i]:.2e}")

if max_diff < 1e-5:
    print(f"\n✓ PASS: nnenum matches onnxruntime (max diff < 1e-5)")
    success = True
else:
    print(f"\n✗ FAIL: nnenum differs from onnxruntime (max diff = {max_diff:.2e})")
    success = False

# Check spec
print(f"\n{'='*80}")
print("SPECIFICATION CHECK")
print('='*80)

from nnenum.specification import Specification, DisjunctiveSpec

spec_objects = [Specification(mat, rhs) for mat, rhs in spec_list]
disjunctive_spec = DisjunctiveSpec(spec_objects)

is_violation_ort = disjunctive_spec.is_violation(output_ort_flat)
is_violation_nnenum = disjunctive_spec.is_violation(output_nnenum_flat)

print(f"\nonnxruntime spec violation: {is_violation_ort}")
print(f"nnenum spec violation: {is_violation_nnenum}")

if is_violation_ort == is_violation_nnenum:
    print(f"✓ PASS: Both agree on spec result")
else:
    print(f"✗ FAIL: Spec results differ!")
    success = False

print(f"\n{'='*80}")
print("FINAL RESULT")
print('='*80)

if success:
    print("✓ ALL CHECKS PASSED")
    print("\nThe flatten fix correctly handles soundnessbench:")
    print("  - Conv layers use explicit padding")
    print("  - Flatten converts HWC → CHW before flattening")
    print("  - Execution matches onnxruntime")
    print("  - Spec evaluation is consistent")
else:
    print("✗ SOME CHECKS FAILED")
    sys.exit(1)
