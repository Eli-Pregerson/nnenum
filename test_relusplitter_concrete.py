#!/usr/bin/env python3
"""Test concrete execution on the relusplitter network to verify padding fix."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np

print("="*80)
print("TESTING RELUSPLITTER NETWORK CONCRETE EXECUTION")
print("="*80)

# Load network
onnx_path = "vnncomp2025_benchmarks/benchmarks/relusplitter/onnx/oval21-benchmark_cifar_base_kw.onnx"
vnnlib_path = "vnncomp2025_benchmarks/benchmarks/relusplitter/vnnlib/oval21-benchmark_cifar_base_kw-img2537-eps0.006666666666666667.vnnlib"

print(f"\nLoading ONNX: {onnx_path}")

try:
    from nnenum.onnx_network import load_onnx_network_optimized
    network = load_onnx_network_optimized(onnx_path)
    print(f"✓ Network loaded successfully with {len(network.layers)} layers")

    # Check conv layers
    from nnenum.network import Convolutional2dLayer

    conv_layers = [(i, layer) for i, layer in enumerate(network.layers)
                   if isinstance(layer, Convolutional2dLayer)]

    print(f"\nFound {len(conv_layers)} convolutional layers:")
    for i, layer in conv_layers:
        print(f"\n  Layer {i}: Convolutional2dLayer")
        print(f"    Input shape: {layer.get_input_shape()}")
        print(f"    Output shape: {layer.get_output_shape()}")
        print(f"    Kernel shape: {layer.kernels[0][0].shape}")
        print(f"    Strides: {layer.strides}")
        print(f"    Mode: {layer.mode}")
        print(f"    Pads: {layer.pads}")

        if layer.pads is None:
            print(f"    ⚠️  WARNING: Using auto-padding instead of explicit ONNX pads!")
        elif layer.pads == (1, 1, 1, 1):
            print(f"    ✓ Using ONNX explicit padding [1,1,1,1]")
        else:
            print(f"    ⚠️  Unexpected padding: {layer.pads}")

except Exception as e:
    print(f"✗ Error loading network: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Load vnnlib spec
print(f"\n{'='*80}")
print("LOADING VNNLIB SPECIFICATION")
print('='*80)

from nnenum.vnnlib import get_num_inputs_outputs, read_vnnlib_simple

num_inputs, num_outputs, dtype = get_num_inputs_outputs(onnx_path)
print(f"  Inputs: {num_inputs}, Outputs: {num_outputs}")

specs = read_vnnlib_simple(vnnlib_path, num_inputs, num_outputs)
box, spec_list = specs[0]

print(f"  Number of specs: {len(spec_list)}")
print(f"  Input box shape: {len(box)} constraints")

# Test with center of input box
print(f"\n{'='*80}")
print("TESTING CONCRETE EXECUTION")
print('='*80)

# Create input at center of box
test_input = np.array([(low + high) / 2.0 for low, high in box], dtype=np.float32)
print(f"\nInput shape: {test_input.shape}")
print(f"Input (first 10): {test_input[:10]}")
print(f"Input (last 10): {test_input[-10:]}")

# Check input is in box
in_box = all(low <= val <= high for val, (low, high) in zip(test_input, box))
print(f"✓ Input is in box: {in_box}")

# Execute network
from nnenum.network import nn_unflatten, nn_flatten

# Unflatten to network input shape
input_shape = network.layers[0].get_input_shape()
print(f"\nNetwork input shape: {input_shape}")

test_input_shaped = nn_unflatten(test_input, input_shape)
print(f"Reshaped input: {test_input_shaped.shape}")

print(f"\nExecuting network...")
output = network.execute(test_input_shaped)
output_flat = nn_flatten(output)

print(f"\nOutput shape: {output.shape}")
print(f"Output (flattened): {output_flat}")

# Check which class has highest output
predicted_class = np.argmax(output_flat)
print(f"\nPredicted class: {predicted_class}")
print(f"Output values:")
for i, val in enumerate(output_flat):
    marker = " <-- MAX" if i == predicted_class else ""
    print(f"  Y_{i} = {val:10.6f}{marker}")

# Check against spec
from nnenum.specification import Specification, DisjunctiveSpec

spec_objects = []
for mat, rhs in spec_list:
    spec_objects.append(Specification(mat, rhs))

disjunctive_spec = DisjunctiveSpec(spec_objects)

is_violation = disjunctive_spec.is_violation(output_flat)

print(f"\n{'='*80}")
print("SPECIFICATION CHECK")
print('='*80)

print(f"\nSpec property: (or (Y_1 <= Y_0) (Y_1 <= Y_2) ... (Y_1 <= Y_9))")
print(f"This asks: Is Y_1 (automobile) <= any other class?")
print(f"\nY_1 = {output_flat[1]:.6f}")

print(f"\nComparisons:")
for i in range(10):
    if i == 1:
        continue
    satisfied = output_flat[1] <= output_flat[i]
    symbol = "✓" if satisfied else "✗"
    print(f"  {symbol} Y_1 <= Y_{i}: {output_flat[1]:.6f} <= {output_flat[i]:.6f}? {satisfied}")

num_satisfied = sum(1 for i in range(10) if i != 1 and output_flat[1] <= output_flat[i])

print(f"\nNumber of satisfied clauses: {num_satisfied}/9")
print(f"Spec is_violation: {is_violation}")

if is_violation:
    print(f"\n✗ COUNTEREXAMPLE FOUND: Network misclassifies this input")
    print(f"  Expected class: 1 (automobile)")
    print(f"  Predicted class: {predicted_class}")
else:
    print(f"\n✓ Property holds: Network correctly predicts class 1")

print(f"\n{'='*80}")
print("SUMMARY")
print('='*80)

print(f"\n1. Padding fix applied: {conv_layers[0][1].pads == (1, 1, 1, 1)}")
print(f"2. Network executes: True")
print(f"3. Center input result: {'COUNTEREXAMPLE' if is_violation else 'SAFE'}")
