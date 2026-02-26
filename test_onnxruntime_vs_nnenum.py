#!/usr/bin/env python3
"""Compare onnxruntime execution vs nnenum execution."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import onnx
import onnxruntime as ort

print("="*80)
print("COMPARING ONNXRUNTIME VS NNENUM EXECUTION")
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

# Reshape to 32x32x3
test_input = test_input_flat.reshape(32, 32, 3)

print(f"\nTest input shape: {test_input.shape}")
print(f"Test input (first channel, first row): {test_input[0, :5, 0]}")

# Execute with onnxruntime
print(f"\n{'='*80}")
print("ONNXRUNTIME EXECUTION")
print('='*80)

onnx_model = onnx.load(onnx_path)
sess = ort.InferenceSession(onnx_model.SerializeToString())

# Get input/output names
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

print(f"Input name: {input_name}")
print(f"Output name: {output_name}")

# ONNX expects NCHW format (batch, channels, height, width)
# Our input is HWC format (height, width, channels)
test_input_nchw = np.transpose(test_input, (2, 0, 1))  # CHW
test_input_nchw = test_input_nchw[np.newaxis, ...]    # BCHW

print(f"ONNX input shape (NCHW): {test_input_nchw.shape}")

output_ort = sess.run([output_name], {input_name: test_input_nchw})[0]
output_ort_flat = output_ort.flatten()

print(f"\nONNXruntime output shape: {output_ort.shape}")
print(f"ONNXruntime output: {output_ort_flat}")

predicted_class_ort = np.argmax(output_ort_flat)
print(f"\nONNXruntime predicted class: {predicted_class_ort}")

# Execute with nnenum
print(f"\n{'='*80}")
print("NNENUM EXECUTION")
print('='*80)

from nnenum.onnx_network import load_onnx_network_optimized
from nnenum.network import nn_flatten

network = load_onnx_network_optimized(onnx_path)

# nnenum expects HWC format
output_nnenum = network.execute(test_input)
output_nnenum_flat = nn_flatten(output_nnenum)

print(f"nnenum output shape: {output_nnenum.shape}")
print(f"nnenum output: {output_nnenum_flat}")

predicted_class_nnenum = np.argmax(output_nnenum_flat)
print(f"\nnnenum predicted class: {predicted_class_nnenum}")

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
for i in range(10):
    print(f"  Y_{i}: ort={output_ort_flat[i]:10.6f}, nnenum={output_nnenum_flat[i]:10.6f}, diff={diff[i]:.2e}")

if max_diff < 1e-5:
    print(f"\n✓ MATCH: onnxruntime and nnenum produce the same results")
else:
    print(f"\n✗ MISMATCH: onnxruntime and nnenum produce different results!")
    print(f"  onnxruntime predicts: class {predicted_class_ort}")
    print(f"  nnenum predicts: class {predicted_class_nnenum}")

# Check spec violation
from nnenum.specification import Specification, DisjunctiveSpec

spec_objects = [Specification(mat, rhs) for mat, rhs in spec_list]
disjunctive_spec = DisjunctiveSpec(spec_objects)

is_violation_ort = disjunctive_spec.is_violation(output_ort_flat)
is_violation_nnenum = disjunctive_spec.is_violation(output_nnenum_flat)

print(f"\n{'='*80}")
print("SPECIFICATION CHECK")
print('='*80)

print(f"\nonnxruntime spec violation: {is_violation_ort}")
print(f"nnenum spec violation: {is_violation_nnenum}")

if is_violation_ort != is_violation_nnenum:
    print(f"\n⚠️  VERIFICATION MISMATCH!")
    print(f"  onnxruntime: {'sat' if is_violation_ort else 'unsat'}")
    print(f"  nnenum: {'sat' if is_violation_nnenum else 'unsat'}")
else:
    print(f"\n✓ Both agree: {'sat' if is_violation_ort else 'unsat'}")
