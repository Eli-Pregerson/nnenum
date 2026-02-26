#!/usr/bin/env python3
"""
Debug soundness issue on relusplitter instance where nnenum=sat but abc=unsat
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from nnenum.onnx_network import load_onnx_network_optimized
from nnenum.vnnlib import read_vnnlib_simple
from nnenum.enumerate import enumerate_network
from nnenum.settings import Settings
import numpy as np

onnx_path = "vnncomp2025_benchmarks/benchmarks/relusplitter/onnx/oval21-benchmark_cifar_base_kw.onnx"
vnnlib_path = "vnncomp2025_benchmarks/benchmarks/relusplitter/vnnlib/oval21-benchmark_cifar_base_kw-img2537-eps0.006666666666666667.vnnlib"

print("="*70)
print("DEBUGGING SOUNDNESS BUG")
print("="*70)
print(f"Instance: relusplitter/oval21-benchmark_cifar_base_kw")
print(f"Expected (ABC): unsat")
print(f"Got (nnenum): sat")
print()

# Try with convolution batching DISABLED first
print("="*70)
print("TEST 1: With convolution batching DISABLED")
print("="*70)

Settings.CONV_BATCHING_FIRST_LAYER_ONLY = True  # Force disable
Settings.PRINT_OUTPUT = True
Settings.TIMEOUT = 30.0
Settings.NUM_PROCESSES = 1

# Load network
print("Loading network...")
network = load_onnx_network_optimized(onnx_path)
print(f"Network: {len(network.layers)} layers")

# Check for conv layers
from nnenum.network import Convolutional2dLayer
conv_layers = [i for i, l in enumerate(network.layers) if isinstance(l, Convolutional2dLayer)]
print(f"Conv layers at: {conv_layers}")

# Load spec
print("\nLoading specification...")
spec_list = read_vnnlib_simple(vnnlib_path, network.get_num_inputs(), network.get_num_outputs())
init_box, spec_data = spec_list[0]

if isinstance(init_box, list):
    init_box = np.array(init_box, dtype=np.float32)

from nnenum.specification import Specification
if isinstance(spec_data, list):
    spec = Specification(spec_data[0][0], spec_data[0][1])
else:
    spec = spec_data

print(f"Input shape: {init_box.shape}")
print(f"Spec shape: {spec.mat.shape}")

# Run verification WITHOUT batching
print("\nRunning verification (batching DISABLED)...")
result_no_batch = enumerate_network(init_box, network, spec)

print(f"\nResult WITHOUT batching: {result_no_batch.result_str}")

# Now try WITH batching enabled
print("\n" + "="*70)
print("TEST 2: With convolution batching ENABLED")
print("="*70)

Settings.CONV_BATCHING_FIRST_LAYER_ONLY = False  # Enable
Settings.LOG_CONV_BATCHING = True  # Log batching stats

# Reload network (fresh state)
network = load_onnx_network_optimized(onnx_path)

print("\nRunning verification (batching ENABLED)...")
result_with_batch = enumerate_network(init_box, network, spec)

print(f"\nResult WITH batching: {result_with_batch.result_str}")

# Compare results
print("\n" + "="*70)
print("COMPARISON")
print("="*70)
print(f"WITHOUT batching: {result_no_batch.result_str}")
print(f"WITH batching:    {result_with_batch.result_str}")
print(f"Expected (ABC):   unsat")
print()

if result_no_batch.result_str != result_with_batch.result_str:
    print("⚠️  CRITICAL: Results differ! Batching optimization introduced bug.")
elif result_with_batch.result_str == 'sat':
    print("⚠️  Both return 'sat' but ABC says 'unsat' - bug exists in baseline code.")
else:
    print("✓ Results match and are correct.")
