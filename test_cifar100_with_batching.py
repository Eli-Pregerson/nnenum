#!/usr/bin/env python3
"""
Test convolution batching on actual CIFAR100 benchmark
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from nnenum.onnx_network import load_onnx_network_optimized
from nnenum.vnnlib import read_vnnlib_simple
from nnenum.enumerate import enumerate_network
from nnenum.settings import Settings
from nnenum.timerutil import Timers
import numpy as np

# Enable convolution batching logging
Settings.LOG_CONV_BATCHING = True
Settings.TIMING_STATS = True

# Settings for cifar100 (image settings + overapprox)
Settings.COMPRESS_INIT_BOX = True
Settings.BRANCH_MODE = Settings.BRANCH_OVERAPPROX
Settings.TRY_QUICK_OVERAPPROX = False
Settings.OVERAPPROX_MIN_GEN_LIMIT = np.inf
Settings.SPLIT_IF_IDLE = False
Settings.OVERAPPROX_LP_TIMEOUT = np.inf
Settings.CONTRACT_ZONOTOPE = False
Settings.CONTRACT_ZONOTOPE_LP = False
Settings.PRINT_OUTPUT = True
Settings.TIMEOUT = 10.0  # Short timeout for testing
Settings.NUM_PROCESSES = 1

# Use a simple CIFAR100 instance
onnx_path = "vnncomp2025_benchmarks/benchmarks/cifar100_2024/onnx/CIFAR100_resnet_medium.onnx"

# Find a simple vnnlib file
import glob
vnnlib_files = glob.glob("vnncomp2025_benchmarks/benchmarks/cifar100_2024/vnnlib/*.vnnlib")
if not vnnlib_files:
    print("No vnnlib files found!")
    sys.exit(1)

vnnlib_path = vnnlib_files[0]  # Use first one

print(f"Testing on:")
print(f"  ONNX: {onnx_path}")
print(f"  VNNLIB: {vnnlib_path}")

print("\n" + "="*60)
print("Loading network...")
print("="*60)
network = load_onnx_network_optimized(onnx_path)

print(f"\nNetwork has {len(network.layers)} layers")

# Check for convolutional layers
from nnenum.network import Convolutional2dLayer
conv_layers = [i for i, l in enumerate(network.layers) if isinstance(l, Convolutional2dLayer)]
print(f"Convolutional layers at indices: {conv_layers[:5]}...")

print("\n" + "="*60)
print("Loading specification...")
print("="*60)
spec_list = read_vnnlib_simple(vnnlib_path, network.get_num_inputs(), network.get_num_outputs())

if len(spec_list) > 1:
    print(f"Multiple specs in vnnlib ({len(spec_list)}), using first one")

init_box, spec_data = spec_list[0]
if isinstance(init_box, list):
    init_box = np.array(init_box, dtype=np.float32)

from nnenum.specification import Specification
if isinstance(spec_data, list):
    spec = Specification(spec_data[0][0], spec_data[0][1])
else:
    spec = spec_data

print(f"Input box shape: {init_box.shape}")
print(f"Spec matrix shape: {spec.mat.shape}")

print("\n" + "="*60)
print("Running verification...")
print("="*60)

# Clear timers
Timers.reset()

result = enumerate_network(init_box, network, spec)

print("\n" + "="*60)
print(f"RESULT: {result.result_str}")
print("="*60)

# Print timing stats
if Settings.TIMING_STATS:
    print("\n" + "="*60)
    print("TIMING STATISTICS")
    print("="*60)
    Timers.print_stats()

print("\nKey metrics:")
print(f"  Total time: {Timers.top_level_timer['enumerate_network']:.4f}s")

# Check convolution times
conv_times = []
for key in Timers.timers:
    if 'conv' in key.lower() or 'batch' in key.lower():
        time_val = sum(Timers.timers[key])
        if time_val > 0:
            print(f"  {key}: {time_val:.4f}s")
            conv_times.append((key, time_val))
