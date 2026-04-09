#!/usr/bin/env python3
"""
Debug soundness issue on CIFAR100 instance where nnenum=sat but abc=unsat
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from nnenum.onnx_network import load_onnx_network_optimized
from nnenum.vnnlib import read_vnnlib_simple
from nnenum.enumerate import enumerate_network
from nnenum.settings import Settings
import numpy as np

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
Settings.TIMEOUT = 30.0
Settings.NUM_PROCESSES = 1

onnx_path = "vnncomp2025_benchmarks/benchmarks/cifar100_2024/onnx/CIFAR100_resnet_medium.onnx"
vnnlib_path = "vnncomp2025_benchmarks/benchmarks/cifar100_2024/vnnlib/CIFAR100_resnet_medium_prop_idx_2132_sidx_6868_eps_0.0039.vnnlib"

print("Loading network...")
network = load_onnx_network_optimized(onnx_path)

print(f"Network has {len(network.layers)} layers")
print(f"Skip connections: {len(network.dag_predecessors)} SkipAddLayers")

# Check if network has skip connections
from nnenum.network import SkipAddLayer
skip_layers = [i for i, l in enumerate(network.layers) if isinstance(l, SkipAddLayer)]
print(f"SkipAddLayer indices: {skip_layers[:5]}...")

print("\nLoading specification...")
spec_list = read_vnnlib_simple(vnnlib_path, network.get_num_inputs(), network.get_num_outputs())

if len(spec_list) > 1:
    print(f"Multiple specs in vnnlib ({len(spec_list)})")

init_box, spec_data = spec_list[0]
if isinstance(init_box, list):
    init_box = np.array(init_box, dtype=np.float32)

# spec_data is a list of disjuncts, use the first one
from nnenum.specification import Specification
if isinstance(spec_data, list):
    spec = Specification(spec_data[0][0], spec_data[0][1])
else:
    spec = spec_data

print(f"Input box shape: {init_box.shape}")
print(f"Spec matrix shape: {spec.mat.shape}")

print("\nRunning verification...")
result = enumerate_network(init_box, network, spec)

print(f"\n{'='*60}")
print(f"RESULT: {result.result_str}")
print(f"{'='*60}")

if result.result_str in ('unsafe', 'unsafe (unconfirmed)'):
    print("\nThis is the SOUNDNESS BUG!")
    print("nnenum claims 'unsafe' (sat) but ABC says 'unsat' (property holds)")
    print("\nThe overapproximation is incorrectly under-approximating the reachable set.")
