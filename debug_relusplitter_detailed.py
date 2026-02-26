#!/usr/bin/env python3
"""
Detailed debugging of the relusplitter instance to see what's going wrong.

We'll load the actual network and check:
1. Are the layers being created with correct padding?
2. Does the network structure match expectations?
3. Are there any other issues?
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import onnx
import numpy as np

onnx_path = "vnncomp2025_benchmarks/benchmarks/relusplitter/onnx/oval21-benchmark_cifar_base_kw.onnx"

print("="*80)
print("DETAILED DEBUGGING: relusplitter network")
print("="*80)

# Load ONNX model
model = onnx.load(onnx_path)

print("\nONNX Network Structure:")
print("-" * 80)

for i, node in enumerate(model.graph.node):
    print(f"\nNode {i}: {node.op_type}")

    if node.op_type == 'Conv':
        # Extract Conv attributes
        pads = None
        strides = None
        kernel_shape = None
        auto_pad = None

        for attr in node.attribute:
            if attr.name == 'pads':
                pads = list(attr.ints)
            elif attr.name == 'strides':
                strides = list(attr.ints)
            elif attr.name == 'kernel_shape':
                kernel_shape = list(attr.ints)
            elif attr.name == 'auto_pad':
                auto_pad = attr.s.decode('utf-8')

        print(f"  kernel_shape: {kernel_shape}")
        print(f"  pads: {pads}")
        print(f"  strides: {strides}")
        print(f"  auto_pad: {auto_pad}")

        # What should nnenum do with this?
        if auto_pad == 'VALID' or (pads and all(p == 0 for p in pads)):
            print(f"  → nnenum: mode='valid', pads=None")
        elif auto_pad in ['SAME_UPPER', 'SAME_LOWER']:
            print(f"  → nnenum: mode='same', pads=None")
        elif pads is not None and not all(p == 0 for p in pads):
            print(f"  → nnenum: mode='same', pads={tuple(pads[:4]) if len(pads) >= 4 else pads}")
        else:
            print(f"  → nnenum: mode='same', pads=None (default)")

# Now try to load with nnenum
print("\n" + "="*80)
print("Loading with nnenum...")
print("="*80)

try:
    from nnenum.onnx_network import load_onnx_network_optimized
    network = load_onnx_network_optimized(onnx_path)

    print(f"\nSuccessfully loaded network with {len(network.layers)} layers")

    # Check conv layers
    from nnenum.network import Convolutional2dLayer, ReluLayer, FullyConnectedLayer, FlattenLayer

    for i, layer in enumerate(network.layers):
        if isinstance(layer, Convolutional2dLayer):
            print(f"\nLayer {i}: Convolutional2dLayer")
            print(f"  Input shape: {layer.get_input_shape()}")
            print(f"  Output shape: {layer.get_output_shape()}")
            print(f"  Kernel shape: {layer.kernels[0][0].shape}")
            print(f"  Strides: {layer.strides}")
            print(f"  Mode: {layer.mode}")
            print(f"  Pads: {layer.pads}")

            # Check if pads is set correctly
            if layer.pads is None:
                print(f"  ⚠️  WARNING: Using auto-padding instead of explicit ONNX pads!")
            else:
                print(f"  ✓ Using ONNX explicit padding")
        elif isinstance(layer, ReluLayer):
            print(f"\nLayer {i}: ReluLayer")
        elif isinstance(layer, FlattenLayer):
            print(f"\nLayer {i}: FlattenLayer")
        elif isinstance(layer, FullyConnectedLayer):
            print(f"\nLayer {i}: FullyConnectedLayer")
            print(f"  Input shape: {layer.get_input_shape()}")
            print(f"  Output shape: {layer.get_output_shape()}")

except Exception as e:
    print(f"Error loading network: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)
print("\nCheck if:")
print("1. Conv layers have pads=(1,1,1,1) set?")
print("2. Output shapes are correct?")
print("3. Any other issues?")
