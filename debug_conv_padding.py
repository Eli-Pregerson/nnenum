#!/usr/bin/env python3
"""Debug convolution padding"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from nnenum.network import Convolutional2dLayer
from scipy.signal import convolve2d

# Simple test: 1 input channel, 1 output filter
input_shape = (4, 4, 1)
kernel_size = 2
onnx_pads = (1, 1, 1, 1)
strides = (1, 1)

np.random.seed(42)
kernels = np.random.randn(1, 1, kernel_size, kernel_size).astype(np.float32) * 0.1
biases = np.array([0.0], dtype=np.float32)  # Zero bias for simplicity

layer = Convolutional2dLayer(0, kernels, biases, input_shape,
                              mode='same', strides=strides, pads=onnx_pads)

print("Simple test: 4x4 input, 2x2 kernel, pads=[1,1,1,1]")
print(f"Kernel:\n{kernels[0,0,:,:]}")

# Test input
test_input = np.arange(16, dtype=np.float32).reshape(4, 4, 1)
print(f"\nInput:\n{test_input[:,:,0]}")

# Layer output
output = layer.execute(test_input)
print(f"\nLayer output shape: {output.shape}")
print(f"Layer output:\n{output[:,:,0]}")

# Manual calculation
# Step 1: Pad
padded = np.pad(test_input[:,:,0], ((1,1), (1,1)), mode='constant')
print(f"\nPadded input (6x6):\n{padded}")

# Step 2: Convolve with valid mode
# Get kernel (already flipped by layer)
kernel_flipped = layer.kernels[0][0]
print(f"\nKernel (flipped for scipy):\n{kernel_flipped}")

conv_result = convolve2d(padded, kernel_flipped, mode='valid')
print(f"\nManual conv result:\n{conv_result}")

diff = np.abs(output[:,:,0] - conv_result)
print(f"\nDifference:\n{diff}")
print(f"Max diff: {np.max(diff)}")
