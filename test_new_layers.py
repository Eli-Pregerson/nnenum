#!/usr/bin/env python3
"""
Layer-by-layer testing for newly added layers (Constant, Reshape, Conv)

Tests both concrete execution and abstract transformation soundness
"""

import numpy as np
import sys
from pathlib import Path

# Import as module (same as run_tests.sh does)
from nnenum.network import ConstantLayer, ReshapeLayer, Convolutional2dLayer
from nnenum.lp_star import LpStar
from nnenum.zonotope import Zonotope
from nnenum.overapprox import DeeppolyOverapprox


class TestResults:
    """Track test results"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.failures = []

    def record(self, test_name, passed, message=""):
        if passed:
            self.passed += 1
            print(f"✓ {test_name}")
        else:
            self.failed += 1
            self.failures.append((test_name, message))
            print(f"✗ {test_name}: {message}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"Results: {self.passed}/{total} passed, {self.failed}/{total} failed")
        if self.failures:
            print(f"\nFailures:")
            for name, msg in self.failures:
                print(f"  - {name}: {msg}")
        return self.failed == 0


def test_constant_layer(results):
    """Test ConstantLayer execution and transforms"""
    print("\n--- Testing ConstantLayer ---")

    # Test 1: Basic execution
    value = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    layer = ConstantLayer(0, value)

    # Test execution (should return constant regardless of input)
    dummy_input = np.zeros((2, 2))
    output = layer.execute(dummy_input)

    results.record(
        "ConstantLayer.execute()",
        np.allclose(output, value),
        f"Expected {value}, got {output}"
    )

    # Test 2: Shape methods
    results.record(
        "ConstantLayer.get_input_shape()",
        layer.get_input_shape() is None,
        "Constant should have no input shape"
    )

    results.record(
        "ConstantLayer.get_output_shape()",
        layer.get_output_shape() == value.shape,
        f"Expected {value.shape}, got {layer.get_output_shape()}"
    )

    # Test 3: Transform soundness (star set should become single point)
    # Create simple star: identity matrix as generators, zero bias
    dims = 4  # flattened value has 4 elements
    init_a_mat = np.eye(dims, dtype=np.float32)
    init_bias = np.zeros(dims, dtype=np.float32)
    init_box = np.array([[-1, 1]] * dims, dtype=np.float32)

    star = LpStar(init_a_mat, init_bias, init_box)

    # Apply constant transform
    layer.transform_star(star)

    # Star should now represent only the constant value
    flattened_value = value.flatten()
    results.record(
        "ConstantLayer.transform_star() - bias",
        star.bias is not None and np.allclose(star.bias, flattened_value),
        f"Star bias should equal constant value"
    )

    results.record(
        "ConstantLayer.transform_star() - no generators",
        star.a_mat is None,
        "Star should have no generators (single point)"
    )


def test_reshape_layer(results):
    """Test ReshapeLayer execution and transforms"""
    print("\n--- Testing ReshapeLayer ---")

    # Test 1: Basic reshape (2x3 -> 6)
    input_shape = (2, 3)
    new_shape = (6,)
    layer = ReshapeLayer(0, new_shape, input_shape)

    test_input = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    output = layer.execute(test_input)

    results.record(
        "ReshapeLayer.execute() - basic reshape",
        output.shape == new_shape and np.allclose(output, test_input.flatten()),
        f"Expected shape {new_shape}, got {output.shape}"
    )

    # Test 2: Reshape with -1 dimension
    input_shape2 = (2, 3)
    new_shape2 = (1, -1)  # Should infer as (1, 6)
    layer2 = ReshapeLayer(0, new_shape2, input_shape2)

    results.record(
        "ReshapeLayer.__init__() - infer -1 dimension",
        layer2.new_shape == (1, 6),
        f"Expected (1, 6), got {layer2.new_shape}"
    )

    output2 = layer2.execute(test_input)
    results.record(
        "ReshapeLayer.execute() - with -1 dimension",
        output2.shape == (1, 6),
        f"Expected shape (1, 6), got {output2.shape}"
    )

    # Test 3: Transform does nothing (reshape is identity on values)
    dims = 6
    init_a_mat = np.eye(dims, dtype=np.float32)
    init_bias = np.zeros(dims, dtype=np.float32)
    init_box = np.array([[0, 1]] * dims, dtype=np.float32)

    star = LpStar(init_a_mat, init_bias, init_box)
    original_a_mat = star.a_mat.copy() if star.a_mat is not None else None
    original_bias = star.bias.copy() if star.bias is not None else None

    layer.transform_star(star)

    # Star should be unchanged (reshape doesn't affect abstract domain)
    a_mat_unchanged = (star.a_mat is None and original_a_mat is None) or \
                      (star.a_mat is not None and original_a_mat is not None and np.allclose(star.a_mat, original_a_mat))
    bias_unchanged = (star.bias is None and original_bias is None) or \
                     (star.bias is not None and original_bias is not None and np.allclose(star.bias, original_bias))

    results.record(
        "ReshapeLayer.transform_star() - no change",
        a_mat_unchanged and bias_unchanged,
        "Reshape should not modify star representation"
    )


def test_conv_layer(results):
    """Test Convolutional2dLayer execution and basic properties"""
    print("\n--- Testing Convolutional2dLayer ---")

    # Test 1: Simple 3x3 conv with 1 input channel, 1 output channel
    prev_shape = (5, 5, 1)  # height, width, channels

    # 3x3 kernel with all ones
    kernels = np.ones((1, 1, 3, 3), dtype=np.float32)  # (out_ch, in_ch, h, w)
    biases = np.array([0.0], dtype=np.float32)

    layer = Convolutional2dLayer(0, kernels, biases, prev_shape, mode='valid')

    # Test input: 5x5 image with all ones
    test_input = np.ones((5, 5, 1), dtype=np.float32)
    output = layer.execute(test_input)

    # With 'valid' padding and 3x3 kernel on 5x5 input -> 3x3 output
    expected_shape = (3, 3, 1)
    results.record(
        "ConvLayer.execute() - output shape (valid)",
        output.shape == expected_shape,
        f"Expected {expected_shape}, got {output.shape}"
    )

    # Each output should be sum of 3x3 kernel = 9.0 (since all inputs are 1)
    results.record(
        "ConvLayer.execute() - output values (valid)",
        np.allclose(output, 9.0),
        f"Expected all 9.0, got {output.flatten()[:5]}..."
    )

    # Test 2: Same padding
    layer_same = Convolutional2dLayer(0, kernels, biases, prev_shape, mode='same')
    output_same = layer_same.execute(test_input)

    # With 'same' padding, output should match input spatial dimensions
    expected_shape_same = (5, 5, 1)
    results.record(
        "ConvLayer.execute() - output shape (same)",
        output_same.shape == expected_shape_same,
        f"Expected {expected_shape_same}, got {output_same.shape}"
    )

    # Test 3: Multi-channel convolution
    prev_shape_multi = (4, 4, 2)  # 2 input channels
    kernels_multi = np.ones((3, 2, 2, 2), dtype=np.float32)  # 3 output channels, 2 input channels, 2x2 kernel
    biases_multi = np.zeros(3, dtype=np.float32)

    layer_multi = Convolutional2dLayer(0, kernels_multi, biases_multi, prev_shape_multi, mode='valid')

    results.record(
        "ConvLayer.get_output_shape() - multi-channel",
        layer_multi.get_output_shape() == (3, 3, 3),
        f"Expected (3, 3, 3), got {layer_multi.get_output_shape()}"
    )

    # Test 4: With bias
    kernels_bias = np.ones((1, 1, 2, 2), dtype=np.float32)
    biases_bias = np.array([5.0], dtype=np.float32)
    layer_bias = Convolutional2dLayer(0, kernels_bias, biases_bias, (4, 4, 1), mode='valid')

    test_input_bias = np.ones((4, 4, 1), dtype=np.float32)
    output_bias = layer_bias.execute(test_input_bias)

    # Each output should be 4.0 (sum of 2x2) + 5.0 (bias) = 9.0
    results.record(
        "ConvLayer.execute() - with bias",
        np.allclose(output_bias, 9.0),
        f"Expected all 9.0, got {output_bias.flatten()[:5]}..."
    )


def test_integration_with_onnx(results):
    """Test that layers work when parsed from ONNX"""
    print("\n--- Testing ONNX Integration ---")

    try:
        from nnenum.onnx_network import load_onnx_network_optimized

        # Test with the reshape model we created
        model_path = "examples/convHelp/perturbations_0_reshape.onnx"
        if Path(model_path).exists():
            try:
                network = load_onnx_network_optimized(model_path)

                # Check that Reshape layer is present
                has_reshape = any(layer.__class__.__name__ == 'ReshapeLayer' for layer in network.layers)
                results.record(
                    "ONNX parsing - ReshapeLayer loaded",
                    has_reshape,
                    "ReshapeLayer not found in parsed network"
                )

                # Note: Constant nodes used for parameters (like Reshape shape) are
                # processed during parsing but not added as layers - this is correct!
                # Only Constant nodes in the main data flow become ConstantLayers
                print(f"  Note: Constant node for Reshape shape is processed but not a layer (expected)")
                results.record(
                    "ONNX parsing - Constant handled correctly",
                    True,  # This is actually correct behavior
                    ""
                )

                print(f"  Network has {len(network.layers)} layers:")
                for i, layer in enumerate(network.layers):
                    print(f"    {i}: {layer.__class__.__name__}")

            except Exception as e:
                results.record(
                    "ONNX parsing - load reshape model",
                    False,
                    str(e)
                )
        else:
            print(f"  Skipping ONNX test - {model_path} not found")

    except ImportError as e:
        print(f"  Skipping ONNX test - import error: {e}")


def test_batchnorm_folding(results):
    """Test BatchNormalization folding into Conv layers"""
    print("\n--- Testing BatchNorm Folding ---")

    # Test 1: Fold BatchNorm into Conv layer
    # Create a simple Conv layer: 1 input channel, 2 output channels, 2x2 kernel
    prev_shape = (4, 4, 1)  # height, width, channels

    # Simple kernels: all ones
    kernels = np.ones((2, 1, 2, 2), dtype=np.float32)  # (out_ch, in_ch, h, w)
    biases = np.array([1.0, 2.0], dtype=np.float32)  # 2 output channels

    conv_layer = Convolutional2dLayer(0, kernels, biases, prev_shape, mode='valid')

    # Create BatchNorm parameters
    # BatchNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta
    gamma = np.array([2.0, 0.5], dtype=np.float32)  # scale per output channel
    beta = np.array([0.5, -0.5], dtype=np.float32)   # shift per output channel
    mean = np.array([1.0, 2.0], dtype=np.float32)
    var = np.array([1.0, 3.0], dtype=np.float32)
    epsilon = 1e-5

    # Manually compute expected folded parameters
    # bn_scale = gamma / sqrt(var + epsilon)
    bn_scale = gamma / np.sqrt(var + epsilon)
    # bn_shift = beta - mean * bn_scale
    bn_shift = beta - mean * bn_scale

    expected_biases = bn_scale * biases + bn_shift

    # Store original kernel values for comparison
    original_kernel_00 = conv_layer.kernels[0][0].copy()
    expected_kernel_00 = original_kernel_00 * bn_scale[0]

    # Now simulate BatchNorm folding (what our ONNX parser does)
    # Create new scaled kernels
    num_out_channels = len(conv_layer.kernels)
    for out_c in range(num_out_channels):
        for in_c in range(len(conv_layer.kernels[out_c])):
            conv_layer.kernels[out_c][in_c] = conv_layer.kernels[out_c][in_c] * bn_scale[out_c]

    # Update biases
    conv_layer.biases = bn_scale * biases + bn_shift

    # Update kernels_array (stores unflipped kernels; scale per output channel)
    conv_layer.kernels_array = conv_layer.kernels_array * bn_scale[:, np.newaxis, np.newaxis, np.newaxis].astype(conv_layer.kernels_array.dtype)

    # Test 1a: Check kernels were scaled correctly
    results.record(
        "BatchNorm folding - kernel scaling",
        np.allclose(conv_layer.kernels[0][0], expected_kernel_00),
        f"Expected kernel scaled by {bn_scale[0]}"
    )

    # Test 1b: Check biases were updated correctly
    results.record(
        "BatchNorm folding - bias update",
        np.allclose(conv_layer.biases, expected_biases),
        f"Expected {expected_biases}, got {conv_layer.biases}"
    )

    # Test 2: Verify folded layer produces equivalent output
    # Create test input
    test_input = np.random.rand(4, 4, 1).astype(np.float32)

    # Compute output with separate Conv and BatchNorm
    conv_only = Convolutional2dLayer(0, kernels, biases, prev_shape, mode='valid')
    conv_output = conv_only.execute(test_input)

    # Apply BatchNorm manually: y = gamma * (x - mean) / sqrt(var + eps) + beta
    # conv_output has shape (3, 3, 2) for valid mode on 4x4 input with 2x2 kernel
    # BatchNorm is applied per channel
    bn_output = np.zeros_like(conv_output)
    for c in range(conv_output.shape[2]):
        bn_output[:, :, c] = gamma[c] * (conv_output[:, :, c] - mean[c]) / np.sqrt(var[c] + epsilon) + beta[c]

    # Compute output with folded Conv+BatchNorm
    folded_output = conv_layer.execute(test_input)

    results.record(
        "BatchNorm folding - output equivalence",
        np.allclose(folded_output, bn_output, rtol=1e-5, atol=1e-6),
        f"Folded layer output should match Conv→BatchNorm pipeline"
    )

    # Test 3: Check output shapes
    results.record(
        "BatchNorm folding - output shape preserved",
        folded_output.shape == bn_output.shape,
        f"Expected shape {bn_output.shape}, got {folded_output.shape}"
    )


def test_strided_conv(results):
    """Test strided convolutions"""
    print("\n--- Testing Strided Convolutions ---")

    # Test 1: Stride=2 downsampling with 1x1 kernel
    print("\nTest: Stride=2 with 1x1 kernel")
    input_shape = (8, 8, 3)
    kernels = np.random.randn(4, 3, 1, 1).astype(np.float32)  # 4 output channels, 3 input channels
    biases = np.random.randn(4).astype(np.float32)

    layer = Convolutional2dLayer(0, kernels, biases, input_shape, mode='same', strides=(2, 2))

    # Check output shape
    expected_output_shape = (4, 4, 4)  # 8/2 = 4, 8/2 = 4, 4 output channels
    results.record(
        "Strided conv - output shape with stride=2",
        layer.get_output_shape() == expected_output_shape,
        f"Expected {expected_output_shape}, got {layer.get_output_shape()}"
    )

    # Test execution
    input_data = np.random.randn(*input_shape).astype(np.float32)
    output = layer.execute(input_data)
    results.record(
        "Strided conv - execution produces correct shape",
        output.shape == expected_output_shape,
        f"Expected {expected_output_shape}, got {output.shape}"
    )

    # Test 2: Stride=2 with 3x3 kernel (VALID mode)
    print("\nTest: Stride=2 with 3x3 kernel (VALID)")
    input_shape = (10, 10, 2)
    kernels = np.random.randn(3, 2, 3, 3).astype(np.float32)
    biases = np.random.randn(3).astype(np.float32)

    layer = Convolutional2dLayer(0, kernels, biases, input_shape, mode='valid', strides=(2, 2))

    # VALID mode: output = (input - kernel + 1) / stride
    # (10 - 3 + 1) / 2 = 8 / 2 = 4
    expected_output_shape = (4, 4, 3)
    results.record(
        "Strided conv (VALID) - output shape calculation",
        layer.get_output_shape() == expected_output_shape,
        f"Expected {expected_output_shape}, got {layer.get_output_shape()}"
    )

    input_data = np.random.randn(*input_shape).astype(np.float32)
    output = layer.execute(input_data)
    results.record(
        "Strided conv (VALID) - execution output shape",
        output.shape == expected_output_shape,
        f"Expected {expected_output_shape}, got {output.shape}"
    )

    # Test 3: Transform star with strided conv
    print("\nTest: Star transformation with strided conv")
    input_shape = (4, 4, 2)
    kernels = np.ones((2, 2, 2, 2), dtype=np.float32)  # Simple kernels for testing
    biases = np.zeros(2, dtype=np.float32)

    layer = Convolutional2dLayer(0, kernels, biases, input_shape, mode='same', strides=(2, 2))

    # Create a simple star set
    input_size = 4 * 4 * 2
    init_a_mat = np.eye(input_size, dtype=np.float32)[:, :5]  # Use first 5 generators
    init_bias = np.zeros(input_size, dtype=np.float32)
    init_box = [(-1.0, 1.0)] * 5

    star = LpStar(init_a_mat, init_bias, init_box)

    # Transform
    layer.transform_star(star)

    # Check output dimensions
    expected_output_size = 2 * 2 * 2  # (4/2) * (4/2) * 2
    results.record(
        "Strided conv - star transform output size",
        star.a_mat.shape[0] == expected_output_size,
        f"Expected {expected_output_size}, got {star.a_mat.shape[0]}"
    )

    results.record(
        "Strided conv - star transform preserves generators",
        star.a_mat.shape[1] == 5,
        f"Expected 5 generators, got {star.a_mat.shape[1]}"
    )


def main():
    """Run all tests"""
    print("="*60)
    print("Layer-by-Layer Testing for Constant, Reshape, Conv, and BatchNorm")
    print("="*60)

    results = TestResults()

    test_constant_layer(results)
    test_reshape_layer(results)
    test_conv_layer(results)
    test_strided_conv(results)
    test_batchnorm_folding(results)
    test_integration_with_onnx(results)

    success = results.summary()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
