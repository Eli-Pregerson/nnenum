#!/usr/bin/env python3
"""
Stress test: larger image, more generators, bigger kernels
to see clearer separation between batched and unbatched methods
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from nnenum.network import Convolutional2dLayer, nn_unflatten, nn_flatten
import time

def benchmark_conv(layer, a_mat, input_shape, use_batching, test_name):
    """Benchmark convolution with or without batching"""

    # Mock the Settings to control batching
    from nnenum.settings import Settings
    original_first_layer_only = Settings.CONV_BATCHING_FIRST_LAYER_ONLY

    if not use_batching:
        # Force skip batching by setting first_layer_only=True and layer_num>0
        Settings.CONV_BATCHING_FIRST_LAYER_ONLY = True
        layer.layer_num = 1  # Temporarily change to skip batching
    else:
        Settings.CONV_BATCHING_FIRST_LAYER_ONLY = False
        layer.layer_num = 0

    class MockStar:
        def __init__(self, a_mat, bias):
            self.a_mat = a_mat
            self.bias = bias

    bias = np.zeros(np.prod(input_shape), dtype=np.float32)
    star = MockStar(a_mat.copy(), bias.copy())

    start = time.time()
    layer.transform_star(star)
    elapsed = time.time() - start

    # Restore
    Settings.CONV_BATCHING_FIRST_LAYER_ONLY = original_first_layer_only
    layer.layer_num = 0

    return elapsed, star.a_mat

# Test configurations with increasing difficulty
configs = [
    {
        'name': 'Small (MNIST-like)',
        'input_shape': (28, 28, 1),
        'num_filters': 32,
        'kernel_size': 5,
        'num_generators': 100,
    },
    {
        'name': 'Medium (CIFAR-10)',
        'input_shape': (32, 32, 3),
        'num_filters': 64,
        'kernel_size': 3,
        'num_generators': 500,
    },
    {
        'name': 'Large (CIFAR-100 style)',
        'input_shape': (32, 32, 3),
        'num_filters': 128,
        'kernel_size': 3,
        'num_generators': 1000,
    },
    {
        'name': 'XLarge (ImageNet-like)',
        'input_shape': (64, 64, 3),
        'num_filters': 64,
        'kernel_size': 7,
        'num_generators': 500,
    },
]

print("="*80)
print("CONVOLUTION BATCHING STRESS TEST")
print("="*80)

results = []

for config in configs:
    print(f"\n{'='*80}")
    print(f"{config['name']}: {config['input_shape']}, {config['num_filters']} filters, " +
          f"{config['kernel_size']}x{config['kernel_size']} kernel, {config['num_generators']} generators")
    print('='*80)

    # Create layer
    input_shape = config['input_shape']
    num_filters = config['num_filters']
    kernel_size = config['kernel_size']
    num_channels = input_shape[2]

    np.random.seed(42)
    kernels = np.random.randn(num_filters, num_channels, kernel_size, kernel_size).astype(np.float32) * 0.1
    biases = np.random.randn(num_filters).astype(np.float32) * 0.1

    layer = Convolutional2dLayer(0, kernels, biases, input_shape, mode='same', strides=(1, 1))

    # Create sparse generators (one-hot)
    num_gens = config['num_generators']
    input_size = np.prod(input_shape)
    a_mat = np.eye(input_size, num_gens, dtype=np.float32)

    sparsity = np.count_nonzero(a_mat[:, 0]) / a_mat.shape[0]
    print(f"First generator sparsity: {sparsity*100:.4f}%")

    # Benchmark with batching
    print("\nBenchmarking WITH batching...")
    time_batched, result_batched = benchmark_conv(layer, a_mat, input_shape, use_batching=True,
                                                    test_name=config['name'])

    # Benchmark without batching
    print("Benchmarking WITHOUT batching...")
    time_unbatched, result_unbatched = benchmark_conv(layer, a_mat, input_shape, use_batching=False,
                                                       test_name=config['name'])

    # Verify correctness
    max_diff = np.max(np.abs(result_batched - result_unbatched))
    correct = np.allclose(result_batched, result_unbatched, rtol=1e-5, atol=1e-6)

    speedup = time_unbatched / time_batched

    print(f"\nResults:")
    print(f"  With batching:    {time_batched:.4f}s")
    print(f"  Without batching: {time_unbatched:.4f}s")
    print(f"  Speedup:          {speedup:.2f}x")
    print(f"  Correctness:      {'✓ PASS' if correct else '✗ FAIL'} (max diff: {max_diff:.2e})")

    results.append({
        'name': config['name'],
        'time_batched': time_batched,
        'time_unbatched': time_unbatched,
        'speedup': speedup,
        'correct': correct,
    })

# Summary table
print(f"\n{'='*80}")
print("SUMMARY")
print('='*80)
print(f"{'Config':<25} {'Batched (s)':<15} {'Unbatched (s)':<15} {'Speedup':<10} {'Status'}")
print('-'*80)

for r in results:
    status = '✓' if r['correct'] else '✗'
    print(f"{r['name']:<25} {r['time_batched']:<15.4f} {r['time_unbatched']:<15.4f} {r['speedup']:<10.2f} {status}")

print()
print(f"Average speedup: {np.mean([r['speedup'] for r in results]):.2f}x")
print(f"Best speedup:    {max([r['speedup'] for r in results]):.2f}x ({max(results, key=lambda x: x['speedup'])['name']})")
