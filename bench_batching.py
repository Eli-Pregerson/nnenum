"""
Benchmark the three batching strategies against each other.

Measures:
  - batch_time: time spent inside _batch_generators_for_conv
  - conv_time:  time spent processing all batches (the actual convolutions)
  - total_time: batch_time + conv_time
  - num_batches, mean_batch_size, compression_ratio

Tested on first conv layer of CIFAR-10 (32x32x3) and a larger 64x64x3 synthetic case,
with varying generator counts to stress the O(n) vs O(n^2) difference.
"""

import sys
import time
import numpy as np

sys.path.insert(0, 'src')

from nnenum.network import Convolutional2dLayer, nn_flatten, nn_unflatten
from nnenum.lp_star import LpStar
from nnenum.settings import Settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_cifar_conv():
    """3x3 conv, 3->32 channels, stride 1, same padding — typical first layer."""
    np.random.seed(42)
    kernels = np.random.randn(32, 3, 3, 3).astype(np.float32) * 0.1
    biases  = np.zeros(32, dtype=np.float32)
    layer = Convolutional2dLayer(0, kernels, biases, (32, 32, 3))
    return layer

def make_identity_star(H, W, C, G):
    """
    Star with G one-hot generators drawn uniformly from the H*W*C input pixels.
    Mimics the initial star for an image input with G perturbed pixels.
    """
    n = H * W * C
    rng = np.random.default_rng(0)
    indices = rng.choice(n, size=min(G, n), replace=False)
    a_mat = np.zeros((n, len(indices)), dtype=np.float32)
    for col, row in enumerate(indices):
        a_mat[row, col] = 1.0
    bias = np.zeros(n, dtype=np.float32)
    return LpStar(a_mat, bias)


def run_batching(layer, star, strategy, max_failures=10, n_repeats=3):
    """Run the batching path and return timing + stats."""
    Settings.CONV_METHOD = 'batching'
    Settings.CONV_BATCHING_ENABLED = True
    Settings.CONV_BATCHING_FIRST_LAYER_ONLY = False
    Settings.CONV_BATCHING_MIN_SPARSITY = 1.0   # never skip due to sparsity
    Settings.CONV_BATCHING_STRATEGY = strategy
    Settings.CONV_BATCHING_MAX_FAILURES = max_failures
    Settings.LOG_CONV_BATCHING = False

    shape = layer.get_input_shape()

    times = []
    num_batches_list = []
    batch_size_list = []

    for _ in range(n_repeats):
        # Fresh copy each repeat so we measure consistently
        import copy
        star_copy = copy.copy(star)
        star_copy.a_mat = star.a_mat.copy()
        star_copy.bias = star.bias.copy()

        t0 = time.perf_counter()
        batches, gen_info = layer._batch_generators_for_conv(star_copy.a_mat, shape)
        t_batch = time.perf_counter() - t0

        # Time the actual convolutions too
        t1 = time.perf_counter()
        G = star_copy.a_mat.shape[1]
        result_columns = [None] * G
        output_shape = layer.get_output_shape()

        for batch in batches:
            if len(batch['indices']) == 1:
                idx = batch['indices'][0]
                col = gen_info[idx]['column']
                mc = nn_unflatten(col, shape)
                mc = layer.execute(mc, zero_bias=True)
                flat = nn_flatten(mc)
                flat = flat.reshape(-1, 1)
                result_columns[idx] = flat
            else:
                combined = np.zeros(shape, dtype=np.float32)
                for idx in batch['indices']:
                    combined += nn_unflatten(gen_info[idx]['column'], shape)
                cr = layer.execute(combined, zero_bias=True)
                for i, idx in enumerate(batch['indices']):
                    or_ = batch['output_regions'][i]
                    if or_ is None:
                        result_columns[idx] = np.zeros((np.prod(output_shape), 1), dtype=np.float32)
                    else:
                        y0, y1, x0, x1 = or_
                        masked = np.zeros(output_shape, dtype=np.float32)
                        masked[y0:y1+1, x0:x1+1, :] = cr[y0:y1+1, x0:x1+1, :]
                        result_columns[idx] = nn_flatten(masked).reshape(-1, 1)

        t_conv = time.perf_counter() - t1

        times.append((t_batch, t_conv))
        num_batches_list.append(len(batches))
        batch_size_list.append(np.mean([len(b['indices']) for b in batches]))

    avg_batch = np.mean([t for t, _ in times])
    avg_conv  = np.mean([t for _, t in times])
    return {
        'strategy':      strategy,
        'batch_time':    avg_batch,
        'conv_time':     avg_conv,
        'total_time':    avg_batch + avg_conv,
        'num_batches':   np.mean(num_batches_list),
        'mean_batch_sz': np.mean(batch_size_list),
        'compression':   star.a_mat.shape[1] / np.mean(num_batches_list),
    }


def run_dense_baseline(layer, star, n_repeats=3):
    """Dense im2col+BLAS baseline — what batching is competing against."""
    Settings.CONV_METHOD = 'dense'
    shape = layer.get_input_shape()
    times = []
    for _ in range(n_repeats):
        import copy
        star_copy = copy.copy(star)
        star_copy.a_mat = star.a_mat.copy()
        star_copy.bias  = star.bias.copy()
        t0 = time.perf_counter()
        layer.transform_star(star_copy)
        times.append(time.perf_counter() - t0)
    return np.mean(times)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def experiment(label, layer, H, W, C, gen_counts, n_repeats=3):
    print(f"\n{'='*70}")
    print(f"  {label}  (input {H}x{W}x{C})")
    print(f"{'='*70}")
    header = f"{'G':>6}  {'strategy':>10}  {'k':>4}  {'t_batch':>8}  {'t_conv':>8}  {'total':>8}  {'batches':>7}  {'sz':>6}  {'ratio':>6}"
    print(header)
    print('-' * len(header))

    for G in gen_counts:
        star = make_identity_star(H, W, C, G)

        dense_time = run_dense_baseline(layer, star, n_repeats)

        for strategy, extra in [('greedy', {}),
                                 ('random', {'max_failures': 5}),
                                 ('random', {'max_failures': 20}),
                                 ('period', {})]:
            r = run_batching(layer, star, strategy, n_repeats=n_repeats, **extra)
            k_str = str(extra.get('max_failures', '-'))
            speedup = dense_time / r['total_time']
            print(f"{G:>6}  {strategy:>10}  {k_str:>4}  "
                  f"{r['batch_time']:>8.4f}  {r['conv_time']:>8.4f}  "
                  f"{r['total_time']:>8.4f}  {r['num_batches']:>7.0f}  "
                  f"{r['mean_batch_sz']:>6.1f}  {r['compression']:>6.1f}x"
                  f"  [vs dense {speedup:.2f}x]")

        print(f"{'':>6}  {'dense':>10}  {'':>4}  {'':>8}  {'':>8}  {dense_time:>8.4f}")
        print()


if __name__ == '__main__':
    np.random.seed(0)

    layer_cifar = make_cifar_conv()

    # CIFAR-scale: small G (normal batching regime)
    experiment(
        "CIFAR 32x32x3, 3x3->32ch conv",
        layer_cifar, 32, 32, 3,
        gen_counts=[100, 500, 1000, 3072],
        n_repeats=3,
    )

    # Larger image: stress the O(n) vs O(n^2) difference
    np.random.seed(0)
    kernels_large = np.random.randn(64, 3, 3, 3).astype(np.float32) * 0.1
    biases_large  = np.zeros(64, dtype=np.float32)
    layer_large = Convolutional2dLayer(0, kernels_large, biases_large, (64, 64, 3))

    experiment(
        "Larger 64x64x3, 3x3->64ch conv",
        layer_large, 64, 64, 3,
        gen_counts=[500, 2000, 5000, 12288],
        n_repeats=3,
    )
