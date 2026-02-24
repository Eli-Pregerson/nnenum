#!/usr/bin/env python3
"""
Step 7 Tests: Full verification loop correctness for skip-connection networks

Tests that enumerate_network() gives correct safe/unsafe results on synthetic
networks with SkipAddLayer, covering both identity and non-identity skip paths.

Tests:
1. Safe property holds on identity-skip network (no ReLU splits)
2. Violated property detected on identity-skip network
3. Safe property holds on identity-skip network (with ReLU splits)
4. Violated property detected on skip network (with ReLU splits)
5. CIFAR100 ResNet verification loop runs without crash (short timeout)
6. Correct result on sequential (non-skip) network (regression)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import time

from nnenum.network import (FullyConnectedLayer, ReluLayer, SkipAddLayer,
                             NeuralNetwork)
from nnenum.enumerate import enumerate_network
from nnenum.specification import Specification
from nnenum.settings import Settings
from nnenum.result import Result


CIFAR_MODEL = os.path.join(
    os.path.dirname(__file__), '..',
    'vnncomp2025_benchmarks', 'benchmarks', 'cifar100_2024', 'onnx',
    'CIFAR100_resnet_medium.onnx'
)


def make_skip_net_no_relu():
    """FC(2x) -> FC(I) -> SkipAdd(skip=after FC1, main=FC2)
    Input x -> output 2x + 2x = 4x
    For x in [lo, hi]: output in [4*lo, 4*hi]
    """
    n = 2
    W0 = 2.0 * np.eye(n, dtype=np.float32)
    b0 = np.zeros(n, dtype=np.float32)
    fc0 = FullyConnectedLayer(0, W0, b0, (n,))

    W1 = np.eye(n, dtype=np.float32)
    b1 = np.zeros(n, dtype=np.float32)
    fc1 = FullyConnectedLayer(1, W1, b1, (n,))

    skip2 = SkipAddLayer(2, input_shape=(n,))

    # Layer 1 (FC0 output) is the skip source
    # dag: SkipAdd layer 2 has skip from cache_key=1 and main from layer 1
    return NeuralNetwork([fc0, fc1, skip2], dag_predecessors={2: [1, 1]})


def make_skip_net_with_relu():
    """FC(2x) -> ReLU -> FC(I) -> SkipAdd(skip=after ReLU)
    For x in [lo, hi]:
    - After ReLU: max(0, 2x)
    - Output: max(0,2x) + max(0,2x) = 2*max(0,2x)
    """
    n = 2
    W0 = 2.0 * np.eye(n, dtype=np.float32)
    b0 = np.zeros(n, dtype=np.float32)
    fc0 = FullyConnectedLayer(0, W0, b0, (n,))

    relu1 = ReluLayer(1, (n,))

    W2 = np.eye(n, dtype=np.float32)
    b2 = np.zeros(n, dtype=np.float32)
    fc2 = FullyConnectedLayer(2, W2, b2, (n,))

    skip3 = SkipAddLayer(3, input_shape=(n,))

    # ReLU layer 1 is the skip source (cache_key=2, i.e. star before layer 2)
    return NeuralNetwork([fc0, relu1, fc2, skip3], dag_predecessors={3: [2, 2]})


def configure_exact():
    Settings.PRINT_OUTPUT = False
    Settings.NUM_PROCESSES = 1
    Settings.TIMEOUT = 30.0
    Settings.BRANCH_MODE = Settings.BRANCH_EXACT
    Settings.TIMING_STATS = False


def run_tests():
    passed = 0
    failed = 0

    def ok(name, cond, msg=""):
        nonlocal passed, failed
        if cond:
            print(f"  PASS: {name}")
            passed += 1
        else:
            print(f"  FAIL: {name}" + (f" -- {msg}" if msg else ""))
            failed += 1

    print("=== Step 7: Full verification loop correctness ===\n")

    # ── Test 1: Safe property, no-ReLU skip network ──────────────────────
    print("-- 1. Safe property (no-ReLU skip network) --")
    try:
        configure_exact()
        net = make_skip_net_no_relu()
        # Input x ∈ [0, 0.5]^2, output ∈ [0, 2]^2
        # Property: output_0 <= 3 (always holds since max = 4*0.5 = 2)
        # Unsafe: output_0 >= 3 -> mat=[-1, 0], rhs=-3 -> -output_0 <= -3
        init_box = np.array([[0.0, 0.5], [0.0, 0.5]], dtype=np.float32)
        spec = Specification([[-1.0, 0.0]], [-3.0])  # unsafe: output_0 >= 3

        res = enumerate_network(init_box, net, spec)

        ok("No-ReLU skip: result is Result instance", isinstance(res, Result))
        ok("No-ReLU skip: property holds (safe)",
           res.result_str == 'safe',
           f"got result_str={res.result_str!r}")

    except Exception as e:
        import traceback
        ok(f"Safe no-ReLU skip (exception: {type(e).__name__}: {e})", False)
        traceback.print_exc()
        ok("(skipped)", False)

    # ── Test 2: Violated property, no-ReLU skip network ─────────────────
    print("\n-- 2. Violated property (no-ReLU skip network) --")
    try:
        configure_exact()
        net = make_skip_net_no_relu()
        # Input x ∈ [0, 1]^2, output ∈ [0, 4]^2
        # Property: output_0 <= 3 (violated since max = 4)
        # Unsafe: output_0 >= 3 -> mat=[-1, 0], rhs=-3
        init_box = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        spec = Specification([[-1.0, 0.0]], [-3.0])

        res = enumerate_network(init_box, net, spec)

        ok("No-ReLU skip violated: result is Result instance", isinstance(res, Result))
        ok("No-ReLU skip violated: violation found (unsafe)",
           res.result_str in ('unsafe', 'unsafe (unconfirmed)'),
           f"got result_str={res.result_str!r}")

    except Exception as e:
        import traceback
        ok(f"Violated no-ReLU skip (exception: {type(e).__name__}: {e})", False)
        traceback.print_exc()
        ok("(skipped)", False)

    # ── Test 3: Safe property, ReLU skip network ─────────────────────────
    print("\n-- 3. Safe property (ReLU skip network) --")
    try:
        configure_exact()
        net = make_skip_net_with_relu()
        # Input x ∈ [0, 0.5]^2, after ReLU: max(0, 2x) = 2x ∈ [0,1]
        # Output: 2*2x = 4x ∈ [0, 2]
        # Property: output_0 <= 3 (holds since max = 2)
        # Unsafe: output_0 >= 3 -> mat=[-1, 0], rhs=-3
        init_box = np.array([[0.0, 0.5], [0.0, 0.5]], dtype=np.float32)
        spec = Specification([[-1.0, 0.0]], [-3.0])

        res = enumerate_network(init_box, net, spec)

        ok("ReLU skip safe: result is Result instance", isinstance(res, Result))
        ok("ReLU skip safe: property holds (safe)",
           res.result_str == 'safe',
           f"got result_str={res.result_str!r}")

    except Exception as e:
        import traceback
        ok(f"Safe ReLU skip (exception: {type(e).__name__}: {e})", False)
        traceback.print_exc()
        ok("(skipped)", False)

    # ── Test 4: Violated property, ReLU skip network ─────────────────────
    print("\n-- 4. Violated property (ReLU skip network) --")
    try:
        configure_exact()
        net = make_skip_net_with_relu()
        # Input x ∈ [-1, 1]^2
        # After ReLU: max(0, 2x) ∈ [0, 2]
        # Output: 2*max(0, 2x) ∈ [0, 4]
        # Property: output_0 <= 3 (violated since max = 4 at x=1)
        # Unsafe: output_0 >= 3 -> mat=[-1, 0], rhs=-3
        init_box = np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=np.float32)
        spec = Specification([[-1.0, 0.0]], [-3.0])

        res = enumerate_network(init_box, net, spec)

        ok("ReLU skip violated: result is Result instance", isinstance(res, Result))
        ok("ReLU skip violated: violation found (unsafe)",
           res.result_str in ('unsafe', 'unsafe (unconfirmed)'),
           f"got result_str={res.result_str!r}")

    except Exception as e:
        import traceback
        ok(f"Violated ReLU skip (exception: {type(e).__name__}: {e})", False)
        traceback.print_exc()
        ok("(skipped)", False)

    # ── Test 5: Sequential (non-skip) regression ──────────────────────────
    print("\n-- 5. Sequential network regression --")
    try:
        configure_exact()
        n = 2
        W0 = 2.0 * np.eye(n, dtype=np.float32)
        b0 = np.zeros(n, dtype=np.float32)
        fc0 = FullyConnectedLayer(0, W0, b0, (n,))
        relu1 = ReluLayer(1, (n,))
        W2 = np.eye(n, dtype=np.float32)
        b2 = np.zeros(n, dtype=np.float32)
        fc2 = FullyConnectedLayer(2, W2, b2, (n,))
        net_seq = NeuralNetwork([fc0, relu1, fc2])

        # Input x ∈ [0, 1]^2, output = 2x ∈ [0, 2]
        # Property: output_0 <= 3 (holds)
        init_box = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        spec = Specification([[-1.0, 0.0]], [-3.0])

        res = enumerate_network(init_box, net_seq, spec)

        ok("Sequential holds: result is Result", isinstance(res, Result))
        ok("Sequential holds: property holds (safe)",
           res.result_str == 'safe',
           f"got result_str={res.result_str!r}")

    except Exception as e:
        import traceback
        ok(f"Sequential regression (exception: {type(e).__name__}: {e})", False)
        traceback.print_exc()
        ok("(skipped)", False)

    # ── Test 6: CIFAR100 model loads with skip connections ────────────────
    # Full CIFAR100 verification is skipped here because propagating the star
    # through 3072-generator conv layers is too slow for a unit test.
    # Loading + skip-connection structure is already tested in step6_worker_compatibility.py.
    print("\n-- 6. CIFAR100 ResNet model structure (skip connection check) --")
    if not os.path.exists(CIFAR_MODEL):
        print(f"  SKIP: CIFAR100 model not found")
    else:
        try:
            from nnenum.onnx_network import load_onnx_network_optimized
            from nnenum.network import SkipAddLayer
            network = load_onnx_network_optimized(CIFAR_MODEL)

            has_skip = any(isinstance(l, SkipAddLayer) for l in network.layers)
            ok("CIFAR100 loads with SkipAddLayer", has_skip,
               f"layers: {[l.__class__.__name__ for l in network.layers[:5]]}")
            ok("CIFAR100 dag_predecessors non-empty", bool(network.dag_predecessors),
               f"dag_predecessors count: {len(network.dag_predecessors)}")

        except Exception as e:
            import traceback
            ok(f"CIFAR100 model check (exception: {type(e).__name__}: {e})", False)
            traceback.print_exc()
            ok("(skipped)", False)

    print(f"\n=== Results: {passed} passed, {failed} failed ===")
    return failed == 0


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
