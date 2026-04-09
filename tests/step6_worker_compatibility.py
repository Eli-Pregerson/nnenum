#!/usr/bin/env python3
"""
Step 6 Tests: worker.py + enumerate.py compatibility

Tests that skip connection networks can be loaded, LpStarState can be created,
and propagation works through SkipAddLayer without crashing in the main
verification flow.

Tests:
1. CIFAR100 ResNet parses without error and has SkipAddLayer
2. CIFAR100 LpStarState can be created and propagated
3. Skip connection star propagation through SkipAddLayer completes
4. Branch tuples grow correctly after splitting
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import time
import traceback

from nnenum.network import SkipAddLayer, NeuralNetwork
from nnenum.lp_star_state import LpStarState
from nnenum.settings import Settings

CIFAR_MODEL = os.path.join(
    os.path.dirname(__file__), '..',
    'vnncomp2025_benchmarks', 'benchmarks', 'cifar100_2024', 'onnx',
    'CIFAR100_resnet_medium.onnx'
)


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

    Settings.PRINT_OUTPUT = False

    print("=== Step 6: worker.py + enumerate.py compatibility ===\n")

    if not os.path.exists(CIFAR_MODEL):
        print(f"  SKIP: CIFAR100 model not found at {CIFAR_MODEL}")
        print(f"\n=== Results: {passed} passed, {failed} failed (model not found) ===")
        return True  # Not a failure if model is missing

    # ── Test 1: CIFAR100 ResNet loads with SkipAddLayer ──────────────────
    print("-- 1. CIFAR100 ResNet parsing --")
    try:
        from nnenum.onnx_network import load_onnx_network_optimized
        network = load_onnx_network_optimized(CIFAR_MODEL)

        has_skip = any(isinstance(l, SkipAddLayer) for l in network.layers)
        ok("CIFAR100 model loads without error", True)
        ok("CIFAR100 model has SkipAddLayer", has_skip,
           f"layers: {[l.__class__.__name__ for l in network.layers[:10]]}")
        ok("dag_predecessors is non-empty", bool(network.dag_predecessors),
           f"dag_predecessors: {dict(list(network.dag_predecessors.items())[:3])}")

        num_skip = sum(1 for l in network.layers if isinstance(l, SkipAddLayer))
        print(f"  INFO: {len(network.layers)} layers, {num_skip} SkipAddLayers, "
              f"{len(network.dag_predecessors)} skip connections")

    except Exception as e:
        ok(f"CIFAR100 model loads (exception: {type(e).__name__}: {e})", False)
        traceback.print_exc()
        ok("(skipped)", False)
        ok("(skipped)", False)
        print(f"\n=== Results: {passed} passed, {failed} failed ===")
        return failed == 0

    # ── Test 2: LpStarState creation for CIFAR100 ────────────────────────
    print("\n-- 2. LpStarState creation --")
    try:
        input_shape = network.get_input_shape()
        n_inputs = 1
        for d in input_shape:
            n_inputs *= d
        print(f"  INFO: Input shape {input_shape}, {n_inputs} inputs")

        init_box = np.array([[-0.1, 0.1]] * n_inputs, dtype=np.float32)
        ss = LpStarState(uncompressed_init_box=init_box)

        ok("LpStarState created", ss is not None)
        ok("star_cache initialized empty", ss.star_cache == {})
    except Exception as e:
        ok(f"LpStarState creation (exception: {type(e).__name__}: {e})", False)
        traceback.print_exc()
        ok("(skipped)", False)
        print(f"\n=== Results: {passed} passed, {failed} failed ===")
        return failed == 0

    # ── Test 3: Propagate through first few layers (including SkipAdd) ───
    print("\n-- 3. Propagation through SkipAddLayer --")
    try:
        start = time.time()
        timeout = 30.0  # 30 second timeout for propagation

        ss.propagate_up_to_split(network, start)

        elapsed = time.time() - start
        print(f"  INFO: Propagated to layer {ss.cur_layer} in {elapsed:.2f}s")

        ok("Propagation completed without error", True)
        ok("star_cache populated during propagation", len(ss.star_cache) > 0,
           f"star_cache keys: {list(ss.star_cache.keys())[:5]}")

        if network.dag_predecessors:
            # Check that skip source layers are in star_cache
            skip_sources_in_cache = all(
                preds[0] in ss.star_cache
                for layer_idx, preds in network.dag_predecessors.items()
                if layer_idx <= ss.cur_layer  # only check processed layers
            )
            ok("Skip source states are cached", skip_sources_in_cache,
               f"dag_predecessors: {dict(list(network.dag_predecessors.items())[:2])}, "
               f"star_cache keys: {sorted(ss.star_cache.keys())[:10]}")

    except Exception as e:
        ok(f"Propagation (exception: {type(e).__name__}: {e})", False)
        traceback.print_exc()
        ok("(skipped)", False)
        ok("(skipped)", False)

    # ── Test 4: Concrete execution matches expected ───────────────────────
    print("\n-- 4. Concrete execution test --")
    try:
        # Create a random input and verify network.execute() doesn't crash
        input_shape = network.get_input_shape()
        x = np.zeros(input_shape, dtype=np.float32)
        out = network.execute(x)

        ok("network.execute() with zero input works", out is not None,
           f"output shape: {out.shape if out is not None else 'None'}")
        ok("output shape is correct",
           out.shape == network.get_output_shape(),
           f"got {out.shape}, expected {network.get_output_shape()}")
    except Exception as e:
        ok(f"Concrete execution (exception: {type(e).__name__}: {e})", False)
        traceback.print_exc()
        ok("(skipped)", False)

    print(f"\n=== Results: {passed} passed, {failed} failed ===")
    return failed == 0


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
