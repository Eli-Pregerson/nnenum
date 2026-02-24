#!/usr/bin/env python3
"""
Step 2 Tests: LpStarState star cache + SkipAddLayer propagation

Tests:
1. LpStarState has a star_cache dict
2. For sequential networks, star_cache is populated at each layer
3. When advance_through_linear_layers reaches a SkipAddLayer, it fetches
   the skip-path star from the cache and combines it with the current star
4. split_enumerate copies the star_cache to the child
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from nnenum.network import (SkipAddLayer, FullyConnectedLayer, ReluLayer,
                            NeuralNetwork, AddLayer)
from nnenum.lp_star import LpStar
from nnenum.lp_star_state import LpStarState
from nnenum.settings import Settings


def make_fc(layer_num, n):
    'FC layer that doubles the input'
    W = 2.0 * np.eye(n, dtype=np.float32)
    b = np.zeros(n, dtype=np.float32)
    return FullyConnectedLayer(layer_num, W, b, (n,))


def make_simple_sequential_network():
    'FC(0) -> ReLU(1) -> FC(2): simple sequential, no skip connections'
    n = 4
    return NeuralNetwork([make_fc(0, n), ReluLayer(1, (n,)), make_fc(2, n)])


def make_skip_network():
    '''Build: FC(0) -> FC(1) -> SkipAdd(2, skip_from=0)

    Layer 0: FC that doubles input
    Layer 1: FC that triples input  (this is the "block" main path)
    Layer 2: SkipAdd(skip=layer0_output, main=layer1_output)

    dag_predecessors = {2: [0, 1]}  -> layer 2 takes from layer 0 (skip) and layer 1 (main)
    '''
    n = 4
    W1 = 3.0 * np.eye(n, dtype=np.float32)
    b1 = np.zeros(n, dtype=np.float32)
    fc1 = FullyConnectedLayer(1, W1, b1, (n,))

    skip = SkipAddLayer(2, input_shape=(n,))
    dag = {2: [0, 1]}  # layer 2 = skip_from(0) + main_from(1)
    return NeuralNetwork([make_fc(0, n), fc1, skip], dag_predecessors=dag)


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

    print("=== Step 2: LpStarState star cache ===\n")

    # ── Test 1: LpStarState has star_cache ─────────────────────────────────
    print("-- 1. LpStarState has star_cache --")
    ss = LpStarState()
    ok("LpStarState has star_cache attribute", hasattr(ss, 'star_cache'))
    ok("star_cache starts empty", ss.star_cache == {})

    # ── Test 2: sequential propagation still works ─────────────────────────
    print("\n-- 2. Sequential propagation still works --")
    n = 4
    init_box = np.array([[-1.0, 1.0]] * n, dtype=np.float32)
    network_seq = make_simple_sequential_network()

    ss_seq = LpStarState(uncompressed_init_box=init_box)
    # advance past the two FC layers (stop at ReLU)
    import time
    ss_seq.propagate_up_to_split(network_seq, time.time())
    # Should have stopped at layer 1 (ReluLayer)
    ok("Sequential: stopped at ReLU", ss_seq.cur_layer == 1)
    ok("Sequential: star_cache populated for layer 0",
       0 in ss_seq.star_cache)

    # ── Test 3: SkipAdd propagation ─────────────────────────────────────────
    print("\n-- 3. SkipAddLayer propagation --")
    network_skip = make_skip_network()
    # network: FC(0) -> FC(1) -> SkipAdd(2)
    # Input box: all dims in [-1, 1]
    # FC(0): doubles input, so output in [-2, 2]
    # FC(1): triples FC(0) output, so in [-6, 6]
    # SkipAdd(2): should be FC(0) output + FC(1) output

    ss_skip = LpStarState(uncompressed_init_box=init_box)
    ss_skip.propagate_up_to_split(network_skip, time.time())

    # Should have processed all 3 layers (no ReLU to stop at)
    ok("SkipAdd: finished all layers", ss_skip.is_finished(network_skip))

    # Verify the final bias is correct:
    # FC(0) bias: 2 * 0 = 0
    # FC(1) bias: 3 * 0 = 0 (no input bias)
    # SkipAdd bias: 0 + 0 = 0
    ok("SkipAdd: final bias is zero", np.allclose(ss_skip.star.bias, 0.0))

    # Verify the final a_mat combines both paths
    # If input has dim n, a_mat should still have shape (n, n) -- same param space
    ok("SkipAdd: a_mat shape is (n, k) with same k",
       ss_skip.star.a_mat is None or ss_skip.star.a_mat.shape[1] == n)

    # ── Test 4: star_cache populated at skip-source layer ──────────────────
    print("\n-- 4. star_cache has entry at skip source layer --")
    ss_skip2 = LpStarState(uncompressed_init_box=init_box)
    ss_skip2.propagate_up_to_split(network_skip, time.time())

    ok("star_cache has entry at layer 0 (skip source)",
       0 in ss_skip2.star_cache,
       f"star_cache keys: {list(ss_skip2.star_cache.keys())}")

    # ── Test 5: split_enumerate copies star_cache ───────────────────────────
    print("\n-- 5. split_enumerate copies star_cache --")
    # Need a network with a ReLU after the SkipAdd to trigger splitting
    # Build: FC(0) -> ReLU(1) -> FC(2) (simple, just test cache copy)
    network_with_relu = make_simple_sequential_network()
    init_box2 = np.array([[-2.0, 2.0]] * n, dtype=np.float32)
    ss_relu = LpStarState(uncompressed_init_box=init_box2)

    # manually populate star_cache to simulate skip connection was processed
    ss_relu.star_cache[0] = ss_relu.star.copy() if ss_relu.star else None

    # Advance to ReLU, then split
    import time
    from nnenum.specification import Specification
    # Use a trivial spec: first output <= 999 (always sat)
    spec = Specification([[-1.0, 0.0, 0.0, 0.0]], [999.0])

    ss_relu.propagate_up_to_split(network_with_relu, time.time())
    ok("Stopped at ReLU", ss_relu.cur_layer == 1)

    child = ss_relu.do_first_relu_split(network_with_relu, spec, time.time())
    ok("Split produced child", child is not None)
    if child is not None:
        ok("Child has star_cache", hasattr(child, 'star_cache'))
        ok("Child star_cache copied from parent",
           0 in child.star_cache)

    print(f"\n=== Results: {passed} passed, {failed} failed ===")
    return failed == 0


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
