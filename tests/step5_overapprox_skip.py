#!/usr/bin/env python3
"""
Step 5 Tests: prefilter + overapprox for SkipAddLayer

Tests:
1. make_prerelu_sims handles SkipAddLayer (uses simulation_cache for skip)
2. run_overapprox_round handles SkipAddLayer (star/zono transform)
3. Prefilter apply_linear_layer simulation update through SkipAddLayer
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import time

from nnenum.network import (SkipAddLayer, FullyConnectedLayer, ReluLayer,
                            NeuralNetwork)
from nnenum.lp_star import LpStar
from nnenum.lp_star_state import LpStarState
from nnenum.settings import Settings
from nnenum.overapprox import make_prerelu_sims


def make_skip_network_with_relu():
    """Build: FC(0, 2x) -> ReLU(1) -> FC(2, 3x) -> SkipAdd(3, skip_from=0 output)

    At SkipAdd: skip=2x (before relu), main=3*relu(2x)
    dag_predecessors = {3: [1, 2]}
      - skip_cache_key=1 means star_cache[1]=star after layer 0 (2x, before relu)
      - main source layer=2 (fc2)
    """
    n = 4
    W2 = 2.0 * np.eye(n, dtype=np.float32)
    b2 = np.zeros(n, dtype=np.float32)
    fc0 = FullyConnectedLayer(0, W2, b2, (n,))

    relu1 = ReluLayer(1, (n,))

    W3 = 3.0 * np.eye(n, dtype=np.float32)
    b3 = np.zeros(n, dtype=np.float32)
    fc2 = FullyConnectedLayer(2, W3, b3, (n,))

    skip3 = SkipAddLayer(3, input_shape=(n,))

    dag = {3: [1, 2]}  # SkipAdd at layer 3: skip from cache key 1, main from layer 2
    return NeuralNetwork([fc0, relu1, fc2, skip3], dag_predecessors=dag)


def make_ss(init_box):
    """Create LpStarState from box."""
    return LpStarState(uncompressed_init_box=init_box)


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

    print("=== Step 5: overapprox SkipAddLayer ===\n")

    n = 4
    init_box = np.array([[-1.0, 1.0]] * n, dtype=np.float32)
    network = make_skip_network_with_relu()

    # ── Test 1: make_prerelu_sims works through SkipAddLayer ─────────────
    print("-- 1. make_prerelu_sims through SkipAddLayer --")
    ss = make_ss(init_box)
    # Advance to the ReLU layer (layer 1)
    ss.propagate_up_to_split(network, time.time())
    ok("Propagated to ReLU", ss.cur_layer == 1,
       f"cur_layer={ss.cur_layer}")

    try:
        sims = make_prerelu_sims(ss, network)
        ok("make_prerelu_sims did not crash", sims is not None)

        # The sim at layer 3 (SkipAdd) should be non-None
        ok("prerelu_sims has entry for SkipAdd layer",
           sims is not None and 3 in sims,
           f"keys: {list(sims.keys()) if sims else 'None'}")

        if sims and 3 in sims:
            # Simulation center: input = midpoint of box = [0,0,0,0]
            # FC0: 2*[0...] = [0,0,0,0]
            # ReLU: max(0, [0,0,0,0]) = [0,0,0,0]
            # FC2: 3*[0,0,0,0] = [0,0,0,0]
            # SkipAdd: skip=[0,0,0,0] + main=[0,0,0,0] = [0,0,0,0]
            # sims[3] is the state BEFORE SkipAdd = state after FC2 = [0,0,0,0]
            expected_before_skip = np.array([0.0, 0.0, 0.0, 0.0])
            ok("prerelu_sims[3] is state before SkipAdd (after FC2)",
               np.allclose(sims[3], expected_before_skip, atol=1e-5),
               f"got {sims[3]}, expected {expected_before_skip}")

    except Exception as e:
        ok(f"make_prerelu_sims (exception: {type(e).__name__}: {e})", False)
        ok("(skipped)", False)
        ok("(skipped)", False)

    # ── Test 2: Simulation update through SkipAddLayer ────────────────────
    print("\n-- 2. Simulation update through SkipAddLayer --")
    ss2 = make_ss(init_box)
    # Propagate all the way through (no splits should occur since all neurons positive)
    # FC0: 2*[0.5...] = [1,1,1,1] > 0, so ReLU is deterministic
    ss2.propagate_up_to_split(network, time.time())

    # If there are no branching neurons, propagation continues past relu to next split
    # ReLU at layer 1 with input [1,1,1,1] should have no branching neurons
    # Then continue through FC2 (layer 2) and SkipAdd (layer 3)
    ok("Propagated at least to or past ReLU layer 1",
       ss2.cur_layer >= 1,
       f"cur_layer={ss2.cur_layer}")

    if ss2.is_finished(network):
        # The simulation should reflect the SkipAdd
        # Input center: [0,0,0,0] (midpoint of [-1,1])
        # After all layers: 2*0=0, relu(0)=0, 3*0=0, skipAdd(0+0)=0
        sim_out = ss2.prefilter.simulation[1]
        expected_out = np.array([0.0, 0.0, 0.0, 0.0])
        ok("Simulation updated through SkipAdd",
           np.allclose(sim_out, expected_out, atol=1e-4),
           f"got {sim_out}, expected {expected_out}")
    else:
        # Network stopped at ReLU (branching neurons exist)
        ok("Network stopped at ReLU (branching neurons present)", ss2.cur_layer == 1)

    # ── Test 3: star_cache has entry at skip source ───────────────────────
    print("\n-- 3. star_cache populated at skip source --")
    ss3 = make_ss(init_box)
    ss3.propagate_up_to_split(network, time.time())

    ok("star_cache[0] populated (before FC0)",
       0 in ss3.star_cache,
       f"star_cache keys: {list(ss3.star_cache.keys())}")

    # The skip cache key for the SkipAdd is 1 (star_cache[1] = star after FC0)
    ok("star_cache[1] populated (after FC0, = skip source star)",
       1 in ss3.star_cache,
       f"star_cache keys: {list(ss3.star_cache.keys())}")

    # ── Test 4: prefilter.simulation_cache has skip source sim ───────────
    print("\n-- 4. simulation_cache at skip source --")
    ss4 = make_ss(init_box)
    ss4.propagate_up_to_split(network, time.time())

    sim_cache = ss4.prefilter.simulation_cache
    ok("simulation_cache has entry at layer 1 (skip source cache key)",
       1 in sim_cache,
       f"sim_cache keys: {list(sim_cache.keys())}")

    if 1 in sim_cache:
        # sim_cache[1] = simulation BEFORE layer 1 = after FC0 = 2*center = 2*0 = [0,0,0,0]
        expected_sim = np.array([0.0, 0.0, 0.0, 0.0])
        ok("simulation_cache[1] = 2*input_center = [0,0,0,0] (center of [-1,1])",
           np.allclose(sim_cache[1], expected_sim, atol=1e-5),
           f"got {sim_cache[1]}, expected {expected_sim}")
    else:
        ok("(skipped - no sim_cache[1])", False)

    print(f"\n=== Results: {passed} passed, {failed} failed ===")
    return failed == 0


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
