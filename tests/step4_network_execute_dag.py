#!/usr/bin/env python3
"""
Step 4 Tests: NeuralNetwork.execute() for DAG (skip connection) networks

Tests:
1. Sequential network.execute() still works correctly
2. DAG network.execute() correctly combines skip + main paths
3. Concrete numerical correctness for skip network
4. SkipAddLayer with identity skip (shortcut) is correct
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from nnenum.network import (SkipAddLayer, FullyConnectedLayer, ReluLayer,
                            NeuralNetwork, AddLayer)


def make_fc(layer_num, n, scale=1.0):
    'FC layer that scales the input'
    W = scale * np.eye(n, dtype=np.float32)
    b = np.zeros(n, dtype=np.float32)
    return FullyConnectedLayer(layer_num, W, b, (n,))


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

    print("=== Step 4: NeuralNetwork.execute() DAG ===\n")

    n = 4
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    # ── Test 1: Sequential execution still works ──────────────────────────
    print("-- 1. Sequential network execute --")
    seq_net = NeuralNetwork([make_fc(0, n, 2.0), make_fc(1, n, 3.0)])
    out = seq_net.execute(x)
    expected = 6.0 * x  # 2 * 3 * x
    ok("Sequential: 2x then 3x = 6x", np.allclose(out, expected),
       f"got {out}, expected {expected}")

    # ── Test 2: DAG network execute - simple skip ──────────────────────────
    print("\n-- 2. DAG skip network execute --")
    # Network: FC(0, 2x) -> FC(1, 3x) -> SkipAdd(2)
    #          └──────────skip──────────────────────┘
    # Expected: SkipAdd output = FC(0) output + FC(1) output
    #         = 2x + 3*2x = 2x + 6x = 8x
    fc0 = make_fc(0, n, 2.0)
    fc1 = make_fc(1, n, 3.0)
    skip = SkipAddLayer(2, input_shape=(n,))
    dag = {2: [1, 2]}  # SkipAdd at layer 2: skip from cache key 1, main from layer 1
    # Note: cache key 1 = star after layer 0 (= FC0 output = 2x)
    # Layer 1 (fc1) processes the FC0 output -> 3*2x = 6x
    # SkipAdd: 2x + 6x = 8x
    dag_net = NeuralNetwork([fc0, fc1, skip], dag_predecessors=dag)

    out_dag = dag_net.execute(x)
    expected_dag = 8.0 * x  # 2x (skip) + 6x (main) = 8x
    ok("DAG: FC(2x) -> FC(3x) -> SkipAdd = 8x",
       np.allclose(out_dag, expected_dag),
       f"got {out_dag}, expected {expected_dag}")

    # ── Test 3: Numerical correctness with known values ───────────────────
    print("\n-- 3. Numerical correctness --")
    # Simple: input=[1,1,1,1], FC0 doubles, FC1 adds bias=[10,10,10,10]
    W0 = 2.0 * np.eye(n, dtype=np.float32)
    b0 = np.zeros(n, dtype=np.float32)
    fc0_b = FullyConnectedLayer(0, W0, b0, (n,))

    W1 = np.eye(n, dtype=np.float32)
    b1 = 10.0 * np.ones(n, dtype=np.float32)
    fc1_b = FullyConnectedLayer(1, W1, b1, (n,))

    skip_b = SkipAddLayer(2, input_shape=(n,))
    dag_b = {2: [1, 2]}  # skip from cache key 1 = after FC0
    net_b = NeuralNetwork([fc0_b, fc1_b, skip_b], dag_predecessors=dag_b)

    x2 = np.ones(n, dtype=np.float32)
    out_b = net_b.execute(x2)
    # FC0: 2*[1,1,1,1] = [2,2,2,2]
    # FC1: [2,2,2,2] + [10,10,10,10] = [12,12,12,12]
    # SkipAdd: [2,2,2,2] (skip=FC0 out) + [12,12,12,12] (main=FC1 out) = [14,14,14,14]
    expected_b = np.array([14.0, 14.0, 14.0, 14.0], dtype=np.float32)
    ok("Numerical: FC0(2x) + [FC0(2x) + 10] = 14 for x=1",
       np.allclose(out_b, expected_b),
       f"got {out_b}, expected {expected_b}")

    # ── Test 4: Identity skip (ResNet style) ──────────────────────────────
    print("\n-- 4. Identity skip (ResNet-style shortcut) --")
    # input -> [main: FC(2x)] -> Add(input, 2x) = 3x
    # This simulates a ResNet shortcut where the skip is the identity
    # In our setup: FC0=Identity, FC1=2x, SkipAdd
    #   skip from FC0 (= x), main from FC1 (= 2x)
    #   output = x + 2x = 3x
    fc_id = make_fc(0, n, 1.0)   # identity
    fc_2x = make_fc(1, n, 2.0)   # 2x
    skip_r = SkipAddLayer(2, input_shape=(n,))
    dag_r = {2: [1, 2]}  # skip from cache key 1 = after fc_id = x, main = 2x
    net_r = NeuralNetwork([fc_id, fc_2x, skip_r], dag_predecessors=dag_r)

    out_r = net_r.execute(x)
    expected_r = 3.0 * x  # x + 2x = 3x
    ok("ResNet-style: identity skip + FC(2x) main = 3x",
       np.allclose(out_r, expected_r),
       f"got {out_r}, expected {expected_r}")

    # ── Test 5: execute() output shape ────────────────────────────────────
    print("\n-- 5. Output shape correct --")
    ok("Sequential output shape", seq_net.execute(x).shape == (n,))
    ok("DAG output shape", out_dag.shape == (n,))

    print(f"\n=== Results: {passed} passed, {failed} failed ===")
    return failed == 0


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
