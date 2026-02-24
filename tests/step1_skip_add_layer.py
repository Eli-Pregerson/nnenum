#!/usr/bin/env python3
"""
Step 1 Tests: SkipAddLayer class and NeuralNetwork DAG support

Tests:
1. SkipAddLayer.execute(state1, state2) - element-wise addition
2. SkipAddLayer.transform_star(star1, star2) - combine two star sets
3. SkipAddLayer.transform_zono(zono1, zono2) - combine two zonotopes
4. SkipAddLayer shape methods
5. NeuralNetwork with dag_predecessors stores connectivity info
6. NeuralNetwork with SkipAddLayer can be created without io check failure
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from nnenum.network import SkipAddLayer, FullyConnectedLayer, ReluLayer, NeuralNetwork
from nnenum.lp_star import LpStar
from nnenum.zonotope import Zonotope


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

    print("=== Step 1: SkipAddLayer ===\n")

    # ── Test 1: basic execute ───────────────────────────────────────────────
    print("-- 1. execute(state1, state2) --")
    shape = (4,)
    layer = SkipAddLayer(layer_num=5, input_shape=shape)

    s1 = np.array([1.0, 2.0, 3.0, 4.0])
    s2 = np.array([10.0, 20.0, 30.0, 40.0])
    out = layer.execute(s1, s2)
    ok("execute sums element-wise", np.allclose(out, s1 + s2))
    ok("execute output shape", out.shape == shape)

    # ── Test 2: shape methods ───────────────────────────────────────────────
    print("\n-- 2. shape methods --")
    shape_2d = (3, 3, 8)
    layer2d = SkipAddLayer(layer_num=7, input_shape=shape_2d)
    ok("get_input_shape", layer2d.get_input_shape() == shape_2d)
    ok("get_output_shape == input_shape", layer2d.get_output_shape() == shape_2d)

    # ── Test 3: transform_star ──────────────────────────────────────────────
    print("\n-- 3. transform_star(star_skip, star_main) --")
    # Both stars share the SAME parameter space u (same LP columns).
    # result = (b_skip + b_main) + (A_skip + A_main)·u
    n = 4  # output dim
    k = 3  # number of parameters (input box dims)
    layer_flat = SkipAddLayer(layer_num=3, input_shape=(n,))

    # skip star: bias=[1,1,1,1], a_mat = 2*ones(n,k)
    a_skip = 2.0 * np.ones((n, k))
    b_skip = np.ones(n)
    star_skip = LpStar(a_skip, b_skip, [(-1.0, 1.0)] * k)

    # main star: bias=[10,10,10,10], a_mat = 3*ones(n,k)
    a_main = 3.0 * np.ones((n, k))
    b_main = 10.0 * np.ones(n)
    star_main = LpStar(a_main, b_main, [(-1.0, 1.0)] * k)

    result_star = layer_flat.transform_star(star_skip, star_main)

    # Expected: bias = b_skip + b_main = 11*ones(n)
    #           a_mat = (2+3)*ones(n,k) = 5*ones(n,k)  [element-wise add]
    ok("transform_star bias = b_skip + b_main",
       np.allclose(result_star.bias, b_skip + b_main))
    ok("transform_star a_mat shape unchanged (same param space)",
       result_star.a_mat.shape == (n, k))
    ok("transform_star a_mat = A_skip + A_main (element-wise)",
       np.allclose(result_star.a_mat, a_skip + a_main))
    ok("transform_star bias at u=0 is sum of biases",
       np.allclose(result_star.bias, np.array([11.0, 11.0, 11.0, 11.0])))

    # ── Test 4: transform_zono ──────────────────────────────────────────────
    print("\n-- 4. transform_zono(zono_skip, zono_main) --")
    # Both zonotopes share the same generator space (same mat_t shape).
    # result: center = c_skip + c_main, mat_t = mat_t_skip + mat_t_main
    n = 3  # output dim
    k = 2  # number of generators (same for both — shared parameter space)
    layer_z = SkipAddLayer(layer_num=4, input_shape=(n,))

    center_skip = np.array([1.0, 2.0, 3.0])
    mat_t_skip = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])  # (n=3, k=2)
    zono_skip = Zonotope(center_skip, mat_t_skip)

    center_main = np.array([10.0, 20.0, 30.0])
    mat_t_main = np.array([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])  # (n=3, k=2)
    zono_main = Zonotope(center_main, mat_t_main)

    result_zono = layer_z.transform_zono(zono_skip, zono_main)

    ok("transform_zono center = sum",
       np.allclose(result_zono.center, center_skip + center_main))
    ok("transform_zono mat_t shape unchanged (same param space)",
       result_zono.mat_t.shape == (n, k))
    ok("transform_zono mat_t = mat_t_skip + mat_t_main (element-wise)",
       np.allclose(result_zono.mat_t, mat_t_skip + mat_t_main))

    # ── Test 5: NeuralNetwork with dag_predecessors ─────────────────────────
    print("\n-- 5. NeuralNetwork dag_predecessors --")
    # Build a tiny network:  FC(0) -> ReLU(1) -> FC(2) -> SkipAdd(3, inputs=[1,2])
    # (SkipAdd takes output of ReLU[1] as skip and output of FC[2] as main)
    # For simplicity we don't actually wire them - just test that the network stores it

    W0 = np.eye(4, dtype=np.float32)
    b0 = np.zeros(4, dtype=np.float32)
    fc0 = FullyConnectedLayer(0, W0, b0, (4,))

    relu1 = ReluLayer(1, (4,))

    W2 = np.eye(4, dtype=np.float32)
    b2 = np.zeros(4, dtype=np.float32)
    fc2 = FullyConnectedLayer(2, W2, b2, (4,))

    skip3 = SkipAddLayer(3, input_shape=(4,))

    # Create network - SkipAdd is last layer, skip predecessor is relu1 (layer 1)
    dag_preds = {3: [1, 2]}  # layer 3 has two inputs: layer 1 (skip) and layer 2 (main)
    network = NeuralNetwork([fc0, relu1, fc2, skip3], dag_predecessors=dag_preds)

    ok("NeuralNetwork accepts dag_predecessors",
       hasattr(network, 'dag_predecessors'))
    ok("dag_predecessors stored correctly",
       network.dag_predecessors == {3: [1, 2]})
    ok("Network has 4 layers",
       len(network.layers) == 4)
    ok("SkipAddLayer is last layer",
       isinstance(network.layers[3], SkipAddLayer))

    # Sequential networks should still work without dag_predecessors
    W_s = np.eye(4, dtype=np.float32)
    b_s = np.zeros(4, dtype=np.float32)
    fc_a = FullyConnectedLayer(0, W_s, b_s, (4,))
    relu_b = ReluLayer(1, (4,))
    seq_net = NeuralNetwork([fc_a, relu_b])
    ok("Sequential NeuralNetwork still works",
       seq_net.dag_predecessors == {})

    print(f"\n=== Results: {passed} passed, {failed} failed ===")
    return failed == 0


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
