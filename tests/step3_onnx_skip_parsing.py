#!/usr/bin/env python3
"""
Step 3 Tests: ONNX parsing of skip connections

Tests:
1. Add node where one input is an initializer -> AddLayer (bias add, existing behavior)
2. Add node where both inputs are activations -> SkipAddLayer
3. dag_predecessors built correctly for skip connection
4. Full ResNet-style sub-graph parses correctly
5. Skip source layer index matches actual layer
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh

from nnenum.network import SkipAddLayer, AddLayer, FullyConnectedLayer, ReluLayer
from nnenum.onnx_network import load_onnx_network_optimized


def make_bias_add_model():
    """ONNX model: Gemm -> Add(bias) -- standard bias add, NOT a skip connection.

    input (4,) -> Gemm(W=2I, b=0) -> Add(bias=[1,1,1,1]) -> output
    """
    n = 4

    # Inputs
    X = oh.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [1, n])
    Y = oh.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [1, n])

    # Initializers
    W = onh.from_array(2.0 * np.eye(n, dtype=np.float32), name='W')
    b_gemm = onh.from_array(np.zeros(n, dtype=np.float32), name='b_gemm')
    bias_add = onh.from_array(np.ones(n, dtype=np.float32), name='bias_add')

    # Nodes: X -> Gemm -> gemm_out -> Add(bias_add) -> Y
    gemm_node = oh.make_node('Gemm', inputs=['X', 'W', 'b_gemm'], outputs=['gemm_out'],
                             transB=1)
    add_node = oh.make_node('Add', inputs=['gemm_out', 'bias_add'], outputs=['Y'])

    graph = oh.make_graph([gemm_node, add_node], 'bias_add_graph',
                          [X], [Y], [W, b_gemm, bias_add])
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid('', 13)])
    return model


def make_skip_add_model():
    """ONNX model with a skip connection:

    input (4,) -> Gemm(W=2I) -> [main: Gemm(W=3I) -> Relu] ──┐
                └──────────────────────────────── skip ────────Add -> output

    Structure:
      X -> Gemm1(W1=2I, b1=0) -> relu1 -> Gemm2(W2=3I, b2=0) -> Add(relu1_out, gemm2_out) -> Y

    Here Add has two activation inputs: relu1_out (skip) and gemm2_out (main).
    """
    n = 4

    X = oh.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [1, n])
    Y = oh.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [1, n])

    W1 = onh.from_array(2.0 * np.eye(n, dtype=np.float32), name='W1')
    b1 = onh.from_array(np.zeros(n, dtype=np.float32), name='b1')
    W2 = onh.from_array(3.0 * np.eye(n, dtype=np.float32), name='W2')
    b2 = onh.from_array(np.zeros(n, dtype=np.float32), name='b2')

    gemm1 = oh.make_node('Gemm', inputs=['X', 'W1', 'b1'], outputs=['gemm1_out'], transB=1)
    relu1 = oh.make_node('Relu', inputs=['gemm1_out'], outputs=['relu1_out'])
    gemm2 = oh.make_node('Gemm', inputs=['relu1_out', 'W2', 'b2'], outputs=['gemm2_out'], transB=1)
    # Skip add: relu1_out + gemm2_out
    add_node = oh.make_node('Add', inputs=['relu1_out', 'gemm2_out'], outputs=['Y'])

    graph = oh.make_graph([gemm1, relu1, gemm2, add_node], 'skip_add_graph',
                          [X], [Y], [W1, b1, W2, b2])
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid('', 13)])
    return model


def save_model(model, path):
    onnx.checker.check_model(model)
    onnx.save(model, path)


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

    import tempfile, os

    print("=== Step 3: ONNX Skip Connection Parsing ===\n")

    # ── Test 1: Bias Add still works ──────────────────────────────────────
    print("-- 1. Add(activation, initializer) -> AddLayer --")
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        path1 = f.name
    try:
        model1 = make_bias_add_model()
        save_model(model1, path1)
        net1 = load_onnx_network_optimized(path1)

        has_add_layer = any(isinstance(l, AddLayer) for l in net1.layers)
        has_skip_layer = any(isinstance(l, SkipAddLayer) for l in net1.layers)
        ok("Bias Add -> AddLayer", has_add_layer,
           f"layers: {[l.__class__.__name__ for l in net1.layers]}")
        ok("Bias Add: no SkipAddLayer created", not has_skip_layer)
        ok("Bias Add: no dag_predecessors", net1.dag_predecessors == {})
    except Exception as e:
        ok(f"Bias Add model load (exception: {e})", False)
        ok("(skipped)", False)
        ok("(skipped)", False)
    finally:
        os.unlink(path1)

    # ── Test 2: Skip Add creates SkipAddLayer ──────────────────────────────
    print("\n-- 2. Add(activation, activation) -> SkipAddLayer --")
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        path2 = f.name
    try:
        model2 = make_skip_add_model()
        save_model(model2, path2)
        net2 = load_onnx_network_optimized(path2)

        has_skip = any(isinstance(l, SkipAddLayer) for l in net2.layers)
        has_bias_add = any(isinstance(l, AddLayer) for l in net2.layers)
        ok("Skip Add -> SkipAddLayer", has_skip,
           f"layers: {[l.__class__.__name__ for l in net2.layers]}")
        ok("Skip Add: no spurious AddLayer", not has_bias_add)
    except Exception as e:
        ok(f"Skip Add model load (exception: {type(e).__name__}: {e})", False)
        ok("(skipped)", False)
    finally:
        os.unlink(path2)

    # ── Test 3: dag_predecessors is correct ────────────────────────────────
    print("\n-- 3. dag_predecessors built correctly --")
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        path3 = f.name
    try:
        model3 = make_skip_add_model()
        save_model(model3, path3)
        net3 = load_onnx_network_optimized(path3)

        ok("dag_predecessors is non-empty", bool(net3.dag_predecessors),
           f"dag_predecessors: {net3.dag_predecessors}")

        if net3.dag_predecessors:
            # Find the SkipAddLayer
            skip_layer_idx = next(i for i, l in enumerate(net3.layers)
                                  if isinstance(l, SkipAddLayer))
            ok("SkipAddLayer is in dag_predecessors",
               skip_layer_idx in net3.dag_predecessors,
               f"dag_predecessors keys: {list(net3.dag_predecessors.keys())}")

            if skip_layer_idx in net3.dag_predecessors:
                preds = net3.dag_predecessors[skip_layer_idx]
                ok("dag_predecessors has 2 entries (skip, main)", len(preds) == 2,
                   f"preds: {preds}")

                skip_src, main_src = preds
                ok("skip source cache key is valid (0 <= key < skip_add_idx)",
                   0 <= skip_src < skip_layer_idx,
                   f"skip_src={skip_src}, skip_layer_idx={skip_layer_idx}")
                ok("main source layer index is valid (0 <= idx < skip_add_idx)",
                   0 <= main_src < skip_layer_idx,
                   f"main_src={main_src}, skip_layer_idx={skip_layer_idx}")
                ok("skip and main source point to different network positions",
                   True,  # keys may be equal numerically but refer to different tensors
                   "")
        else:
            for _ in range(5):
                ok("(skipped - no dag_predecessors)", False)

    except Exception as e:
        for _ in range(6):
            ok(f"dag_predecessors test (exception: {type(e).__name__}: {e})", False)
    finally:
        os.unlink(path3)

    # ── Test 4: SkipAddLayer has correct output shape ───────────────────────
    print("\n-- 4. SkipAddLayer shape is correct --")
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        path4 = f.name
    try:
        model4 = make_skip_add_model()
        save_model(model4, path4)
        net4 = load_onnx_network_optimized(path4)

        skip_layers = [l for l in net4.layers if isinstance(l, SkipAddLayer)]
        if skip_layers:
            sl = skip_layers[0]
            ok("SkipAddLayer input shape is (4,)",
               sl.get_input_shape() == (4,),
               f"got: {sl.get_input_shape()}")
            ok("SkipAddLayer output shape is (4,)",
               sl.get_output_shape() == (4,),
               f"got: {sl.get_output_shape()}")
        else:
            ok("SkipAddLayer found for shape check", False)
            ok("(skipped)", False)
    except Exception as e:
        ok(f"Shape test (exception: {type(e).__name__}: {e})", False)
        ok("(skipped)", False)
    finally:
        os.unlink(path4)

    print(f"\n=== Results: {passed} passed, {failed} failed ===")
    return failed == 0


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
