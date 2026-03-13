"""
Preprocess an ONNX network by replacing MaxPool operations with equivalent
ReLU-based computations. No dnnv/TensorFlow required.

Mathematical identity used:
    max(a, b) = ReLU(a - b) + b

Applied in a binary tree over the kernel window:
- Phase 1: conv with kernel_shape to compute pairwise (diff, val, neg_diff) triples
- Phase 2: 1x1 convs to merge pairs recursively
- Phase 3: 1x1 conv to extract final max, followed by ReLU

Usage:
    python3 reluify_maxpool.py input.onnx output.onnx

Limitations:
    - ceil_mode is not supported
    - dilations != 1 are not supported (these MaxPools are left unchanged)
    - Only spatial 2D MaxPool supported
"""

import sys
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto, helper
from pathlib import Path

_ONNX_TO_NUMPY = {
    TensorProto.FLOAT: np.float32,
    TensorProto.DOUBLE: np.float64,
    TensorProto.FLOAT16: np.float16,
    TensorProto.INT32: np.int32,
    TensorProto.INT64: np.int64,
}


def _get_tensor_dtype(graph, tensor_name):
    """Return numpy dtype of a named tensor in the graph (from value_info or inputs)."""
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        if vi.name == tensor_name:
            elem_type = vi.type.tensor_type.elem_type
            return _ONNX_TO_NUMPY.get(elem_type, np.float32)
    return np.float32  # fallback


def _get_tensor_shape(graph, tensor_name):
    """Return shape tuple of a named tensor (None for dynamic dims)."""
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        if vi.name == tensor_name:
            dims = vi.type.tensor_type.shape.dim
            return tuple(d.dim_value if d.dim_value != 0 else None for d in dims)
    return None


def _make_initializer(name, array):
    """Create an ONNX initializer from a numpy array."""
    return numpy_helper.from_array(array.astype(array.dtype, copy=False), name=name)


def _make_conv_node(input_name, weight_name, output_name,
                    kernel_shape=None, strides=None, pads=None, group=1):
    """Build an ONNX Conv node."""
    attrs = {"group": group}
    if kernel_shape is not None:
        attrs["kernel_shape"] = list(kernel_shape)
    if strides is not None:
        attrs["strides"] = list(strides)
    if pads is not None:
        attrs["pads"] = list(pads)
    return helper.make_node("Conv", inputs=[input_name, weight_name], outputs=[output_name], **attrs)


def _make_relu_node(input_name, output_name):
    return helper.make_node("Relu", inputs=[input_name], outputs=[output_name])


def reluify_maxpool_subgraph(node, graph, uid_counter):
    """
    Replace a single MaxPool node with Conv+ReLU chains.
    Returns (new_nodes, new_initializers, output_name) or None if unsupported.
    uid_counter: list with one int, incremented to generate unique names.
    """
    def uid():
        uid_counter[0] += 1
        return uid_counter[0]

    # Parse attributes
    ceil_mode = 0
    dilations = None
    kernel_shape = None
    pads = None
    strides = None

    for attr in node.attribute:
        if attr.name == "ceil_mode":
            ceil_mode = attr.i
        elif attr.name == "dilations":
            dilations = list(attr.ints)
        elif attr.name == "kernel_shape":
            kernel_shape = list(attr.ints)
        elif attr.name == "pads":
            pads = list(attr.ints)
        elif attr.name == "strides":
            strides = list(attr.ints)

    if ceil_mode:
        return None  # unsupported
    if dilations and any(d != 1 for d in dilations):
        return None  # unsupported
    if kernel_shape is None:
        return None

    k = int(np.prod(kernel_shape))
    if k == 1:
        return None  # no-op

    if strides is None:
        strides = [1] * len(kernel_shape)
    if pads is None:
        pads = [0] * (2 * len(kernel_shape))

    # Determine num_channels from input shape
    input_name = node.input[0]
    shape = _get_tensor_shape(graph, input_name)
    dtype = _get_tensor_dtype(graph, input_name)
    if shape is None or shape[1] is None:
        return None  # can't determine channels statically
    num_channels = shape[1]

    new_nodes = []
    new_inits = []
    cur_input = input_name

    # -------------------------
    # Phase 1: initial conv with kernel_shape
    # -------------------------
    num_comparisons = int(np.ceil(k / 2))
    out_channels_1 = num_channels * num_comparisons * 3
    w1 = np.zeros((out_channels_1, num_channels, k), dtype=dtype)

    for ch in range(num_channels):
        for cmp_i in range(num_comparisons):
            index = np.ravel_multi_index((ch, cmp_i, 0), (num_channels, num_comparisons, 3))
            idx1 = 2 * cmp_i
            idx2 = 2 * cmp_i + 1
            if idx2 < k:
                w1[index + 0, ch, idx1] = 1
                w1[index + 0, ch, idx2] = -1
                w1[index + 1, ch, idx2] = 1
                w1[index + 2, ch, idx2] = -1
            else:
                w1[index + 1, ch, idx1] = 1
                w1[index + 2, ch, idx1] = -1

    # Reshape to (out_ch, in_ch, *kernel_shape)
    w1 = w1.reshape(out_channels_1, num_channels, *kernel_shape)

    w1_name = f"__reluify_w1_{uid()}"
    conv1_out = f"__reluify_conv1_{uid()}"
    relu1_out = f"__reluify_relu1_{uid()}"

    new_inits.append(_make_initializer(w1_name, w1))
    new_nodes.append(_make_conv_node(cur_input, w1_name, conv1_out,
                                     kernel_shape=kernel_shape, strides=strides, pads=pads))
    new_nodes.append(_make_relu_node(conv1_out, relu1_out))
    cur_input = relu1_out

    # -------------------------
    # Phase 2: 1x1 merge convs
    # -------------------------
    while num_comparisons != 1:
        num_values = num_comparisons
        num_comparisons = int(np.ceil(num_values / 2))
        out_channels_m = num_channels * num_comparisons * 3
        in_channels_m = num_channels * num_values * 3
        wm = np.zeros((out_channels_m, in_channels_m, 1, 1), dtype=dtype)

        for ch in range(num_channels):
            for cmp_i in range(num_comparisons):
                index = np.ravel_multi_index((ch, cmp_i, 0), (num_channels, num_comparisons, 3))
                idx1 = int(np.ravel_multi_index((ch, 2 * cmp_i, 0), (num_channels, num_values, 3)))
                if 2 * cmp_i + 1 < num_values:
                    idx2 = int(np.ravel_multi_index((ch, 2 * cmp_i + 1, 0), (num_channels, num_values, 3)))
                    wm[index + 0, idx1 + 0, 0, 0] = 1
                    wm[index + 0, idx1 + 1, 0, 0] = 1
                    wm[index + 0, idx1 + 2, 0, 0] = -1
                    wm[index + 0, idx2 + 0, 0, 0] = -1
                    wm[index + 0, idx2 + 1, 0, 0] = -1
                    wm[index + 0, idx2 + 2, 0, 0] = 1
                    wm[index + 1, idx2 + 0, 0, 0] = 1
                    wm[index + 1, idx2 + 1, 0, 0] = 1
                    wm[index + 1, idx2 + 2, 0, 0] = -1
                    wm[index + 2, idx2 + 0, 0, 0] = -1
                    wm[index + 2, idx2 + 1, 0, 0] = -1
                    wm[index + 2, idx2 + 2, 0, 0] = 1
                else:
                    wm[index + 1, idx1 + 0, 0, 0] = 1
                    wm[index + 1, idx1 + 1, 0, 0] = 1
                    wm[index + 1, idx1 + 2, 0, 0] = -1
                    wm[index + 2, idx1 + 0, 0, 0] = -1
                    wm[index + 2, idx1 + 1, 0, 0] = -1
                    wm[index + 2, idx1 + 2, 0, 0] = 1

        wm_name = f"__reluify_wm_{uid()}"
        convm_out = f"__reluify_convm_{uid()}"
        relum_out = f"__reluify_relum_{uid()}"

        new_inits.append(_make_initializer(wm_name, wm))
        new_nodes.append(_make_conv_node(cur_input, wm_name, convm_out, kernel_shape=[1, 1]))
        new_nodes.append(_make_relu_node(convm_out, relum_out))
        cur_input = relum_out

    # -------------------------
    # Phase 3: final extraction conv + ReLU
    # -------------------------
    wf = np.zeros((num_channels, num_channels * 3, 1, 1), dtype=dtype)
    for ch in range(num_channels):
        index = np.ravel_multi_index((ch, 0, 0), (num_channels, 1, 3))
        wf[ch, index + 0, 0, 0] = 1
        wf[ch, index + 1, 0, 0] = 1
        wf[ch, index + 2, 0, 0] = -1

    wf_name = f"__reluify_wf_{uid()}"
    convf_out = f"__reluify_convf_{uid()}"
    # Use the original MaxPool output name for the final ReLU so downstream nodes connect correctly
    final_out = node.output[0]

    new_inits.append(_make_initializer(wf_name, wf))
    new_nodes.append(_make_conv_node(cur_input, wf_name, convf_out, kernel_shape=[1, 1]))
    new_nodes.append(_make_relu_node(convf_out, final_out))

    return new_nodes, new_inits


def reluify_onnx(input_path: str, output_path: str):
    model = onnx.load(input_path)

    # Run shape inference so we can query channel counts
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"Warning: shape inference failed ({e}), proceeding anyway")

    graph = model.graph
    uid_counter = [0]

    new_nodes = []
    new_inits = list(graph.initializer)
    replaced = 0

    for node in graph.node:
        if node.op_type == "MaxPool":
            result = reluify_maxpool_subgraph(node, graph, uid_counter)
            if result is not None:
                nodes, inits = result
                new_nodes.extend(nodes)
                new_inits.extend(inits)
                replaced += 1
                print(f"  Replaced MaxPool '{node.name or node.output[0]}' "
                      f"with {len(nodes)} Conv/ReLU nodes")
                continue
            else:
                print(f"  Skipped MaxPool '{node.name or node.output[0]}' (unsupported config)")
        new_nodes.append(node)

    if replaced == 0:
        print("No MaxPool nodes were replaced.")
    else:
        print(f"Replaced {replaced} MaxPool node(s).")

    # Collect existing graph input names (network inputs only, not initializers)
    existing_input_names = {inp.name for inp in graph.input}
    # In older ONNX models, initializers also appear as graph inputs; collect them
    existing_init_names = {init.name for init in graph.initializer}

    # Build graph inputs: keep original network inputs + add new initializers as inputs
    # (required by ONNX IR >= 4 validation when initializer-only tensors are used)
    graph_inputs = list(graph.input)
    for init in new_inits:
        if init.name not in existing_init_names and init.name not in existing_input_names:
            # Add as graph input (shape + type from the initializer)
            shape = list(init.dims)
            dtype = init.data_type
            graph_inputs.append(
                helper.make_tensor_value_info(init.name, dtype, shape)
            )

    # Rebuild graph
    new_graph = helper.make_graph(
        new_nodes,
        graph.name,
        graph_inputs,
        list(graph.output),
        initializer=new_inits,
    )
    new_model = helper.make_model(new_graph, opset_imports=model.opset_import)
    new_model.ir_version = model.ir_version

    # Re-run shape inference on the new model
    try:
        new_model = onnx.shape_inference.infer_shapes(new_model)
    except Exception as e:
        print(f"Warning: shape inference on output failed ({e})")

    onnx.checker.check_model(new_model)
    onnx.save(new_model, output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python3 {sys.argv[0]} input.onnx output.onnx")
        sys.exit(1)
    print(f"Loading: {sys.argv[1]}")
    reluify_onnx(sys.argv[1], sys.argv[2])
