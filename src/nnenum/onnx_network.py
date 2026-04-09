'''
functions related to loading onnx networks
'''

import time
import numpy as np

from scipy.sparse import csc_matrix, csr_matrix
from scipy import sparse

import onnx
from onnx import version_converter
import onnxruntime as ort

from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs, select_model_inputs_outputs
from onnx.helper import ValueInfoProto, make_graph, make_model

from nnenum.network import NeuralNetwork, ConstantLayer, AddLayer, FlattenLayer, ReshapeLayer, ReluLayer, MatMulLayer, FullyConnectedLayer, Convolutional2dLayer, SkipAddLayer, ScaleLayer
from nnenum.network import nn_unflatten, nn_flatten
from nnenum.settings import Settings

def _tensor_to_numpy(init):
    """Read an ONNX TensorProto initializer as a float32 numpy array.

    Handles both raw_data (compact binary) and float_data (repeated field) storage formats.
    Models like VGGNet use float_data; most other models use raw_data.
    """
    if len(init.raw_data) > 0:
        return np.frombuffer(init.raw_data, dtype='<f4')
    return np.array(init.float_data, dtype=np.float32)

from nnenum.util import Freezable

class LinearOnnxSubnetworkLayer(Freezable):
    '''a linear layer consisting of multiple onnx operators

    this uses the onnx runtime to execute
    '''

    def __init__(self, layer_num, onnx_submodel):

        self.layer_num = layer_num
        self.network = None # populated later when constructing network

        # Ensure opset is 23 (submodel extraction may not preserve converted opset)
        for opset in onnx_submodel.opset_import:
            if opset.domain in ['', 'ai.onnx'] and opset.version > 23:
                opset.version = 23

        initializers = [i.name for i in onnx_submodel.graph.initializer]
        inputs = [i for i in onnx_submodel.graph.input if i.name not in initializers]

        assert len(inputs) == 1
        assert len(onnx_submodel.graph.output) == 1

        self.input_name = inputs[0].name

        #print(onnx_submodel)

        self.model_str = onnx_submodel.SerializeToString()
        self.sess = ort.InferenceSession(self.model_str)

        # execute to find output shape and zero_output
        t = inputs[0].type.tensor_type.elem_type

        if t == onnx.TensorProto.FLOAT:
            self.dtype = np.float32
        else:
            assert t == onnx.TensorProto.DOUBLE
            self.dtype = np.float64
        
        input_map = {}

        inp = inputs[0]
        
        shape = tuple(d.dim_value for d in inp.type.tensor_type.shape.dim)
        input_map[inp.name] = np.zeros(shape, dtype=self.dtype)

        assert input_map, "didn't find input?"
        self.input_shape = input_map[inp.name].shape
        
        self.zero_output = self.sess.run(None, input_map)[0]
        self.output_shape = self.zero_output.shape

        self.freeze_attrs()

    def __str__(self):
        return f'[LinearOnnxSubnetworkLayer with input {self.get_input_shape()} and output {self.get_output_shape()}]'

    def get_input_shape(self):
        'get the input shape to this layer'

        return self.input_shape

    def get_output_shape(self):
        'get the output shape from this layer'

        return self.output_shape

    def transform_star(self, star):
        'transform the star'

        if star.a_mat is None:
            dims = star.lpi.get_num_cols()
            star.a_mat = np.identity(dims, dtype=self.dtype)
            star.bias = np.zeros(dims, dtype=self.dtype)

        cols = []

        for col in range(star.a_mat.shape[1]):
            #print(f".transforming star: {col} / {star.a_mat.shape[1]})")
            vec = star.a_mat[:, col]
            vec = nn_unflatten(vec, self.input_shape)
            
            res = self.execute(vec)
            res = res - self.zero_output
            res = nn_flatten(res)

            cols.append(res)

        dtype = star.bias.dtype
        star.a_mat = np.array(cols, dtype=dtype).transpose()

        vec = nn_unflatten(star.bias, self.input_shape)
        res = self.execute(vec)
        star.bias = nn_flatten(res)

    def transform_zono(self, zono):
        'transform the zono'

        cols = []

        for col in range(zono.mat_t.shape[1]):
            #print(f".transforming zono: {col} / {zono.mat_t.shape[1]})")
            vec = zono.mat_t[:, col]
            vec = nn_unflatten(vec, self.input_shape)

            res = self.execute(vec)

            res = res - self.zero_output
            res = nn_flatten(res)

            cols.append(res)

        dtype = zono.center.dtype
        zono.mat_t = np.array(cols, dtype=dtype).transpose()

        start_center = nn_unflatten(zono.center, self.input_shape)
        end_center = self.execute(start_center)
        zono.center = nn_flatten(end_center)

    def transform_deeppoly(self, deeppoly):
        'transform the deeppoly'

        # ####### Version 2.0 #######
        # cols = []

        # for col in range(deeppoly.ubcoef.shape[1]):
        #     #print(f".transforming zono: {col} / {zono.mat_t.shape[1]})")
        #     vec = deeppoly.ubcoef[:, col]
        #     vec = nn_unflatten(vec, self.input_shape)

        #     res = self.execute(vec)

        #     res = res - self.zero_output
        #     res = nn_flatten(res)

        #     cols.append(res)

        # dtype = deeppoly.ubconst.dtype
        # deeppoly.ubcoef = np.array(cols, dtype=dtype).transpose()

        # start_center = nn_unflatten(deeppoly.ubconst, self.input_shape)
        # end_center = self.execute(start_center)
        # deeppoly.ubconst = nn_flatten(end_center)
        # # np.save(f'deeppoly.ubcoef_{self.layer_num}.npy', deeppoly.ubcoef)
        # # print(f'deeppoly.ubcoef.shape: {deeppoly.ubcoef.shape}')

        # cols = []

        # for col in range(deeppoly.lbcoef.shape[1]):
        #     #print(f".transforming zono: {col} / {zono.mat_t.shape[1]})")
        #     vec = deeppoly.lbcoef[:, col]
        #     vec = nn_unflatten(vec, self.input_shape)

        #     res = self.execute(vec)

        #     res = res - self.zero_output
        #     res = nn_flatten(res)

        #     cols.append(res)

        # dtype = deeppoly.lbconst.dtype
        # deeppoly.lbcoef = np.array(cols, dtype=dtype).transpose()

        # start_center = nn_unflatten(deeppoly.lbconst, self.input_shape)
        # end_center = self.execute(start_center)
        # deeppoly.lbconst = nn_flatten(end_center)

        # deeppoly.ubs = np.where(deeppoly.ubcoef >= 0, deeppoly.ubcoef, 0) @ deeppoly.inputbounds[:, 1]
        # deeppoly.ubs += np.where(deeppoly.ubcoef < 0, deeppoly.ubcoef, 0) @ deeppoly.inputbounds[:, 0]
        # deeppoly.ubs += deeppoly.ubconst
        # deeppoly.lbs = np.where(deeppoly.lbcoef >= 0, deeppoly.lbcoef, 0) @ deeppoly.inputbounds[:, 0]
        # deeppoly.lbs += np.where(deeppoly.lbcoef < 0, deeppoly.lbcoef, 0) @ deeppoly.inputbounds[:, 1]
        # deeppoly.lbs += deeppoly.lbconst

        ###### Version 1.0 #######
        # dims = np.prod(self.network.layers[self.layer_num].get_output_shape()[1:])
        dims = np.prod(self.input_shape[1:])
        weights = np.identity(dims, dtype=self.dtype)
        biases = np.zeros(dims, dtype=self.dtype)

        cols = []

        for col in range(weights.shape[1]):
            #print(f".transforming star: {col} / {star.a_mat.shape[1]})")
            vec = weights[:, col]
            vec = nn_unflatten(vec, self.input_shape)
            
            res = self.execute(vec)
            res = res - self.zero_output
            res = nn_flatten(res)

            cols.append(res)

        dtype = biases.dtype
        weights = np.array(cols, dtype=dtype).transpose()

        vec = nn_unflatten(biases, self.input_shape)
        res = self.execute(vec)
        biases = nn_flatten(res)

        # back substitution
        updated_ubcoef_nl = np.where(weights >= 0, weights, 0) @ deeppoly.ubcoef
        updated_ubcoef_nl += np.where(weights < 0, weights, 0) @ deeppoly.lbcoef
        updated_ubconst_nl = np.where(weights >= 0, weights, 0) @ deeppoly.ubconst
        updated_ubconst_nl += np.where(weights < 0, weights, 0) @ deeppoly.lbconst
        updated_ubconst_nl += biases

        updated_lbcoef_nl = np.where(weights >= 0, weights, 0) @ deeppoly.lbcoef
        updated_lbcoef_nl += np.where(weights < 0, weights, 0) @ deeppoly.ubcoef
        updated_lbconst_nl = np.where(weights >= 0, weights, 0) @ deeppoly.lbconst
        updated_lbconst_nl += np.where(weights < 0, weights, 0) @ deeppoly.ubconst
        updated_lbconst_nl += biases
        
        deeppoly.ubcoef = updated_ubcoef_nl
        deeppoly.ubconst = updated_ubconst_nl
        deeppoly.lbcoef = updated_lbcoef_nl
        deeppoly.lbconst = updated_lbconst_nl
        # np.save(f'deeppoly.ubcoef_{self.layer_num}_lg.npy', deeppoly.ubcoef)
        # print(f'deeppoly.ubcoef.shape: {deeppoly.ubcoef.shape}')

        deeppoly.ubs = np.where(deeppoly.ubcoef >= 0, deeppoly.ubcoef, 0) @ deeppoly.inputbounds[:, 1]
        deeppoly.ubs += np.where(deeppoly.ubcoef < 0, deeppoly.ubcoef, 0) @ deeppoly.inputbounds[:, 0]
        deeppoly.ubs += deeppoly.ubconst
        deeppoly.lbs = np.where(deeppoly.lbcoef >= 0, deeppoly.lbcoef, 0) @ deeppoly.inputbounds[:, 0]
        deeppoly.lbs += np.where(deeppoly.lbcoef < 0, deeppoly.lbcoef, 0) @ deeppoly.inputbounds[:, 1]
        deeppoly.lbs += deeppoly.lbconst


    def execute(self, state):
        '''execute on a concrete state
 
        returns output
        '''

        assert state.dtype == self.dtype, f"onnx subgraph dtype was {self.dtype}, execute() input was {state.dtype}"
        assert state.shape == self.input_shape, f"expected input shape {self.input_shape}, got {state.shape}"

        rv = self.sess.run(None, {self.input_name: state})[0]

        assert rv.shape == self.output_shape

        return rv

def find_node_with_input(graph, input_name, init_map=None):
    '''Find the next node on the main path that accepts input_name.

    For skip connections, a tensor can be consumed by two nodes: the main-path
    next node AND the skip-merge Add node.  When multiple consumers exist we
    return the non-skip-merge node (the main-path continuation).  A skip-merge
    Add is an Add node where BOTH inputs are activations (neither is an
    initializer).
    '''

    candidates = []
    for n in graph.node:
        if input_name in n.input:
            candidates.append(n)

    if len(candidates) == 0:
        return None
    elif len(candidates) == 1:
        return candidates[0]
    else:
        # Multiple consumers: skip-connection case.
        # Filter out skip-merge Add nodes (Add where both inputs are activations).
        def is_skip_merge_add(n):
            return (n.op_type == 'Add' and
                    init_map is not None and
                    all(inp not in init_map for inp in n.input))

        non_merge = [n for n in candidates if not is_skip_merge_add(n)]
        if len(non_merge) == 1:
            return non_merge[0]
        # If still ambiguous, fall back to first candidate (may error later)
        assert len(non_merge) > 0, \
            f"all consumers of '{input_name}' are skip-merge Adds — cannot determine main path"
        return non_merge[0]

def find_node_with_output(graph, output_name):
    'find the onnx node with the given output, can return None'

    for n in graph.node:
        for o in n.output:
            if o == output_name:
                return n

    return None

def _trace_skip_branch(graph, skip_inp, tensor_to_layer, init_map, max_depth=20):
    '''Trace backward from skip_inp to find the branch-start tensor.

    Returns (branch_start_tensor, nodes_in_forward_order) where:
      - branch_start_tensor is a tensor already in tensor_to_layer
      - nodes_in_forward_order is the ordered list of ONNX nodes on the skip
        path (from branch start to skip_inp)
    Returns None if tracing fails.
    '''
    chain = []  # (node, main_input_name, output_name)
    cur_out = skip_inp
    depth = 0

    while cur_out not in tensor_to_layer and depth < max_depth:
        producing_node = find_node_with_output(graph, cur_out)
        if producing_node is None:
            return None
        # Find the one non-initializer input (the "data" input)
        main_inputs = [inp for inp in producing_node.input if inp not in init_map]
        if len(main_inputs) != 1:
            return None  # multi-activation input, can't trace linearly
        chain.append((producing_node, main_inputs[0], cur_out))
        cur_out = main_inputs[0]
        depth += 1

    if cur_out not in tensor_to_layer:
        return None  # exceeded depth or couldn't find branch start

    chain.reverse()  # now forward order: branch_start → skip_inp
    nodes_forward = [c[0] for c in chain]
    return cur_out, nodes_forward


def _parse_skip_branch_nodes(nodes, init_map, branch_shape, onnx_type_float):
    '''Parse ONNX nodes on a skip branch (all linear) into nnenum layer objects.

    BatchNorm is folded into the preceding Conv/FC layer.
    Returns a list of layer objects, or raises RuntimeError if parsing fails.
    '''
    from nnenum.network import Convolutional2dLayer, FullyConnectedLayer, FlattenLayer, AddLayer

    skip_layers = []
    prev_shape = branch_shape
    i = 0

    while i < len(nodes):
        node = nodes[i]
        op = node.op_type
        layer_num = len(skip_layers)  # dummy index (not in main layers list)

        if op == 'Conv':
            weight_init = init_map.get(node.input[1])
            if weight_init is None or weight_init.data_type != onnx_type_float:
                raise RuntimeError(f"Skip branch Conv node has unsupported weight type")

            kernels = np.frombuffer(weight_init.raw_data, dtype='<f4').copy()
            kernel_shape = tuple(d for d in weight_init.dims)
            kernels = kernels.reshape(kernel_shape)

            if len(node.input) >= 3 and node.input[2] in init_map:
                bias_init = init_map[node.input[2]]
                biases = np.frombuffer(bias_init.raw_data, dtype='<f4').copy()
            else:
                biases = np.zeros(kernel_shape[0], dtype=np.float32)

            mode = 'same'
            strides = (1, 1)
            for attr in node.attribute:
                if attr.name == 'auto_pad':
                    auto_pad = attr.s.decode('utf-8')
                    if auto_pad == 'VALID':
                        mode = 'valid'
                elif attr.name == 'pads':
                    pads = list(attr.ints)
                    if all(p == 0 for p in pads):
                        mode = 'valid'
                elif attr.name == 'strides':
                    strides = tuple(attr.ints)

            layer = Convolutional2dLayer(layer_num, kernels, biases, prev_shape, mode=mode, strides=strides)

            # Fold following BatchNorm into this Conv
            if i + 1 < len(nodes) and nodes[i + 1].op_type == 'BatchNormalization':
                bn = nodes[i + 1]
                scale = np.frombuffer(init_map[bn.input[1]].raw_data, dtype='<f4')
                bn_bias = np.frombuffer(init_map[bn.input[2]].raw_data, dtype='<f4')
                mean = np.frombuffer(init_map[bn.input[3]].raw_data, dtype='<f4')
                var = np.frombuffer(init_map[bn.input[4]].raw_data, dtype='<f4')
                epsilon = 1e-5
                for attr in bn.attribute:
                    if attr.name == 'epsilon':
                        epsilon = attr.f
                bn_scale = scale / np.sqrt(var + epsilon)
                bn_shift = bn_bias - mean * bn_scale
                num_out_ch = len(layer.kernels)
                for out_c in range(num_out_ch):
                    for in_c in range(len(layer.kernels[out_c])):
                        layer.kernels[out_c][in_c] = layer.kernels[out_c][in_c] * bn_scale[out_c]
                layer.biases = bn_scale * layer.biases + bn_shift
                i += 1  # consumed the BN node

            skip_layers.append(layer)
            prev_shape = layer.get_output_shape()

        elif op == 'BatchNormalization':
            # Should have been consumed by the preceding Conv; skip standalone BN
            pass

        elif op == 'Gemm':
            weight_init = init_map.get(node.input[1])
            bias_init = init_map.get(node.input[2])
            if weight_init is None or bias_init is None:
                raise RuntimeError("Skip branch Gemm node missing weight/bias")
            b = np.frombuffer(weight_init.raw_data, dtype='<f4')
            shape = tuple(d for d in reversed(weight_init.dims))
            weight_mat = nn_unflatten(b, shape, order='F')
            b = np.frombuffer(bias_init.raw_data, dtype='<f4')
            shape2 = tuple(d for d in reversed(bias_init.dims))
            bias_vec = nn_unflatten(b, shape2, order='F')
            for a in node.attribute:
                if a.name == 'transB' and a.i == 1:
                    weight_mat = weight_mat.transpose().copy()
            layer = FullyConnectedLayer(layer_num, weight_mat, bias_vec, prev_shape)
            skip_layers.append(layer)
            prev_shape = layer.get_output_shape()

        elif op == 'MatMul':
            weight_init = init_map.get(node.input[1])
            if weight_init is None:
                raise RuntimeError("Skip branch MatMul missing weight")
            from nnenum.network import MatMulLayer
            b = np.frombuffer(weight_init.raw_data, dtype='<f4')
            shape = tuple(d for d in reversed(weight_init.dims))
            weight_mat = nn_unflatten(b, shape, order='F')
            layer = MatMulLayer(layer_num, weight_mat, prev_shape)
            skip_layers.append(layer)
            prev_shape = layer.get_output_shape()

        elif op == 'Flatten':
            layer = FlattenLayer(layer_num, prev_shape)
            skip_layers.append(layer)
            prev_shape = layer.get_output_shape()

        elif op in ['Add', 'Sub']:
            # Bias add on skip path (one input is initializer)
            inp0, inp1 = node.input[0], node.input[1]
            if inp0 in init_map or inp1 in init_map:
                bias_inp = inp0 if inp0 in init_map else inp1
                init = init_map[bias_inp]
                b = np.frombuffer(init.raw_data, dtype='<f4')
                shape = tuple(d for d in init.dims)
                b = nn_unflatten(b, shape, order='F')
                if op == 'Sub':
                    b = -1 * b
                layer = AddLayer(layer_num, b)
                skip_layers.append(layer)
                prev_shape = layer.get_output_shape()
            else:
                raise RuntimeError(f"Skip branch has an activation Add (not bias), unsupported")

        elif op == 'Relu':
            raise RuntimeError("Skip branch contains Relu — non-linear skip unsupported")

        else:
            raise RuntimeError(f"Unsupported op '{op}' on skip branch")

        i += 1

    return skip_layers


def parse_onnx_int_tensor(data_type, raw_data):
    'parse an ONNX integer tensor from raw bytes, returns numpy array'

    if data_type == 7:  # INT64
        return np.frombuffer(raw_data, dtype='<i8')  # little endian int64
    elif data_type == 2:  # INT32
        return np.frombuffer(raw_data, dtype='<i4')  # little endian int32
    else:
        assert False, f"Unsupported integer data type: {data_type}"

def convert_model_type_unused(model, from_type=onnx.TensorProto.FLOAT, to_type=onnx.TensorProto.DOUBLE, check_model=True):
    'convert a float32 model to a float64 one'

    assert from_type in [onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE]
    assert to_type in [onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE]

    # make a copy
    s = model.SerializeToString()
    model = onnx.ModelProto.FromString(s)

    # we must convert inputs, outputs, initiaizers, and nodes
    new_nodes = []
    new_inputs = []
    new_outputs = []
    new_init = []

    for inp in model.graph.input:
        if inp.type.tensor_type.elem_type == from_type:
            inp.type.tensor_type.elem_type = to_type
            
        new_inputs.append(inp)

    for out in model.graph.output:
        if out.type.tensor_type.elem_type == from_type:
            out.type.tensor_type.elem_type = to_type
            
        new_outputs.append(out)

    for node in model.graph.node:
    #    for a in node.attribute:
    #        print(f"attribute:\n{a}")
    #        if a.type == from_type:
    #            a.type = to_type
           
        new_nodes.append(node)

    for init in model.graph.initializer:
        if init.data_type == from_type:
            init.data_type = to_type

            from_dtype = '<f4' if from_type == onnx.TensorProto.FLOAT else '<f8'
            to_dtype = '<f4' if to_type == onnx.TensorProto.FLOAT else '<f8'
            
            # convert raw_data
            b = np.frombuffer(init.raw_data, dtype=from_dtype)
            init.raw_data = b.astype(np.dtype(to_dtype)).tobytes()

        new_init.append(init)

    graph = make_graph(new_nodes, model.graph.name, new_inputs,
                        new_outputs, new_init)

    onnx_model = make_model_with_graph(model, graph, check_model=check_model)

    return onnx_model

def load_onnx_network_optimized(filename):
    '''load an onnx network from a filename and return the NeuralNetwork object

    this has optimized implementation for linear transformations, but it supports less
    layer types than the non-optimized version
    '''

    model = onnx.load(filename)

    # Convert to opset 23 if needed (opset 25+ is not fully supported by onnxruntime)
    if any(opset.version > 23 for opset in model.opset_import):
        model = version_converter.convert_version(model, 23)

    onnx.checker.check_model(model)

    graph = model.graph

    #print(graph)

    # find the node with input "input"
    all_input_names = sum([[str(i) for i in n.input] for n in graph.node], [])

    #print(f"all input names: {all_input_names}")

    all_initializer_names = [i.name for i in graph.initializer]
    all_output_names = sum([[str(o) for o in n.output] for n in graph.node], [])

    # the input to the network is the one not in all_inputs_list and not in all_outputs_list
    network_input = None

    for i in set(all_input_names):  # Use set to avoid checking duplicates
        if i and i not in all_initializer_names and i not in all_output_names:  # Skip empty strings
            assert network_input is None, f"multiple onnx network inputs {network_input} and {i}"
            network_input = i

    assert network_input, "did not find onnx network input"

    assert len(graph.output) == 1, "onnx network defined multiple outputs"
    network_output = graph.output[0].name

    #print(f"input: '{network_input}', output: '{network_output}'")
    
    #assert network_input == graph.input[0].name, \
    #    f"network_input ({network_input}) != graph.input[0].name ({graph.input[0].name})"
    ##########

    # map names -> structs
    input_map = {i.name: i for i in graph.input}
    init_map = {i.name: i for i in graph.initializer}

    i = input_map[network_input]

    # find the node which takes the input (probably node 0)
    cur_node = find_node_with_input(graph, network_input, init_map)
    cur_input_name = network_input

    # ok! now proceed recursively
    layers = []

    # Map from ONNX output tensor name to the star_cache key that holds the
    # star AT that tensor.  key = len(layers) right after the producing layer
    # is appended, which equals the cur_layer at which _maybe_cache_star will
    # capture the star produced by that layer.
    tensor_to_layer = {network_input: 0}

    # Map from ONNX tensor name to its nnenum shape (tuple) at that tensor.
    # Used to determine the branch-point shape when tracing non-identity skip paths.
    # We'll populate it after computing prev_shape for the first node.
    tensor_to_shape = {}  # filled in as we parse

    # DAG predecessors: skip_add_layer_idx -> [skip_src_cache_key, main_src_layer_idx]
    dag_predecessors = {}

    # data types
    onnx_type_float = 1
    onnx_type_int = 2

    while cur_node is not None:
        op = cur_node.op_type

        # For Add/Sub with skip connections, cur_input_name may appear at index 1
        # instead of index 0.  Handle this before the general assertion.
        if op in ['Add', 'Sub'] and cur_node.input[0] != cur_input_name:
            assert cur_input_name in cur_node.input, \
                f"cur_input_name ({cur_input_name}) not in Add node inputs {list(cur_node.input)}"
        else:
            assert cur_node.input[0] == cur_input_name, \
                f"cur_node.input[0] ({cur_node.input[0]}) should be previous output " \
                f"({cur_input_name}) in node:\n{cur_node.name}"

        layer = None

        if layers:
            prev_shape = layers[-1].get_output_shape()
        else:
            s_node = graph.input[0].type.tensor_type.shape
            # ONNX format: (batch, channels, height, width) -> nnenum format: (height, width, channels)
            # Extract all dimensions (treating 0 or missing as 1)
            all_dims = tuple(d.dim_value if d.dim_value != 0 else 1 for d in s_node.dim)

            if len(all_dims) == 4:
                # Image input: (batch, channels, height, width) -> (height, width, channels)
                prev_shape = (all_dims[2], all_dims[3], all_dims[1])
            elif len(all_dims) == 2:
                # Flat input: (batch, features) -> (features,)
                prev_shape = (all_dims[1],)
            else:
                # Unexpected format, keep as-is (may fail later with better error message)
                prev_shape = all_dims
            # Record the network-input shape for skip-branch tracing
            tensor_to_shape[cur_input_name] = prev_shape
            
        if op in ['Add', 'Sub']:
            assert len(cur_node.input) == 2

            b = _tensor_to_numpy(init)
                # note shapes are not reversed here... acasxu input is 1, 1, 1, 5, but dim_value is 1, 1, 1, 5
            shape = tuple(d for d in init.dims) # note dims reversed, acasxu has 5, 50 but want 5 cols
            b = nn_unflatten(b, shape, order='F')
            # Determine which input is the "other" (not the main path activation).
            # For standard bias Add: one input is an initializer.
            # For skip-connection Add: both inputs are activations.
            inp0, inp1 = cur_node.input[0], cur_node.input[1]

            if inp0 in init_map or inp1 in init_map:
                # Bias add: one input is an initializer (constant bias)
                bias_inp = inp0 if inp0 in init_map else inp1
                init = init_map[bias_inp]
                assert init.data_type == onnx_type_float

                b = np.frombuffer(init.raw_data, dtype='<f4')
                shape = tuple(d for d in init.dims)
                b = nn_unflatten(b, shape, order='F')

                if op == 'Sub':
                    b = -1 * b

                layer = AddLayer(len(layers), b)
            else:
                # Skip-connection Add: both inputs are activations.
                # Identify which is the main-path (= cur_input_name) and which is skip.
                if inp1 == cur_input_name:
                    main_inp, skip_inp = inp1, inp0
                else:
                    main_inp, skip_inp = inp0, inp1

                skip_cache_key = tensor_to_layer.get(skip_inp)

                if skip_cache_key is not None:
                    # Identity (or previously-parsed) skip: branch point star is cached.
                    layer = SkipAddLayer(len(layers), input_shape=prev_shape)
                else:
                    # Non-identity skip: the skip tensor was produced by a side branch
                    # (e.g. a 1x1 Conv+BN projection in ResNets) that was not visited by
                    # the main-path parser.  Trace back, parse the side-branch nodes, and
                    # fold them into SkipAddLayer.skip_layers.
                    trace = _trace_skip_branch(graph, skip_inp, tensor_to_layer, init_map)
                    assert trace is not None, (
                        f"Skip-source tensor '{skip_inp}' not found in tensor_to_layer "
                        f"and skip branch could not be traced")

                    branch_start_tensor, skip_nodes = trace
                    skip_cache_key = tensor_to_layer[branch_start_tensor]
                    branch_shape = tensor_to_shape.get(branch_start_tensor, prev_shape)

                    skip_layers = _parse_skip_branch_nodes(
                        skip_nodes, init_map, branch_shape, onnx_type_float)

                    layer = SkipAddLayer(len(layers), input_shape=prev_shape,
                                        skip_layers=skip_layers,
                                        skip_branch_shape=branch_shape)

                # Record DAG: SkipAdd layer index -> [skip_src_cache_key, main_src_layer_idx]
                dag_predecessors[len(layers)] = [skip_cache_key, len(layers) - 1]
        elif op == 'Constant':
            # Extract constant value from node's 'value' attribute
            value_attr = None
            for attr in cur_node.attribute:
                if attr.name == 'value':
                    value_attr = attr
                    break

            assert value_attr is not None, "Constant node must have 'value' attribute"

            # Get tensor from attribute
            const_tensor = value_attr.t
            assert const_tensor.data_type == onnx_type_float, "Constant must be float type"

            # Extract value
            value = np.frombuffer(const_tensor.raw_data, dtype='<f4')  # little endian float32
            shape = tuple(d for d in const_tensor.dims)
            value = nn_unflatten(value, shape, order='F')

            layer = ConstantLayer(len(layers), value)
        elif op == 'Flatten':
            # axis attribute defaults to 1 per ONNX spec (flatten all dims after batch)
            axis = 1
            if cur_node.attribute:
                axis = cur_node.attribute[0].i
            assert axis == 1  # flatten along columns

            layer = FlattenLayer(len(layers), prev_shape)
            
        elif op == 'MatMul':
            assert len(cur_node.input) == 2
            init = init_map[cur_node.input[1]]
            
            assert init.data_type == onnx_type_float

            b = _tensor_to_numpy(init)
            shape = tuple(d for d in reversed(init.dims)) # note dims reversed, acasxu has 5, 50 but want 5 cols

            b = nn_unflatten(b, shape, order='F')

            layer = MatMulLayer(len(layers), b, prev_shape)
            
        elif op == 'Relu':
            assert layers, "expected previous layer before relu layer"
            
            layer = ReluLayer(len(layers), prev_shape)
        elif op == 'Gemm':
            assert len(cur_node.input) == 3
            
            weight_init = init_map[cur_node.input[1]]
            bias_init = init_map[cur_node.input[2]]

            # weight
            assert weight_init.data_type == onnx_type_float
            b = _tensor_to_numpy(weight_init)
            shape = tuple(d for d in reversed(weight_init.dims)) # note dims reversed, acasxu has 5, 50 but want 5 cols
            weight_mat = nn_unflatten(b, shape, order='F')

            # bias
            assert bias_init.data_type == onnx_type_float
            b = _tensor_to_numpy(bias_init)
            shape = tuple(d for d in reversed(bias_init.dims)) # note dims reversed, acasxu has 5, 50 but want 5 cols
            bias_vec = nn_unflatten(b, shape, order='F')

            for a in cur_node.attribute:
                assert a.name in ['alpha', 'beta', 'transA', 'transB'], "general Gemm node unsupported"

                if a.name in ['alpha', 'beta']:
                    assert a.f == 1.0
                    assert a.type == onnx_type_float
                elif a.name == 'transA':
                    assert a.i == 0, "transA=1 not supported"
                elif a.name == 'transB':
                    assert a.type == onnx_type_int
                    assert a.i == 1
                    weight_mat = weight_mat.transpose().copy()

            layer = FullyConnectedLayer(len(layers), weight_mat, bias_vec, prev_shape)
        elif op == 'Conv':
            # Conv has 2 or 3 inputs: data, weights, and optionally biases
            assert len(cur_node.input) in [2, 3], "Conv should have 2 or 3 inputs"

            # Get weights
            weight_init = init_map[cur_node.input[1]]
            assert weight_init.data_type == onnx_type_float

            # ONNX Conv weight shape: (out_channels, in_channels, kernel_h, kernel_w)
            kernels = _tensor_to_numpy(weight_init)
            kernel_shape = tuple(d for d in weight_init.dims)
            kernels = kernels.reshape(kernel_shape)

            # Get biases (if present)
            if len(cur_node.input) == 3:
                bias_init = init_map[cur_node.input[2]]
                assert bias_init.data_type == onnx_type_float
                biases = _tensor_to_numpy(bias_init)
            else:
                # No bias provided, use zeros
                biases = np.zeros(kernel_shape[0], dtype=np.float32)

            # Parse attributes to determine padding mode and stride
            mode = 'same'  # default
            auto_pad = None
            pads = None
            strides = (1, 1)  # default
            onnx_pads = None  # Explicit ONNX padding values

            for attr in cur_node.attribute:
                if attr.name == 'auto_pad':
                    auto_pad = attr.s.decode('utf-8')
                elif attr.name == 'pads':
                    pads = list(attr.ints)
                elif attr.name == 'strides':
                    strides = tuple(attr.ints)

            # Determine mode and padding
            if auto_pad == 'VALID' or (pads and all(p == 0 for p in pads)):
                mode = 'valid'
                onnx_pads = None  # No padding
            elif auto_pad in ['SAME_UPPER', 'SAME_LOWER']:
                mode = 'same'
                onnx_pads = None  # Use scipy's auto-padding for 'same'
            elif pads is not None and not all(p == 0 for p in pads):
                # Explicit non-zero padding from ONNX
                # ONNX pads format: [top, left, bottom, right] for 2D
                # or [top_begin, left_begin, top_end, left_end] in general
                if len(pads) == 4:
                    onnx_pads = (pads[0], pads[1], pads[2], pads[3])  # [top, left, bottom, right]
                else:
                    # Fallback to 'same' if padding format is unexpected
                    mode = 'same'
                    onnx_pads = None
            else:
                # auto_pad is None or 'NOTSET' and no explicit pads - default to 'same'
                mode = 'same'
                onnx_pads = None

            layer = Convolutional2dLayer(len(layers), kernels, biases, prev_shape,
                                        mode=mode, strides=strides, pads=onnx_pads)
        # ConvTranspose support disabled — falls back to general (onnxruntime) path.
        # Re-enable when conv optimizations are extended to ConvTranspose (see GOALS.md #2).
        # elif op == 'ConvTranspose':
        #     ...
        elif op == 'Reshape':
            assert len(cur_node.input) == 2, "Reshape should have 2 inputs (data and shape)"

            # Get the shape from the second input (can be initializer or Constant node)
            shape_input_name = cur_node.input[1]

            if shape_input_name in init_map:
                # Shape is from an initializer
                shape_init = init_map[shape_input_name]
                new_shape = parse_onnx_int_tensor(shape_init.data_type, shape_init.raw_data)
            else:
                # Shape might be from a Constant node
                const_node = find_node_with_output(graph, shape_input_name)
                assert const_node is not None, f"Reshape shape input '{shape_input_name}' not found"
                assert const_node.op_type == 'Constant', \
                    f"Reshape shape must be from initializer or Constant node, got {const_node.op_type}"

                # Extract value from Constant node's 'value' attribute
                value_attr = None
                for attr in const_node.attribute:
                    if attr.name == 'value':
                        value_attr = attr
                        break

                assert value_attr is not None, "Constant node must have 'value' attribute"
                shape_tensor = value_attr.t
                new_shape = parse_onnx_int_tensor(shape_tensor.data_type, shape_tensor.raw_data)

            new_shape = tuple(int(d) for d in new_shape)

            # Handle ONNX batch dimension and channel ordering
            # ONNX shapes for images: (batch, channels, height, width)
            # nnenum expects: (height, width, channels) - no batch dimension

            prev_size = 1
            for d in prev_shape:
                prev_size *= d

            # Check if new_shape has batch dimension (first dim is 1 or -1)
            if len(new_shape) >= 2 and new_shape[0] in (1, -1):
                # Calculate size without first dimension, treating -1 as a wildcard
                known_size = 1
                has_wildcard = False
                for d in new_shape[1:]:
                    if d == -1:
                        has_wildcard = True
                    else:
                        known_size *= d

                # Strip batch dim if:
                # - sizes match exactly (no wildcard), OR
                # - there's a wildcard and known dims divide prev_size (wildcard would cover the rest)
                if (not has_wildcard and known_size == prev_size) or \
                   (has_wildcard and known_size > 0 and prev_size % known_size == 0):
                    new_shape = new_shape[1:]

            # Keep ONNX CHW shape - ReshapeLayer.execute() will convert to HWC
            # (conversion must happen during reshape, not just by changing shape tuple)
            layer = ReshapeLayer(len(layers), new_shape, prev_shape)
        elif op == 'BatchNormalization':
            # Fold BatchNorm into previous Conv or FC layer when possible.
            # BatchNorm: y = scale * (x - mean) / sqrt(var + epsilon) + bias
            # This can be folded as: y = (scale / sqrt(var + eps)) * x + (bias - mean * scale / sqrt(var + eps))
            # When folding is not possible, create a standalone ScaleLayer.

            assert len(layers) > 0, "BatchNormalization cannot be the first layer"
            prev_layer = layers[-1]

            # BatchNorm inputs: X, scale, B, input_mean, input_var
            assert len(cur_node.input) == 5, f"BatchNorm should have 5 inputs, got {len(cur_node.input)}"

            scale_init = init_map[cur_node.input[1]]  # gamma
            bias_init = init_map[cur_node.input[2]]   # beta
            mean_init = init_map[cur_node.input[3]]   # running_mean
            var_init = init_map[cur_node.input[4]]    # running_var

            scale = _tensor_to_numpy(scale_init)
            bias = _tensor_to_numpy(bias_init)
            mean = _tensor_to_numpy(mean_init)
            var = _tensor_to_numpy(var_init)

            # Get epsilon from attributes (default 1e-5)
            epsilon = 1e-5
            for attr in cur_node.attribute:
                if attr.name == 'epsilon':
                    epsilon = attr.f

            # Compute BatchNorm scaling: scale / sqrt(var + epsilon)
            bn_scale = scale / np.sqrt(var + epsilon)
            bn_shift = bias - mean * bn_scale

            # Default: no new layer (folded case)
            layer = None

            if isinstance(prev_layer, Convolutional2dLayer):
                # For Conv: scale each output channel's kernel and bias
                # prev_layer.kernels is list[list[2D array]]: kernels[out_ch][in_ch] = 2D kernel
                # bn_scale/bn_shift shape: (out_channels,)

                num_out_channels = len(prev_layer.kernels)
                assert len(bn_scale) == num_out_channels, \
                    f"BatchNorm scale size {len(bn_scale)} doesn't match Conv output channels {num_out_channels}"

                # Scale kernels: create new scaled kernels (arrays are frozen/read-only)
                for out_c in range(num_out_channels):
                    for in_c in range(len(prev_layer.kernels[out_c])):
                        prev_layer.kernels[out_c][in_c] = prev_layer.kernels[out_c][in_c] * bn_scale[out_c]

                # kernels_array stores unflipped weights; scale each output channel row
                prev_layer.kernels_array = prev_layer.kernels_array * bn_scale[:, np.newaxis, np.newaxis, np.newaxis].astype(prev_layer.kernels_array.dtype)

                # Update biases: create new bias array (b_new = bn_scale * b_old + bn_shift)
                prev_layer.biases = bn_scale * prev_layer.biases + bn_shift

            elif isinstance(prev_layer, FullyConnectedLayer):
                # For FC: scale weights and biases
                # prev_layer.weights shape: (out_features, in_features)
                # bn_scale/bn_shift shape: (out_features,)

                num_out_features = prev_layer.weights.shape[0]
                assert len(bn_scale) == num_out_features, \
                    f"BatchNorm scale size {len(bn_scale)} doesn't match FC output features {num_out_features}"

                # Scale weights: each output row multiplied by its bn_scale
                prev_layer.weights = prev_layer.weights * bn_scale[:, np.newaxis]

                # Update biases: b_new = bn_scale * b_old + bn_shift
                prev_layer.biases = bn_scale * prev_layer.biases + bn_shift

            else:
                # Cannot fold into previous layer — create a standalone ScaleLayer instead.
                # Handles BatchNorm after Reshape, ReLU, Add, or any other non-parameterized layer.
                prev_output_shape = prev_layer.get_output_shape()

                if len(prev_output_shape) == 3:
                    # 3D HWC output: BN params are per-channel (C = last dim in HWC)
                    h, w, c = prev_output_shape
                    assert len(bn_scale) == c, \
                        f"BatchNorm scale size {len(bn_scale)} doesn't match output channels {c}"
                    # Broadcast per-channel values to full HWC-flat vector
                    flat_scale = np.broadcast_to(bn_scale, (h, w, c)).reshape(-1).astype(np.float64)
                    flat_shift = np.broadcast_to(bn_shift, (h, w, c)).reshape(-1).astype(np.float64)
                else:
                    # 1D or other flat output: BN params must match flat size
                    flat_size = int(np.prod(prev_output_shape))
                    assert len(bn_scale) == flat_size, \
                        f"BatchNorm scale size {len(bn_scale)} doesn't match output size {flat_size}"
                    flat_scale = bn_scale.astype(np.float64)
                    flat_shift = bn_shift.astype(np.float64)

                layer = ScaleLayer(len(layers), flat_scale, flat_shift, prev_output_shape)
        elif op == 'Dropout':
            # Dropout is identity at inference time — no layer needed, just pass through
            layer = None
        else:
            assert False, f"unsupported onnx op_type {op} in node {cur_node.name}"

        # Only append if we created a new layer (BatchNorm doesn't create a layer)
        if layer is not None:
            layers.append(layer)

        assert len(cur_node.output) == 1, f"multiple output at onnx node {cur_node.name}"
        cur_input_name = cur_node.output[0]

        # Record the star_cache key for this output tensor.
        # star_cache[len(layers)] will hold the star AT cur_input_name when it
        # is populated by _maybe_cache_star at the start of the next layer.
        # For BatchNorm (layer=None), len(layers) hasn't changed, so the BN
        # output maps to the same cache key as the preceding Conv/FC output.
        tensor_to_layer[cur_input_name] = len(layers)

        # Track the nnenum shape at this tensor for skip-branch parsing.
        if layers:
            tensor_to_shape[cur_input_name] = layers[-1].get_output_shape()

        cur_node = find_node_with_input(graph, cur_input_name, init_map)

    assert cur_input_name == network_output, \
        f"output without node {cur_input_name} is not network output {network_output}"

    return NeuralNetwork(layers, dag_predecessors=dag_predecessors)

def stan_select_model_inputs_outputs(model, dtype, inputs, outputs, io_shapes):
    """
    a modificiation of select_model_input_outputs from sklearn-on

    Takes a model and changes its inputs and outputs
    :param model: *ONNX* model
    :param inputs: new inputs
    :return: modified model
    The function removes unneeded nodes.
    """

    if dtype == np.float32:
        elem_type = onnx.TensorProto.FLOAT
    else:
        assert dtype == np.float64
        elem_type = onnx.TensorProto.DOUBLE
    
    
    if inputs is None:
        raise NotImplementedError("Parameter inputs cannot be empty.")
    if outputs is None:
        raise NotImplementedError("Parameter inputs cannot be empty.")
    
    if not isinstance(inputs, list):
        inputs = [inputs]

    if not isinstance(outputs, list):
        outputs = [outputs]

    ##########

    mark_var = {} # keys are (input or node output) names, vals 1 = keep, 0 = delete
    
    for out in enumerate_model_node_outputs(model):
        mark_var[out] = 0
        
    for inp in model.graph.input:
        mark_var[inp.name] = 0

    for out in outputs:
        if out not in mark_var:
            raise ValueError("Desired Output '{}' not found in model.".format(out))

    initializers = [i.name for i in model.graph.initializer]
        
    for inp in inputs:
        if inp not in mark_var:
            raise ValueError("Desired Input '{}' not found in model.".format(inp))

        if inp not in initializers:
            mark_var[inp] = 1

    nodes = list(enumerate(model.graph.node))
    
    mark_op = {} # these are the marks for the node indices, 1 = keep, 0 = delete
    for node in nodes:
        mark_op[node[0]] = 0

    # We mark all the nodes we need to keep.
    nb = 1 # number marked... used as a termination condition

    keep_initializers = []

    while nb > 0:
        nb = 0

        for index, node in nodes:
            
            if mark_op[index] == 1: # node was already processed, skip
                continue
            
            mod = False # is this a newly-marked node?

            node_initializers = []
            
            for inp in node.input:
                if inp in outputs:
                    continue
                
                if not inp in mark_var or mark_var.get(inp, 0) == 0:
                    node_initializers.append(inp) # was initializer
                elif mark_var[inp] == 1:
                    # make the node because its input was marked
                    mark_op[index] = 1
                    mod = True

            for out in node.output:
                if out in inputs:
                    continue
                
                if mark_var[out] == 1:
                    # mark the node because the output was marked
                    mark_op[index] = 1
                    mod = True
                
            if not mod: # none of the node's inputs were marked, skip it
                continue

            keep_initializers += node_initializers

            nb += 1 # mark the node and all its inputs / outputs
            
            for out in node.output:
                if mark_var.get(out, 0) == 1:
                    continue
                
                if out in outputs:
                    continue

                mark_var[out] = 1
                nb += 1

            for inp in node.input:
                if mark_var.get(inp, 0) == 1:
                    continue
                
                if inp in inputs:
                    continue

                mark_var[inp] = 1
                nb += 1

    # All nodes verifies mark_op[node.name] == 1
    keep_nodes = [node[1] for node in nodes if mark_op[node[0]] == 1]

    var_in = []
    for inp in inputs:
        nt = onnx.TypeProto()
        nt.tensor_type.elem_type = elem_type

        # inputs need shape info, which is not in the graph!
        shape = io_shapes[inp]

        for s in shape:
            nt.tensor_type.shape.dim.add()
            nt.tensor_type.shape.dim[-1].dim_value = s

        value_info = ValueInfoProto(type=nt)
        value_info.name = inp
        
        var_in.append(value_info)

    # add initializers to inputs
    for i in model.graph.input:
        if i.name in keep_initializers:
            var_in.append(i)

    var_out = []
    for out in outputs:
        nt = onnx.TypeProto()
        nt.tensor_type.elem_type = elem_type

        # inputs need shape info, which is not in the graph!
        shape = io_shapes[out]

        for s in shape:
            nt.tensor_type.shape.dim.add()
            nt.tensor_type.shape.dim[-1].dim_value = s

        value_info = ValueInfoProto(type=nt)
            
        value_info.name = out
        var_out.append(value_info)
            

    init_out = [init for init in model.graph.initializer if init.name in keep_initializers] 

    graph = make_graph(keep_nodes, model.graph.name, var_in,
                       var_out, init_out)

    #print(f"making model with inputs {inputs} / outputs {outputs} and nodes len: {len(keep_nodes)}")
    # Skip SSA validation for submodels - ResNets have skip connections where outputs
    # are used by multiple downstream nodes, which the strict SSA checker rejects
    onnx_model = make_model_with_graph(model, graph, check_model=False)

    return onnx_model

def extract_ordered_relus(model, start):
    '''extract the relu nodes in a topological order

    returns an ordered list of relu nodes (from model.graph.node)
    '''

    relu_nodes = []
    marked_values = [start]
    marked_nodes = []

    modified = True

    while modified:
        modified = False

        for index, node in enumerate(model.graph.node):

            # node was already processed
            if index in marked_nodes:
                continue

            should_process = False

            for inp in node.input:
                if inp in marked_values:
                    should_process = True
                    break

            # none of the node's inputs were marked
            if not should_process:
                continue

            # process the node!
            modified = True
            marked_nodes.append(index)

            if node.op_type == 'Relu':
                relu_nodes.append(node)

            for out in node.output:
                if out in marked_values:
                    continue

                marked_values.append(out)

    return relu_nodes

def extract_ordered_split_nodes(model, start):
    '''extract nodes that should split the network (ReLU, Add, etc.) in topological order

    For ResNets, Add nodes merge skip connections and must be treated as split points
    to avoid creating submodels with multiple inputs.

    returns an ordered list of (node, node_type) tuples
    '''

    split_ops = {'Relu', 'Add'}  # Operations that split the network into sections
    split_nodes = []
    marked_values = [start]
    marked_nodes = []

    modified = True

    while modified:
        modified = False

        for index, node in enumerate(model.graph.node):

            # node was already processed
            if index in marked_nodes:
                continue

            should_process = False

            for inp in node.input:
                if inp in marked_values:
                    should_process = True
                    break

            # none of the node's inputs were marked
            if not should_process:
                continue

            # process the node!
            modified = True
            marked_nodes.append(index)

            if node.op_type in split_ops:
                split_nodes.append((node, node.op_type))

            for out in node.output:
                if out in marked_values:
                    continue

                marked_values.append(out)

    return split_nodes

def make_model_with_graph(model, graph, check_model=True):
    'copy a model with a new graph'

    onnx_model = make_model(graph)
    onnx_model.ir_version = model.ir_version
    onnx_model.producer_name = model.producer_name
    onnx_model.producer_version = model.producer_version
    onnx_model.domain = model.domain
    onnx_model.model_version = model.model_version
    onnx_model.doc_string = model.doc_string
    
    if len(model.metadata_props) > 0:
        values = {p.key: p.value for p in model.metadata_props}
        onnx.helper.set_model_props(onnx_model, values)

    #if len(onnx_model.graph.input) != len(model.graph.input):
    #    raise RuntimeError("Input mismatch {} != {}".format(
    #                    len(onnx_model.input), len(model.input)))

    # fix opset import
    for oimp in model.opset_import:
        op_set = onnx_model.opset_import.add()
        op_set.domain = oimp.domain
        op_set.version = oimp.version

    #print(". making model -------------")
    #onnx.save_model(onnx_model, 'temp.onnx')
    #with open('temp.txt', 'w') as f:
    #    f.write(str(graph))

    if check_model:
        onnx.checker.check_model(onnx_model, full_check=True)

    return onnx_model

def get_io_shapes(model):
    """returns map io_name -> shape"""

    rv = {}

    intermediate_outputs = list(enumerate_model_node_outputs(model))

    initializers = [i.name for i in model.graph.initializer]
    inputs = [i for i in model.graph.input if i.name not in initializers]
    assert len(inputs) == 1

    t = inputs[0].type.tensor_type.elem_type
    assert t == onnx.TensorProto.FLOAT
    dtype = np.float32

    if dtype == np.float32:
        elem_type = onnx.TensorProto.FLOAT
    else:
        assert dtype == np.float64
        elem_type = onnx.TensorProto.DOUBLE

    # create inputs as zero tensors
    input_map = {}

    for inp in inputs:            
        shape = tuple(d.dim_value if d.dim_value != 0 else 1 for d in inp.type.tensor_type.shape.dim)
        
        input_map[inp.name] = np.zeros(shape, dtype=dtype)

        # also save it's shape
        rv[inp.name] = shape

    new_out = []

    # add all old outputs
    for out in model.graph.output:
        new_out.append(out)
        
    for out_name in intermediate_outputs:
        if out_name in rv: # inputs were already added
            continue

        # create new output
        #nt = onnx.TypeProto()
        #nt.tensor_type.elem_type = elem_type

        value_info = ValueInfoProto()
        value_info.name = out_name
        new_out.append(value_info)

    # ok run once and get all outputs
    graph = make_graph(model.graph.node, model.graph.name, model.graph.input,
                       new_out, model.graph.initializer)

    # this model is not a valud model since the outputs don't have shape type info...
    # but it still will execute! skip the check_model step
    new_onnx_model = make_model_with_graph(model, graph, check_model=False)

    # Ensure opset is 23 (make_model_with_graph may create inconsistent state)
    for opset in new_onnx_model.opset_import:
        if opset.domain in ['', 'ai.onnx'] and opset.version > 23:
            opset.version = 23

    sess = ort.InferenceSession(new_onnx_model.SerializeToString())

    res = sess.run(None, input_map)
    names = [o.name for o in sess.get_outputs()]
    out_map = {name: output for name, output in zip(names, res)}

    for out_name in intermediate_outputs:
        if out_name in rv: # inputs were already added
            continue

        rv[out_name] = out_map[out_name].shape
        
    return rv

def remove_unused_initializers(model):
    'return a modified model'

    new_init = []

    for init in model.graph.initializer:
        found = False
        
        for node in model.graph.node:
            for i in node.input:
                if init.name == i:
                    found = True
                    break

            if found:
                break

        if found:
            new_init.append(init)

    graph = make_graph(model.graph.node, model.graph.name, model.graph.input,
                        model.graph.output, new_init)

    onnx_model = make_model_with_graph(model, graph)

    return onnx_model
    

def load_onnx_network(filename):
    '''load an onnx network from a filename and return the NeuralNetwork object

    This newer version will use the onnx runtime to execute all linear layers,
    spliting the network into parts on the relu layers
    '''

    model = onnx.load(filename)

    # Convert to opset 23 if needed (opset 25+ is not fully supported by onnxruntime)
    if any(opset.version > 23 for opset in model.opset_import):
        model = version_converter.convert_version(model, 23)

    onnx.checker.check_model(model)

    passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
    
    model = remove_unused_initializers(model)
    onnx.checker.check_model(model)

    io_shapes = get_io_shapes(model)

    dtype = np.float32

    initializers = [i.name for i in model.graph.initializer]
    inputs = [i.name for i in model.graph.input if i.name not in initializers]
    
    assert len(inputs) == 1, f"expected one input in onnx model, got {inputs}"
    assert len(model.graph.output) == 1, "expected single output in onnx model"

    graph = model.graph

    for node in graph.node:
        o = node.op_type
        assert o not in Settings.ONNX_BLACKLIST, f"Onnx model contains node with op {o}, which is unsupported. " + \
          "Updated Settings.BLACKLIST if you want to override this."

        assert o in Settings.ONNX_WHITELIST, f"Onnx model contains node with op {o}, which may not be a linear operation. " + \
          "Updated Settings.WHITELIST if you want to override this."
    
    # check to see if only linear nodes are being used
    relus = extract_ordered_relus(model, inputs[0])
    assert relus, "expected at least one relu layer in network"
    
    # split the network input parts along the relus
    prev_input = inputs[0]

    layers = []

    while relus:
        r = relus[0]

        assert r.op_type == 'Relu'
        assert len(r.input) == 1 and len(r.output) == 1
        end_node = r.input[0]

        # in case network starts with relu or there are two relus in a row
        if end_node != prev_input:
            submodel = stan_select_model_inputs_outputs(model, dtype, prev_input, end_node, io_shapes)

            l = LinearOnnxSubnetworkLayer(len(layers), submodel)
            layers.append(l)

        end_shape = io_shapes[end_node]
        l = ReluLayer(len(layers), end_shape)
        layers.append(l)
        
        prev_input = r.output[0]

        # next!
        relus = relus[1:]

    # done with relus... do last subnetwork
    if prev_input != model.graph.output[0].name:
        end_node = model.graph.output[0].name

        submodel = stan_select_model_inputs_outputs(model, dtype, prev_input, end_node, io_shapes)

        l = LinearOnnxSubnetworkLayer(len(layers), submodel)
        layers.append(l)

    return NeuralNetwork(layers)
