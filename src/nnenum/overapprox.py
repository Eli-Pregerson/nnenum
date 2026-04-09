'''
Overapproximation analysis
'''

import time
import numpy as np

from nnenum.settings import Settings
from nnenum.timerutil import Timers
from nnenum.util import Freezable
from nnenum.prefilter import update_bounds_lp, sort_splits
from nnenum.specification import DisjunctiveSpec
from nnenum.network import ReluLayer, FullyConnectedLayer, MatMulLayer, AddLayer, \
    FlattenLayer, ReshapeLayer, Convolutional2dLayer, nn_flatten, nn_unflatten, SkipAddLayer


def try_ibp(init_box, network, spec):
    '''Interval Bound Propagation precheck.

    Propagates just two vectors (lb, ub) through the network — O(neurons) memory,
    no generator matrix. If all output constraints are provably satisfied under the
    interval overapproximation, returns True (proven safe). Otherwise returns False
    (inconclusive — fall through to star enumeration).

    Effective when epsilon is tiny, e.g. VGGNet specs with eps=2e-7 to 2e-5.
    '''

    init_box = np.array(init_box, dtype=np.float32)
    lb = init_box[:, 0]
    ub = init_box[:, 1]

    for layer in network.layers:
        if isinstance(layer, ReluLayer):
            lb = np.maximum(0, lb)
            ub = np.maximum(0, ub)
        elif isinstance(layer, FullyConnectedLayer):
            W, b = layer.weights, layer.biases
            lb_new = np.where(W > 0, W, 0) @ lb + np.where(W < 0, W, 0) @ ub + b
            ub_new = np.where(W > 0, W, 0) @ ub + np.where(W < 0, W, 0) @ lb + b
            lb, ub = lb_new, ub_new
        elif isinstance(layer, MatMulLayer):
            W = layer.mat
            lb_new = np.where(W > 0, W, 0) @ lb + np.where(W < 0, W, 0) @ ub
            ub_new = np.where(W > 0, W, 0) @ ub + np.where(W < 0, W, 0) @ lb
            lb, ub = lb_new, ub_new
        elif isinstance(layer, AddLayer):
            lb = lb + layer.vec.ravel()
            ub = ub + layer.vec.ravel()
        elif isinstance(layer, (FlattenLayer, ReshapeLayer)):
            pass  # lb/ub stay flat vectors
        elif isinstance(layer, Convolutional2dLayer):
            shape = layer.get_input_shape()
            mid = (lb + ub) / 2
            rad = (ub - lb) / 2
            mid_out = nn_flatten(layer.execute(nn_unflatten(mid, shape)))
            orig_k = layer.kernels.copy()
            orig_b = layer.biases.copy()
            layer.kernels = np.abs(orig_k)
            layer.biases = np.zeros_like(orig_b)
            rad_out = nn_flatten(layer.execute(nn_unflatten(rad, shape)))
            layer.kernels = orig_k
            layer.biases = orig_b
            lb = mid_out - rad_out
            ub = mid_out + rad_out
        # other layer types (ConstantLayer etc.) are rare; skip and stay conservative

    # Check each Specification in the DisjunctiveSpec (or a single Specification).
    # A Specification is violated if ALL rows of (mat @ output <= rhs) hold simultaneously.
    # IBP proves a Specification cannot be violated if for at least one row,
    # the minimum of (mat @ output) over [lb, ub] exceeds rhs — making that row impossible.
    # Safe overall = every Specification in spec_list cannot be violated.
    from nnenum.specification import DisjunctiveSpec, Specification
    if isinstance(spec, DisjunctiveSpec):
        spec_list = spec.spec_list
    else:
        assert isinstance(spec, Specification)
        spec_list = [spec]
    for single_spec in spec_list:
        mat = single_spec.mat.astype(np.float32)
        rhs = single_spec.rhs
        # minimum of each row of mat @ output over [lb, ub]
        min_vals = np.where(mat > 0, mat, 0) @ lb + np.where(mat < 0, mat, 0) @ ub
        # if any row's min > rhs, that row can never be satisfied → spec cannot be violated
        if not np.any(min_vals > rhs + 1e-6):
            return False  # cannot rule out this spec being violated → inconclusive

    return True  # every spec proven impossible → safe

def try_ibp_from_bounds(lb, ub, network, cur_layer, spec):
    '''Run IBP from mid-network interval bounds through remaining layers.

    lb, ub: per-neuron float32 arrays at the output of layer cur_layer - 1.
    Returns True if proven safe, False if inconclusive.
    '''
    import types
    init_box = np.stack([lb, ub], axis=1)  # (N, 2)
    partial_network = types.SimpleNamespace()
    partial_network.layers = network.layers[cur_layer:]
    return try_ibp(init_box, partial_network, spec)

def try_quick_overapprox(ss, network, spec, start_time):
    'try a quick overapproximation, return is_safe, concrete_io_tuple'

    Timers.tic('try_quick_overapprox')
    
    overapprox_types = Settings.QUICK_OVERAPPROX_TYPES

    def check_cancel_func():
        'worker cancel func. can raise OverapproxCanceledException'

        diff = time.perf_counter() - start_time

        if diff > Settings.TIMEOUT:
            raise OverapproxCanceledException('timeout exceeded')
        
    try:
        check_cancel_func()
        
        prerelu_sims = make_prerelu_sims(ss, network)

        check_cancel_func()

        if Settings.PRINT_OUTPUT and Settings.PRINT_OVERAPPROX_OUTPUT:
            print(f"Doing quick overapprox with {len(overapprox_types)} rounds...")
        
        rr = do_overapprox_rounds(ss, network, spec, prerelu_sims, check_cancel_func=check_cancel_func,
                                  overapprox_types=overapprox_types)

        rv = rr.is_safe, rr.concrete_io_tuple
    except OverapproxCanceledException as e:
        if Settings.PRINT_OUTPUT:
            print(f"Overapprox canceled ({e})")
        rv = False, None

    Timers.toc('try_quick_overapprox')

    return rv

def make_prerelu_sims(ss, network):
    '''compute the prerelu simulation values at each remaining layer

    this only saves the state for the remaining layers, before relu is executed
    the output of the network is stored at index len(network.layers)

    returns a dict, layer_num -> sim_vector
    '''

    if ss.prefilter.simulation is None:
        rv = None
    else:
        rv = {}

        state = ss.prefilter.simulation[1].copy()
        layer_num = ss.cur_layer

        if layer_num < len(network.layers):
            layer = network.layers[layer_num]
            rv[layer_num] = state

            # current layer may be partially processed
            if isinstance(layer, ReluLayer):
                state = np.clip(state, 0, np.inf)

        while layer_num + 1 < len(network.layers):
            layer_num += 1

            layer = network.layers[layer_num]
            rv[layer_num] = state

            if isinstance(layer, SkipAddLayer) and network.dag_predecessors:
                # SkipAdd: combine skip-path simulation with current (main-path) simulation
                skip_cache_key = network.dag_predecessors[layer_num][0]
                skip_sim = ss.prefilter.simulation_cache.get(skip_cache_key)
                if skip_sim is not None:
                    state = state + skip_sim
                # else: skip sim not available, leave state unchanged (conservative)
            elif not isinstance(layer, SkipAddLayer):
                shape = layer.get_input_shape()
                input_tensor = nn_unflatten(state, shape).astype(ss.star.a_mat.dtype)
                output_tensor = layer.execute(input_tensor)
                state = nn_flatten(output_tensor)

        # save final output
        rv[len(network.layers)] = state

    return rv

def check_round(ss, sets, spec_arg, check_cancel_func=None):
    '''check overapproximation result of one round against spec

    this may modify ss.safe_spec_list is part of the spec is proven as safe

    This returns is_safe?, violation_stars, violation_indices
    '''

    Timers.tic('overapprox_check_round')

    if check_cancel_func is None:
        check_cancel_func = lambda: False
    
    whole_safe = True

    unsafe_violation_stars = [] # list of violation stars for each part of the disjunctive spec
    unsafe_violation_indices = [] # index in spec_list

    # break it apart disjunctive specs, as quicker overapproximation may work for some parts and not others
    spec_list = spec_arg.spec_list if isinstance(spec_arg, DisjunctiveSpec) else [spec_arg]

    for i, single_spec in enumerate(spec_list):

        if ss.safe_spec_list is not None and ss.safe_spec_list[i]:
            continue
        
        single_safe = False

        violation_star = None
        
        for s in sets:
            single_safe = s.check_spec(single_spec, check_cancel_func)

            if isinstance(s, StarOverapprox) and not single_safe:
                violation_star = s.violation_star

            if single_safe:
                if ss.safe_spec_list is not None:
                    ss.safe_spec_list[i] = True
                
                break # done with this spec!

        if not single_safe:
            whole_safe = False

            if violation_star is not None:
                unsafe_violation_stars.append(violation_star)
                unsafe_violation_indices.append(i)

            # just need one violation star
            break

    Timers.toc('overapprox_check_round')

    return whole_safe, unsafe_violation_stars, unsafe_violation_indices

class RoundsResult:
    'result of do_overapprox_rounds'

    def __init__(self):

        self.is_safe = False
        self.round_generators = [] # list of lists for each round
        self.round_ms = [] # ms for each round
        self.concrete_io_tuple = None

    def __str__(self):

        if Settings.SAVE_BRANCH_TUPLES_TIMES:
            rv = ", ".join([f"{max(r)} ({round(ms, 1)} ms)" for r, ms in zip(self.round_generators, self.round_ms)])
        else:
            rv = ", ".join([f"{max(r)}" for r in self.round_generators])
        
        # comma seperated for each round
        # {round(diff * 1000, 1)}
        # {round_max_gens} (100 ms)
        
        return rv

    def get_max_gens(self):
        'get the maximum number of generators from all the reprsentations'

        rv = -np.inf

        for gen_list in self.round_generators:
            rv = max(rv, max(gen_list))

        return rv

def test_abstract_violation(dims, vstars, vindices, network, spec):
    '''test concrete executions for specification violations

    returns abstract_ios, (concrete_io_tuple or None)
    '''
    
    concrete_io_tuple = None
    abstract_ios = []

    Timers.tic('try_abstract_violation')

    for vstar, vindex in zip(vstars, vindices):
        if isinstance(spec, DisjunctiveSpec):
            cur_spec = spec.spec_list[vindex]
        else:
            cur_spec = spec

        # try all rows
        # rows = cur_spec.mat

        # try sum of all rows
        rows = []

        sum_row = np.zeros(cur_spec.mat.shape[1])

        for row in cur_spec.mat:
            sum_row += row

        rows.append(sum_row)
        
        for row in rows:
            # this one is almost free since objective direction is None
            cinput, coutput = vstar.minimize_vec(None, return_io=True)
            assert cur_spec.is_violation(coutput, tol_rhs=1e-4)

            trimmed_input = cinput[:dims]
            
            full_input = vstar.to_full_input(trimmed_input)
            exec_output = network.execute(full_input)
            flat_output = np.ravel(exec_output)

            if cur_spec.is_violation(flat_output):
                if Settings.PRINT_OUTPUT:
                    print("Found unsafe from first concrete execution of abstract counterexample")

                concrete_io_tuple = (full_input, flat_output)
                break

            # this one is worst violation, use row as objective function
            cinput, coutput = vstar.minimize_vec(row, return_io=True)
            assert cur_spec.is_violation(coutput, tol_rhs=1e-4)
            
            abstract_ios.append((cinput, coutput))

            trimmed_input = cinput[:dims]
            full_input = vstar.to_full_input(trimmed_input)
            exec_output = network.execute(full_input)
            flat_output = np.ravel(exec_output)

            if cur_spec.is_violation(flat_output):
                if Settings.PRINT_OUTPUT:
                    print("Found unsafe from second concrete execution of abstract counterexample")

                concrete_io_tuple = (full_input, flat_output)
                break

        if concrete_io_tuple is not None:
            break

    Timers.toc('try_abstract_violation')

    return abstract_ios, concrete_io_tuple
        
def do_overapprox_rounds(ss, network, spec, prerelu_sims, check_cancel_func=None, gen_limit=np.inf,
                         overapprox_types=None):
    '''do the multi-round overapproximation analysis

    returns an instance of RoundsResult:
    'is_safe' -> bool
    'branch_label' -> list of 2-tuples for each round (max_generators [int], milliseconds [int])
    'max_gens' -> int

    returns (is_safe, branch_label. max_gens) 
    '''

    if overapprox_types is None:
        overapprox_types = Settings.OVERAPPROX_TYPES

    rv = RoundsResult()

    first_round = True
    sets = []

    for round_num, types in enumerate(overapprox_types):
        assert isinstance(types, list), f"types was not list: {types}"
        sets.clear()

        for type_str in types:
            if type_str.startswith('zono.'):
                z = ZonoOverapprox(ss, type_str, gen_limit)
                sets.append(z)
            elif type_str.startswith('deeppoly.'):
                z = DeeppolyOverapprox(ss, type_str, gen_limit)
                sets.append(z)
            else:
                assert type_str.startswith('star.'), f"unknown type_str: {type_str}"
                s = StarOverapprox(ss, type_str, gen_limit)
                sets.append(s)

        start = time.perf_counter()

        if not ss.branch_tuples and Settings.PRINT_OUTPUT and Settings.PRINT_OVERAPPROX_OUTPUT:
            print(f"Overapprox Round {round_num+1}/{len(overapprox_types)} has {len(sets)} set(s)")

        try:
            if ss.cur_layer < len(network.layers):
                run_overapprox_round(network, ss, sets, prerelu_sims, check_cancel_func)
            
            diff = time.perf_counter() - start if Settings.SAVE_BRANCH_TUPLES_TIMES else 0
        except OverapproxCanceledException as e:
            diff = time.perf_counter() - start if Settings.SAVE_BRANCH_TUPLES_TIMES else 0

            msg = f"canceled after {round(diff * 1000, 1)} ms"

            raise OverapproxCanceledException(f"{e}; {rv}, {msg}")

        gens = [s.get_num_gens() for s in sets]
        rv.round_generators.append(gens)
        rv.round_ms.append(diff * 1000)

        start = time.perf_counter()
        rv.is_safe, vstars, vindices = check_round(ss, sets, spec, check_cancel_func)

        if rv.is_safe:
            break

        if vstars:
            dims = ss.star.lpi.get_num_cols()
                
            _abstract_ios, rv.concrete_io_tuple = test_abstract_violation(dims, vstars, vindices, network, spec)

        if first_round:
            first_round = False
        
    return rv

def run_overapprox_round(network, ss_init, sets, prerelu_sims, check_cancel_func=None):
    '''
    run overapproximation analysis through the network (a single round with multiple sets)

    ss_init - the exact star set, at a split point in the network
    sets - a list of overapproximation set representations like ZonoOverapprox or StarOverapprox
    check_cancel_func - a function that can be called for long operations which may raise a OverapproxCanceledException

    this modifies the passed-in sets in place
    '''

    if check_cancel_func is None:
        check_cancel_func = lambda: False

    layer_num = ss_init.cur_layer
    depth = len(ss_init.branch_tuples)

    # precondition is that ss_init is at a split in the network
    assert layer_num < len(network.layers)
    assert isinstance(network.layers[layer_num], ReluLayer)
    assert ss_init.prefilter.output_bounds is not None
    assert ss_init.prefilter.output_bounds.branching_neurons.size > 0
    assert len(sets) > 0, "need at least one type of overapproximation set"

    split_indices = ss_init.prefilter.output_bounds.branching_neurons
    zero_indices = np.array([], dtype=int) # no zero assignments needed (already done eagerly)
    layer_bounds = ss_init.prefilter.output_bounds.layer_bounds

    # run first layer with existing bounds
    for s in sets:
        s.execute_with_bounds(layer_num, layer_bounds, split_indices, zero_indices)

    layer_num += 1

    # Determine which layer indices are skip sources needed by downstream SkipAddLayers
    # that will be reached during this round (i.e., beyond the starting ReLU layer).
    # These won't be in ss_init's caches (propagation stopped at ss_init.cur_layer),
    # so we must save intermediate overapprox states as we pass through them.
    skip_source_layers = set()
    for l_idx in range(layer_num, len(network.layers)):
        l = network.layers[l_idx]
        if isinstance(l, SkipAddLayer) and l_idx in network.dag_predecessors:
            src = network.dag_predecessors[l_idx][0]
            if src >= layer_num:
                skip_source_layers.add(src)

    # Cache: layer_num -> list of state copies (one per s in `sets`), saved BEFORE
    # that layer is applied.  Used by downstream SkipAddLayers in this round.
    overapprox_skip_states = {}

    # run remaining layers with newly-computed bounds
    remaining_layers = network.layers[layer_num:]

    for layer_index, layer in enumerate(remaining_layers):
        check_cancel_func()

        # Save overapprox state BEFORE processing this layer if it is a skip source.
        if layer_num in skip_source_layers:
            overapprox_skip_states[layer_num] = [s.get_skip_state_copy() for s in sets]

        if not ss_init.branch_tuples and Settings.PRINT_OUTPUT and Settings.PRINT_OVERAPPROX_OUTPUT:
            layer_start = time.perf_counter()
            extra = ''

            if isinstance(sets[0], ZonoOverapprox):
                extra = f' (zono shape: {sets[0].zono.mat_t.shape})'

            print(f"Layer {layer_index + 1}/{len(remaining_layers)}: {type(layer).__name__}{extra}...", end='', flush=True)

        if isinstance(layer, ReluLayer):
            sim = None if prerelu_sims is None else prerelu_sims[layer_num]
            split_indices = None
            layer_bounds = None

            for s in sets:
                if not ss_init.branch_tuples and Settings.PRINT_OUTPUT and layer_bounds is not None \
                                             and Settings.PRINT_OVERAPPROX_OUTPUT:
                    if isinstance(s, StarOverapprox) and s.do_lp:
                        print(f"\nUsing LP to check {len(make_split_indices(layer_bounds))}/{len(layer_bounds)} " + \
                              "potential ReLU splits...", end='', flush=True)

                layer_bounds, split_indices = s.tighten_bounds(layer_bounds, split_indices, sim,
                                                               check_cancel_func, depth)

            # bounds are now as tight as they will get
            if split_indices is None:
                split_indices = make_split_indices(layer_bounds)

            split_indices = sort_splits(layer_bounds, split_indices)
            zero_indices = np.nonzero(layer_bounds[:, 1] < -Settings.SPLIT_TOLERANCE)[0]

            for s in sets:
                s.execute_with_bounds(layer_num, layer_bounds, split_indices, zero_indices)
        else:
            # non-relu layer
            Timers.tic('transform_linear')

            if isinstance(layer, SkipAddLayer) and network.dag_predecessors:
                # SkipAdd: combine skip-path state with current (main-path) state.
                # Prefer states saved during this overapprox round (for skip sources
                # that are beyond the initial propagation's stopping point).
                # Fall back to ss_init's exact caches for earlier skip sources.
                skip_cache_key = network.dag_predecessors[layer_num][0]
                saved_states = overapprox_skip_states.get(skip_cache_key)
                for i, s in enumerate(sets):
                    if saved_states is not None:
                        s.transform_skip_linear_from_saved(layer, saved_states[i])
                    else:
                        s.transform_skip_linear(layer, skip_cache_key, ss_init)
                    check_cancel_func()
            else:
                for s in sets:
                    s.transform_linear(layer)
                    check_cancel_func()

            Timers.toc('transform_linear')

        if not ss_init.branch_tuples and Settings.PRINT_OUTPUT and Settings.PRINT_OVERAPPROX_OUTPUT:
            diff = time.perf_counter() - layer_start
            print(f" {round(diff, 3)} sec")

        layer_num += 1

def make_split_indices(layer_bounds):
    'make split indices from layer bounds'

    Timers.tic('make_split_indices')
    split_indices = np.nonzero(np.logical_and(layer_bounds[:, 0] < -Settings.SPLIT_TOLERANCE, \
                                              layer_bounds[:, 1] > Settings.SPLIT_TOLERANCE))[0]
    Timers.toc('make_split_indices')

    return split_indices

class StarOverapprox(Freezable):
    '''
    star set (triangle) overapproximation set representation
    star sets support efficient affine transformation and intersection
    '''

    def __init__(self, ss, type_string, max_gens=np.inf):
        assert max_gens >= Settings.OVERAPPROX_MIN_GEN_LIMIT

        self.star = ss.star.copy()

        assert type_string in ['star.lp', 'star.quick']

        self.type_string = type_string
        self.do_lp = type_string == 'star.lp'

        self.violation_star = None # assigned in check_spec

        self.max_gens = max_gens

    def __str__(self):
        return f"[StarOverapprox ({self.type_string})]"

    def execute_with_bounds(self, layer_num, layer_bounds, split_indices, zero_indices):
        'do the layer overapproximation with the passed-in bounds'

        if self.get_num_gens() + len(split_indices) > self.max_gens:
            raise OverapproxCanceledException(f'star gens exceeds limit (> {self.max_gens})')

        self.star.execute_relus_overapprox(layer_num, layer_bounds, split_indices, zero_indices)

    def transform_linear(self, layer):
        'affine transformation'

        layer.transform_star(self.star)

    def get_skip_state_copy(self):
        'return a copy of the current star for use as a skip-source during an overapprox round'

        return self.star.copy()

    def transform_skip_linear(self, layer, skip_cache_key, ss_init):
        'affine transformation for SkipAddLayer using cached skip-path star'

        star_skip = ss_init.star_cache.get(skip_cache_key)
        assert star_skip is not None, (
            f"SkipAddLayer: skip source {skip_cache_key} not found in ss_init.star_cache "
            f"(keys: {list(ss_init.star_cache.keys())}). "
            "The skip source must be cached before the overapprox round starts.")
        layer.transform_star(star_skip, self.star)

    def transform_skip_linear_from_saved(self, layer, saved_star):
        'apply SkipAdd using a star saved during this overapprox round at the skip-source layer'

        layer.transform_star(saved_star, self.star)

    def tighten_bounds(self, layer_bounds, split_indices, sim, check_cancel_func, depth):
        '''
        update the passed-in layer bounds

        layer_bounds and/or split_indices may be None

        returns (layer_bounds, split_indices), split_indices can be None
        '''

        if layer_bounds is None:
            num_neurons = self.star.a_mat.shape[0]            
            layer_bounds = np.array([[-np.inf, np.inf] for _ in range(num_neurons)], dtype=float)
        elif split_indices is None:
            split_indices = make_split_indices(layer_bounds)

        if self.do_lp:
            both_bounds = Settings.OVERAPPROX_BOTH_BOUNDS

            split_indices = update_bounds_lp(layer_bounds, self.star, sim, split_indices, depth,
                                             check_cancel_func=check_cancel_func, both_bounds=both_bounds)

        return layer_bounds, split_indices

    def check_spec(self, spec, check_cancel_func):
        'returns issafe?'

        # todo: evaluate whether this helps
        check_cancel_func()

        self.violation_star = spec.get_violation_star(self.star, domain_contraction=False)

        return self.violation_star is None

    def get_num_gens(self):
        'get the number of generators in the overapproximation (None if inapplicable)'

        return self.star.a_mat.shape[1]

class ZonoOverapprox(Freezable):
    '''
    Zonotope overapproximation
    '''

    def __init__(self, ss, type_string, max_gens=np.inf):
        '''
        initialize from an (exact) StarState,

        type_string is the type of overapproximation 'zono.area', 'zono.ybloat', or 'zono.interval'

        if gens exceeds max_gens, an OverapproxCanceledException is raised
        '''

        assert max_gens >= Settings.OVERAPPROX_MIN_GEN_LIMIT

        self.zono = ss.prefilter.zono.deep_copy()
        
        self.type_string = type_string
        self.max_gens = max_gens

        if type_string == 'zono.area':
            self.relu_update_func = relu_update_best_area_zono
        elif type_string == 'zono.ybloat':
            self.relu_update_func = relu_update_ybloat_zono
        else:
            assert type_string == 'zono.interval'
            self.relu_update_func = relu_update_interval_zono

    def __str__(self):
        return f"[ZonoOverapprox ({self.type_string})]"

    def execute_with_bounds(self, _layer_num, layer_bounds, split_indices, zero_indices):
        'do the layer overapproximation with the passed-in bounds'

        if self.get_num_gens() + len(split_indices) > self.max_gens:
            raise OverapproxCanceledException(f'{self.type_string} gens exceeds limit (> {self.max_gens})')
        
        update_zono(self.zono, self.relu_update_func, layer_bounds, split_indices, zero_indices)

    def transform_linear(self, layer):
        'affine transformation'

        layer.transform_zono(self.zono)

    def get_skip_state_copy(self):
        'return a copy of the current zonotope for use as a skip-source during an overapprox round'

        return self.zono.deep_copy()

    def transform_skip_linear(self, layer, skip_cache_key, ss_init):
        'affine transformation for SkipAddLayer using cached skip-path zonotope'

        zono_skip = ss_init.prefilter.zono_cache.get(skip_cache_key)
        assert zono_skip is not None, (
            f"SkipAddLayer: skip source {skip_cache_key} not found in ss_init.prefilter.zono_cache "
            f"(keys: {list(ss_init.prefilter.zono_cache.keys())}). "
            "The skip source must be cached before the overapprox round starts.")
        layer.transform_zono(zono_skip, self.zono)

    def transform_skip_linear_from_saved(self, layer, saved_zono):
        'apply SkipAdd using a zonotope saved during this overapprox round at the skip-source layer'

        layer.transform_zono(saved_zono, self.zono)

    def tighten_bounds(self, layer_bounds, _split_indices, _sim, _check_cancel_func, _depth):
        '''
        update the passed-in layer bounds

        layer_bounds and/or split_indices may be None

        returns (layer_bounds, split_indices), split_indices can be None
        '''

        box_bounds = self.zono.box_bounds()

        if layer_bounds is None:
            layer_bounds = box_bounds
        else:
            layer_bounds[:, 0] = np.maximum(layer_bounds[:, 0], box_bounds[:, 0])
            layer_bounds[:, 1] = np.minimum(layer_bounds[:, 1], box_bounds[:, 1])

        return layer_bounds, None

    def check_spec(self, spec, _check_cancel_func):
        'returns is_safe?'

        may_be_unsafe = spec.zono_might_violate_spec(self.zono)

        return not may_be_unsafe
        
    def get_num_gens(self):
        'get the number of generators in the overapproximation (-1 if inapplicable)'

        return self.zono.mat_t.shape[1]

def _update_zono_sparse(z, relu_update_func, bounds, splits, zeros):
    '''Sparse-aware ReLU update — called when mat_t is sparse and densifying would OOM.

    Operations on mat_t rows:
      - zeros: zero out entire rows (neurons clamped to 0)
      - splits: apply relu_update_func which modifies one row and writes one value into new_gen

    Each split adds exactly one new generator column with at most one nonzero (at output_dim).

    The exact per-split math (row scaling, center update, new_gen value) depends on which
    relu_update_func is passed — the caller fills that in.  This stub handles the sparse
    bookkeeping; the math is delegated to relu_update_func via a dense proxy row.
    '''
    from scipy.sparse import issparse, csc_matrix, csr_matrix, hstack as sp_hstack

    center = z.center

    # --- Zero rows (neurons fully clamped to 0) ---
    center[zeros] = 0
    if len(zeros) > 0:
        # CSR makes row zeroing efficient: just zero the data slice for each row
        mat_csr = z.mat_t.tocsr()
        for row_idx in zeros:
            s, e = mat_csr.indptr[row_idx], mat_csr.indptr[row_idx + 1]
            mat_csr.data[s:e] = 0
        mat_csr.eliminate_zeros()
        z.mat_t = mat_csr.tocsc()

    if splits.size == 0:
        return

    # --- Split rows: apply relu_update_func row-by-row ---
    # Each call to relu_update_func may:
    #   - scale gen_mat_t[output_dim] (a row)
    #   - update center[output_dim]
    #   - write exactly one nonzero into new_gen[output_dim]
    #
    # We use a dense proxy for the single row touched per split, then write it back.
    # New generator columns are collected as (value, row_index) pairs and bulk-inserted.

    mat_csc = z.mat_t.tocsc()
    rows_total = mat_csc.shape[0]
    G = mat_csc.shape[1]

    new_gen_values = []   # (value, row_index) for each new column
    modified_rows = {}    # row_index -> new dense row array (only for scaled rows)

    class _RowProxy:
        '''Proxy so relu_update_func can index with proxy[output_dim] as if it were a 2D matrix row.
        proxy[output_dim] returns/sets the row_dense array, regardless of the index value.
        This matches the 2D semantics: mat[row_index] returns the full row.
        '''
        __slots__ = ('_row',)
        def __init__(self, row): self._row = row
        def __getitem__(self, i): return self._row
        def __setitem__(self, i, v): self._row[:] = v

    for i, split_index in enumerate(splits):
        lb, ub = bounds[split_index]

        # Extract the row as a dense vector so relu_update_func can mutate it
        row_dense = np.asarray(mat_csc.getrow(split_index).todense()).ravel().astype(z.dtype)
        new_gen_col = np.zeros(rows_total, dtype=z.dtype)

        # Pass proxy so relu_update_func sees proxy[output_dim] = full row (not a scalar)
        proxy = _RowProxy(row_dense)
        relu_update_func(lb, ub, int(split_index), proxy, center, new_gen_col)

        # Check if the row was modified (interval zeros it; best_area scales it)
        modified_rows[int(split_index)] = row_dense

        # new_gen_col has at most one nonzero at split_index
        val = new_gen_col[split_index]
        if val != 0.0:
            new_gen_values.append((val, int(split_index)))

    # Rebuild mat_csc with modified rows by converting to LIL (efficient row assignment)
    if modified_rows:
        mat_lil = mat_csc.tolil()
        for row_idx, row_dense in modified_rows.items():
            mat_lil[row_idx] = row_dense
        mat_csc = mat_lil.tocsc()
        mat_csc.eliminate_zeros()

    # Build new generator columns: each is a sparse (rows_total, 1) column
    S = len(splits)
    if new_gen_values:
        col_indices = [c for c in range(len(new_gen_values))]
        nz_rows = [r for (_, r) in new_gen_values]
        nz_vals = [v for (v, _) in new_gen_values]
        new_cols = csc_matrix((nz_vals, (nz_rows, col_indices)), shape=(rows_total, S), dtype=z.dtype)
    else:
        new_cols = csc_matrix((rows_total, S), dtype=z.dtype)

    z.mat_t = sp_hstack([mat_csc, new_cols], format='csc')
    z.init_bounds += [(-1, 1) for _ in range(S)]


def update_zono(z, relu_update_func, bounds, splits, zeros):
    'update a zono with the current bounds'

    # this assumes apply_linear_map was done first, so that only ReLU processing remains
    lb_len = bounds.shape[0]
    assert len(z.center) == lb_len, "zonotope dims ({len(z.center)}) doesn't match layer_bounds {lb_len}"

    from scipy.sparse import issparse, csc_matrix
    is_sparse = issparse(z.mat_t)

    if is_sparse:
        # Phase 1: memory guard — never densify if it would exceed the memory budget.
        rows, cols = z.mat_t.shape
        dense_bytes = rows * cols * 4  # float32
        budget_bytes = Settings.MEMORY_BUDGET_GB * 1e9

        if Settings.SPARSE_DEBUG:
            nnz = z.mat_t.nnz
            density = nnz / max(rows * cols, 1)
            print(f"[update_zono] mat_t={rows}x{cols} nnz={nnz} density={density:.4%} "
                  f"dense_would_be={dense_bytes/1e9:.2f}GB budget={Settings.MEMORY_BUDGET_GB}GB "
                  f"splits={len(splits)} zeros={len(zeros)}", flush=True)

        if dense_bytes > budget_bytes:
            # Cannot densify — use the sparse ReLU path
            _update_zono_sparse(z, relu_update_func, bounds, splits, zeros)
            return

        # Phase 3: capture pre-ReLU nnz to avoid a full matrix scan after densification
        pre_relu_nnz = z.mat_t.nnz
        z.mat_t = z.mat_t.toarray().astype(np.float32)

    gen_mat_t = z.mat_t
    center = z.center

    # these are the bounds on the input for each neuron in the current layer
    Timers.tic('assign_zeros')
    center[zeros] = 0
    gen_mat_t[zeros, :] = 0
    Timers.toc('assign_zeros')

    if splits.size > 0:
        new_generators = np.zeros((gen_mat_t.shape[0], len(splits)), dtype=z.dtype)

        Timers.tic('relu_update')
        for i, split_index in enumerate(splits):
            lb, ub = bounds[split_index]

            # need to add a new generator for the overapproximation
            relu_update_func(lb, ub, split_index, gen_mat_t, center, new_generators[:, i])
        Timers.toc('relu_update')

        Timers.tic('stack_new_generators')
        # need to update zonotope with new generators
        z.init_bounds += [(-1, 1) for _ in range(len(splits))]

        z.mat_t = np.hstack([z.mat_t, new_generators])

        Timers.toc('stack_new_generators')

    # Phase 3: reconvert to sparse using pre-ReLU nnz estimate (avoids O(rows*cols) count_nonzero).
    # Row-zeroing only reduces nnz, row-scaling doesn't increase it, new_generators add len(splits)
    # columns each with 1 nonzero — so post-ReLU nnz <= pre_relu_nnz + len(splits).
    if is_sparse:
        post_nnz_estimate = pre_relu_nnz + len(splits)
        total = z.mat_t.shape[0] * z.mat_t.shape[1]
        if total > 0 and post_nnz_estimate / total <= Settings.CONV_BATCHING_MIN_SPARSITY:
            # Use np.nonzero for O(nnz) CSC construction rather than O(rows*cols) csc_matrix scan
            nz_rows, nz_cols = np.nonzero(z.mat_t)
            nz_vals = z.mat_t[nz_rows, nz_cols]
            z.mat_t = csc_matrix((nz_vals, (nz_rows, nz_cols)), shape=z.mat_t.shape)

def relu_update_interval_zono(_lb, ub, output_dim, gen_mat_t, center, new_gen):
    '''update one dimension (output) of a zonotope due to a relu split
    This function produces the interval zonotope.
    '''

    gen_mat_t[output_dim] = 0

    y_offset = ub / 2.0
    center[output_dim] = y_offset

    new_gen[output_dim] = y_offset

def relu_update_ybloat_zono(lb, _ub, output_dim, _gen_mat_t, center, new_gen):
    '''update one dimension (output) of a zonotope due to a relu split
    This function produces the ybloat zonotope (new generator is vertical).
    '''

    y_offset = -lb / 2.0

    center[output_dim] += y_offset
    new_gen[output_dim] = y_offset

def relu_update_best_area_zono(lb, ub, output_dim, gen_mat_t, center, new_gen):
    '''update one dimension (output) of a zonotope due to a relu split
    This function produces the best-area zonotope.
    '''

    assert lb < Settings.SPLIT_TOLERANCE
    assert ub > -Settings.SPLIT_TOLERANCE

    slope_lambda = ub / (ub - lb)
    gen_mat_t[output_dim] *= slope_lambda

    # add new generator value to bm
    mu = -1 * (ub * lb) / (2 * (ub - lb))
    new_gen[output_dim] = mu

    # modify center
    center[output_dim] = center[output_dim] * slope_lambda + mu

class OverapproxCanceledException(Exception):
    'an exception used for when overapproximation analysis is canceled'


class DeeppolyOverapprox(Freezable):
    '''
    Deeppoly overapproximation
    '''

    def __init__(self, ss, type_string, max_gens=np.inf):
        '''
        initialize from an (exact) StarState,

        type_string is the type of overapproximation 'zono.area', 'zono.ybloat', or 'zono.interval'

        if gens exceeds max_gens, an OverapproxCanceledException is raised
        '''

        assert max_gens >= Settings.OVERAPPROX_MIN_GEN_LIMIT

        self.zono = ss.prefilter.zono.deep_copy()

        # Densify mat_t if sparse — DeepPoly arithmetic assumes dense numpy arrays
        from scipy.sparse import issparse
        mat_t_dense = self.zono.mat_t.toarray().astype(np.float32) \
            if issparse(self.zono.mat_t) else self.zono.mat_t

        self.ubcoef = np.copy(mat_t_dense)   # upper bounds coefficients of upper bound equations
        self.ubconst = np.copy(self.zono.center) # upper bounds constants of upper bound equations
        self.lbcoef = np.copy(mat_t_dense)   # lower bounds coefficients of lower bound equations
        self.lbconst = np.copy(self.zono.center) # lower bounds constants of lower bound equations
        self.inputbounds = np.copy(self.zono.init_bounds)    # input bounds (at this stage)
        self.ubs = np.copy(ss.prefilter.output_bounds.layer_bounds[:, 1])    # upper bounds
        self.lbs = np.copy(ss.prefilter.output_bounds.layer_bounds[:, 0])    # lower bounds


        self.type_string = type_string
        self.max_gens = max_gens

    def __str__(self):
        return f"[ZonoOverapprox ({self.type_string})]"

    def execute_with_bounds(self, _layer_num, layer_bounds, split_indices, zero_indices):
        'do the layer overapproximation with the passed-in bounds'

        if self.get_num_gens() + len(split_indices) > self.max_gens:
            raise OverapproxCanceledException(f'{self.type_string} gens exceeds limit (> {self.max_gens})')

        # case1: negative upper bound
        idx = np.where(self.ubs <= 0)
        if np.size(idx) > 0:
            self.ubcoef[idx] = 0.0
            self.ubconst[idx] = 0.0
            self.lbcoef[idx] = 0.0
            self.lbconst[idx] = 0.0

            self.ubs[idx], self.lbs[idx] = 0.0, 0.0

        # case2: positive lower bound -> do nothing

        # case3: convex approximation
        if self.type_string == 'deeppoly.area':
            idx = np.where((self.ubs <= -1 * self.lbs) & (self.ubs > 0) & (self.lbs < 0))
            if np.size(idx) > 0:
                factor = self.ubs[idx]/(self.ubs[idx] - self.lbs[idx])
                bias = -self.lbs[idx] * self.ubs[idx]/(self.ubs[idx] - self.lbs[idx])

                self.ubcoef[idx] = factor.reshape(-1, 1) * self.ubcoef[idx]
                self.ubconst[idx] = factor * self.ubconst[idx] + bias

                self.lbcoef[idx] = 0.0
                self.lbconst[idx] = 0.0

                self.ubs[idx], self.lbs[idx] = self.ubs[idx], 0.0

            idx = np.where((self.ubs > -1 * self.lbs) & (self.lbs < 0))
            if np.size(idx) > 0:
                factor = self.ubs[idx]/(self.ubs[idx] - self.lbs[idx])
                bias = -self.lbs[idx] * self.ubs[idx]/(self.ubs[idx] - self.lbs[idx])

                self.ubcoef[idx] = factor.reshape(-1, 1) * self.ubcoef[idx]
                self.ubconst[idx] = factor * self.ubconst[idx] + bias

                self.ubs[idx], self.lbs[idx] = self.ubs[idx], self.lbs[idx]
        elif self.type_string == 'deeppoly.upper':
            idx = np.where((self.ubs > 0) & (self.lbs < 0))
            if np.size(idx) > 0:
                factor = self.ubs[idx]/(self.ubs[idx] - self.lbs[idx])
                bias = -self.lbs[idx] * self.ubs[idx]/(self.ubs[idx] - self.lbs[idx])

                self.ubcoef[idx] = factor.reshape(-1, 1) * self.ubcoef[idx]
                self.ubconst[idx] = factor * self.ubconst[idx] + bias

                self.ubs[idx], self.lbs[idx] = self.ubs[idx], self.lbs[idx]
        elif self.type_string == 'deeppoly.lower':
            idx = np.where((self.ubs > 0) & (self.lbs < 0))
            if np.size(idx) > 0:
                factor = self.ubs[idx]/(self.ubs[idx] - self.lbs[idx])
                bias = -self.lbs[idx] * self.ubs[idx]/(self.ubs[idx] - self.lbs[idx])

                self.ubcoef[idx] = factor.reshape(-1, 1) * self.ubcoef[idx]
                self.ubconst[idx] = factor * self.ubconst[idx] + bias

                self.lbcoef[idx] = 0.0
                self.lbconst[idx] = 0.0

                self.ubs[idx], self.lbs[idx] = self.ubs[idx], 0.0
        elif self.type_string == 'deeppoly.middle':
            idx = np.where((self.ubs > 0) & (self.lbs < 0))
            if np.size(idx) > 0:
                factor = self.ubs[idx]/(self.ubs[idx] - self.lbs[idx])
                bias = -self.lbs[idx] * self.ubs[idx]/(self.ubs[idx] - self.lbs[idx])

                self.ubcoef[idx] = factor.reshape(-1, 1) * self.ubcoef[idx]
                self.ubconst[idx] = factor * self.ubconst[idx] + bias

                factor_lb = 0.5 * np.ones_like(factor) # bias_lb = 0.0
                self.lbcoef[idx] = factor_lb.reshape(-1, 1) * self.lbcoef[idx]
                self.lbconst[idx] = factor_lb * self.lbconst[idx]

                self.ubs[idx], self.lbs[idx] = self.ubs[idx], self.lbs[idx]/2.0
        else:
            raise UnknownType(f'unknown type string for deeppoly overapprox: {self.type_string}')

    def transform_linear(self, layer):
        'affine transformation'

        layer.transform_deeppoly(self)

    def transform_skip_linear(self, layer, skip_cache_key, ss_init):
        'affine transformation for SkipAddLayer (DeepPoly not yet supported for skip connections)'

        raise NotImplementedError(
            "DeepPoly overapproximation does not yet support SkipAddLayer. "
            "Use zono or star overapprox for skip connection networks.")

        # ubcoef_nl, ubconst_nl, lbcoef_nl, lbconst_nl = layer.transform_deeppoly()

        # # back substitution
        # updated_ubcoef_nl = np.where(ubcoef_nl >= 0, ubcoef_nl, 0) @ self.ubcoef
        # updated_ubcoef_nl += np.where(ubcoef_nl < 0, ubcoef_nl, 0) @ self.lbcoef
        # updated_ubconst_nl = np.where(ubcoef_nl >= 0, ubcoef_nl, 0) @ self.ubconst
        # updated_ubconst_nl += np.where(ubcoef_nl < 0, ubcoef_nl, 0) @ self.lbconst
        # updated_ubconst_nl += ubconst_nl

        # updated_lbcoef_nl = np.where(lbcoef_nl >= 0, lbcoef_nl, 0) @ self.lbcoef
        # updated_lbcoef_nl += np.where(lbcoef_nl < 0, lbcoef_nl, 0) @ self.ubcoef
        # updated_lbconst_nl = np.where(lbcoef_nl >= 0, lbcoef_nl, 0) @ self.lbconst
        # updated_lbconst_nl += np.where(lbcoef_nl < 0, lbcoef_nl, 0) @ self.ubconst
        # updated_lbconst_nl += lbconst_nl
        
        # self.ubcoef = updated_ubcoef_nl
        # self.ubconst = updated_ubconst_nl
        # self.lbcoef = updated_lbcoef_nl
        # self.lbconst = updated_lbconst_nl

        # self.ubs = np.where(self.ubcoef >= 0, self.ubcoef, 0) @ self.inputbounds[:, 1]
        # self.ubs += np.where(self.ubcoef < 0, self.ubcoef, 0) @ self.inputbounds[:, 0]
        # self.ubs += self.ubconst
        # self.lbs = np.where(self.lbcoef >= 0, self.lbcoef, 0) @ self.inputbounds[:, 0]
        # self.lbs += np.where(self.lbcoef < 0, self.lbcoef, 0) @ self.inputbounds[:, 1]
        # self.lbs += self.lbconst

    def tighten_bounds(self, layer_bounds, _split_indices, _sim, _check_cancel_func, _depth):
        '''
        update the passed-in layer bounds

        layer_bounds and/or split_indices may be None

        returns (layer_bounds, split_indices), split_indices can be None
        '''

        box_bounds = np.column_stack((self.lbs, self.ubs))

        if layer_bounds is None:
            layer_bounds = box_bounds
        else:
            layer_bounds[:, 0] = np.maximum(layer_bounds[:, 0], box_bounds[:, 0])
            layer_bounds[:, 1] = np.minimum(layer_bounds[:, 1], box_bounds[:, 1])
        
            # update ubs & lbs based on layer_bounds  
            self.ubs = np.copy(layer_bounds[:, 1])
            self.lbs = np.copy(layer_bounds[:, 0])
        
        return layer_bounds, None

    def check_spec(self, spec, _check_cancel_func):
        'returns is_safe?'

        coef_nl = spec.mat
        
        # back substitution
        updated_lbcoef_nl = np.where(coef_nl >= 0, coef_nl, 0) @ self.lbcoef
        updated_lbcoef_nl += np.where(coef_nl < 0, coef_nl, 0) @ self.ubcoef
        updated_lbconst_nl = np.where(coef_nl >= 0, coef_nl, 0) @ self.lbconst
        updated_lbconst_nl += np.where(coef_nl < 0, coef_nl, 0) @ self.ubconst
        min_vals = np.where(updated_lbcoef_nl >= 0, updated_lbcoef_nl, 0) @ self.inputbounds[:, 0]
        min_vals += np.where(updated_lbcoef_nl < 0, updated_lbcoef_nl, 0) @ self.inputbounds[:, 1]
        min_vals += updated_lbconst_nl

        might_violate = True
        for i, row in enumerate(updated_lbcoef_nl):
            if min_vals[i] > spec.rhs[i]:
                might_violate = False
                # print('verified by deeppoly')
                return not might_violate
        return not might_violate
        
    def get_num_gens(self):
        'get the number of generators in the overapproximation (-1 if inapplicable)'
        
        return -1

class UnknownType(RuntimeError):
    'raised if type_string is unknown'