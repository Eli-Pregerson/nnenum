'''
Computation Settings. Change settings by assigning directly to the class attributes.

For example, to run single-threaded:
Settings.NUM_PROCESSES = 1
'''

import os
import multiprocessing

import numpy as np

from nnenum.util import FreezableMeta

class Settings(metaclass=FreezableMeta):
    '''enumeration settings. Access these using, for example, Settings.NUM_PROCESSES

    These get initialized by the metaclass to the values in the reset() class method.
    '''

    BRANCH_OVERAPPROX, BRANCH_EGO, BRANCH_EGO_LIGHT, BRANCH_EXACT = range(4) # used for BRANCH_MODE
    SPLIT_LARGEST, SPLIT_ONE_NORM, SPLIT_SMALLEST, SPLIT_INORDER = range(4) # used for SPLIT_ORDER
    #TODO: one norm should acutally be called inf norm

    @classmethod
    def reset(cls):
        'assign default settings'
        
        cls.LP_SOLVER = "GLPK" # options: 'GLPK', 'Gurobi' 

        # settings / optimizations
        num_cores = multiprocessing.cpu_count()

        try:
            num_cores = len(os.sched_getaffinity(0)) # doesn't work on some unix platforms
        except AttributeError:
            pass
        
        cls.NUM_PROCESSES = num_cores # use multiple cores
        cls.TIMEOUT = np.inf # verification timeout, in seconds (np.inf = no timeout)

        cls.SINGLE_SET = False # only do single-set overapproximation (no splitting)

        cls.PRINT_OUTPUT = True # print anything to stdout? (controls all output)

        cls.RESULT_SAVE_POLYS = False # save 2-d projections of output polygons to Result.polys?
        cls.RESULT_SAVE_POLYS_DIMS = (0, 1) # (x_dim, y_dim) of 2-d projections, used if RESULT_SAVE_POLYGONS is True

        cls.RESULT_SAVE_STARS = False # save LpStar objects in result?

        cls.RESULT_SAVE_TIMERS = [] # list of timers to record in Result.timers; TIMING_STATS must be True

        cls.FIND_CONCRETE_COUNTEREXAMPLES = True # should we try to find concrete counterexamples if spec violated?

        #########################
        ### advanced settings ###
        cls.PRINT_PROGRESS = True # print periodic progress updates
        cls.PRINT_INTERVAL = 0.1 # print interval in seconds (0 = no printing)
        cls.TIMING_STATS = False # compute and print detailed timing stats

        cls.LOG_CONV_BATCHING = False # log convolution batching statistics

        cls.CONV_BATCHING_ENABLED = True # set to False to disable conv generator batching entirely
        cls.CONV_BATCHING_MIN_SPARSITY = 0.05 # skip batching if generator sparsity > 5% (too dense)
        cls.CONV_BATCHING_FIRST_LAYER_ONLY = False # only apply batching to first conv layer

        # Batching strategy when CONV_METHOD='batching':
        #   'greedy'  - original O(n*B) scan: assign each gen to first compatible batch
        #   'random'  - O(n*k) randomized: try k random batches before creating a new one
        #   'period'  - O(n) grid-period: assign by (y%period, x%period) — zero conflict checking
        cls.CONV_BATCHING_STRATEGY = 'greedy'

        # Max consecutive failures before creating a new batch (only for 'random' strategy)
        cls.CONV_BATCHING_MAX_FAILURES = 10

        # Conv transformation method for abstract domain (star/zono) generator columns.
        # 'dense':    vectorized im2col matmul on all generators at once (default)
        # 'sparse':   prebuilt Toeplitz sparse matrix W; scipy sparse @ gen_mat
        # 'batching': greedy spatial grouping + im2col batched matmul (legacy; now slower than
        #             dense because the grouping overhead exceeds any FLOP savings)
        # Note: CONV_BATCHING_MIN_SPARSITY threshold is applied for 'batching' and 'sparse';
        # both fall back to 'dense' when generators are too dense to benefit.
        cls.CONV_METHOD = 'sparse'

        # Maximum memory (GB) allowed for a dense mat_t materialization.
        # If .toarray() on a sparse mat_t would exceed this, take the sparse path instead.
        cls.MEMORY_BUDGET_GB = 8.0

        # Print per-layer sparse stats (density, nnz, timing) for debugging the sparse path.
        cls.SPARSE_DEBUG = False

        # Minimum number of generators required to use the sparse conv path.
        # _build_conv_matrix has a large one-time cost (e.g. ~21s for VGGNet layer 0).
        # With few generators, dense im2col is faster because the build cost doesn't amortize.
        # For VGGNet layer 0: dense ~0.002s/gen, build ~21.5s → breakeven ~10750 generators.
        # Default 5000 is conservative (sparse wins clearly above this).
        cls.CONV_SPARSE_MIN_GENERATORS = 5000

        # Allow star.a_mat to remain as a scipy sparse CSR matrix through conv layers.
        # Densification is deferred to FC/MatMul layers where W @ a_mat produces a dense result anyway.
        # Disable for small networks (no benefit, avoids any sparse overhead).
        cls.SPARSE_STAR = False

        # Run interval bound propagation (IBP) as a fast precheck before star enumeration.
        # IBP propagates just two vectors (lb, ub) through the network — O(neurons) memory.
        # If IBP proves the property, return immediately without constructing the star/zono.
        # Very effective when epsilon is tiny. Off by default; enable per benchmark profile.
        cls.TRY_IBP = False

        # When SPARSE_STAR=True and a conv layer would produce a star.a_mat exceeding
        # MEMORY_BUDGET_GB, collapse the star to a per-neuron interval (diagonal) star
        # and continue propagation. Overapproximation — sound for unsat, may miss sat.
        # If TRY_IBP is also True, attempts IBP from the interval bounds first.
        cls.SPARSE_INTERVAL_FALLBACK = False

        cls.CHECK_SINGLE_THREAD_BLAS = False
        # idea... replace this with threadpoolctl: https://github.com/joblib/threadpoolctl
        
        cls.UPDATE_SHARED_VARS_INTERVAL = 0.05 # interval for each thread to update shared state

        cls.COMPRESS_INIT_BOX = True

        cls.EAGER_BOUNDS = True
        
        cls.CONTRACT_ZONOTOPE = False # try domain contraction on zonotopes (more accurate prefilter, but slower)
        cls.CONTRACT_ZONOTOPE_LP = True # contract zonotope using LPs (even more accurate prefilter, but even slower)
        cls.CONTRACT_LP_OPTIMIZED = True # use optimized lp contraction
        cls.CONTRACT_LP_TRACK_WITNESSES = True # track box bounds witnesses to reduce LP solving
        cls.CONTRACT_LP_CHECK_EPSILON = 1e-4 # numerical error tol['star.lp'],erated when doing contractions before error, None=skip

        # the types of overapproximation to use in each round
        cls.OVERAPPROX_TYPES = [['zono.area'], 
                                ['zono.area', 'zono.ybloat', 'zono.interval'],
                                ['zono.area', 'zono.ybloat', 'zono.interval', 'star.lp']] #['deeppoly.area'], , 'star.lp'

        cls.OVERAPPROX_NEAR_ROOT_MAX_SPLITS = 2
        cls.OVERAPPROX_TYPES_NEAR_ROOT = cls.OVERAPPROX_TYPES

        cls.OVERAPPROX_GEN_LIMIT_MULTIPLIER = 1.5 # don't try approx star if multizono.gens > THIS * last_safe_gens
        cls.OVERAPPROX_MIN_GEN_LIMIT = 50 # minimum generators to use as cap
        cls.OVERAPPROX_LP_TIMEOUT = 1.0 # timeout for LP part of overapproximation, use np.inf for unbounded
        cls.OVERAPPROX_BOTH_BOUNDS = False # should overapprox star method compute both bounds or just reject branches?

        cls.SAVE_BRANCH_TUPLES_FILENAME = None
        cls.SAVE_BRANCH_TUPLES_TIMES = True # when saving branch tuples, also include runtimes
        cls.BRANCH_MODE = cls.BRANCH_OVERAPPROX
        cls.PRINT_BRANCH_TUPLES = False

        cls.TRY_QUICK_OVERAPPROX = True
        cls.QUICK_OVERAPPROX_TYPES = [['zono.area'],
                                      ['zono.area', 'zono.ybloat', 'zono.interval']]
        cls.PRINT_OVERAPPROX_OUTPUT = True # print progress on first overapprox

        # one_norm is especially good at finding counterexamples
        cls.SPLIT_ORDER = cls.SPLIT_ONE_NORM # rearrange splitting order within each layer
        
        cls.RESULT_SAVE_POLYS_EPSILON = 1e-7 # accuracy of vertices when projecting polygons for Kamenev method

        cls.OFFLOAD_CLOSEST_TO_ROOT = True # when offloading work to other threads, use stars closest to root of search

        cls.SPLIT_TOLERANCE = 1e-8 # small outputs get rounded to zero when deciding if splitting is possible
        cls.TEST_FUNC_BEFORE_ASSIGNMENT = None # function to call before eager assignement, used for unit testing

        cls.SPLIT_IF_IDLE = True # force splitting (rather than overapproximation) if there are idle processes

        cls.SHUFFLE_TIME = None # shuffle star sets after some time (improves unsafe specs)

        cls.GLPK_TIMEOUT = 60 # maximum allowed seconds for each indivudal LP run
        cls.GLPK_FIRST_PRIMAL = True # first try primal LP... if that fails do dual
        cls.GLPK_RESET_BEFORE_MINIMIZE = False # reset the lp basis before minimize

        cls.SKIP_COMPRESSED_CHECK = False # sanity check for compressed inputs when COMPRESS_INIT_BOX is False
        ####
        cls.UNDERFLOW_BEHAVIOR = 'ignore' # np.seterr behavior for floating-point underflow
        cls.SKIP_CONSTRAINT_NORMALIZATION = False # disable constraint normalization in LP (may reduce stability) 

        ####
        cls.NUM_LP_PROCESSES = 1 # if > 1, then force multiprocessing during lp step
        cls.PARALLEL_ROOT_LP = True # near the root of the search, use parallel lp, override NUM_LP_PROCESES if true

        ####
        # generally it should be safe to add any linear layers to the whitelist
        cls.ONNX_WHITELIST = ['Add', 'AveragePool', 'Constant', 'Concat', 'Conv', 'Flatten', 'Gather', \
                              'Gemm', 'MatMul', 'Mul', 'Reshape', 'Relu', 'Shape', 'Sub', 'Unsqueeze', 'Slice', \
                              'Dropout', 'BatchNormalization', 'ConvTranspose', 'Upsample']

        cls.ONNX_BLACKLIST = ['Atan', 'MaxPool', 'Sigmoid', 'Tanh'] # unsupported nonlinear laters
