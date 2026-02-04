import numpy as np
from nnenum.settings import Settings


def set_mnist_fc_settings():
    'set settings for "mnist_fc" benchmarks'

    Settings.LP_SOLVER = "Gurobi"
    Settings.COMPRESS_INIT_BOX = True
    Settings.BRANCH_MODE = Settings.BRANCH_OVERAPPROX
    Settings.TRY_QUICK_OVERAPPROX = False
    
    Settings.OVERAPPROX_MIN_GEN_LIMIT = np.inf
    Settings.SPLIT_IF_IDLE = False
    Settings.OVERAPPROX_LP_TIMEOUT = np.inf
    #Settings.TIMING_STATS = True

    # contraction doesn't help in high dimensions
    #Settings.OVERAPPROX_CONTRACT_ZONO_LP = False
    Settings.CONTRACT_ZONOTOPE = False
    Settings.CONTRACT_ZONOTOPE_LP = False