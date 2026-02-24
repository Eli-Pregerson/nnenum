# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**nnenum** (pronounced *en-en-en-um*) is a high-performance neural network verification tool that verifies ReLU networks using abstraction refinement. It uses multiple levels of abstraction: zonotopes, star sets (triangle overapproximations), and efficient parallelized ReLU case splitting. Written in Python 3, uses GLPK for LP solving, and accepts ONNX network files and vnnlib specifications.

## Running and Testing

### Basic Usage
```bash
# Run verification on an ONNX model with vnnlib specification
python3 -m nnenum.nnenum -o <onnx_file> -v <vnnlib_file> [-t timeout] [-f outfile] [-p processes] [-s settings]

# Example
python3 -m nnenum.nnenum -o examples/acasxu/data/ACASXU_run2a_3_3_batch_2000.onnx -v examples/acasxu/data/prop_9.vnnlib
```

### Running Tests
```bash
./run_tests.sh  # Runs all example verifications
```

Test verification outputs:
- "violated" = property violated (unsafe)
- "holds" = property holds (safe)

## Architecture

### Core Verification Flow
1. **ONNX Loading** (`onnx_network.py`) - Parse ONNX model into layer representation
2. **Specification** (`vnnlib.py`, `specification.py`) - Parse vnnlib property specifications
3. **Enumeration** (`enumerate.py`) - Main verification loop with parallelized case splitting
4. **Abstraction** (`overapprox.py`, `lp_star.py`, `zonotope.py`) - Multiple abstraction levels
5. **Result** (`result.py`) - Verification outcome (safe/unsafe/timeout)

### Network Layer Architecture

**Location**: `src/nnenum/network.py` and `src/nnenum/onnx_network.py`

All layers inherit from `Freezable` base class and implement standard interface:
- `__init__(layer_num, ...)` - Initialize with layer number and parameters
- `get_input_shape()` / `get_output_shape()` - Return input/output tensor shapes
- `execute(state)` - Concrete execution on numpy array
- `transform_star(star)` - Transform star set representation (exact verification)
- `transform_zono(zono)` - Transform zonotope representation (overapproximation)
- `transform_deeppoly(deeppoly)` - Transform deeppoly representation (overapproximation)

**Layer Types**:
- `ConstantLayer` - Outputs fixed value (no input)
- `AddLayer` - Element-wise addition with bias vector
- `FlattenLayer` - Reshape multi-dimensional to 1D
- `ReshapeLayer` - Arbitrary shape transformation (supports -1 dimension inference)
- `MatMulLayer` - Matrix multiplication without bias
- `FullyConnectedLayer` - Affine transformation (Wx + b)
- `ReluLayer` - ReLU activation (triggers branching in verification)
- `Convolutional2dLayer` - 2D convolution with kernels, biases, and stride support (for downsampling)
- `PoolingLayer` - Max/mean pooling (not in optimized path)

**ONNX Parsing**:
- **Optimized path** (`load_onnx_network_optimized`): Supports Add, Sub, Constant, Flatten, Reshape, MatMul, Relu, Gemm, Conv
- **General path** (`load_onnx_network`): Wraps unsupported linear ops in `LinearOnnxSubnetworkLayer` (uses ONNX runtime)
- Falls back to general path if optimized parsing fails

**Note on opset versions**: Models with ONNX opset >23 are automatically converted to opset 23 at load time (opset 25+ not fully supported by onnxruntime).

### Adding New Layer Types

1. **Define layer class** in `network.py`:
   - Inherit from `Freezable`
   - Implement required methods (see interface above)
   - Call `self.freeze_attrs()` at end of `__init__`

2. **Add ONNX parsing** in `onnx_network.py`:
   - Update imports: add layer to import from `nnenum.network`
   - Add `elif op == 'YourOp':` clause in `load_onnx_network_optimized` (around line 440-560)
   - Extract parameters from `init_map[cur_node.input[...]]` (weights/biases from initializers)
   - Handle node attributes via `cur_node.attribute`
   - Create layer instance: `layer = YourLayer(len(layers), params, prev_shape)`

3. **Handle special cases**:
   - If layer needs special verification logic (like ReLU), add `isinstance(layer, YourLayer)` checks in `lp_star_state.py` and `overapprox.py`
   - Linear layers typically need no special handling (transform methods do nothing or just reshape)

### Verification Modes

**Branch modes** (Settings.BRANCH_MODE):
- `BRANCH_OVERAPPROX` - Overapproximation only (fast, may be imprecise)
- `BRANCH_EXACT` - Exact verification with LP (slower, complete)
- `BRANCH_EGO` / `BRANCH_EGO_LIGHT` - Eager overapproximation with refinement

**Abstraction types** (Settings.OVERAPPROX_TYPES):
- `'zono.area'` - Zonotope with area heuristic
- `'zono.ybloat'` - Zonotope with y-bloat
- `'zono.interval'` - Interval zonotope
- `'star.lp'` - Star set with LP
- `'deeppoly.area'` - DeepPoly abstraction

### Key Components

- **Star Sets** (`lp_star.py`, `lp_star_state.py`) - Exact representation using convex polytopes with LP
- **Zonotopes** (`zonotope.py`) - Overapproximation using zonotope geometry
- **LP Solving** (`lpinstance.py`, `lpinstance_glpk.py`) - Linear programming interface (GLPK or Gurobi)
- **Prefiltering** (`prefilter.py`) - Fast bounds propagation before exact analysis
- **Workers** (`worker.py`) - Parallel verification worker processes

### Settings Configuration

Modify `Settings` class attributes to control verification:
```python
from nnenum.settings import Settings

Settings.NUM_PROCESSES = 4      # Parallel workers
Settings.TIMEOUT = 60.0         # Seconds
Settings.BRANCH_MODE = Settings.BRANCH_EXACT
Settings.PRINT_OUTPUT = True    # Enable/disable output
```

Pre-configured settings in `setting_cat.py` (e.g., `set_exact_settings()`, `set_overapprox_settings()`).

## Dependencies

Core: numpy, scipy, onnx, onnxruntime, skl2onnx, swiglpk (GLPK), termcolor

See `requirements.txt` for complete list.

## VNN-COMP 2025 Benchmark Results

The `vnncomp2025_benchmarks/` directory contains critical performance data:

### Benchmark CSV Files

Each CSV has format: `benchmark_title, onnx_path, vnnlib_path, overhead_time, result, verification_time`

- **`abc_results.csv`** - Results from alpha-beta-CROWN (multi-year winning tool) on this device
  - Use as performance target/comparison baseline
  - Shows state-of-the-art performance on each benchmark

- **`app_results_pre.csv`** - nnenum results BEFORE recent changes
  - Baseline for measuring improvement from optimizations

- **`all_results.csv`** - nnenum results AFTER recent changes (updated during experiments)
  - Current performance with Constant/Reshape/Conv optimizations

### Using Benchmark Data

To identify optimization opportunities:
```python
# Compare nnenum vs alpha-beta-CROWN
# Look for: benchmarks where nnenum times out but abc succeeds
# Or: benchmarks with large time differences (abc: 5s, nnenum: 50s)

# Compare pre vs post changes
# Measure: improvement from Constant/Reshape/Conv optimizations
```

**Priority**: Focus optimization on benchmarks where:
1. Alpha-beta-CROWN succeeds but nnenum times out
2. Large time gap between tools (indicates room for improvement)
3. Conv-heavy models (nnenum's main performance gap)

## File Organization

- `src/nnenum/` - Main source code
  - `nnenum.py` - CLI entry point
  - `network.py` - Layer definitions
  - `onnx_network.py` - ONNX parsing
  - `enumerate.py` - Main verification algorithm
  - `settings.py` - Configuration
- `examples/` - Test networks and properties (acasxu, mnist, cifar)
- `vnncomp_scripts/` - VNN-COMP competition scripts
- `vnncomp2025_benchmarks/` - Benchmark results and comparison data
- `test_new_layers.py` - Layer-by-layer correctness tests

## Known Issues

### Skip Connection Soundness Bug (CRITICAL - Under Investigation)

**Status**: Identified but not yet fixed. Being tracked on separate branch for future work.

**Problem**: When using overapproximation mode (`BRANCH_OVERAPPROX`) on ResNet-style networks with skip connections, nnenum produces **unsound results** - reporting "sat" (property violated) when the actual answer is "unsat" (property holds).

**Impact**: 348 mismatches vs alpha-beta-CROWN results:
- 256 instances: nnenum=sat, abc=unsat ← **CRITICAL SOUNDNESS BUG**
- 46 instances: nnenum=sat, abc=timeout
- 46 instances: nnenum=sat, abc=no_result_in_file

**Affected Benchmarks**:
- cifar100_2024 (ResNet with skip connections): 171 mismatches
- malbeware: 77 mismatches
- relusplitter: 51 mismatches
- metaroom_2023: 48 mismatches

**Root Cause Hypothesis**: Parameter space mismatch in `SkipAddLayer.transform_star()` when combining skip and main paths.

The issue occurs when:
1. Skip path star is saved early in the network with N generators (one per input dimension)
2. Main path undergoes ReLU splits, adding K new generator columns (N+K total generators)
3. At SkipAddLayer, `transform_star(star_skip, star_main)` tries to add the generator matrices
4. Assertion at [network.py:1022](src/nnenum/network.py#L1022) requires `star_skip.a_mat.shape == star_main.a_mat.shape`
5. **Shape mismatch**: skip has (output_dim, N), main has (output_dim, N+K)

**Why Unit Tests Passed**:
- Tests in `tests/step*.py` use tiny 2-4 neuron networks
- With tight input boxes, overapproximation produces zero ReLU splits
- Parameter spaces remain same size → assertion passes
- Real CIFAR100 networks: 3072 inputs, deep ResNet, many splits → shape mismatch

**Temporary Workaround**:
The code has a fallback at [overapprox.py:518-519](src/nnenum/overapprox.py#L518) that silently ignores skip connections when the skip source is unavailable:
```python
else:
    # Skip source not available; leave star unchanged (unsound but graceful)
    pass
```
This prevents crashes but causes **unsound under-approximation** of the reachable set.

**Fix Required**:
Proper fix requires aligning parameter spaces before addition, either by:
1. Padding skip star with zero-columns for ReLU splits that happened on main path
2. Restructuring skip connection handling in overapproximation to maintain common parameter space
3. Using a different abstract domain that handles skip connections more naturally

This is a significant architectural issue requiring careful design and extensive testing.

**Current Status**:
- Bug identified and documented
- Work being done on separate branch
- Main branch focusing on convolution performance optimizations first
- Will return to fix this soundness issue after performance work is complete

**Validation**: Use `validate_results.py` to compare nnenum vs ABC results after any changes to skip connection handling.
