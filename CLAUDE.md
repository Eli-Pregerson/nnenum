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

## Convolution Optimization (Generator Batching)

**Added**: 2025-02-24

A significant optimization for convolutional layers that batches sparse generators to reduce the number of convolution operations.

### Key Insight

In image verification, initial generators are typically one-hot vectors (one per input pixel-channel). For a 32×32×3 CIFAR image, this means 3072 extremely sparse generators. Without optimization, each generator requires a separate convolution operation (3072 convolutions per layer).

Since convolution is **linear** (`conv(a + b) = conv(a) + conv(b)`) and generators with non-overlapping regions produce non-overlapping outputs, we can:
1. **Batch** generators whose output regions don't overlap
2. **Combine** them into a single tensor
3. Perform **one convolution** instead of N
4. **Extract** each generator's contribution by masking its output region

### Implementation

**Location**: `src/nnenum/network.py` - `Convolutional2dLayer` class

**Methods**:
- `_compute_output_region()` - Maps input nonzero region to output region after convolution
- `_batch_generators_for_conv()` - Greedy batching algorithm that groups non-conflicting generators
- Modified `transform_star()` and `transform_zono()` - Apply batching with adaptive threshold

**Adaptive Batching**: The optimization automatically skips batching when generators are too dense:
- Checks sparsity: `nonzeros / total_elements`
- If sparsity > `Settings.CONV_BATCHING_MIN_SPARSITY` (default 5%), uses unbatched path
- Avoids overhead when batching won't help (later conv layers)

### Performance

**CIFAR-scale (32×32×3)**:
- **First conv layer**: 3072 generators → 27 batches (~114x compression) → **4.7x speedup**
- **Second conv layer**: Generators become denser (~0.6% sparsity) → 5.3x compression → still beneficial
- **Third conv layer**: Further degradation (~1.4% sparsity) → 4.0x compression → marginal benefit
- **Dense layers** (>5% sparsity): Batching skipped automatically to avoid overhead

### Settings

```python
from nnenum.settings import Settings

# Enable logging to see batching statistics
Settings.LOG_CONV_BATCHING = True

# Adjust sparsity threshold (default: 0.05 = 5%)
Settings.CONV_BATCHING_MIN_SPARSITY = 0.02  # More aggressive

# Only batch first conv layer (for very deep networks)
Settings.CONV_BATCHING_FIRST_LAYER_ONLY = True
```

**Recommendation**: Use defaults. The adaptive sparsity check handles most cases automatically.

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
