# nnenum Work Items

Ordered by current priority (subject to change). Remove items when complete; add new ones as discovered.

---

## 1. VGGNet Support via Full Sparse Conv-ReLU Path (Highest Priority)

**Goal**: Get results on `vggnet16_2022` (18 instances, 224×224×3 input, 13 conv + 5 MaxPool + 3 dense layers). VGGNet is memory-bound for dense tools — the hope is to pass instances no other tool can handle by staying in sparse representation through the early layers.

**Why sparse matters here**: A one-hot generator has 150K elements but only 64–576 nonzeros through the first two conv blocks. Dense representation requires `150K × G` floats — prohibitive for large G. Sparse stays feasible. Generators cross the 5% density threshold only after pool2 (layer ~16 of 41), so ~40% of the network benefits.

**Two bugs to fix first**:
1. **float_data vs raw_data** (`src/nnenum/onnx_network.py`): VGGNet stores weights in `float_data`, not `raw_data`. Add `_tensor_to_numpy(init)` helper and replace all `np.frombuffer(init.raw_data, ...)` calls. `onnx.numpy_helper.to_array()` handles this correctly and could be used as alternative.
2. **MaxPool**: Handled externally by user (preprocessing ONNX to replace MaxPool with ReLU-based equivalent).

**Sparse conv-ReLU path** (`CONV_METHOD='sparse'`, keep mat_t as CSC through layers):
- `_apply_conv_to_mat_sparse` → return `csc_matrix` instead of calling `.toarray()`
- `transform_zono` → store sparse `mat_t`; convert to dense when density > `CONV_BATCHING_MIN_SPARSITY`
- `update_zono` (`overapprox.py`) → sparse-aware: row zeroing (LIL conversion), row scaling (CSR conversion), new generator hstack (`scipy.sparse.hstack`)
- `box_bounds()` and other zonotope ops → audit `mat_t` usages, add `.toarray()` where 1D dense needed
- `DeepPolyOverapprox.__init__` → densify `mat_t` before copying to `ubcoef`/`lbcoef`
- Star `a_mat` stays dense (LP construction assumes dense — too invasive to change)

**Implementation order**: (1) fix float_data bug → (2) verify loading with preprocessed VGGNet → (3) return CSC from sparse conv → (4) store sparse mat_t + density gate → (5) sparse-aware ReLU update → (6) zonotope op audit → (7) end-to-end test

**Relevant code**:
- `src/nnenum/onnx_network.py`: weight loading, ~lines 449, 489, 508, 514, 539, 547, 671–674
- `src/nnenum/network.py`: `_apply_conv_to_mat_sparse`, `transform_zono`
- `src/nnenum/overapprox.py`: `update_zono`, `relu_update_best_area_zono`, `DeepPolyOverapprox.__init__`
- `src/nnenum/zonotope.py`: `box_bounds()` and all `mat_t` usages

**Status**: Not started (plan written 2026-03-08)

---

## 2. Conv Timing Analysis (deprioritized — dense is now default, batching obsolete)

**Goal**: Find the hardest conv instances nnenum can still decide, profile them to understand where time is spent, and assess how much further speedup is achievable in the conv path.

**Approach**:
- Identify candidates from `vnncomp2025_benchmarks/all_results_batching.csv`: slow decided instances in conv-heavy benchmarks (`malbeware`, `metaroom_2023`, `relusplitter`, `cgan_2023`, `collins_rul_cnn_2022`)
- Current hardest decided instances (~top runtimes):
  - `metaroom_2023`: up to 76s
  - `cgan_2023`: up to 75s (now uses optimized path with ConvTranspose support)
  - `malbeware`: up to 57s
- Run with `Settings.TIMING_STATS = True` to get per-operation breakdown
- Key question: what fraction of time is in `transform_star_batched_conv` vs `transform_star_unbatched` vs LP solving?

**Relevant code**:
- `src/nnenum/network.py`: `Convolutional2dLayer.transform_star/zono`, `_batch_generators_for_conv`
- `src/nnenum/settings.py`: `CONV_BATCHING_*` settings, `TIMING_STATS`
- `src/nnenum/timerutil.py`: timing infrastructure

**Status**: Not started

---

## 2. Extend Conv Optimizations to ConvTranspose Layers

**Goal**: Apply generator batching (and potentially the sparse matrix method) to ConvTranspose layers in `cgan_2023`, which currently have batching disabled.

**Context**: ConvTranspose is exclusively used in `cgan_2023` (17 instances across 5 ONNX files). Batching is geometrically valid — two generators with non-overlapping input regions produce non-overlapping output regions (in fact output regions are larger than input due to upsampling, making non-overlap easier to satisfy for sparse generators). cgan starts with only 5 inputs (3 non-zero), so early ConvTranspose generators are very sparse and should compress well.

**Approach**:
- Adapt `_compute_output_region` for ConvTranspose geometry: `out_region = [in_h * stride, in_h * stride + kernel - 1]` per spatial dim
- Adapt `_batch_generators_for_conv` to use the ConvTranspose output region formula
- Remove the `elif self.is_transpose: should_batch = False` guard in `transform_star` and `transform_zono`
- The `_apply_conv_to_mat` batched path already handles ConvTranspose correctly (im2col with flipped kernels)

**Affected benchmarks**: `cgan_2023` only (17 instances)

**Status**: Not started

---

## 3. Sparse Matrix Conv Method

**Status**: Partially complete — `CONV_METHOD='sparse'` implemented in `network.py` (sparse Toeplitz W, `_apply_conv_to_mat_sparse`, `_build_conv_matrix`). Benchmarking showed sparse wins for large-image early layers (0.68–0.89x of dense). The next step — propagating sparse mat_t through ReLU — is now item 1.

---

## 4. LP Solver Comparison

**Goal**: Evaluate alternative LP solvers against the current GLPK backend to measure their impact on verification throughput.

**Context**: nnenum uses GLPK (via `swiglpk`) for LP solving, which is the performance bottleneck for hard instances requiring many ReLU splits. The `Settings.LP_SOLVER` setting already supports `'GLPK'` and `'Gurobi'` as options. Other solvers (HiGHS, CPLEX, Mosek, SoPlex) may offer significant speedups, especially on the dense LPs that arise from deep splitting.

**Approach**:
- Identify the slowest decided instances where LP time dominates (low conv%, high total time)
- Profile LP call count and time per call using `Settings.TIMING_STATS = True`
- Add support for at least one additional solver (HiGHS is free, fast, and has a Python interface via `highspy`)
- Run the same instances with each solver and compare total verification time
- Key files: `src/nnenum/lpinstance.py`, `src/nnenum/lpinstance_glpk.py`, `src/nnenum/settings.py`

**Relevant code**:
- `src/nnenum/lpinstance.py`: LP abstraction layer
- `src/nnenum/lpinstance_glpk.py`: GLPK implementation
- `src/nnenum/lpinstance_gb.py`: Gurobi implementation (exists but requires license)
- `src/nnenum/settings.py`: `LP_SOLVER`, `GLPK_TIMEOUT`, `GLPK_FIRST_PRIMAL`

**Status**: Not started

---

## 5. Skip Connections / DAG Structures

**Goal**: Support residual/skip connections (Add nodes where one input is not the immediately preceding layer's output) and other non-sequential graph topologies.

**Context**: Currently the optimized ONNX parser (`load_onnx_network_optimized`) assumes a strictly linear node chain. ResNet-style architectures (`cifar100_2024`, 8 Add nodes with skip connections) fail with:
```
AssertionError: multiple onnx nodes accept network input 121
```

**Approach** (being worked on in a separate branch):
- Replace the sequential `cur_input_name` pointer with a proper topological sort + DAG traversal
- Introduce a `SkipConnectionLayer` or extend `AddLayer` to accept two inputs (one from current chain, one from a stored intermediate output)
- Store intermediate layer outputs by name during forward pass

**Affected benchmarks**: `cifar100_2024` (ResNet, 19 BN layers, 8 Add skip nodes)

**Status**: In progress on separate branch — not to be worked on here until merged

---

## Recently Completed

- **ConvTranspose layer support** (`cgan_2023`): Added `is_transpose` flag to `Convolutional2dLayer`; weights swapped (axes 0↔1) and not pre-flipped; `execute()` upsamples input then uses `convolve2d(mode='full')`; batching disabled for transpose layers; added `ConvTranspose` ONNX handler. All cgan networks now use the optimized path. *(2026-03-03)*

- **BatchNorm after non-Conv/FC layers** (`cgan_2023`): Added `ScaleLayer` class to `network.py`; updated BatchNorm handler in `onnx_network.py` to fall back to `ScaleLayer` instead of raising `RuntimeError`. Fixes `Reshape → BatchNorm` pattern. *(2026-03-02)*

- **FlattenLayer / ReshapeLayer HWC→CHW permutation**: Fixed `transform_star` and `transform_zono` for 3D→1D transitions to apply `_hwc_to_chw_permutation`. Eliminated 47 false SATs in `metaroom_2023`. *(2026-02-26)*

- **Conv generator batching**: Added `_batch_generators_for_conv` to `Convolutional2dLayer`; +248 decided instances vs pre-optimization baseline, 0 soundness violations. *(2026-02-26)*

- **Constant/Reshape/BatchNorm layer support**: Added `ConstantLayer`, fixed `ReshapeLayer`, added BatchNorm folding into Conv/FC. *(2026-02-16)*
