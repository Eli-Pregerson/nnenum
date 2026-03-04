# nnenum Work Items

Ordered by current priority (subject to change). Remove items when complete; add new ones as discovered.

---

## 1. Conv Timing Analysis (Highest Priority)

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

**Goal**: Implement an alternative conv path using sparse matrix representations and benchmark against the current generator-batching approach.

**Approach**:
- Express convolution as a sparse matrix-vector product: build the Toeplitz-like sparse matrix for the conv kernel, then apply it to a batch of generators at once
- Use `scipy.sparse` (already imported in `onnx_network.py`) for the sparse matrix
- Compare: (a) current batching approach, (b) sparse matmul approach, (c) naive unbatched baseline
- Key trade-off: batching avoids the overhead of constructing the sparse matrix; sparse matmul may win for layers where batching compression is low

**Relevant code**:
- `src/nnenum/network.py`: `Convolutional2dLayer` — add alternative `transform_star_sparse()` method
- `src/nnenum/onnx_network.py`: `from scipy.sparse import csc_matrix, csr_matrix` already imported
- `src/nnenum/settings.py`: add `CONV_METHOD` setting to switch between approaches

**Prerequisite**: Complete item 1 first — timing analysis will clarify whether the conv path is the bottleneck and which layer indices matter most.

**Status**: Not started

---

## 3. Skip Connections / DAG Structures

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
