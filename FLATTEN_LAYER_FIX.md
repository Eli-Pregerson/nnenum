# Flatten Layer HWC→CHW Conversion Bug Fix

## Issue

After fixing the ONNX padding bug, nnenum execution still differed dramatically from onnxruntime (max diff ≈ 1.96) on the relusplitter benchmark.

## Root Cause

**Layout mismatch between convolutional layers and fully-connected layers**:

1. nnenum uses **HWC format** (Height, Width, Channels) for convolutional layers
2. ONNX uses **CHW format** (Channels, Height, Width) throughout
3. When Flatten layer flattens HWC data with row-major order ('C'), it produces:
   ```
   [h0_w0_c0, h0_w0_c1, ..., h0_w0_cN, h0_w1_c0, h0_w1_c1, ...]
   ```
4. But FC layer weights were trained expecting CHW-flattened data:
   ```
   [c0_h0_w0, c0_h0_w1, ..., c0_hM_wN, c1_h0_w0, c1_h0_w1, ...]
   ```
5. These orderings are **completely different**, causing wrong FC layer outputs

## Investigation

Tested layer-by-layer execution:
- Conv layers: Statistics match (mean/std), but element order differs due to HWC vs CHW layout (expected)
- After flatten: nnenum fed HWC-ordered data to FC layers trained on CHW-ordered data
- Result: Massive errors in final output (max diff 1.96)

## Fix

**Modified**: `src/nnenum/network.py` - `FlattenLayer.execute()`

Added HWC → CHW conversion before flattening:

```python
def execute(self, state):
    '''execute the layer on a concrete state

    returns output
    '''

    # ONNX Flatten expects CHW format, but nnenum uses HWC
    # Need to convert HWC (H, W, C) → CHW (C, H, W) before flattening
    if len(state.shape) == 3:
        # Convert from HWC to CHW
        state = np.transpose(state, (2, 0, 1))

    rv = nn_flatten(state)
    assert rv.shape == self.output_shape

    return rv
```

## Testing

**Before fix**:
- nnenum output: `[0.481, -0.830, 0.489, 0.998, -0.700, 1.267, -3.258, 1.734, -1.610, 1.431]`
- onnxruntime: `[2.400, 1.131, -0.345, -0.374, -0.305, 0.185, -3.772, 2.498, -2.325, 0.907]`
- Max diff: **1.96** (WRONG!)

**After fix**:
- nnenum output: `[2.400062, 1.130798, -0.344512, -0.373939, -0.305053, 0.185067, -3.772373, 2.497964, -2.324973, 0.906920]`
- onnxruntime:   `[2.400062, 1.130799, -0.344512, -0.373939, -0.305053, 0.185066, -3.772374, 2.497963, -2.324973, 0.906920]`
- Max diff: **7.52e-07** (CORRECT!)

## Impact

This fix resolves a **critical soundness bug** that affected:
- **ALL networks** with conv layers followed by flatten + FC layers
- The bug caused nnenum to compute completely wrong network outputs
- Led to both false positives (spurious counterexamples) and false negatives (missed violations)

## Files Modified

1. `src/nnenum/network.py`:
   - `FlattenLayer.execute()`: Added HWC → CHW transpose before flattening

## Related Fixes

This fix builds on the ONNX padding fix (see `ONNX_PADDING_BUG_FIX.md`). Both were discovered while investigating the relusplitter benchmark discrepancy with alpha-beta-CROWN.

## Remaining Question

After both fixes, nnenum now matches onnxruntime exactly. Both find a counterexample at the center of the epsilon ball for `relusplitter/oval21-benchmark_cifar_base_kw-img2537-eps0.006666666666666667`, but alpha-beta-CROWN reports `unsat`. This requires further investigation:
- Is alpha-beta-CROWN using a different epsilon?
- Is there preprocessing/normalization we're missing?
- Is the vnnlib file incorrect?
- Does alpha-beta-CROWN have a completeness issue for this instance?
