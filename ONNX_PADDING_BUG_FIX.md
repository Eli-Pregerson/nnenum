# ONNX Convolution Padding Bug Fix

## Issue

**Soundness bug** discovered: nnenum reported `sat` on `relusplitter/oval21-benchmark_cifar_base_kw-img2537` when alpha-beta-CROWN (correct) reported `unsat`.

## Root Cause

The bug was in **baseline nnenum's ONNX Conv handling** (not in the convolution batching optimization):

1. **ONNX parsing** (`onnx_network.py`) ignored explicit `pads` attribute values
2. Only checked if `pads` were all zero (→ `mode='valid'`) or not (→ `mode='same'`)
3. For networks with `pads=[1,1,1,1]` and `kernel=4x4`:
   - ONNX expected padding: `[top=1, left=1, bottom=1, right=1]`
   - scipy `mode='same'` produced: different (asymmetric) padding
4. This caused **incorrect convolution results**, leading to wrong verification

**Timeline**:
- Before: Network had `error_exit_code_1` (parsing failed due to other bugs)
- After fixing Constant/Reshape/Conv parsing: Network ran but produced wrong result (`sat` instead of `unsat`)
- The ONNX padding bug existed all along but was hidden by parsing errors

## Investigation Process

1. **Suspected convolution batching** (our recent optimization)
2. **Ran comprehensive tests**: All 27 tests passed with max diff = 0.0
3. **Conclusion**: Batching algorithm is correct
4. **Analyzed ONNX parsing**: Found that explicit padding values were ignored
5. **Verified bug**: scipy `mode='same'` produces different output than ONNX `pads=[1,1,1,1]` for 4x4 kernel

## Fix

### Changes Made

**1. `network.py` - `Convolutional2dLayer.__init__`**
- Added `pads` parameter to store ONNX explicit padding values
- Format: `(top, left, bottom, right)` or `None` for auto-padding

**2. `network.py` - `Convolutional2dLayer.execute`**
- Check if `self.pads is not None`
- If yes: manually pad input with `np.pad()`, then use `mode='valid'`
- If no: use original scipy auto-padding (`mode='same'` or `mode='valid'`)

**3. `network.py` - `Convolutional2dLayer.get_output_shape`**
- Account for explicit padding when calculating output dimensions
- Formula: `output = (input + top + bottom - kernel) / stride + 1`

**4. `network.py` - `Convolutional2dLayer._compute_output_region`**
- Handle explicit padding in batching output region calculation
- Convert input coordinates to padded coordinates before computing output region

**5. `onnx_network.py` - Conv parsing**
- Extract actual `pads` values from ONNX attributes
- Pass them to `Convolutional2dLayer` constructor
- Only use auto-padding for `auto_pad='SAME_UPPER/LOWER'` or when pads are all zero

## Testing

### Test Suite

**test_onnx_padding_fix.py** - Explicit padding correctness
- ✓ Output shapes match ONNX spec
- ✓ Execution matches manual ONNX-style computation (max diff: 1.04e-07)
- ✓ ONNX pads ≠ scipy 'same' (confirms bug was real)
- ✓ Batching works with explicit padding

**test_relusplitter_config.py** - Failing network configuration
- ✓ Input: 32x32x3, kernel: 4x4, stride: 2, pads: [1,1,1,1]
- ✓ Output shape: 16x16x16 (correct)
- ✓ Batching: 200 generators, max diff: 0.0
- ✓ Second conv layer works

**test_conv_batching_comprehensive.py** - Regression tests
- ✓ All 27 existing tests still pass
- ✓ No regressions introduced

## Results

**Before fix**:
- relusplitter instance: `sat` (WRONG)
- ABC (correct): `unsat`

**After fix**:
- Should now produce correct result
- Need full verification run to confirm (requires dependencies)

## Files Modified

1. `src/nnenum/network.py`:
   - `Convolutional2dLayer.__init__`: Added `pads` parameter
   - `Convolutional2dLayer.execute`: Handle explicit padding
   - `Convolutional2dLayer.get_output_shape`: Account for explicit pads
   - `Convolutional2dLayer._compute_output_region`: Handle explicit pads in batching

2. `src/nnenum/onnx_network.py`:
   - Conv parsing: Extract and pass ONNX `pads` attribute

## Impact

This fix resolves a **baseline soundness bug** that affected any ONNX Conv layer with:
- Explicit non-zero padding
- Kernel size where `pads ≠ kernel_size // 2` (e.g., 4x4 kernel with pads=[1,1,1,1])

The bug was previously hidden by parsing errors that prevented affected networks from running.

## Verification

To fully verify the fix resolves the soundness issue, run:
```bash
python -m nnenum.nnenum \
  -o vnncomp2025_benchmarks/benchmarks/relusplitter/onnx/oval21-benchmark_cifar_base_kw.onnx \
  -v vnncomp2025_benchmarks/benchmarks/relusplitter/vnnlib/oval21-benchmark_cifar_base_kw-img2537-eps0.006666666666666667.vnnlib
```

Expected result: `unsat` (matching ABC)
