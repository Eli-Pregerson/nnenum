#!/usr/bin/env python3
"""Debug why the counterexample is accepted as valid."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from nnenum.specification import Specification, DisjunctiveSpec

# The output from the debug run
output = np.array([0.48119778327422336, -0.830470387596345, 0.48899494090638684,
                   0.9982284001529036, -0.700484942487842, 1.2665249388062616,
                   -3.2582534573228172, 1.7339393974217263, -1.6104245669788997,
                   1.430659871609989])

print("="*80)
print("COUNTEREXAMPLE VALIDATION DEBUG")
print("="*80)

print(f"\nOutput vector:")
for i, val in enumerate(output):
    print(f"  Y_{i} = {val:.6f}")

# Test each clause of the disjunctive property
# (or (and (<= Y_1 Y_0)) (and (<= Y_1 Y_2)) ... (and (<= Y_1 Y_9)))

print(f"\n{'='*80}")
print("Testing individual clauses:")
print('='*80)

Y_1 = output[1]
print(f"\nY_1 = {Y_1:.6f}\n")

spec_list = []

for i in range(10):
    if i == 1:
        continue  # Skip Y_1 <= Y_1

    # Create specification for (<= Y_1 Y_i)
    # Encoded as: Y_1 - Y_i <= 0
    mat = np.zeros((1, 10))
    mat[0, 1] = 1.0   # Y_1
    mat[0, i] = -1.0  # Y_i
    rhs = np.array([0.0])

    spec = Specification(mat, rhs)
    spec_list.append(spec)

    # Compute result
    result = np.dot(mat, output)[0]
    Y_i = output[i]

    is_viol = spec.is_violation(output)

    print(f"Clause: Y_1 <= Y_{i}")
    print(f"  Y_{i} = {Y_i:.6f}")
    print(f"  Actual: {Y_1:.6f} <= {Y_i:.6f}? {Y_1 <= Y_i}")
    print(f"  Matrix form: {result:.6f} <= 0? {result <= 0}")
    print(f"  spec.is_violation(output) = {is_viol}")

    if is_viol:
        print(f"  ✓ This clause is SATISFIED (counterexample for this clause)")
    else:
        print(f"  ✗ This clause is NOT satisfied")
    print()

# Create disjunctive spec
disjunctive = DisjunctiveSpec(spec_list)

print(f"{'='*80}")
print("DISJUNCTIVE SPEC RESULT")
print('='*80)

is_violation = disjunctive.is_violation(output)
print(f"\ndisjunctive.is_violation(output) = {is_violation}")

if is_violation:
    print("\n⚠️  BUG: Disjunctive spec says this IS a violation!")
    print("But we know it's NOT because Y_1 is not <= Y_6 or Y_8")
else:
    print("\n✓ Disjunctive spec correctly says this is NOT a violation")

print(f"\n{'='*80}")
print("ANALYSIS")
print('='*80)

print("\nFor this to be a valid counterexample, at least ONE clause must be satisfied.")
print("That means Y_1 must be <= at least one other output.")
print(f"\nY_1 = {Y_1:.6f}")
print("\nChecking against all other outputs:")

for i in range(10):
    if i == 1:
        continue
    Y_i = output[i]
    satisfied = Y_1 <= Y_i
    symbol = "✓" if satisfied else "✗"
    print(f"  {symbol} Y_1 <= Y_{i}: {Y_1:.6f} <= {Y_i:.6f}? {satisfied}")

num_satisfied = sum(1 for i in range(10) if i != 1 and Y_1 <= output[i])
print(f"\nNumber of satisfied clauses: {num_satisfied}")

if num_satisfied > 0:
    print("✓ This IS a valid counterexample")
else:
    print("✗ This is NOT a valid counterexample (BUG!)")
