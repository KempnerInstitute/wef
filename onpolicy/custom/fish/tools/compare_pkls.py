#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

import numpy as np
import pandas as pd

def values_equal(a, b, float_tol=1e-8):
    """Safely compare two cell values, including array-valued cells."""

    # Fast path: identical object
    if a is b:
        return True

    # Numpy arrays (including dtype=object arrays)
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.shape != b.shape:
            return False
        # If numeric-ish, allclose; else elementwise recursive
        if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
            return np.allclose(a, b, atol=float_tol, equal_nan=True)
        return all(values_equal(x, y, float_tol) for x, y in zip(a.ravel(), b.ravel()))

    # Lists / tuples
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(values_equal(x, y, float_tol) for x, y in zip(a, b))

    # Scalars: handle NaN / NaT safely
    if pd.isna(a) and pd.isna(b):
        return True

    # Numeric scalars with tolerance
    if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
        return bool(np.isclose(a, b, atol=float_tol, equal_nan=True))

    # Fallback
    return a == b



def main(pkl1, pkl2, float_tol=1e-8, max_report=10):
    print(f"Loading:\n  {pkl1}\n  {pkl2}\n")

    df1 = pd.read_pickle(pkl1)
    df2 = pd.read_pickle(pkl2)

    print("=== STRUCTURE CHECKS ===")
    print(f"Same shape:   {df1.shape == df2.shape}")
    print(f"Same columns: {df1.columns.equals(df2.columns)}")
    print(f"Same index:   {df1.index.equals(df2.index)}")
    print()

    aligned = (
        df1.shape == df2.shape
        and df1.columns.equals(df2.columns)
        and df1.index.equals(df2.index)
    )

    # ---- Exact-ish equality ----
    print("=== VALUE CHECKS ===")

    if not aligned:
        print("DataFrames not aligned — skipping value comparison.")
        return

    unequal = []

    for col in tqdm(df1.columns):
        print(f"Comparing column: {col}")
        for idx in df1.index:
            v1 = df1.at[idx, col]
            v2 = df2.at[idx, col]

            try:
                eq = values_equal(v1, v2, float_tol)
            except Exception as e:
                eq = False

            if not eq:
                unequal.append((idx, col, v1, v2))
                if len(unequal) >= max_report:
                    break
        if len(unequal) >= max_report:
            break

    if not unequal:
        print("All values equal (array-safe comparison). ✅")
    else:
        print(f"Found {len(unequal)} differing cells (showing up to {max_report}):\n")
        for idx, col, v1, v2 in unequal:
            print(f"Index: {idx}, Column: {col}")
            print(f"  df1: {type(v1)} {v1}")
            print(f"  df2: {type(v2)} {v2}")
            print()

    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_pkls.py <file1.pkl> <file2.pkl>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
