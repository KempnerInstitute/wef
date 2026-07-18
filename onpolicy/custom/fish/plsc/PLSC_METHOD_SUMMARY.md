# PLSC Math and Method Summary

This implementation ports the PLSC analysis from `hongw-lab/code_for_2024_zhang-phi` into Python/NumPy.

## Inputs

PLSC compares two time-aligned datasets:

- `X`: shape `(n_samples, n_features_x)`
- `Y`: shape `(n_samples, n_features_y)`

Rows must correspond to the same time samples or observations. Columns are features, such as cells, neural traces, or behavioral variables.

The original MATLAB code assumes both inputs are z-scored column-wise before analysis. The Python helper `zscore` uses sample standard deviation (`ddof=1`) to match MATLAB's default convention.

## Cross-Covariance Matrix

PLSC starts by computing the cross-covariance matrix:

```text
C = X.T @ Y / (n_samples - 1)
```

`C[i, j]` measures covariance between feature `i` in `X` and feature `j` in `Y`.

## Singular Value Decomposition

The shared latent space is found with SVD:

```text
C = U S V.T
```

Where:

- `U` contains loadings for `X`
- `V` contains loadings for `Y`
- `S` contains singular values

Each singular value measures the covariance captured by one paired latent variable. Larger singular values indicate stronger shared structure between the datasets.

## Projected Scores

The datasets are projected into their PLSC spaces:

```text
T_X = X @ U
T_Y = Y @ V
```

Matching columns of `T_X` and `T_Y` are paired latent-variable time courses. The implementation computes the correlation between matching projected columns to quantify how tightly each paired latent variable co-varies over time.

## Temporal-Shift Null Test

To estimate which PLSC dimensions are significant, the method builds a null distribution by circularly shifting `Y` in time:

```text
Y_perm = circular_shift(Y, random_shift)
```

The shift preserves the internal temporal structure of `Y` but disrupts time alignment between `X` and `Y`.

For each permutation:

1. Shift `Y`.
2. Recompute PLSC between `X` and shifted `Y`.
3. Store singular values.
4. Store projected-score correlations.

This produces two null distributions per latent dimension:

- covariance null distribution
- projected-correlation null distribution

## Significance Rule

For each dimension, the observed covariance and observed projected correlation are compared against permutation thresholds.

A dimension is significant only if:

```text
observed_covariance > covariance_threshold
observed_correlation > correlation_threshold
```

Counting proceeds in order from the first latent dimension and stops at the first non-significant dimension. This follows the ordered latent-variable logic used in the MATLAB code.

## Shared and Unique Spaces

After estimating `num_sig_dims`, the first significant PLSC dimensions are treated as the shared space:

```text
U_shared = U[:, :num_sig_dims]
V_shared = V[:, :num_sig_dims]
X_shared = X @ U_shared
Y_shared = Y @ V_shared
```

The remaining PLSC dimensions are treated as residual structure:

```text
X_residual = X @ U[:, num_sig_dims:]
Y_residual = Y @ V[:, num_sig_dims:]
```

PCA is then applied separately to each residual matrix to summarize dataset-specific unique structure.

## Interpretation

PLSC identifies paired directions in two datasets that maximize cross-dataset covariance.

- High singular value: strong shared covariance.
- High projected-score correlation: paired latent variables vary together over samples.
- Significant dimension: observed shared structure exceeds what is expected after disrupting temporal alignment.
- Shared space: dimensions jointly supported by both covariance and correlation null tests.
- Unique space: residual structure left after removing significant shared dimensions.

## Practical Notes

- Z-score features before running PLSC unless raw feature scale is scientifically meaningful.
- Use enough permutations for stable thresholds. The upstream MATLAB default is `2000`.
- Choose a temporal-shift lag large enough to break alignment but smaller than the recording length.
- PLSC requires matched rows; missing samples or mismatched time bases should be handled before analysis.

