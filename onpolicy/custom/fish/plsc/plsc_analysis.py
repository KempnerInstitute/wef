"""Partial least squares correlation (PLSC) analysis utilities.

This module ports the PLSC analysis routines from the MATLAB code in
hongw-lab/code_for_2024_zhang-phi to Python/NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


ArrayLike = np.ndarray


@dataclass(frozen=True)
class PLSCResult:
    """Result of a PLSC decomposition."""

    loading1: np.ndarray
    loading2: np.ndarray
    singular_values: np.ndarray
    projected_data1: np.ndarray
    projected_data2: np.ndarray


@dataclass(frozen=True)
class SharedNullResult:
    """Permutation-test result for significant shared PLSC dimensions."""

    num_sig_dims: int
    observed_cov: np.ndarray
    observed_corr: np.ndarray
    cov_dist: np.ndarray
    corr_dist: np.ndarray
    cov_thresholds: np.ndarray
    corr_thresholds: np.ndarray


@dataclass(frozen=True)
class SpaceProjection:
    """Loadings and scores for one part of a decomposed space."""

    loading1: np.ndarray
    loading2: np.ndarray
    projected_data1: np.ndarray
    projected_data2: np.ndarray


@dataclass(frozen=True)
class NeuralSpace:
    """Shared and unique spaces derived from PLSC loadings."""

    shared: SpaceProjection
    unique: SpaceProjection


def _as_2d_float(name: str, data: ArrayLike) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array with shape (samples, features).")
    if arr.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two samples.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr


def _validate_pair(data1: ArrayLike, data2: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    x = _as_2d_float("data1", data1)
    y = _as_2d_float("data2", data2)
    if x.shape[0] != y.shape[0]:
        raise ValueError("data1 and data2 must have the same number of rows.")
    return x, y


def zscore(data: ArrayLike, ddof: int = 1) -> np.ndarray:
    """Z-score columns, leaving constant columns as zeros.

    Parameters
    ----------
    data:
        Matrix with shape ``(samples, features)``.
    ddof:
        Delta degrees of freedom for the standard deviation. The default
        matches MATLAB's sample-standard-deviation convention.
    """

    x = _as_2d_float("data", data)
    centered = x - np.mean(x, axis=0, keepdims=True)
    scale = np.std(centered, axis=0, ddof=ddof, keepdims=True)
    return np.divide(centered, scale, out=np.zeros_like(centered), where=scale > 0)


def plsc(data1: ArrayLike, data2: ArrayLike) -> PLSCResult:
    """Compute PLSC loadings that maximize covariance between two datasets.

    Inputs are expected to be z-scored arrays with shape
    ``(samples, features)``. The decomposition follows the MATLAB routine:
    ``cov_mat = data1.T @ data2 / (n_samples - 1)``, then SVD.
    """

    x, y = _validate_pair(data1, data2)
    cov_mat = x.T @ y / (x.shape[0] - 1)
    loading1, singular_values, loading2_t = np.linalg.svd(cov_mat, full_matrices=True)
    loading2 = loading2_t.T
    return PLSCResult(
        loading1=loading1,
        loading2=loading2,
        singular_values=singular_values,
        projected_data1=x @ loading1,
        projected_data2=y @ loading2,
    )


def paired_column_corr(data1: ArrayLike, data2: ArrayLike) -> np.ndarray:
    """Correlation between matching columns of two matrices."""

    x, y = _validate_pair(data1, data2)
    n_dims = min(x.shape[1], y.shape[1])
    x = x[:, :n_dims] - np.mean(x[:, :n_dims], axis=0, keepdims=True)
    y = y[:, :n_dims] - np.mean(y[:, :n_dims], axis=0, keepdims=True)
    denom = np.linalg.norm(x, axis=0) * np.linalg.norm(y, axis=0)
    return np.divide(np.sum(x * y, axis=0), denom, out=np.zeros(n_dims), where=denom > 0)


def temp_shift(
    data: ArrayLike,
    lag: int = 60,
    *,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, int]:
    """Circularly shift a time-series matrix by a random lag.

    This mirrors the upstream MATLAB ``tempShift`` helper. The selected shift
    is at least roughly ``lag / 2`` samples away from zero and no more than
    roughly ``n_samples - lag / 2``.
    """

    x = _as_2d_float("data", data)
    if not isinstance(lag, int) or lag <= 0:
        raise ValueError("lag must be a positive integer.")
    if x.shape[0] <= lag:
        raise ValueError("lag must be smaller than the number of samples.")
    generator = np.random.default_rng() if rng is None else rng
    shift = int(generator.integers(1, x.shape[0] - lag + 1) + lag // 2)
    return np.roll(x, shift=shift, axis=0), shift


def compute_shared_null_distribution(
    data1: ArrayLike,
    data2: ArrayLike,
    num_sims: int = 2000,
    sig_thr: float = 0.975,
    lag: int = 60,
    *,
    rng: Optional[np.random.Generator] = None,
) -> SharedNullResult:
    """Estimate significant shared PLSC dimensions with temporal permutations.

    A dimension is counted as significant only while both its observed
    covariance and projected-score correlation exceed the corresponding
    permutation threshold. Counting stops at the first non-significant
    dimension, matching the ordered latent-variable logic in the MATLAB code.
    """

    x, y = _validate_pair(data1, data2)
    if not isinstance(num_sims, int) or num_sims <= 0:
        raise ValueError("num_sims must be a positive integer.")
    if not 0 < sig_thr < 1:
        raise ValueError("sig_thr must be between 0 and 1.")

    generator = np.random.default_rng() if rng is None else rng
    observed = plsc(x, y)
    n_dims = min(x.shape[1], y.shape[1])
    observed_cov = observed.singular_values[:n_dims]
    observed_corr = paired_column_corr(observed.projected_data1, observed.projected_data2)

    cov_dist = np.empty((n_dims, num_sims), dtype=float)
    corr_dist = np.empty((n_dims, num_sims), dtype=float)

    for sim in range(num_sims):
        y_perm, _ = temp_shift(y, lag=lag, rng=generator)
        permuted = plsc(x, y_perm)
        cov_dist[:, sim] = permuted.singular_values[:n_dims]
        corr_dist[:, sim] = paired_column_corr(
            permuted.projected_data1,
            permuted.projected_data2,
        )

    quantile_index = int(np.clip(round(sig_thr * num_sims) - 1, 0, num_sims - 1))
    cov_thresholds = np.sort(cov_dist, axis=1)[:, quantile_index]
    corr_thresholds = np.sort(corr_dist, axis=1)[:, quantile_index]
    passes = (observed_cov > cov_thresholds) & (observed_corr > corr_thresholds)

    first_fail = np.flatnonzero(~passes)
    num_sig_dims = int(first_fail[0]) if first_fail.size else int(np.count_nonzero(passes))

    return SharedNullResult(
        num_sig_dims=num_sig_dims,
        observed_cov=observed_cov,
        observed_corr=observed_corr,
        cov_dist=cov_dist,
        corr_dist=corr_dist,
        cov_thresholds=cov_thresholds,
        corr_thresholds=corr_thresholds,
    )


def _pca_scores(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if data.shape[1] == 0:
        return np.empty((0, 0)), np.empty((data.shape[0], 0))
    centered = data - np.mean(data, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    coeff = vt.T
    return coeff, centered @ coeff


def get_neural_space(data1: ArrayLike, data2: ArrayLike, num_sig_dims: int) -> NeuralSpace:
    """Split PLSC projections into significant shared and residual unique spaces."""

    x, y = _validate_pair(data1, data2)
    if not isinstance(num_sig_dims, int) or num_sig_dims < 0:
        raise ValueError("num_sig_dims must be a non-negative integer.")

    result = plsc(x, y)
    shared_dims = min(num_sig_dims, result.loading1.shape[1], result.loading2.shape[1])

    shared_l1 = result.loading1[:, :shared_dims]
    shared_l2 = result.loading2[:, :shared_dims]
    shared = SpaceProjection(
        loading1=shared_l1,
        loading2=shared_l2,
        projected_data1=x @ shared_l1,
        projected_data2=y @ shared_l2,
    )

    residual1 = x @ result.loading1[:, shared_dims:]
    residual2 = y @ result.loading2[:, shared_dims:]
    unique_l1, unique_scores1 = _pca_scores(residual1)
    unique_l2, unique_scores2 = _pca_scores(residual2)
    unique = SpaceProjection(
        loading1=unique_l1,
        loading2=unique_l2,
        projected_data1=unique_scores1,
        projected_data2=unique_scores2,
    )
    return NeuralSpace(shared=shared, unique=unique)

