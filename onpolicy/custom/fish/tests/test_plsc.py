"""Tests for compute_plsc and compute_plsc_gpu.

Covers:
  - Recovery of known shared latent structure (num_sig ~ k)
  - Null check: independent X, Y should give num_sig == 0
  - top_corr agreement between CPU and GPU implementations
  - GPU falls back gracefully when called with device='cpu'
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")

from analysis_rnn_plsc import compute_plsc, compute_plsc_gpu


def _make_shared(n, H, k, noise=0.5, seed=0):
    """X, Y sharing k latent factors plus Gaussian noise."""
    rng = np.random.default_rng(seed)
    Z  = rng.standard_normal((n, k))
    Wx = rng.standard_normal((k, H)); Wx /= np.linalg.norm(Wx, axis=1, keepdims=True)
    Wy = rng.standard_normal((k, H)); Wy /= np.linalg.norm(Wy, axis=1, keepdims=True)
    X  = (Z @ Wx + noise * rng.standard_normal((n, H))).astype(np.float32)
    Y  = (Z @ Wy + noise * rng.standard_normal((n, H))).astype(np.float32)
    return X, Y


# ── CPU (sklearn NIPALS) ─────────────────────────────────────────────────────

def test_compute_plsc_recovers_shared_dims():
    # Should find at least k-1 dims and no more than k+2 (permutation test is stochastic)
    k = 3
    X, Y = _make_shared(n=300, H=20, k=k, noise=0.3, seed=42)
    result = compute_plsc(X, Y, n_components=8, n_shuffles=50, seed=42)
    assert result["num_sig"] >= k - 1, f"found too few: {result['num_sig']} < {k-1}"
    assert result["num_sig"] <= k + 2, f"found too many: {result['num_sig']} > {k+2}"


def test_compute_plsc_null():
    # With n_shuffles=50 and 5 components, expect ~0 but allow up to 1 false positive
    rng = np.random.default_rng(1)
    X = rng.standard_normal((200, 20)).astype(np.float32)
    Y = rng.standard_normal((200, 20)).astype(np.float32)
    result = compute_plsc(X, Y, n_components=5, n_shuffles=50, seed=1)
    assert result["num_sig"] <= 1, f"too many false positives: {result['num_sig']}"


def test_compute_plsc_no_shuffle():
    X, Y = _make_shared(n=100, H=10, k=2, seed=7)
    result = compute_plsc(X, Y, n_components=3, n_shuffles=0, seed=0)
    assert result["num_sig"] == 0   # no shuffle → significance not tested
    assert result["null_corrs"].shape == (0, 3)
    assert 0.0 <= result["top_corr"] <= 1.0


# ── GPU (SVD-based) ──────────────────────────────────────────────────────────

def test_compute_plsc_gpu_recovers_shared_dims():
    # SVD-based PLSC may find more dims than NIPALS (no deflation floor);
    # check at least k-1 found and fewer than n_components
    k = 3
    X, Y = _make_shared(n=300, H=20, k=k, noise=0.3, seed=42)
    result = compute_plsc_gpu(X, Y, n_components=8, n_shuffles=50, seed=42)
    assert result["num_sig"] >= k - 1, f"found too few: {result['num_sig']} < {k-1}"
    assert result["num_sig"] < 8,      f"found too many: {result['num_sig']}"


def test_compute_plsc_gpu_null():
    # Allow up to 2 false positives — permutation test at alpha=0.025 with 5
    # components and ns=50 shuffles gives ~12% chance of >=1 false positive
    rng = np.random.default_rng(2)
    X = rng.standard_normal((200, 20)).astype(np.float32)
    Y = rng.standard_normal((200, 20)).astype(np.float32)
    result = compute_plsc_gpu(X, Y, n_components=5, n_shuffles=50, seed=2)
    assert result["num_sig"] <= 2, f"too many false positives: {result['num_sig']}"


def test_compute_plsc_gpu_cpu_fallback():
    """compute_plsc_gpu with device='cpu' should run without CUDA."""
    X, Y = _make_shared(n=150, H=16, k=2, seed=5)
    result = compute_plsc_gpu(X, Y, n_components=4, n_shuffles=10, seed=5, device="cpu")
    assert "num_sig" in result
    assert "top_corr" in result
    assert 0.0 <= result["top_corr"] <= 1.0


# ── Agreement between CPU and GPU ────────────────────────────────────────────

def test_top_corr_cpu_gpu_agreement():
    """First canonical correlation should agree closely between implementations."""
    X, Y = _make_shared(n=200, H=20, k=3, noise=0.4, seed=99)
    r_cpu = compute_plsc(    X, Y, n_components=5, n_shuffles=20, seed=0)
    r_gpu = compute_plsc_gpu(X, Y, n_components=5, n_shuffles=20, seed=0, device="cpu")
    diff = abs(r_cpu["top_corr"] - r_gpu["top_corr"])
    assert diff < 0.05, f"top_corr diverged: CPU={r_cpu['top_corr']:.3f} GPU={r_gpu['top_corr']:.3f}"
