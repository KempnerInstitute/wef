"""
Accelerated backends for electric field measurement.
Selected via ELECTRIC_CONSTANTS["electric_backend"] in cfg.py: "original" | "fast" | "numba".
"""

import numpy as np

try:
    from numba import njit
except ImportError:
    raise ImportError("electric_accelerated numba backend requires numba: pip install numba")

from cfg import ELECTRIC_CONSTANTS

_K_COULOMB = ELECTRIC_CONSTANTS["k_coulomb"]
_CM_TO_M   = 0.01

_EMPTY_POS = np.zeros((0, 2), dtype=np.float64)
_EMPTY_Q   = np.zeros((0,),   dtype=np.float64)
_EMPTY_MOM = np.zeros((0, 2), dtype=np.float64)


# --- fast (vectorised NumPy) ---

def measure_electric_field_fast(
    measurement_positions_cm,
    monopole_positions_cm=None,
    monopole_charges=None,
    dipole_positions_cm=None,
    dipole_moments=None,
    measurement_noise=None,  # deprecated
    eps_m=1e-5,
):
    measurement_positions = measurement_positions_cm * _CM_TO_M
    m = measurement_positions.shape[0]
    out = np.zeros((m, 2), dtype=measurement_positions.dtype)

    if monopole_positions_cm is not None and monopole_positions_cm.shape[0] > 0:
        src = monopole_positions_cm * _CM_TO_M
        dx = measurement_positions[:, None, 0] - src[None, :, 0]
        dy = measurement_positions[:, None, 1] - src[None, :, 1]
        r      = np.sqrt(dx * dx + dy * dy) + eps_m
        inv_r  = 1.0 / r
        inv_r3 = inv_r * inv_r * inv_r
        w = (_K_COULOMB * monopole_charges)[None, :] * inv_r3
        out[:, 0] += np.sum(w * dx, axis=1)
        out[:, 1] += np.sum(w * dy, axis=1)

    if dipole_positions_cm is not None and dipole_positions_cm.shape[0] > 0:
        src = dipole_positions_cm * _CM_TO_M
        mx  = dipole_moments[:, 0]
        my  = dipole_moments[:, 1]
        dx  = measurement_positions[:, None, 0] - src[None, :, 0]
        dy  = measurement_positions[:, None, 1] - src[None, :, 1]
        r      = np.sqrt(dx * dx + dy * dy) + eps_m
        inv_r  = 1.0 / r
        inv_r2 = inv_r  * inv_r
        inv_r3 = inv_r2 * inv_r
        inv_r5 = inv_r3 * inv_r2
        mdot = dx * mx[None, :] + dy * my[None, :]
        c1   = 3.0 * mdot * inv_r5
        out[:, 0] += _K_COULOMB * np.sum(c1 * dx - mx[None, :] * inv_r3, axis=1)
        out[:, 1] += _K_COULOMB * np.sum(c1 * dy - my[None, :] * inv_r3, axis=1)

    return out


# --- numba (serial JIT) ---

@njit(cache=True)
def _mef_kernel(meas_m, mp_m, mc, dp_m, dm, eps_m, k_coulomb):
    """
    Serial JIT kernel.  All positions in metres, all arrays float64.
      meas_m : (M, 2)
      mp_m   : (P, 2)  may be empty (P=0)
      mc     : (P,)
      dp_m   : (D, 2)  may be empty (D=0)
      dm     : (D, 2)
    Returns (M, 2) electric field vectors.
    """
    M = meas_m.shape[0]
    P = mp_m.shape[0]
    D = dp_m.shape[0]
    out = np.zeros((M, 2))

    for i in range(M):
        mx_i = meas_m[i, 0]
        my_i = meas_m[i, 1]
        ex   = 0.0
        ey   = 0.0

        for j in range(P):
            dx = mx_i - mp_m[j, 0]
            dy = my_i - mp_m[j, 1]
            r      = (dx * dx + dy * dy) ** 0.5 + eps_m
            inv_r3 = 1.0 / (r * r * r)
            w      = k_coulomb * mc[j] * inv_r3
            ex    += w * dx
            ey    += w * dy

        for j in range(D):
            dx     = mx_i - dp_m[j, 0]
            dy     = my_i - dp_m[j, 1]
            r      = (dx * dx + dy * dy) ** 0.5 + eps_m
            inv_r  = 1.0 / r
            inv_r3 = inv_r  * inv_r  * inv_r
            inv_r5 = inv_r3 * inv_r  * inv_r
            mdot   = dx * dm[j, 0] + dy * dm[j, 1]
            c1     = 3.0 * mdot * inv_r5
            ex    += k_coulomb * (c1 * dx - dm[j, 0] * inv_r3)
            ey    += k_coulomb * (c1 * dy - dm[j, 1] * inv_r3)

        out[i, 0] = ex
        out[i, 1] = ey

    return out


def measure_electric_field_numba(
    measurement_positions_cm,
    monopole_positions_cm=None,
    monopole_charges=None,
    dipole_positions_cm=None,
    dipole_moments=None,
    measurement_noise=None,  # deprecated
    eps_m=1e-5,
):
    """Drop-in replacement using the serial numba kernel."""
    meas_m = np.ascontiguousarray(measurement_positions_cm, dtype=np.float64) * _CM_TO_M

    if monopole_positions_cm is not None and monopole_positions_cm.shape[0] > 0:
        mp_m = np.ascontiguousarray(monopole_positions_cm, dtype=np.float64) * _CM_TO_M
        mc   = np.ascontiguousarray(monopole_charges,      dtype=np.float64)
    else:
        mp_m = _EMPTY_POS
        mc   = _EMPTY_Q

    if dipole_positions_cm is not None and dipole_positions_cm.shape[0] > 0:
        dp_m = np.ascontiguousarray(dipole_positions_cm, dtype=np.float64) * _CM_TO_M
        dm   = np.ascontiguousarray(dipole_moments,      dtype=np.float64)
    else:
        dp_m = _EMPTY_POS
        dm   = _EMPTY_MOM

    return _mef_kernel(meas_m, mp_m, mc, dp_m, dm, eps_m, _K_COULOMB)


def warmup():
    """Pre-compile the numba kernel. Call once at startup."""
    _mef_kernel(
        np.zeros((2, 2), dtype=np.float64),
        np.zeros((1, 2), dtype=np.float64),
        np.zeros((1,),   dtype=np.float64),
        np.zeros((1, 2), dtype=np.float64),
        np.zeros((1, 2), dtype=np.float64),
        1e-5,
        _K_COULOMB,
    )


## --------------------------------------- TESTS --------------------------------------- ##

def _make_inputs(rng, num_positions, num_monopoles, num_dipoles):
    meas = rng.uniform(-50, 50, (num_positions, 2)).astype(np.float64)
    mp   = rng.uniform(-50, 50, (num_monopoles, 2)).astype(np.float64) if num_monopoles else np.zeros((0, 2), dtype=np.float64)
    mc   = rng.uniform(-1,   1, (num_monopoles,)).astype(np.float64)   if num_monopoles else np.zeros((0,),   dtype=np.float64)
    dp   = rng.uniform(-50, 50, (num_dipoles,   2)).astype(np.float64) if num_dipoles   else np.zeros((0, 2), dtype=np.float64)
    dm   = rng.uniform(-1,   1, (num_dipoles,   2)).astype(np.float64) if num_dipoles   else np.zeros((0, 2), dtype=np.float64)
    return meas, mp, mc, dp, dm


def check_backends_match(
    num_positions=100,
    num_monopoles=10,
    num_dipoles=100,
    num_runs=500,
    seed=0,
    rtol=1e-10,
    atol=1e-12,
    min_speedup_fast=None,
    min_speedup_numba=None,
):
    """Correctness and speed check: fast and numba vs original."""
    import timeit
    from numpy.testing import assert_allclose
    from electric import measure_electric_field_original  # local import avoids circular at module level

    rng = np.random.default_rng(seed)

    CASES = [
        ("both",           num_positions, num_monopoles, num_dipoles),
        ("monopoles_only", num_positions, num_monopoles, 0),
        ("dipoles_only",   num_positions, 0,             num_dipoles),
        ("empty",          num_positions, 0,             0),
    ]
    backends = [("fast", measure_electric_field_fast), ("numba", measure_electric_field_numba)]

    print("=== Correctness ===")
    for label, m, p, d in CASES:
        meas, mp, mc, dp, dm = _make_inputs(rng, m, p, d)
        ref = measure_electric_field_original(meas, mp, mc, dp, dm)
        for name, fn in backends:
            got     = fn(meas, mp, mc, dp, dm)
            diff    = np.abs(got - ref)
            max_abs = diff.max() if diff.size else 0.0
            max_ref = np.abs(ref).max() if ref.size else 0.0
            max_rel = max_abs / (max_ref + 1e-12)
            status  = "ok" if max_abs <= atol + rtol * max_ref else "FAIL"
            print(f"  [{name:<5}] [{label:<15}]  max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  {status}")
            if status == "FAIL":
                idx = np.unravel_index(diff.argmax(), diff.shape)
                print(f"           idx={idx}  ref={ref[idx]:.6e}  got={got[idx]:.6e}")
            assert_allclose(ref, got, rtol=rtol, atol=atol, err_msg=f"[{name}] mismatch: {label}")

    print("\n=== Speed ===")
    meas, mp, mc, dp, dm = _make_inputs(rng, num_positions, num_monopoles, num_dipoles)
    t_orig = timeit.timeit(lambda: measure_electric_field_original(meas, mp, mc, dp, dm), number=num_runs)
    print(f"  [{'original':<5}]  {t_orig:.4f}s  ({t_orig/num_runs*1000:.3f} ms/call)")

    speedups = {}
    for name, fn in backends:
        t = timeit.timeit(lambda fn=fn: fn(meas, mp, mc, dp, dm), number=num_runs)
        speedups[name] = t_orig / max(t, 1e-12)
        print(f"  [{name:<5}]  {t:.4f}s  ({t/num_runs*1000:.3f} ms/call)  {speedups[name]:.1f}x")

    if min_speedup_fast  is not None:
        assert speedups["fast"]  >= min_speedup_fast,  f"fast  speedup {speedups['fast']:.1f}x < min {min_speedup_fast}"
    if min_speedup_numba is not None:
        assert speedups["numba"] >= min_speedup_numba, f"numba speedup {speedups['numba']:.1f}x < min {min_speedup_numba}"


if __name__ == "__main__":
    check_backends_match()
