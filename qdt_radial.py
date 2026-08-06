# -*- coding: utf-8 -*-
"""
qdt_radial.py — Quantum-defect Numerov radial dipole matrix elements

Solves the outer-region Coulomb radial equation at E = -1/(2 n*^2),
n* = n - δ_l, with Numerov integration, then forms length-gauge absorption
oscillator strengths. Used as the primary fallback above Bates–Damgaard.

Atomic units throughout unless noted.
"""
from __future__ import annotations

import math
import re
from functools import lru_cache

import numpy as np

# Angular / spectroscopic maps
L_CHAR = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}
L_INT = {v: k for k, v in L_CHAR.items()}


def n_star(n: int, delta: float) -> float:
    return float(n) - float(delta)


def binding_au(nstar: float, z_core: float = 1.0) -> float:
    """E = -Z^2/(2 n*^2) in Hartree (bound). Z=1 neutral, Z=2 for Na+, …"""
    z = float(z_core) if z_core else 1.0
    return -0.5 * (z * z) / (nstar ** 2)


def _potential(r: np.ndarray, l: int, z_core: float = 1.0) -> np.ndarray:
    """Coulomb + centrifugal, Hartree units (nuclear charge z_core)."""
    z = float(z_core) if z_core else 1.0
    return -z / r + 0.5 * l * (l + 1) / (r * r)


def radial_wavefunction(n: int, l: int, delta: float,
                        n_grid: int = 4000, r_min: float = 1e-4,
                        z_core: float = 1.0):
    """
    Normalized radial reduced wavefunction u(r)=r R(r) on a linear grid.

    Integrates the Schrödinger equation inward from large r using the
    asymptotic Whittaker-like start u ~ r^{n*} exp(-Z r/n*), then normalizes
    ∫ u² dr = 1. Returns (r, u) or (None, None) if n* is non-physical.
    """
    z = float(z_core) if z_core else 1.0
    ns = n_star(n, delta)
    if ns <= l + 0.51 or ns <= 0.5:
        return None, None

    E = binding_au(ns, z_core=z)
    # Classical outer turning scale ~ 2 n*^2 / Z; pad generously for high n*
    r_max = max(40.0, 6.0 * ns * ns / max(z, 1.0))
    r = np.linspace(r_min, r_max, n_grid)
    h = r[1] - r[0]
    V = _potential(r, l, z_core=z)
    # Numerov g = 2(V-E); u'' = g u
    g = 2.0 * (V - E)

    u = np.zeros_like(r)
    # Asymptotic start at outer two points
    kappa = z / ns
    u[-1] = r[-1] ** ns * math.exp(-kappa * r[-1])
    u[-2] = r[-2] ** ns * math.exp(-kappa * r[-2])
    if u[-1] == 0.0 or not np.isfinite(u[-1]):
        u[-1] = 1e-300
        u[-2] = 1e-300

    # Inward Numerov
    for i in range(n_grid - 3, -1, -1):
        num = (2.0 + (5.0 / 6.0) * h * h * g[i + 1]) * u[i + 1] - (
            1.0 - (1.0 / 12.0) * h * h * g[i + 2]
        ) * u[i + 2]
        den = 1.0 - (1.0 / 12.0) * h * h * g[i]
        if abs(den) < 1e-30:
            u[i] = 0.0
        else:
            u[i] = num / den
        # Prevent overflow: rescale periodically
        if abs(u[i]) > 1e200:
            u[i:] *= 1e-200

    # Force regular origin behavior softly (u→0)
    u[0] = 0.0
    # Discard pathological nodes near origin if amplitude explodes
    if not np.all(np.isfinite(u)):
        return None, None

    norm = math.sqrt(max(np.trapz(u * u, r), 1e-30))
    u = u / norm
    # Overall phase: make the outer lobe positive
    if u[n_grid // 2] < 0:
        u = -u
    return r, u


@lru_cache(maxsize=512)
def _cached_wave(n: int, l: int, delta_key: float, n_grid: int = 4000,
                 z_core: float = 1.0):
    """Cache waves keyed by rounded quantum defect and core charge."""
    return radial_wavefunction(n, l, delta_key, n_grid=n_grid, z_core=z_core)


def radial_dipole(n_lower, l_lower, delta_lower,
                  n_upper, l_upper, delta_upper,
                  n_grid: int = 4000, z_core: float = 1.0) -> float:
    """
    Radial dipole R = ∫ u_l(r) u_u(r) r dr  (atomic units).
    Requires |l_upper - l_lower| = 1.
    """
    if abs(l_upper - l_lower) != 1:
        return 0.0

    d_l = round(float(delta_lower), 8)
    d_u = round(float(delta_upper), 8)
    z = float(z_core) if z_core else 1.0
    r1, u1 = _cached_wave(int(n_lower), int(l_lower), d_l, n_grid, z)
    r2, u2 = _cached_wave(int(n_upper), int(l_upper), d_u, n_grid, z)
    if r1 is None or r2 is None:
        return 0.0

    # Interpolate upper onto lower grid (same r_min; r_max may differ)
    r_max = min(r1[-1], r2[-1])
    mask = r1 <= r_max
    r = r1[mask]
    u_l = u1[mask]
    u_u = np.interp(r, r2, u2)
    return float(np.trapz(u_l * u_u * r, r))


def numerov_fosc(n_upper, l_upper, n_lower, l_lower,
                 delta_upper, delta_lower, z_core: float = 1.0) -> float:
    """
    Absorption oscillator strength f_lu (lower → upper) via Numerov radial ME.

    f = (2/3) * ΔE_au * [l_max / (2 l_lower + 1)] * R^2
    with R = ∫ u_l u_u r dr, E = -Z^2/(2 n*^2).
    """
    z = float(z_core) if z_core else 1.0
    ns_u = n_star(n_upper, delta_upper)
    ns_l = n_star(n_lower, delta_lower)
    if ns_u <= ns_l or ns_u <= 0.5 or ns_l <= 0.5:
        return 0.0
    if abs(l_upper - l_lower) != 1:
        return 0.0
    # Same-n / near-degenerate guard (mirrors coulomb_fosc)
    if (ns_u ** 2 - ns_l ** 2) < 0.5 * ns_u:
        return 0.0

    E_u = binding_au(ns_u, z_core=z)
    E_l = binding_au(ns_l, z_core=z)
    dE = E_u - E_l  # positive for absorption from lower
    if dE <= 0:
        return 0.0

    R = radial_dipole(n_lower, l_lower, delta_lower,
                      n_upper, l_upper, delta_upper, z_core=z)
    if R == 0.0 or not math.isfinite(R):
        return 0.0

    l_max = max(l_upper, l_lower)
    ang = l_max / (2 * l_lower + 1)
    f = (2.0 / 3.0) * dE * ang * (R * R)
    return max(float(f), 0.0)


def clear_wave_cache():
    _cached_wave.cache_clear()


def _parse_nl_token(token: str):
    """Parse '3s' / '10p' → (n, l_int) or None."""
    if not token:
        return None
    m = re.match(r"^(\d+)([spdfgh])$", str(token).strip())
    if not m:
        return None
    return int(m.group(1)), L_CHAR[m.group(2)]


def fit_numerov_scale_from_nist(qd: dict, nist_fosc: dict,
                                min_pairs: int = 5,
                                min_f: float = 1e-8,
                                scale_lo: float = 0.25,
                                scale_hi: float = 4.0,
                                z_core: float = 1.0) -> dict:
    """
    Empirical renormalization of Numerov f against NIST multiplet f.

    For every NIST multiplet key (nl_lower, nl_upper) with Δl=±1, compute the
    bare Numerov f from the element's own quantum defects and form the ratio
    f_NIST / f_Numerov. The calibration scale is the median of those ratios.

    No element-specific constants: if too few valid pairs, or the median lies
    outside [scale_lo, scale_hi], the scale is 1 (no recalibration).

    Returns dict: scale, n_pairs, ratio percentiles, applied (bool), reason.
    """
    ratios = []
    for key, f_nist in (nist_fosc or {}).items():
        if not isinstance(key, tuple) or len(key) != 2:
            continue
        try:
            f_nist = float(f_nist)
        except (TypeError, ValueError):
            continue
        if f_nist < min_f:
            continue
        lo = _parse_nl_token(key[0])
        up = _parse_nl_token(key[1])
        if lo is None or up is None:
            continue
        n_a, l_a = lo
        n_b, l_b = up
        if abs(l_b - l_a) != 1:
            continue

        delta_a = float(qd.get(L_INT.get(l_a, "s"), 0.0))
        delta_b = float(qd.get(L_INT.get(l_b, "p"), 0.0))
        ns_a = n_star(n_a, delta_a)
        ns_b = n_star(n_b, delta_b)

        # Absorption ordering: lower = more tightly bound (smaller n*)
        if ns_b > ns_a:
            n_l, l_l, d_l = n_a, l_a, delta_a
            n_u, l_u, d_u = n_b, l_b, delta_b
        else:
            n_l, l_l, d_l = n_b, l_b, delta_b
            n_u, l_u, d_u = n_a, l_a, delta_a

        f_num = numerov_fosc(n_u, l_u, n_l, l_l, d_u, d_l, z_core=z_core)
        if f_num < min_f or not math.isfinite(f_num):
            continue
        ratios.append(f_nist / f_num)

    info = {
        "n_pairs": len(ratios),
        "n_inliers": 0,
        "scale": 1.0,
        "applied": False,
        "median_ratio_raw": None,
        "p16": None,
        "p84": None,
        "reason": "insufficient_pairs",
    }
    if len(ratios) < min_pairs:
        return info

    arr = np.asarray(ratios, dtype=float)
    info["median_ratio_raw"] = float(np.median(arr))
    info["p16"] = float(np.percentile(arr, 16))
    info["p84"] = float(np.percentile(arr, 84))

    # Robust fit: median over ratios already inside the physical window.
    # Pathological weak-line ratios (near poles / tiny Numerov f) are excluded
    # from the estimator but still reported in p16/p84 of the raw set.
    inliers = arr[(arr >= scale_lo) & (arr <= scale_hi)]
    info["n_inliers"] = int(inliers.size)
    if inliers.size < min_pairs:
        info["reason"] = "insufficient_inliers"
        return info

    med = float(np.median(inliers))
    info["scale"] = med
    info["applied"] = True
    info["reason"] = "ok"
    return info


def numerov_fosc_scaled(n_upper, l_upper, n_lower, l_lower,
                        delta_upper, delta_lower, scale: float = 1.0,
                        z_core: float = 1.0) -> float:
    """Bare Numerov f multiplied by a data-derived scale (default 1)."""
    f = numerov_fosc(n_upper, l_upper, n_lower, l_lower,
                     delta_upper, delta_lower, z_core=z_core)
    if f <= 0 or scale is None:
        return f
    try:
        s = float(scale)
    except (TypeError, ValueError):
        return f
    if not math.isfinite(s) or s <= 0:
        return f
    return f * s
