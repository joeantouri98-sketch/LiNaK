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

# np.trapz was removed in NumPy 2.4; np.trapezoid exists from NumPy 2.0
_trapezoid = getattr(np, 'trapezoid', None) or np.trapz

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


# Radial step used when n_grid is not given. A fixed 4000-point grid has a step of
# ~3.5 a0 at n* ~ 48 (r_max = 6 n*^2) and cannot resolve the core region that
# controls far-pair matrix elements: Na 5s->50p came out at 0.30 of the converged value
# (0.73 of ARC) and 3d->40f at 0.89. A step of 0.125 a0 converges all tested pairs
# to <= 0.7 % (grid scaling is capped at N_GRID_MAX).
H_TARGET = 0.125
N_GRID_MIN = 4000
N_GRID_MAX = 150000


def radial_wavefunction(n: int, l: int, delta: float,
                        n_grid=None, r_min: float = 1e-4,
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
    if n_grid is None:
        n_grid = int(min(N_GRID_MAX, max(N_GRID_MIN, math.ceil(r_max / H_TARGET))))
    r = np.linspace(r_min, r_max, n_grid)
    h = r[1] - r[0]
    V = _potential(r, l, z_core=z)
    # Numerov g = 2(V-E); u'' = g u
    g = 2.0 * (V - E)

    u = np.zeros_like(r)
    # Asymptotic start at outer two points
    kappa = z / ns
    # Log-space start: r**ns overflows for n* >~ 70 (the old code then returned ~1e-233 or 0).
    # Only the ratio of the two start values matters (the solution is normalised afterwards).
    _a1 = ns * math.log(r[-1]) - kappa * r[-1]
    _a2 = ns * math.log(r[-2]) - kappa * r[-2]
    u[-1] = 1.0
    u[-2] = math.exp(max(min(_a2 - _a1, 600.0), -600.0))

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
        if abs(u[i]) > 1e100:     # keep u*u (and its sum over up to 1.5e5 points) below float overflow
            u[i:] *= 1e-100

    # Force regular origin behavior softly (u→0)
    u[0] = 0.0
    # Discard pathological nodes near origin if amplitude explodes
    if not np.all(np.isfinite(u)):
        return None, None

    # The inward solution at a non-integer n* is the decaying Whittaker function,
    # which is IRREGULAR at the origin (u ~ r^-l). Inside the classical inner
    # turning point it has no physical meaning (and the real core is non-Coulombic
    # anyway), but it can dominate ∫u² for l >= 2 with small defects (e.g. Na 6d:
    # 39 % of the norm; 6f: 100 %), shrinking the physical outer lobes by up to
    # orders of magnitude. Standard QDT remedy: use the Coulomb function only
    # outside r_c and normalise there (agrees with the analytic Seaton norm to <0.5 %).
    # Inner classical turning point for charge Z: (n*^2 / Z) (1 - sqrt(1 - l(l+1)/n*^2));
    # the 1 a0 floor scales as 1/Z (hydrogenic scaling r -> Z r).
    disc = 1.0 - l * (l + 1) / (ns * ns)
    r_in = (ns * ns / z) * (1.0 - math.sqrt(disc)) if disc > 0 else ns * ns / z
    r_c = max(r_in, 1.0 / z)
    u = np.where(r >= r_c, u, 0.0)

    norm = math.sqrt(max(_trapezoid(u * u, r), 1e-30))
    u = u / norm
    # Overall phase: make the outer lobe positive
    if u[n_grid // 2] < 0:
        u = -u
    return r, u


@lru_cache(maxsize=512)
def _cached_wave(n: int, l: int, delta_key: float, n_grid=None,
                 z_core: float = 1.0):
    """Cache waves keyed by rounded quantum defect and core charge."""
    return radial_wavefunction(n, l, delta_key, n_grid=n_grid, z_core=z_core)


def radial_dipole(n_lower, l_lower, delta_lower,
                  n_upper, l_upper, delta_upper,
                  n_grid=None, z_core: float = 1.0) -> float:
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
    return float(_trapezoid(u_l * u_u * r, r))


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
    # No near-degeneracy guard here: it exists in coulomb_fosc because the closed form diverges
    # as n*_u -> n*_l, but the numerical radial integral is finite and well behaved for
    # same-n l -> l+-1 pairs (checked against hydrogen). Dropping it restores the microwave
    # channels (e.g. nd -> nf) that blackbody coupling needs. f stays > 0 only if dE > 0.

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


# ---------------------------------------------------------------------------
# Fine-structure (LS-coupling) line strengths and per-transition-type scale
# ---------------------------------------------------------------------------

def _fact(n2: int) -> float:
    """(n2/2)! for a non-negative integer n2 that is even (i.e. integer argument)."""
    return float(math.factorial(n2 // 2))


def _tri2(a2: int, b2: int, c2: int) -> float:
    """Triangle coefficient Delta(abc) with doubled arguments (2a, 2b, 2c)."""
    return math.sqrt(_fact(a2 + b2 - c2) * _fact(a2 - b2 + c2) * _fact(-a2 + b2 + c2)
                     / _fact(a2 + b2 + c2 + 2))


def wigner_6j(j1x2: int, j2x2: int, j3x2: int, j4x2: int, j5x2: int, j6x2: int) -> float:
    """
    Wigner 6j symbol {j1 j2 j3; j4 j5 j6} by the Racah formula.
    Arguments are DOUBLED angular momenta (2j), so half-integers are plain ints.
    Returns 0.0 if any triad violates the triangle/parity conditions.
    """
    a = (j1x2, j2x2, j3x2)
    b = (j1x2, j5x2, j6x2)
    c = (j4x2, j2x2, j6x2)
    d = (j4x2, j5x2, j3x2)
    for tr in (a, b, c, d):
        x, y, z = tr
        if (x + y + z) % 2 or z > x + y or z < abs(x - y):
            return 0.0
    pref = _tri2(*a) * _tri2(*b) * _tri2(*c) * _tri2(*d)
    t_min = max(sum(a), sum(b), sum(c), sum(d)) // 2
    t_max = min(j1x2 + j2x2 + j4x2 + j5x2, j2x2 + j3x2 + j5x2 + j6x2,
                j3x2 + j1x2 + j6x2 + j4x2) // 2
    s = 0.0
    for t in range(t_min, t_max + 1):
        den = (math.factorial(t - sum(a) // 2) * math.factorial(t - sum(b) // 2)
               * math.factorial(t - sum(c) // 2) * math.factorial(t - sum(d) // 2)
               * math.factorial((j1x2 + j2x2 + j4x2 + j5x2) // 2 - t)
               * math.factorial((j2x2 + j3x2 + j5x2 + j6x2) // 2 - t)
               * math.factorial((j3x2 + j1x2 + j6x2 + j4x2) // 2 - t))
        s += (-1) ** t * math.factorial(t + 1) / den
    return pref * s


def ls_line_weight(Lu: int, Ju2: int, Ll: int, Jl2: int) -> float:
    """
    Fraction of the multiplet decay rate of an upper level (Lu, J_u) that goes to
    the lower fine-structure level (Ll, J_l)  [spin S = 1/2]:

        w = (2 Lu + 1)(2 Jl + 1) { Lu Ju S ; Jl Ll 1 }^2

    Ju2/Jl2 are doubled J. Sum over Jl for fixed Ju is 1 (checked in the tests).
    """
    s = wigner_6j(2 * Lu, Ju2, 1, Jl2, 2 * Ll, 2)
    return (2 * Lu + 1) * (Jl2 + 1) * s * s


def allowed_J2(L: int):
    """Doubled J values for one electron with orbital L and spin 1/2."""
    return [1] if L == 0 else [2 * L - 1, 2 * L + 1]


def parse_J2_set(J_str, L: int):
    """
    '3/2' -> [3]; '5/2,7/2' -> [5, 7]; None/'' -> every allowed J of L
    (an unresolved multiplet: statistical average over its levels).
    Values not allowed for L are dropped; empty result falls back to all allowed.
    """
    allowed = allowed_J2(L)
    if not J_str:
        return allowed
    out = []
    for tok in str(J_str).split(','):
        tok = tok.strip()
        try:
            if '/' in tok:
                num, den = tok.split('/')
                v = (2 * int(num)) // int(den)
            else:
                v = 2 * int(tok)
        except ValueError:
            continue
        if v in allowed and v not in out:
            out.append(v)
    return out or allowed


def ls_group_branching(Lu: int, Ju2_set, Ll: int, Jl2_set):
    """
    Rate of an upper *group* of levels (statistical mix of Ju2_set) into a lower
    group (Jl2_set), as a fraction of the multiplet-to-multiplet rate A_mult.

    Returns (ratio, g_upper, g_lower) with g = sum(2J+1) over each group.
    ratio = sum_Ju (g_Ju / g_U) sum_Jl w(Ju->Jl); ratio is 1 for full multiplets
    and 0 for fully forbidden (|dJ| > 1) combinations.
    """
    gU = sum(j + 1 for j in Ju2_set)
    gL = sum(j + 1 for j in Jl2_set)
    ratio = 0.0
    for ju in Ju2_set:
        for jl in Jl2_set:
            ratio += (ju + 1) / gU * ls_line_weight(Lu, ju, Ll, jl)
    return ratio, gU, gL


# Gap dependence of the Numerov renormalisation.  NIST/Numerov is NOT a constant for
# a transition type: for Na s->p it is 1.005 (11 pairs) when the n* gap is < 1.5 and
# 1.36 (18 pairs) for gaps > 4, while the remaining types are flat within noise.  The
# QDT Numerov integral with a cut-off Coulomb core is accurate when the overlap is in
# the outer region (small gap) and misses the core-penetration contribution for
# distant pairs.  A single type-wide scale (1.27 for s->p) therefore over-scales the
# near-resonant channels (20s->20p) that dominate Rydberg polarizabilities by ~27 %.
GAP_KNOTS = (1.5, 4.0)       # n* gap below/above which the near/far value applies
NEAR_DEG_FRAC = 0.5          # n*_u^2 - n*_l^2 < NEAR_DEG_FRAC * n*_u  ==  same-n / near-degenerate pair
GAP_MIN_BIN = 5              # inliers required in EACH bin to switch the profile on
GAP_MIN_CONTRAST = 0.20      # |far/near - 1| must exceed ~2x the Numerov model uncertainty (0.10)


def is_near_degenerate(ns_u: float, ns_l: float) -> bool:
    """Same-n style pair (e.g. nd->nf) whose Coulomb closed form diverges; the Numerov integral does not."""
    return (ns_u ** 2 - ns_l ** 2) < NEAR_DEG_FRAC * ns_u


def _gap_profile(vals, scale_lo, scale_hi):
    """Near/far Numerov scales for one transition type, or {'gap_dependent': False}."""
    g1, g2 = GAP_KNOTS
    pts = [(r, g) for (r, g) in vals if scale_lo <= r <= scale_hi]
    near = [r for r, g in pts if g < g1]
    far = [r for r, g in pts if g >= g2]
    out = {"gap_dependent": False, "n_near": len(near), "n_far": len(far),
           "scale_near": None, "scale_far": None}
    if near:
        out["scale_near"] = float(np.median(near))
    if far:
        out["scale_far"] = float(np.median(far))
    if (len(near) >= GAP_MIN_BIN and len(far) >= GAP_MIN_BIN
            and abs(out["scale_far"] / out["scale_near"] - 1.0) > GAP_MIN_CONTRAST):
        out["gap_dependent"] = True
    return out


def fit_numerov_scales_by_type(qd: dict, nist_fosc: dict,
                               min_pairs: int = 5, min_f: float = 1e-8,
                               scale_lo: float = 0.25, scale_hi: float = 4.0,
                               z_core: float = 1.0, gate: float = 2.5) -> dict:
    """
    Per-transition-type renormalisation of Numerov multiplet f against NIST.

    Same estimator as fit_numerov_scale_from_nist (median of inlier ratios
    f_NIST / f_Numerov over NIST multiplet pairs) but

      * pairs are restricted to the SAME domain in which the scale is applied
        (n*_lower > gate and n*_upper > gate), so deep-core levels such as
        Na 3s/3p, which the Numerov branch never handles, no longer bias it;
      * the median is taken separately per (l_lower, l_upper) type; a type with
        fewer than `min_pairs` inliers falls back to the pooled (gated) median.

    Returns {'global': info, 'types': {(l_lo, l_up): info}} where each info has
    scale, n_pairs, n_inliers, p16, p84, rel_spread (= (p84-p16)/(2 median)).
    """
    pairs = {}
    near_deg = []
    for key, f_nist in (nist_fosc or {}).items():
        if not isinstance(key, tuple) or len(key) != 2:
            continue
        try:
            f_nist = float(f_nist)
        except (TypeError, ValueError):
            continue
        if f_nist < min_f:
            continue
        a = _parse_nl_token(key[0]); b = _parse_nl_token(key[1])
        if a is None or b is None or abs(a[1] - b[1]) != 1:
            continue
        da = float(qd.get(L_INT.get(a[1], "s"), 0.0))
        db = float(qd.get(L_INT.get(b[1], "p"), 0.0))
        nsa, nsb = n_star(a[0], da), n_star(b[0], db)
        if nsb > nsa:
            lo, up, dl, du = a, b, da, db
        else:
            lo, up, dl, du = b, a, db, da
        if min(nsa, nsb) <= gate:
            continue
        if is_near_degenerate(max(nsa, nsb), min(nsa, nsb)):
            near_deg.append(f_nist)   # kept out of the type fits (see numerov_scale_for)
            continue
        f_num = numerov_fosc(up[0], up[1], lo[0], lo[1], du, dl, z_core=z_core)
        if f_num < min_f or not math.isfinite(f_num):
            continue
        pairs.setdefault((L_INT[lo[1]], L_INT[up[1]]), []).append(
            (f_nist / f_num, abs(nsb - nsa)))

    def _info(vals):
        arr = np.asarray([v[0] if isinstance(v, tuple) else v for v in vals], dtype=float)
        inl = arr[(arr >= scale_lo) & (arr <= scale_hi)]
        info = {"n_pairs": int(arr.size), "n_inliers": int(inl.size), "scale": 1.0,
                "applied": False, "p16": None, "p84": None, "rel_spread": None}
        if arr.size:
            info["p16"], info["p84"] = (float(x) for x in np.percentile(arr, [16, 84]))
        if inl.size >= min_pairs:
            med = float(np.median(inl))
            lo16, hi84 = np.percentile(inl, [16, 84])
            info.update(scale=med, applied=True,
                        rel_spread=float((hi84 - lo16) / (2.0 * med)))
        return info

    allv = [r for v in pairs.values() for r in v]
    glob = _info(allv) if allv else {"n_pairs": 0, "n_inliers": 0, "scale": 1.0,
                                     "applied": False, "p16": None, "p84": None,
                                     "rel_spread": None}
    types = {}
    for k, v in pairs.items():
        info = _info(v)
        info["source"] = "type" if info["applied"] else "global_fallback"
        if not info["applied"]:
            info["scale"] = glob["scale"]
        info.update(_gap_profile(v, scale_lo, scale_hi))
        types[k] = info
    return {"global": glob, "types": types, "gap_knots": list(GAP_KNOTS),
            "near_degenerate_pairs_excluded": len(near_deg)}


def numerov_scale_for(cal: dict, l_lower: str, l_upper: str, gap=None, n_star_upper=None) -> float:
    """
    Scale for a (lower l, upper l) type from fit_numerov_scales_by_type().

    `gap` = n*_upper - n*_lower (>0). When given and the type has a fitted gap
    profile (info['gap_dependent']), the scale is the near value below GAP_KNOTS[0],
    the far value above GAP_KNOTS[1] and linear in between; otherwise the
    type-wide median is returned.

    Near-degenerate (same-n) pairs, recognised from gap and `n_star_upper`, get scale 1.0: they were
    excluded from the fits, the NIST/Numerov ratio of such pairs measured ~1.00 (Na d->p 0.95-1.04),
    and the unscaled radial integral agrees with hydrogen to ~1%.
    """
    if not cal:
        return 1.0
    if gap is not None and n_star_upper is not None and is_near_degenerate(
            float(n_star_upper), float(n_star_upper) - abs(float(gap))):
        return 1.0
    t = cal.get("types", {}).get((l_lower, l_upper))
    if t is not None:
        if gap is not None and t.get("gap_dependent"):
            g1, g2 = cal.get("gap_knots", GAP_KNOTS)
            w = min(1.0, max(0.0, (abs(float(gap)) - g1) / (g2 - g1)))
            return float((1.0 - w) * t["scale_near"] + w * t["scale_far"])
        return float(t["scale"])
    return float(cal.get("global", {}).get("scale", 1.0))


def build_true_multiplet_f(fosc_jl: dict) -> dict:
    """
    Lower-multiplet-averaged absorption f for nl -> n'l' from NIST LINE data:

        f_mult = sum_{Ju,Jl} g_Jl f_line(Jl -> Ju) / sum_{Jl} g_Jl

    fosc_jl maps (lo_nl, up_nl, J_up, J_lo) -> (f_line, acc). A missing line of an
    otherwise listed upper level is completed with the LS 6j ratio
    (f_line ~ w(Ju,Jl)/(2Jl+1)); a pair is skipped if any allowed upper level has
    no listed line at all. Returns {(lo_nl, up_nl): f_mult}.

    NOTE: alpha_core's nist_fosc (sum over J_up of the lower-J-AVERAGED mean) equals
    this only when the lower or the upper level has a single J (s->p, p->s); for
    p->d, d->f, d->p, f->d it is larger by 1.45-1.48 (checked on Na NIST lines).
    """
    def _j2(s):
        a, b = s.split('/') if '/' in s else (s, '1')
        return (2 * int(a)) // int(b)

    groups = {}
    for (lo, up, Ju, Jl), (f, _acc) in (fosc_jl or {}).items():
        try:
            groups.setdefault((lo, up), {})[(_j2(Ju), _j2(Jl))] = float(f)
        except (ValueError, ZeroDivisionError):
            continue
    out = {}
    for (lo, up), d in groups.items():
        a = _parse_nl_token(lo); b = _parse_nl_token(up)
        if a is None or b is None or abs(a[1] - b[1]) != 1:
            continue
        Ll, Lu = a[1], b[1]
        num, ok = 0.0, True
        for ju in allowed_J2(Lu):
            lines = {}
            for jl in allowed_J2(Ll):
                w = ls_line_weight(Lu, ju, Ll, jl)
                if w > 1e-12:
                    lines[jl] = w
            if not lines:
                continue
            known = [jl for jl in lines if (ju, jl) in d]
            if not known:
                ok = False
                break
            jk = known[0]
            for jl, w in lines.items():
                if (ju, jl) in d:
                    f = d[(ju, jl)]
                else:
                    f = d[(ju, jk)] * (w / (jl + 1)) / (lines[jk] / (jk + 1))
                num += (jl + 1) * f
        if ok and num > 0:
            out[(lo, up)] = num / sum(j + 1 for j in allowed_J2(Ll))
    return out


def ls_line_f(f_mult: float, Lu: int, ju2: int, Ll: int, jl2: int) -> float:
    """
    Absorption f of ONE fine-structure line (Ll, Jl) -> (Lu, Ju) from the multiplet f
    (lower-multiplet averaged) in LS coupling:

        f_line = f_mult (2Ll+1)/(2Lu+1) * w(Ju->Jl) * (2Ju+1)/(2Jl+1)
    """
    return (f_mult * (2 * Ll + 1) / (2 * Lu + 1)
            * ls_line_weight(Lu, ju2, Ll, jl2) * (ju2 + 1) / (jl2 + 1))


def nist_line_f(fosc_jl: dict, lo_nl: str, up_nl: str, Lu: int, ju2: int, Ll: int, jl2: int):
    """
    One NIST line f for (lo_nl, Jl) -> (up_nl, Ju).  fosc_jl[(lo, up, 'J_up', 'J_lo')] = (f, acc).
    Returns (f, acc, reconstructed): the listed line, else the value implied by another listed
    lower-J line of the same upper level through f ~ w(Ju,Jl)/(2Jl+1) (LS 6j), else None.
    A line forbidden in LS (|dJ| > 1) returns (0.0, None, False).
    """
    w = ls_line_weight(Lu, ju2, Ll, jl2)
    if w <= 1e-12:
        return (0.0, None, False)
    Ju = f"{ju2}/2"
    hit = (fosc_jl or {}).get((lo_nl, up_nl, Ju, f"{jl2}/2"))
    if hit:
        return (hit[0], hit[1], False)
    for j2 in allowed_J2(Ll):
        if j2 == jl2:
            continue
        wk = ls_line_weight(Lu, ju2, Ll, j2)
        if wk <= 1e-12:
            continue
        other = (fosc_jl or {}).get((lo_nl, up_nl, Ju, f"{j2}/2"))
        if other:
            return (other[0] * (w / (jl2 + 1)) / (wk / (j2 + 1)), other[1], True)
    return None
