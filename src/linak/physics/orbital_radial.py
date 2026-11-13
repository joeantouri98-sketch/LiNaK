"""Radial functions for the orbital viewer.

Three methods, chosen per state:

* ``hydrogenic`` — one-electron species (H, He+, ...): the analytic radial function. Exact.
* ``model``      — alkalis Li–Fr: the l-dependent Marinescu–Sadeghpour–Dalgarno potential
  (Phys. Rev. A 49, 982 (1994)) with core polarization and spin–orbit, the same potential
  as the extrapolation model (``physics/extrapolation/exact.py``). Solved for the state with
  n − l − 1 nodes, so the inner shells are present; the eigenvalue reproduces the level.
* ``coulomb``    — everything else: the decaying Coulomb (Whittaker) function at the level's
  n*, exact outside the ionic core only. Inside r_c (inner classical turning point, at least
  1/Z a0) it is continued as u ∝ r^(l+1); the core is not modelled.

All functions return ``u(r) = r R(r)`` in atomic units, normalised to ∫u² dr = 1.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import eval_genlaguerre

from linak.physics.extrapolation.exact import ALPHA, MARINESCU, RCUT, _numerov

H_STEP = 0.002          # log-grid step (same as the extrapolation model)
R_MIN = 1e-5            # a0
CORE_FRAC = 0.99        # core radius: the sphere holding 99 % of the core electrons


# ── which method applies ──────────────────────────────────────────────────────
def model_supported(symbol: str, l: int, charge: int = 0) -> bool:
    """Marinescu parameters exist (neutral Li–Fr; Fr d and f are not fitted)."""
    if charge != 0 or symbol not in MARINESCU:
        return False
    return not (symbol == "Fr" and l >= 2)


def method_for(symbol: str, Z: int, charge: int, l: int, prefer: str = "model") -> str:
    if Z - charge == 1:
        return "hydrogenic"
    if prefer == "model" and model_supported(symbol, l, charge):
        return "model"
    return "coulomb"


# ── Marinescu model potential ────────────────────────────────────────────────
def _pidx(l: int) -> int:
    return min(l, 3)        # l > 3 uses the f parameters (the ARC convention)


def zeff_l(symbol: str, l: int, r):
    """Effective charge Z_l(r) of the model potential (→ Z at r → 0, → 1 outside the core)."""
    p, i = MARINESCU[symbol], _pidx(l)
    return (1.0 + (p["Z"] - 1.0) * np.exp(-p["a1"][i] * r)
            - r * (p["a3"][i] + p["a4"][i] * r) * np.exp(-p["a2"][i] * r))


def _vc(symbol: str, l: int, r):
    p, i = MARINESCU[symbol], _pidx(l)
    return -zeff_l(symbol, l, r) / r - p["alphaC"] / (2 * r**4) * (1 - np.exp(-((r / p["rc"][i]) ** 6)))


def so_scale(symbol: str, l: int) -> float:
    """Spin–orbit scale of the p channel, as fitted for the extrapolation model (1 otherwise)."""
    if l != 1:
        return 1.0
    try:
        from linak.physics.extrapolation.params import species_params
        return float(species_params(symbol).get("so_scale", 1.0))
    except Exception:
        return 1.0


def model_potential(symbol: str, l: int, j: float, r):
    """V_l(r) + spin–orbit (α²/2) L·S (1/r) dV/dr, the derivative cut off below RCUT.

    The p-channel spin–orbit term carries the fitted scale of the extrapolation model, so the
    fine-structure splitting matches the levels.
    """
    v = _vc(symbol, l, r)
    if l == 0:
        return v
    ls = (j * (j + 1) - l * (l + 1) - 0.75) / 2.0
    re = np.maximum(r, RCUT)
    h = 1e-5 * re
    dv = (_vc(symbol, l, re + h) - _vc(symbol, l, re - h)) / (2 * h)
    return v + ALPHA**2 / 2 * ls * dv / re * so_scale(symbol, l)


def core_radius(symbol: str, l: int) -> float:
    """Radius of the sphere holding 99 % of the core electrons in the l-channel potential.

    By Gauss's law Z_l(r) − 1 is the core charge outside r, so this is the outermost r with
    |Z_l(r) − 1| ≥ 1 % of the Z − 1 core electrons (Na 1.7 a0, Cs 2.6 a0).
    """
    r = np.linspace(0.01, 30.0, 30000)
    bad = np.abs(zeff_l(symbol, l, r) - 1.0) >= (1.0 - CORE_FRAC) * (MARINESCU[symbol]["Z"] - 1.0)
    return float(r[np.nonzero(bad)[0][-1]]) if bad.any() else 0.0


def model_state(symbol: str, n: int, l: int, j: float, nu_hint: float | None = None):
    """(E_Ha, r, u) of the bound state with n − l − 1 nodes (u normalised)."""
    rmax = 2.0 * n * n + 60.0 * n + 40
    x = np.arange(math.log(R_MIN), math.log(rmax), H_STEP)
    r = np.exp(x)
    target = n - l - 1
    V = model_potential(symbol, l, j, r)
    r2 = 2.0 * r * r
    lp = (l + 0.5) ** 2
    u0, u1 = math.exp((l + 0.5) * x[0]), math.exp((l + 0.5) * x[1])

    def solve(E):
        return _numerov(r2 * (V - E) + lp, H_STEP, u0, u1)

    def nodes_and_tail(E):
        u = solve(E)
        sg = np.sign(u)
        return int(np.sum(sg[1:] * sg[:-1] < 0)), u[-1]

    scans = []
    if nu_hint:
        scans.append(np.linspace(max(nu_hint - 0.4, 0.3), nu_hint + 0.4, 41))
    scans.append(np.linspace(max(0.3, n - 6.0), n + 1.5, 500))
    for nus in scans:
        prev_nodes, e_prev = None, None
        for E in -0.5 / nus**2:
            nd, _ = nodes_and_tail(E)
            if prev_nodes == target and nd == target + 1:
                eb = brentq(lambda ee: nodes_and_tail(ee)[1], e_prev, E, xtol=1e-15)
                y = solve(eb)
                u = y * np.sqrt(r)
                k0 = int(np.searchsorted(r, 1.0 / abs(eb)))
                cut = max(k0 + int(np.argmin(np.abs(u)[k0:])), k0 + 1)
                u, rr, xx = u[:cut], r[:cut], x[:cut]
                u = u / math.sqrt(np.trapezoid(u * u * rr, xx))
                if u[np.argmax(np.abs(u))] < 0:
                    u = -u
                return float(eb), rr, u
            prev_nodes, e_prev = nd, E
    raise RuntimeError(f"no bound state found for {symbol} n={n} l={l} j={j}")


# ── hydrogenic and Coulomb (QDT) ──────────────────────────────────────────────
def hydrogenic_state(n: int, l: int, Z: float = 1.0):
    """Analytic u_nl(r) for nuclear charge Z, on a log grid out to where it is negligible."""
    rmax = (2.0 * n * n + 30.0 * n + 40) / Z
    x = np.arange(math.log(R_MIN / Z), math.log(rmax), H_STEP)
    r = np.exp(x)
    rho = 2.0 * Z * r / n
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        u = r * np.exp(-rho / 2) * rho**l * eval_genlaguerre(n - l - 1, 2 * l + 1, rho)
    u = np.nan_to_num(u)
    u = u / math.sqrt(np.trapezoid(u * u * r, x))
    if u[np.argmax(np.abs(u))] < 0:
        u = -u
    return -0.5 * Z * Z / (n * n), r, u


def coulomb_cutoff(nstar: float, l: int, z: float = 1.0) -> float:
    disc = 1.0 - l * (l + 1) / (nstar * nstar)
    r_in = (nstar * nstar / z) * (1.0 - math.sqrt(disc)) if disc > 0 else nstar * nstar / z
    return max(r_in, 1.0 / z)


def coulomb_state(nstar: float, l: int, z: float = 1.0):
    """Decaying Whittaker function at E = −z²/(2n*²) by inward Numerov; u ∝ r^(l+1) inside r_c."""
    rc = coulomb_cutoff(nstar, l, z)
    rstart = (2.0 * nstar * nstar + 30.0 * nstar + 40) / z
    x = np.arange(math.log(R_MIN / z), math.log(rstart), H_STEP)
    r = np.exp(x)
    E = -0.5 * z * z / (nstar * nstar)
    Q = (2.0 * r * r * (-z / r - E) + (l + 0.5) ** 2)
    # inward: integrate the reversed arrays with the asymptotic start y ~ r^(n* - 1/2) e^(-z r / n*)
    lny = (nstar - 0.5) * x - z * r / nstar
    yr = _numerov(Q[::-1], -H_STEP, 1.0, math.exp(lny[-2] - lny[-1]))
    y = yr[::-1]
    u = y * np.sqrt(r)
    k = int(np.searchsorted(r, rc))
    u[:k] = u[k] * (r[:k] / r[k]) ** (l + 1)
    if not np.all(np.isfinite(u)):
        raise RuntimeError("Coulomb function overflowed")
    u = u / math.sqrt(np.trapezoid(u * u * r, x))
    if u[np.argmax(np.abs(u))] < 0:
        u = -u
    return E, r, u, rc


# ── summary numbers ──────────────────────────────────────────────────────────
def node_radii(r, u, tail_frac: float = 1e-6) -> list[float]:
    """Radii of the sign changes of u (linear interpolation), ignoring the numerical tail."""
    a = np.abs(u)
    keep = a > tail_frac * a.max()
    idx = np.nonzero(keep)[0]
    lo, hi = idx[0], idx[-1]
    s = np.sign(u[lo:hi + 1])
    k = np.nonzero(s[1:] * s[:-1] < 0)[0] + lo
    return [float(r[i] - u[i] * (r[i + 1] - r[i]) / (u[i + 1] - u[i])) for i in k]


def expectation(r, u, power: float) -> float:
    x = np.log(r)
    return float(np.trapezoid(u * u * r**power * r, x))


def enclosed_radius(r, u, frac: float) -> float:
    """Radius enclosing the probability ``frac``."""
    x = np.log(r)
    w = u * u * r
    c = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(x))])
    c /= c[-1]
    return float(np.interp(frac, c, r))


def probability_inside(r, u, rc: float) -> float:
    if rc <= r[0]:
        return 0.0
    x = np.log(r)
    k = int(np.searchsorted(r, rc))
    return float(np.trapezoid(u[:k] ** 2 * r[:k], x[:k]))


def turning_points(nstar: float, l: int, z: float = 1.0) -> tuple[float, float]:
    """Classical turning points of the Coulomb problem at n* (inner 0 for s)."""
    disc = max(1.0 - l * (l + 1) / (nstar * nstar), 0.0)
    s = math.sqrt(disc)
    return (nstar * nstar / z) * (1 - s), (nstar * nstar / z) * (1 + s)


def resample(r, u, npts: int):
    """u on a uniform log-r grid from the first non-negligible point to the end of the tail."""
    a = np.abs(u)
    big = np.nonzero(a > 1e-7 * a.max())[0]
    lo = max(int(big[0]) - 1, 0)
    w = u * u * r
    c = np.cumsum(w)
    c /= c[-1]
    hi = min(int(np.searchsorted(c, 1 - 1e-9)) + 1, len(r) - 1)
    hi = max(hi, int(big[-1]))
    xs = np.linspace(math.log(r[lo]), math.log(r[hi]), npts)
    return xs, np.interp(xs, np.log(r), u)


def state_radial(symbol: str, Z: int, charge: int, z_core: float, n: int, l: int, j: float,
                 nstar: float, prefer: str = "model") -> dict:
    """Radial function and summary numbers for one level (atomic units)."""
    method = method_for(symbol, Z, charge, l, prefer)
    rc = None
    if method == "model":
        try:
            E, r, u = model_state(symbol, n, l, j, nu_hint=nstar)
            rc = core_radius(symbol, l)
        except RuntimeError:
            method = "coulomb"
    if method == "hydrogenic":
        E, r, u = hydrogenic_state(n, l, float(Z))
        rc = 0.0
    if method == "coulomb":
        E, r, u, rc = coulomb_state(nstar, l, z_core)
    nu = z_core / math.sqrt(-2.0 * E)
    nodes = node_radii(r, u)
    tin, tout = turning_points(nstar, l, z_core)
    return dict(
        method=method, r=r, u=u, nu=nu, nodes=nodes, rc=rc,
        r1=expectation(r, u, 1), r2=expectation(r, u, 2), rm1=expectation(r, u, -1),
        rm3=expectation(r, u, -3) if l > 0 else None,
        p_core=probability_inside(r, u, rc) if rc else 0.0,
        r_box=enclosed_radius(r, u, 0.999), t_in=tin, t_out=tout,
    )
