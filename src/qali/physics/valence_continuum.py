"""Photoionization continuum of the valence electron, as pseudo-oscillators for α(ω).

The polarizability sums run over the bound np (or ns, nd) levels up to n = 50; the valence
oscillator strength that goes to the continuum is missing from them. For Li 2s that is 0.26 of the
total, so α(0) came out 0.8 % and C₆ 2.6 % low; for the heavier alkalis it is 0.015–0.05.

Here the bound state and the energy-normalised continuum εl' are solved in the same
ℓ-dependent Marinescu model potential as the orbital viewer (``orbital_radial``), and

    df/dε = (2/3) ω Σ_j' (2j'+1) {l j ½; j' l' 1}² l_> |∫ u_nlj r u_εl'j' dr|²,   ω = ε + E_bind,

which reduces to (2/3) ω l_>/(2l+1) R² when the radial integral does not depend on j'. df/dε is
integrated on a log grid from threshold to 20 Hartree (plus an ε^-7/2 tail) and each grid point
becomes one line at ω = ε + E_bind, with E_bind the level's measured binding energy so the
threshold sits at the true ionization limit.

Only species with model parameters (neutral Li–Fr) and final channels with parameters get a
continuum; Fr d and f are not fitted, so Fr p → εd is left out.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache

import numpy as np

from qali.physics.atomic_structure.angular import sixj2
from qali.physics.extrapolation.exact import _numerov
from qali.physics.orbital_radial import H_STEP, R_MIN, model_potential, model_state, model_supported

E_MAX_HA = 20.0                 # photoelectron energy where the grid stops; an ε^-7/2 tail follows
REL_SIGMA = 0.10                # 1σ given to the model continuum f-values
_LABEL = re.compile(r'^(\d+)([spdfghi])(?:(\d+)/2)?')
_L = 'spdfghi'


def energy_grid() -> np.ndarray:
    """Photoelectron energies ε (Hartree): dense near threshold, logarithmic to E_MAX_HA."""
    return np.concatenate([np.geomspace(1e-4, 0.05, 20), np.geomspace(0.05, E_MAX_HA, 44)[1:]])


def continuum_u(symbol: str, l: int, j: float, eps: float, x: np.ndarray) -> np.ndarray:
    """Energy-normalised continuum u_εl(r) (∫u_ε u_ε' dr = δ(ε − ε'), Hartree) on the log grid x.

    Outward Numerov from the origin. Outside the core u = A sqrt(2/(πk)) sin φ with the local
    k(r) = sqrt(2(ε − V) − l(l+1)/r²) (WKB), so A² is the phase average of π k u² over the outer half
    of the grid (weight dφ = k dr, many periods); A = 1 is energy normalisation.
    """
    r = np.exp(x)
    V = model_potential(symbol, l, j, r)
    y = _numerov(2.0 * r * r * (V - eps) + (l + 0.5) ** 2, x[1] - x[0],
                 math.exp((l + 0.5) * x[0]), math.exp((l + 0.5) * x[1]))
    u = y * np.sqrt(r)
    u = u / np.max(np.abs(u[len(u) // 2:]))
    k = np.sqrt(np.maximum(2.0 * (eps - V) - l * (l + 1) / r**2, 1e-30))
    outer = r > 0.5 * r[-1]
    w = k[outer] * r[outer]                         # dφ = k dr = k r dx on the log grid
    amp2 = float(np.sum(math.pi * k[outer] * u[outer] ** 2 * w) / np.sum(w))
    return u / math.sqrt(amp2)


def radial_grid(eps: float) -> np.ndarray:
    """Log grid for the continuum at ε: out to 40 wavelengths (at least 60 a0), with at least 25
    points per wavelength at the outer edge."""
    k = math.sqrt(2.0 * eps)
    rmax = max(60.0, 40.0 * 2.0 * math.pi / k)
    h = min(H_STEP, 2.0 * math.pi / (k * rmax * 25.0))
    return np.arange(math.log(R_MIN), math.log(rmax), h)


def _final_channels(symbol: str, l: int, j2: int):
    """(l', j', weight) with Σ weight = l_>/(2l+1) over the j' of each l'; j2 = 2j, or 0 for an
    nl level without fine structure (then the j' weights are the statistical (2j'+1)/(2(2l'+1)))."""
    out = []
    for lf in (l - 1, l + 1):
        if lf < 0 or not model_supported(symbol, lf):
            continue
        lg = max(l, lf)
        for jf2 in (2 * lf - 1, 2 * lf + 1):
            if jf2 < 1:
                continue
            if j2:
                if abs(jf2 - j2) > 2:
                    continue
                w = (jf2 + 1) * sixj2(2 * l, j2, 1, jf2, 2 * lf, 2) ** 2 * lg
            else:
                w = (jf2 + 1) / (2.0 * (2 * lf + 1)) * lg / (2 * l + 1)
            if w > 0:
                out.append((lf, jf2 / 2.0, w))
    return out


@lru_cache(maxsize=128)
def continuum_lines(symbol: str, n: int, l: int, j2: int, binding_au: float) -> tuple[tuple[float, float], ...]:
    """((ω, f), ...) pseudo-lines for the level n l j (j2 = 2j, 0 if not resolved), absorption from it.

    Empty when the species or level has no model parameters, or the bound state is not found.
    """
    if not model_supported(symbol, l) or binding_au <= 0:
        return ()
    j = j2 / 2.0 if j2 else (l + 0.5 if l else 0.5)
    try:
        _, rb, ub = model_state(symbol, n, l, j, nu_hint=1.0 / math.sqrt(2.0 * binding_au))
    except RuntimeError:
        return ()
    chans = _final_channels(symbol, l, j2)
    if not chans:
        return ()
    eps = energy_grid()
    dfde = np.zeros_like(eps)
    for i, e in enumerate(eps):
        x = radial_grid(float(e))
        r = np.exp(x)
        ubi = np.interp(r, rb, ub, right=0.0)
        s = sum(w * float(np.trapezoid(ubi * r * continuum_u(symbol, lf, jf, e, x) * r, x)) ** 2
                for lf, jf, w in chans)
        dfde[i] = (2.0 / 3.0) * (e + binding_au) * s
    wts = np.zeros_like(eps)
    d = np.diff(eps)
    wts[:-1] += d / 2
    wts[1:] += d / 2
    wts[0] += eps[0]                              # from threshold to the first grid point
    lines = [(float(e + binding_au), float(f * w)) for e, f, w in zip(eps, dfde, wts, strict=True) if f * w > 0]
    tail = dfde[-1] * eps[-1] / 2.5               # ∫_εmax^∞ of df/dε ∝ ε^-7/2
    if tail > 0:
        lines.append((float(eps[-1] * 5.0 / 3.0 + binding_au), float(tail)))   # its mean ε
    return tuple(lines)


def lines_for_state(symbol: str, charge: int, label: str, binding_au: float):
    """Continuum pseudo-lines for a state label such as '2s', '5p3/2' or '49p' (unresolved)."""
    if charge:
        return ()
    m = _LABEL.match(str(label))
    if not m:
        return ()
    n, l = int(m.group(1)), _L.index(m.group(2))
    j2 = int(m.group(3)) if m.group(3) else 0
    if j2 and abs(j2 - 2 * l) != 1:
        j2 = 0
    return continuum_lines(symbol, n, l, j2, round(float(binding_au), 10))
