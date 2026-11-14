"""Dual-kinetic-balance B-spline basis for the radial Dirac equation.

Basis functions (Shabaev, Tupitsyn, Yerokhin & Plunien, PRL 93, 130405 (2004)):

    u_i = ( B_i , (B_i' + kappa B_i / r) / 2c )   and   ( (B_i' - kappa B_i / r) / 2c , B_i )

so the generalised eigenproblem  H u = eps S u  (energies shifted by -c^2) has no
spurious states.  B-splines of degree 7 on knots uniform in s = ln r + beta r;
integrals use Gauss-Legendre points inside every knot interval (exact for the
polynomial parts), and cumulative integrals a per-interval spectral integration
matrix, so Hartree/exchange potentials are accurate to ~1e-14.

Validated: hydrogen-like Dirac levels exact to ~1e-10 for Z = 1, 55, 87
(tests/test_atomic_structure.py).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import eigh

from qali.core.constants import A0_SI, ALPHA_FS_INV

C_AU = ALPHA_FS_INV                    # speed of light in atomic units
BOHR_FM = A0_SI * 1e15                 # 1 bohr in fm

_GW = {m: np.polynomial.legendre.leggauss(m)[1] for m in range(4, 25)}


def _r_of_s(s: np.ndarray, beta: float) -> np.ndarray:
    """Invert s = ln r + beta r (vectorised Newton)."""
    r = np.exp(np.minimum(s, 50.0))
    if beta > 0:
        r = np.minimum(r, np.maximum(s / beta, 1e-12))
        for _ in range(60):
            r = r - (np.log(r) + beta * r - s) / (1 / r + beta)
            r = np.maximum(r, 1e-300)
    return r


def _gauss_tools(m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes/weights on [-1, 1] and the integration matrix
    Mi[j, k] = int_{-1}^{x_j} L_k(x) dx (L_k: Lagrange polynomials on the nodes)."""
    x, wq = np.polynomial.legendre.leggauss(m)
    coef = np.linalg.inv(np.vander(x, m, increasing=True))
    p = np.arange(m)
    integ = (x[:, None] ** (p + 1) - (-1.0) ** (p + 1)) / (p + 1)
    return x, wq, integ @ coef


class Basis:
    """B-spline DKB basis and quadrature on [0, rmax]."""

    def __init__(self, Z: float, rmax: float, n_int: int = 120, beta: float | None = None,
                 degree: int = 7, r0: float | None = None, pts: int = 12):
        self.Z, self.rmax, self.d = Z, rmax, degree
        r0 = r0 if r0 is not None else 1e-5 / Z
        beta = beta if beta is not None else (0.0 if rmax < 100 else 0.02)
        s0, s1 = math.log(r0) + beta * r0, math.log(rmax) + beta * rmax
        brk = _r_of_s(np.linspace(s0, s1, n_int + 1), beta)
        brk[-1] = rmax
        self.t = np.concatenate(([0.0] * (degree + 1), brk[:-1], [rmax] * (degree + 1)))
        nb = len(self.t) - degree - 1
        edges = np.concatenate(([0.0], brk))
        xg, wg, mi = _gauss_tools(pts)
        lo, hi = edges[:-1], edges[1:]
        half = 0.5 * (hi - lo)
        self.r = (lo[:, None] + half[:, None] * (xg[None, :] + 1)).ravel()
        self.w = (half[:, None] * wg[None, :]).ravel()
        self._half, self._mi, self._m, self._nint = half, mi, pts, len(lo)
        spl = BSpline(self.t, np.eye(nb), degree, extrapolate=False)
        keep = slice(1, nb - 1)                      # B(0) = B(rmax) = 0
        self.B = np.nan_to_num(spl(self.r))[:, keep].T
        self.B1 = np.nan_to_num(spl.derivative(1)(self.r))[:, keep].T
        self.B2 = np.nan_to_num(spl.derivative(2)(self.r))[:, keep].T
        self.nb = self.B.shape[0]

    def integrate(self, f: np.ndarray) -> float:
        return float(np.sum(f * self.w))

    def cumint(self, f: np.ndarray) -> np.ndarray:
        """int_0^r f dr' at every quadrature point (along the last axis)."""
        sh = f.shape
        F = f.reshape((*sh[:-1], self._nint, self._m))
        part = np.einsum("...ik,jk->...ij", F, self._mi) * self._half[:, None]
        full = np.einsum("...ik,k->...i", F, _GW[self._m]) * self._half
        off = np.concatenate((np.zeros((*sh[:-1], 1)), np.cumsum(full, axis=-1)[..., :-1]),
                             axis=-1)
        return (part + off[..., None]).reshape(sh)

    def components(self, kappa: int, c: float = C_AU):
        """(P, Q, DP) of the 2*nb DKB functions as rows; DP = (d/dr + kappa/r) P."""
        r, B, B1, B2 = self.r, self.B, self.B1, self.B2
        P = np.vstack([B, (B1 - kappa * B / r) / (2 * c)])
        Q = np.vstack([(B1 + kappa * B / r) / (2 * c), B])
        DP = np.vstack([B1 + kappa * B / r, (B2 - kappa * (kappa - 1) * B / r ** 2) / (2 * c)])
        return P, Q, DP


def dirac_matrices(basis: Basis, kappa: int, V: np.ndarray, c: float = C_AU):
    """H (local potential V on both components; energies shifted by -c^2), S, (P, Q)."""
    P, Q, DP = basis.components(kappa, c)
    W = basis.w
    S = (P * W) @ P.T + (Q * W) @ Q.T
    H = (P * (W * V)) @ P.T + (Q * (W * (V - 2 * c * c))) @ Q.T
    T = c * ((DP * W) @ Q.T)
    H = H + T + T.T
    return 0.5 * (H + H.T), 0.5 * (S + S.T), (P, Q)


def dirac_floor(Z: float, kappa: int, c: float = C_AU) -> float:
    """Lowest bound energy (shifted by -c^2) of channel kappa for a bare point nucleus Z.

    The lowest n of a channel is |kappa| for kappa < 0 and kappa + 1 for kappa > 0 (n_r = 0 or 1).
    A screened, finite nucleus binds less, so no physical state of an atom lies below this; a
    lower positive-energy eigenvalue is a basis intruder (the DKB spurious state of kappa > 0 near
    a point nucleus at large Z)."""
    g = math.sqrt(max(kappa * kappa - (Z / c) ** 2, 0.0))
    n_r = 0 if kappa < 0 else 1
    return c * c * (1.0 / math.sqrt(1.0 + ((Z / c) / (n_r + g)) ** 2) - 1.0)


INTRUDER_MARGIN = 0.02      # 2 % below the point-nucleus floor still counts as physical


def physical_states(eps: np.ndarray, Z: float, kappa: int, c: float = C_AU) -> np.ndarray:
    """Indices of the positive-energy eigenvalues that are physical (above -c^2 and not intruders
    below the point-nucleus floor of the channel, ``dirac_floor``)."""
    floor = dirac_floor(Z, kappa, c) * (1.0 + INTRUDER_MARGIN)
    return np.where(eps > max(-c * c, floor))[0]


def n_intruders(eps: np.ndarray, Z: float, kappa: int, c: float = C_AU) -> int:
    """Positive-energy eigenvalues below the point-nucleus floor of the channel (basis intruders)."""
    floor = dirac_floor(Z, kappa, c) * (1.0 + INTRUDER_MARGIN)
    return int(np.sum((eps > -c * c) & (eps <= floor)))


def solve_channel(H: np.ndarray, S: np.ndarray, comps, nstates: int, c: float = C_AU):
    """Lowest positive-energy states: eps, P, Q (normalised), coefficient vectors."""
    eps, vec = eigh(H, S)
    sel = np.where(eps > -c * c)[0][:nstates]
    P, Q = comps
    return eps[sel], vec[:, sel].T @ P, vec[:, sel].T @ Q, vec[:, sel]


def evaluate(basis: Basis, kappa: int, coef: np.ndarray, r: np.ndarray, c: float = C_AU):
    """P(r), Q(r) of a DKB expansion at arbitrary radii (zero beyond rmax)."""
    nb_full = len(basis.t) - basis.d - 1
    full = np.zeros((nb_full, 2))
    full[1:nb_full - 1, 0] = coef[:basis.nb]
    full[1:nb_full - 1, 1] = coef[basis.nb:]
    spl = BSpline(basis.t, full, basis.d, extrapolate=False)
    B = np.nan_to_num(spl(r))
    B1 = np.nan_to_num(spl.derivative(1)(r))
    P = B[:, 0] + (B1[:, 1] - kappa * B[:, 1] / r) / (2 * c)
    Q = (B1[:, 0] + kappa * B[:, 0] / r) / (2 * c) + B[:, 1]
    inside = r <= basis.rmax
    return np.where(inside, P, 0.0), np.where(inside, Q, 0.0)


def psi_k(basis: Basis, f: np.ndarray, k: int) -> np.ndarray:
    """int (r<^k / r>^(k+1)) f(r') dr' for the rows of f (Hartree/exchange kernels)."""
    r = basis.r
    inner = basis.cumint(f * r ** k)
    outer_cum = basis.cumint(f / r ** (k + 1))
    outer = outer_cum[..., -1:] - outer_cum
    return inner / r ** (k + 1) + outer * r ** k


def nuclear_radius(mass_number: float) -> float:
    """Uniform-sphere nuclear radius R = 1.2 fm A^(1/3), in bohr."""
    return 1.2 * mass_number ** (1 / 3) / BOHR_FM


def default_mass_number(Z: int) -> int:
    """A rough mass number on the valley of stability, A ≈ 2Z + 0.006 Z² (Fr: 219, Cs: 128), for a
    finite nucleus when no isotope is given: R ∝ A^(1/3), so 5 % in A is under 2 % in R."""
    return max(1, round(2 * Z + 0.006 * Z * Z)) if Z > 1 else 1


def nuclear_potential(r: np.ndarray, Z: float, R: float) -> np.ndarray:
    """-Z/r outside, uniform charged sphere inside (R in bohr; R = 0: point nucleus)."""
    V = -Z / r
    if R > 0:
        inside = r < R
        V[inside] = -Z / (2 * R) * (3 - (r[inside] / R) ** 2)
    return V
