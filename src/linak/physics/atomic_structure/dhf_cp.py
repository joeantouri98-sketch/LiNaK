"""Dirac-Hartree-Fock frozen core + core polarisation for one-valence atoms.

Model
-----
* Core: closed-shell Dirac-Hartree-Fock for the M+ ion (exact non-local exchange),
  j-j subshells with 2j+1 electrons; Pulay DIIS, then damped iteration.
* Valence: V^(N-1) — the same core Fock operator for every kappa channel, in a
  3000-bohr box, so Rydberg series reach n ~ 30 and the discretised continuum makes
  sums over states complete.
* Core polarisation (Norcross, PRA 7, 606 (1973); Mitroy, Safronova & Clark,
  J. Phys. B 43, 202001 (2010)):
      V_cp(r) = -alpha_c / (2 r^4) [1 - exp(-(r/rho)^6)],
  with the consistent dipole-operator correction
      r -> r [1 - alpha_c / r^3 (1 - exp(-(r/rho)^6))].
  rho for s is fixed by the measured ionisation energy; optional per-kappa cutoffs
  are fixed by measured levels (the pipeline passes the NIST lowest d3/2, d5/2).

Not included: Breit, QED, correlation beyond the dipole polarisation potential,
non-adiabatic (energy-dependent) core polarisation.

Engine contract: inputs as arguments, results as frozen dataclasses; no files,
prints or plots.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq

from linak.core.constants import AU_TIME_S, CM1_TO_EV, HARTREE_TO_EV, HC_EV_NM
from linak.physics.atomic_structure.angular import (
    exchange_coefficients,
    kappa_label,
    kappa_lj,
    kappas_of_l,
    reduced_c,
)
from linak.physics.atomic_structure.dirac_basis import (
    C_AU,
    Basis,
    dirac_matrices,
    evaluate,
    nuclear_potential,
    nuclear_radius,
    psi_k,
)

HARTREE_TO_CM1 = HARTREE_TO_EV / CM1_TO_EV
VALENCE_KAPPAS = (-1, 1, -2, 2, -3, 3, -4)            # s1/2 ... f7/2

# Subshell filling order (Madelung) used to build noble-gas cores.
_MADELUNG = ((1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (3, 2), (4, 1), (5, 0), (4, 2),
             (5, 1), (6, 0), (4, 3), (5, 2), (6, 1), (7, 0))
_NOBLE_GAS_ELECTRONS = (2, 10, 18, 36, 54, 86)


def one_valence_configuration(Z: int) -> tuple[tuple[tuple[int, int], ...], int]:
    """(closed core subshells (n, l), valence n) for an atom with one s electron outside a
    noble-gas core (H excluded).  Raises ValueError for any other atom."""
    n_core = Z - 1
    if n_core not in _NOBLE_GAS_ELECTRONS:
        raise ValueError(f"Z={Z} is not one s electron outside a noble-gas core")
    core: list[tuple[int, int]] = []
    left = n_core
    for n, l in _MADELUNG:
        if left == 0:
            return tuple(core), n
        cap = 2 * (2 * l + 1)
        core.append((n, l))
        left -= cap
    raise ValueError(f"Z={Z} beyond the supported range")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Level:
    label: str                 # '6p3/2'
    n: int
    l: int
    j: float
    kappa: int
    energy_cm: float           # above the ground state
    binding_eV: float
    n_star: float
    quantum_defect: float
    fitted: bool               # True if this level was an input of the model


@dataclass(frozen=True)
class Transition:
    lower: str
    upper: str
    wavelength_nm_vac: float
    reduced_dipole_au: float   # |<lower||d||upper>|, core-polarisation-corrected
    f_absorption: float
    A_per_s: float


@dataclass(frozen=True)
class Lifetime:
    state: str
    tau_ns: float


@dataclass(frozen=True)
class DHFResult:
    Z: int
    valence: str
    ionization_cm: float
    alpha_core_au: float
    cpp_cutoff_bohr: float
    cpp_cutoffs_kappa_bohr: dict[str, float]
    nuclear_radius_bohr: float
    core_scf_iterations: int
    alpha0_au: float
    alpha_dynamic_au: dict[str, float]
    levels: tuple[Level, ...]
    transitions: tuple[Transition, ...]
    lifetimes: tuple[Lifetime, ...]
    quantum_defects: dict[str, tuple[tuple[int, float], ...]]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class _State:
    __slots__ = ("P", "Q", "eps", "kappa", "n")

    def __init__(self, kappa, n, eps, P, Q):
        self.kappa, self.n, self.eps, self.P, self.Q = kappa, n, eps, P, Q


class _Model:
    def __init__(self, Z, core_nl, nv, ip_au, alpha_core, mass_number, *, c=C_AU,
                 rmax=3000.0, n_int_val=260, n_int_core=110):
        self.Z, self.c, self.nv, self.ip, self.alpha_c = Z, c, nv, ip_au, alpha_core
        self.core = [(n, kap, kappa_lj(kap)[1] + 1) for n, l in core_nl for kap in kappas_of_l(l)]
        self.core_kappas = sorted({k for _, k, _ in self.core}, key=lambda k: (kappa_lj(k)[0], k > 0))
        self.R_nuc = nuclear_radius(mass_number) if mass_number else 0.0
        self.bc = Basis(Z, rmax=40.0, n_int=n_int_core, beta=0.0)
        # first knot at 1e-3/Z: inside the nucleus, but avoids the dynamic range that
        # would cost high-n Rydberg levels their last digits
        self.bv = Basis(Z, rmax=rmax, n_int=n_int_val, beta=0.02, r0=1e-3 / Z)
        self.rho = 1.0
        self.rho_k: dict[int, float] = {}

    # ---- core ------------------------------------------------------------
    def _direct(self, basis, orbs):
        dens = np.zeros_like(basis.r)
        for (_, _, occ), (P, Q) in orbs.items():
            dens += occ * (P * P + Q * Q)
        return psi_k(basis, dens[None, :], 0)[0]

    def _exchange(self, basis, kappa, comps, orbs, idx=None):
        P, Q = comps[0], comps[1]
        if idx is not None:
            P, Q = P[idx], Q[idx]
        K = np.zeros((P.shape[0], P.shape[0]))
        for (_, kb, _), (Pb, Qb) in orbs.items():
            F = P * Pb + Q * Qb
            for k, lam in exchange_coefficients(kappa, kb):
                K += lam * ((F * basis.w) @ psi_k(basis, F, k).T)
        return 0.5 * (K + K.T)

    def scf_core(self, maxit=120, tol=1e-8, ndiis=8):
        b, c, Z = self.bc, self.c, self.Z
        r = b.r
        Vn = nuclear_potential(r, Z, self.R_nuc)
        x = r / (0.8853 * Z ** (-1 / 3))               # Thomas-Fermi start (Tietz form)
        phi = 1 / (1 + 0.02747 * x ** 0.5 + 1.243 * x - 0.1486 * x ** 1.5
                   + 0.2302 * x ** 2 + 0.007298 * x ** 2.5 + 0.006944 * x ** 3)
        Vscr = (Z - 1) / r * (1 - phi)
        comps = {k: b.components(k, c) for k in self.core_kappas}
        nsub = {k: sorted(n for n, kk, _ in self.core if kk == k) for k in self.core_kappas}
        hist_F: list[dict] = []
        hist_E: list[np.ndarray] = []
        orbs = None
        eps_old = None
        use_diis = True
        Fprev: dict = {}
        self.core_vec: dict = {}
        it = 0
        for it in range(maxit):
            V = Vn + Vscr
            Fk, Sk = {}, {}
            for kap in self.core_kappas:
                H, S, _ = dirac_matrices(b, kap, V, c)
                if orbs is not None:
                    H = H - self._exchange(b, kap, comps[kap], orbs)
                Fk[kap], Sk[kap] = H, S
            if orbs is not None and not use_diis:
                # final phase: damped plain iteration (near convergence the DIIS error
                # vectors are dominated by roundoff from the near-nucleus Fock entries)
                Fk = {k: 0.3 * Fprev[k] + 0.7 * Fk[k] for k in Fk}
            if orbs is not None and use_diis:
                err = []
                for kap in self.core_kappas:
                    C = np.array([self.core_vec[(n, kap)][0] for n in nsub[kap]]).T
                    D = C @ C.T
                    err.append((Fk[kap] @ D @ Sk[kap] - Sk[kap] @ D @ Fk[kap]).ravel())
                hist_F = ([*hist_F, {k: v.copy() for k, v in Fk.items()}])[-ndiis:]
                hist_E = ([*hist_E, np.concatenate(err)])[-ndiis:]
                m = len(hist_E)
                if m >= 2:
                    Bm = -np.ones((m + 1, m + 1))
                    Bm[m, m] = 0.0
                    for i in range(m):
                        for j in range(m):
                            Bm[i, j] = hist_E[i] @ hist_E[j]
                    rhs = np.zeros(m + 1)
                    rhs[m] = -1.0
                    try:
                        cf = np.linalg.solve(Bm, rhs)[:m]
                        Fk = {k: sum(cf[i] * hist_F[i][k] for i in range(m)) for k in Fk}
                    except np.linalg.LinAlgError:
                        pass
            Fprev = Fk
            new_orbs, eps_all = {}, []
            for kap in self.core_kappas:
                eps, vec = eigh(Fk[kap], Sk[kap])
                sel = np.where(eps > -c * c)[0][:len(nsub[kap])]
                occ = kappa_lj(kap)[1] + 1
                for i, n in enumerate(nsub[kap]):
                    v = vec[:, sel[i]]
                    new_orbs[(n, kap, occ)] = (v @ comps[kap][0], v @ comps[kap][1])
                    self.core_vec[(n, kap)] = (v, eps[sel[i]])
                    eps_all.append(eps[sel[i]])
            orbs = new_orbs
            Vscr = self._direct(b, orbs)
            eps_arr = np.array(eps_all)
            if eps_old is not None:
                # relative criterion: deep core levels sit at the eigensolver's roundoff
                # floor (~1e-9 relative) long before an absolute 1e-9 is reachable
                rel = np.max(np.abs(eps_arr - eps_old) / np.maximum(1.0, np.abs(eps_arr)))
                if rel < tol:
                    break
                if use_diis and rel < 1e-6:
                    use_diis = False
            eps_old = eps_arr
        self.core_iterations = it + 1

    # ---- valence ---------------------------------------------------------
    def setup_valence(self):
        bv, c = self.bv, self.c
        r = bv.r
        self.core_orbs_v = {key: evaluate(self.bc, key[1], self.core_vec[key[:2]][0], r, c)
                            for key in self.core}
        V0 = nuclear_potential(r, self.Z, self.R_nuc) + self._direct(bv, self.core_orbs_v)
        dens = sum(P * P + Q * Q for P, Q in self.core_orbs_v.values())
        rcore = r[np.where(dens > 1e-16)[0][-1]]
        self.F0, self.S, self.comps = {}, {}, {}
        for kap in VALENCE_KAPPAS:
            comps = bv.components(kap, c)
            H, S, _ = dirac_matrices(bv, kap, V0, c)
            # exchange only couples basis functions that overlap the core
            first = np.argmax(np.abs(comps[0]) > 0, axis=1)
            idx = np.where(r[first] < rcore)[0]
            H[np.ix_(idx, idx)] -= self._exchange(bv, kap, comps, self.core_orbs_v, idx)
            self.F0[kap], self.S[kap], self.comps[kap] = H, S, comps

    def vcp(self, rho):
        r = self.bv.r
        return -self.alpha_c / (2 * r ** 4) * (1 - np.exp(-(r / rho) ** 6))

    def channel(self, kap):
        H = self.F0[kap]
        if self.alpha_c > 0:
            P, Q, _ = self.comps[kap]
            w = self.bv.w * self.vcp(self.rho_k.get(kap, self.rho))
            H = H + (P * w) @ P.T + (Q * w) @ Q.T
        eps, vec = eigh(H, self.S[kap])
        sel = np.where(eps > -self.c ** 2)[0]
        P, Q, _ = self.comps[kap]
        return eps[sel], vec[:, sel].T @ P, vec[:, sel].T @ Q

    def ncore(self, kap):
        return sum(1 for _, k, _ in self.core if k == kap)

    def fit_rho_s(self):
        if self.alpha_c <= 0:
            return
        k0 = self.ncore(-1)

        def miss(rho):
            self.rho = rho
            return self.channel(-1)[0][k0] + self.ip
        if miss(0.3) * miss(8.0) > 0:
            raise ValueError("cannot match the ionization energy with a cutoff in [0.3, 8] bohr")
        self.rho = brentq(miss, 0.3, 8.0, xtol=1e-12)

    def fit_rho_kappa(self, levels_cm):
        g_e = self.channel(-1)[0][self.ncore(-1)]
        for (n, kap), e_cm in levels_cm.items():
            idx = n - kappa_lj(kap)[0] - 1          # core states of kappa come first

            def miss(rho, kap=kap, idx=idx, e_cm=e_cm):
                self.rho_k[kap] = rho
                return (self.channel(kap)[0][idx] - g_e) * HARTREE_TO_CM1 - e_cm
            if miss(0.2) * miss(10.0) > 0:
                raise ValueError(f"cannot fit a cutoff for {kappa_label(n, kap)} to {e_cm} cm-1")
            self.rho_k[kap] = brentq(miss, 0.2, 10.0, xtol=1e-10)

    def collect_states(self):
        self.states = {}
        for kap in VALENCE_KAPPAS:
            eps, P, Q = self.channel(kap)
            nc = self.ncore(kap)
            l = kappa_lj(kap)[0]
            self.states[kap] = [_State(kap, nc + l + 1 + i, eps[nc + i], P[nc + i], Q[nc + i])
                                for i in range(len(eps) - nc)]

    # ---- E1 --------------------------------------------------------------
    def reff(self):
        r = self.bv.r
        if self.alpha_c <= 0:
            return r
        return r - self.alpha_c / r ** 2 * (1 - np.exp(-(r / self.rho) ** 6))

    def red_d(self, a, b, reff):
        rc = reduced_c(a.kappa, b.kappa)
        return 0.0 if rc == 0.0 else rc * self.bv.integrate((a.P * b.P + a.Q * b.Q) * reff)

    def level(self, n, kap):
        return next(s for s in self.states[kap] if s.n == n)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def compute(Z: int, ionization_cm: float, alpha_core_au: float, *,
            mass_number: float | None = None,
            fit_levels_cm: Mapping[tuple[int, int], float] | None = None,
            n_extra: int = 6, n_defects: int = 30,
            omegas_eV: Sequence[float] = ()) -> DHFResult:
    """DHF + core-polarisation spectrum and E1 properties of a one-valence atom.

    ionization_cm   measured first ionization limit (fixes the s cutoff)
    alpha_core_au   M+ core dipole polarisability (0 -> pure frozen-core DHF)
    fit_levels_cm   {(n, kappa): energy above ground in cm-1}: per-kappa cutoffs
    n_extra         levels / transitions / lifetimes are reported up to n_valence + n_extra
    """
    core_nl, nv = one_valence_configuration(Z)
    m = _Model(Z, core_nl, nv, ionization_cm / HARTREE_TO_CM1, alpha_core_au, mass_number)
    m.scf_core()
    m.setup_valence()
    m.fit_rho_s()
    fitted = dict(fit_levels_cm or {})
    if fitted and alpha_core_au > 0:
        m.fit_rho_kappa(fitted)
    m.collect_states()
    reff = m.reff()
    g = m.level(nv, -1)
    nmax = nv + n_extra

    report = sorted((s for kap in VALENCE_KAPPAS for s in m.states[kap] if s.eps < 0 and s.n <= nmax),
                    key=lambda s: s.eps)
    levels = []
    for s in report:
        ns = 1 / math.sqrt(-2 * s.eps)
        l, j2 = kappa_lj(s.kappa)
        levels.append(Level(kappa_label(s.n, s.kappa), s.n, l, j2 / 2, s.kappa,
                            float((s.eps - g.eps) * HARTREE_TO_CM1),
                            float(-s.eps * HARTREE_TO_EV), float(ns), float(s.n - ns),
                            (s.n, s.kappa) in fitted))

    alpha3 = 1 / m.c ** 3
    transitions = []
    A_tot = {kappa_label(s.n, s.kappa): 0.0 for s in report}
    for i, lo in enumerate(report):
        for up in report[i + 1:]:
            if reduced_c(up.kappa, lo.kappa) == 0.0:
                continue
            d = m.red_d(lo, up, reff)
            w = up.eps - lo.eps
            f = 2.0 / 3.0 * w * d * d / (kappa_lj(lo.kappa)[1] + 1)
            A = 4.0 / 3.0 * alpha3 * w ** 3 * d * d / (kappa_lj(up.kappa)[1] + 1) / AU_TIME_S
            up_lab = kappa_label(up.n, up.kappa)
            A_tot[up_lab] += A
            transitions.append(Transition(kappa_label(lo.n, lo.kappa), up_lab,
                                          float(HC_EV_NM / (w * HARTREE_TO_EV)), float(abs(d)),
                                          float(f), float(A)))
    lifetimes = tuple(Lifetime(lab, float(1e9 / A)) for lab, A in A_tot.items() if A > 0)

    def alpha_at(omega):
        jv2 = kappa_lj(g.kappa)[1]
        tot = 0.0
        for kap in VALENCE_KAPPAS:
            if reduced_c(kap, g.kappa) == 0.0:
                continue
            for s in m.states[kap]:
                d = m.red_d(s, g, reff)
                w = s.eps - g.eps
                tot += d * d * w / (w * w - omega * omega)
        return float(2.0 / (3 * (jv2 + 1)) * tot + alpha_core_au)

    defects = {}
    for kap in VALENCE_KAPPAS:
        defects[kappa_label(0, kap)[1:]] = tuple(
            (s.n, float(s.n - 1 / math.sqrt(-2 * s.eps)))
            for s in m.states[kap] if s.eps < 0 and s.n <= n_defects)

    return DHFResult(
        Z=Z, valence=kappa_label(nv, -1), ionization_cm=ionization_cm, alpha_core_au=alpha_core_au,
        cpp_cutoff_bohr=float(m.rho),
        cpp_cutoffs_kappa_bohr={kappa_label(0, k)[1:]: float(v) for k, v in sorted(m.rho_k.items())},
        nuclear_radius_bohr=m.R_nuc, core_scf_iterations=m.core_iterations,
        alpha0_au=alpha_at(0.0),
        alpha_dynamic_au={f"{w:g}": alpha_at(w / HARTREE_TO_EV) for w in omegas_eV},
        levels=tuple(levels), transitions=tuple(transitions), lifetimes=lifetimes,
        quantum_defects=defects,
    )
