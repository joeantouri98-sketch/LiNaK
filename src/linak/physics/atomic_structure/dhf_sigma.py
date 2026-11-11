"""Second-order correlation potential Sigma(2)(E) and Brueckner orbitals for one-valence atoms.

An energy-dependent alternative to the static core-polarisation potential of ``dhf_cp``
(energies only):

    (h_DHF + lambda_kappa * Sigma(2)(E)) phi = E phi,     E solved self-consistently per level.

Sigma(2)
--------
Valence second-order energy (m-scheme; Johnson, Dzuba, Safronova & Safronova,
arXiv:physics/0310141, eq. 7):

    E2_v = - sum_{b m n} v_mnvb  vt_vbmn / (e_m + e_n - e_v - e_b)
           + sum_{a b m} v_mvab  vt_abmv / (e_m + e_v - e_a - e_b),     vt_ijkl = v_ijkl - v_ijlk

with a, b core and m, n every non-core state of a finite DHF basis (V^(N-1), l <= lmax, 40-bohr
cavity).  Sigma_pq(E) replaces v by p (left) and q (right) and e_v by E.  The first term is
core polarisation (it contains the -alpha_c/2r^4 tail and its non-adiabatic, energy-dependent
corrections); the second is the exchange-type term.  The j-coupled form below agrees with a
brute-force m-scheme sum (``mscheme_second_order_energy``) and, for Cs, reproduces 97-99% of
the Sigma(2) correlation energies of Ginges et al., arXiv:1603.09116.

Sigma is built on a few energies (Chebyshev nodes), interpolated in E, projected onto the
3000-bohr valence basis of ``dhf_cp`` and added to the frozen-core DHF Hamiltonian.

Scaling
-------
Sigma(2) over-correlates the heavy species and under-correlates Na (unscaled errors: s, p
1-3%, the inner (n-1)d of Ca+/Sr+/Ba+ 7-28%).  With measured levels, one factor lambda per
kappa is fitted to the lowest level of that kappa (the s factor to the ionization energy);
every other level is predicted, and the energy dependence of Sigma carries the series.  A
kappa without a measured level takes the mean factor of the nearest fitted l.  Without
measured levels lambda = 1: a pure ab initio (DHF + second-order) spectrum.

Not included: Breit, QED, all-order (ladder / screening) correlation.  E1 matrix elements need
a matching RPA dipole and are not computed here.

Engine contract: inputs as arguments, results as frozen dataclasses; no files, prints or plots.
The Sigma matrices can be kept between runs in a caller-owned mapping (``cache``).
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq

from linak.core.constants import HARTREE_TO_EV
from linak.physics.atomic_structure.angular import kappa_label, kappa_lj, reduced_c, sixj2, threej2
from linak.physics.atomic_structure.dhf_cp import (
    HARTREE_TO_CM1,
    VALENCE_KAPPAS,
    _Model,
    one_valence_configuration,
)
from linak.physics.atomic_structure.dirac_basis import (
    Basis,
    dirac_matrices,
    evaluate,
    nuclear_potential,
    psi_k,
)

SIGMA2_VERSION = 1                  # bump when the Sigma matrices change (invalidates caches)
N_NODES = 7                         # Chebyshev energies Sigma is built on
SCALE_RANGE = (0.2, 1.6)            # allowed lambda


def _j2(kap: int) -> int:
    return kappa_lj(kap)[1]


def _phase(n: int) -> float:
    return -1.0 if n % 2 else 1.0


# ------------------------------------------------------------------ MBPT basis
@dataclass
class Orb:
    kappa: int
    eps: float
    P: np.ndarray
    Q: np.ndarray
    core: bool
    coef: np.ndarray


def _rho(a: Orb, b: Orb) -> np.ndarray:
    return a.P * b.P + a.Q * b.Q


class MBPT:
    """DHF V^(N-1) orbitals of every kappa (l <= lmax) in a cavity, for the second-order sums."""

    def __init__(self, model: _Model, lmax: int = 5, rmax: float = 40.0, n_int: int = 40):
        self.m = model
        self.c = model.c
        self.lmax = lmax
        b = Basis(model.Z, rmax=rmax, n_int=n_int, beta=0.0)
        self.b = b
        r = b.r
        core = {key: evaluate(model.bc, key[1], model.core_vec[key[:2]][0], r, self.c) for key in model.core}
        V0 = nuclear_potential(r, model.Z, model.R_nuc) + model._direct(b, core)
        self.kappas = [k for l in range(lmax + 1) for k in ([-1] if l == 0 else [l, -l - 1])]
        self.orbs: dict[int, list[Orb]] = {}
        for kap in self.kappas:
            comps = b.components(kap, self.c)
            H, S, _ = dirac_matrices(b, kap, V0, self.c)
            H = H - model._exchange(b, kap, comps, core)
            eps, vec = eigh(H, S)
            sel = np.where(eps > -self.c ** 2)[0]
            ncore = model.ncore(kap)
            self.orbs[kap] = [Orb(kap, float(eps[idx]), vec[:, idx] @ comps[0], vec[:, idx] @ comps[1],
                                  i < ncore, vec[:, idx]) for i, idx in enumerate(sel)]
        self.core = [o for kap in self.kappas for o in self.orbs[kap] if o.core]
        self.w = b.w

    def excited(self, kap: int) -> list[Orb]:
        return [o for o in self.orbs.get(kap, []) if not o.core]


# ------------------------------------------------------------------ brute force (checks only)
def mscheme_second_order_energy(mb: MBPT, v: Orb, terms: tuple[int, ...] = (1, 2)) -> float:
    """E2_v from the m-scheme sum (slow; small bases only).  Reference for the j-coupled Sigma."""
    w = mb.w
    ycache: dict = {}

    def R(i, j, k_, l, k):     # R^k(ij;kl) = int rho_ik(1) Y^k_jl(1)
        key = (id(j), id(l), k)
        if key not in ycache:
            ycache[key] = psi_k(mb.b, _rho(j, l)[None, :], k)[0]
        return float(np.sum(w * _rho(i, k_) * ycache[key]))

    def cme(a, ma, k, q, c_, mc):   # <a ma| C^k_q | c mc>
        return (_phase((_j2(a.kappa) - ma) // 2) * threej2(_j2(a.kappa), 2 * k, _j2(c_.kappa), -ma, 2 * q, mc)
                * reduced_c(a.kappa, c_.kappa, k))

    def cv(i, mi, j, mj, k_, mk, l, ml):
        tot = 0.0
        for k in range(2 * mb.lmax + 4):
            ang = sum(_phase(q) * cme(i, mi, k, q, k_, mk) * cme(j, mj, k, -q, l, ml) for q in range(-k, k + 1))
            if ang:
                tot += ang * R(i, j, k_, l, k)
        return tot

    def ms(o):
        return range(-_j2(o.kappa), _j2(o.kappa) + 1, 2)

    exc = [o for kap in mb.kappas for o in mb.excited(kap)]
    mv = _j2(v.kappa)
    e2 = 0.0
    for b in (mb.core if 1 in terms else []):
        for mbb in ms(b):
            for m in exc:
                for mm in ms(m):
                    for n in exc:
                        for mn in ms(n):
                            if mm + mn != mv + mbb:
                                continue
                            a1 = cv(m, mm, n, mn, v, mv, b, mbb)
                            if a1 == 0.0:
                                continue
                            t = cv(v, mv, b, mbb, m, mm, n, mn) - cv(v, mv, b, mbb, n, mn, m, mm)
                            e2 -= a1 * t / (m.eps + n.eps - v.eps - b.eps)
    for a in (mb.core if 2 in terms else []):
        for ma in ms(a):
            for b in mb.core:
                for mbb in ms(b):
                    for m in exc:
                        for mm in ms(m):
                            if mm + mv != ma + mbb:
                                continue
                            a1 = cv(m, mm, v, mv, a, ma, b, mbb)
                            if a1 == 0.0:
                                continue
                            t = cv(a, ma, b, mbb, m, mm, v, mv) - cv(a, ma, b, mbb, v, mv, m, mm)
                            e2 += a1 * t / (m.eps + v.eps - a.eps - b.eps)
    return e2


# ------------------------------------------------------------------ j-coupled Sigma(2)
def sigma2_matrices(mb: MBPT, kv: int, energies, terms: tuple[int, ...] = (1, 2)) -> np.ndarray:
    """Sigma(2)_pq(E) over the excited MBPT states p, q of kappa kv: array [E, p, q]."""
    w = mb.w
    jv = _j2(kv)
    E = np.asarray(energies, float)
    V = mb.excited(kv)
    PV = np.array([o.P for o in V]) * w
    QV = np.array([o.Q for o in V]) * w
    S = np.zeros((len(E), len(V), len(V)))
    kmax = 2 * mb.lmax + 3

    def Y(a: Orb, x: Orb, k: int) -> np.ndarray:
        return psi_k(mb.b, _rho(a, x)[None, :], k)[0]

    def ints(Ys, orbs):
        """[x, p, y]: int rho_{p y} Y_x for a stack of potentials Ys[x] and orbitals y."""
        Py = np.array([o.P for o in orbs])
        Qy = np.array([o.Q for o in orbs])
        return (Ys[:, None, :] * PV[None]) @ Py.T + (Ys[:, None, :] * QV[None]) @ Qy.T

    # ---- term 1 (core polarisation): core b, excited m (kappa_m), n (kappa_n)
    for b in (mb.core if 1 in terms else []):
        kb, jb = b.kappa, _j2(b.kappa)
        for km in mb.kappas:
            M = mb.excited(km)
            jm = _j2(km)
            for kn in mb.kappas:
                N = mb.excited(kn)
                jn = _j2(kn)
                ks = [k for k in range(kmax) if reduced_c(kv, km, k) and reduced_c(kb, kn, k)]
                if not (M and N and ks):
                    continue
                kxs = [k for k in range(kmax) if reduced_c(km, kb, k) and reduced_c(kn, kv, k)]
                Dk = {k: ints(np.array([Y(b, n, k) for n in N]), M) for k in ks}       # [n, p, m]
                Xk = {k: ints(np.array([Y(b, mm, k) for mm in M]), N) for k in kxs}    # [m, q, n]
                den = b.eps - np.array([o.eps for o in M])[:, None] - np.array([o.eps for o in N])[None, :]
                W = 1.0 / (E[:, None, None] + den[None])                                # [e, m, n]
                ph = _phase((jm + jn - jv - jb) // 2)
                for k in ks:
                    left = _phase(k) * reduced_c(kv, km, k) * reduced_c(kb, kn, k) * Dk[k]
                    right = _phase(k) * reduced_c(km, kv, k) * reduced_c(kn, kb, k) * Dk[k]
                    for kx in kxs:
                        sj = sixj2(jm, jv, 2 * k, jn, jb, 2 * kx)
                        if sj:
                            cx = _phase(kx) * reduced_c(km, kb, kx) * reduced_c(kn, kv, kx)
                            right = right + (2 * k + 1) * sj * cx * np.transpose(Xk[kx], (2, 1, 0))
                    pref = ph / ((jv + 1) * (2 * k + 1))
                    S += pref * np.einsum('npm,emn,nqm->epq', left, W, right)

    # ---- term 2 (exchange type): core a, b; excited m
    for a in (mb.core if 2 in terms else []):
        ka, ja = a.kappa, _j2(a.kappa)
        for b in mb.core:
            kb, jb = b.kappa, _j2(b.kappa)
            for km in mb.kappas:
                M = mb.excited(km)
                jm = _j2(km)
                ks = [k for k in range(kmax) if reduced_c(km, ka, k) and reduced_c(kv, kb, k)]
                if not (M and ks):
                    continue
                kxs = [k for k in range(kmax) if reduced_c(ka, kv, k) and reduced_c(kb, km, k)]
                H = {k: np.einsum('xg,pg,g->xp', np.array([Y(a, mm, k) for mm in M]), PV, b.P)
                     + np.einsum('xg,pg,g->xp', np.array([Y(a, mm, k) for mm in M]), QV, b.Q)
                     for k in ks}                                                   # [m, p]: R_k(mp;ab)
                G = {k: np.einsum('xg,pg,g->xp', np.array([Y(b, mm, k) for mm in M]), PV, a.P)
                     + np.einsum('xg,pg,g->xp', np.array([Y(b, mm, k) for mm in M]), QV, a.Q)
                     for k in kxs}                                                  # [m, q]: R_k'(ab;qm)
                den = np.array([o.eps for o in M]) - a.eps - b.eps
                W2 = 1.0 / (E[:, None] + den[None])                                 # [e, m]
                ph = _phase((jm + jv - ja - jb) // 2)
                for k in ks:
                    left = _phase(k) * reduced_c(km, ka, k) * reduced_c(kv, kb, k) * H[k]
                    right = _phase(k) * reduced_c(ka, km, k) * reduced_c(kb, kv, k) * H[k]
                    for kx in kxs:
                        sj = sixj2(ja, jm, 2 * k, jb, jv, 2 * kx)
                        if sj:
                            cx = _phase(kx) * reduced_c(ka, kv, kx) * reduced_c(kb, km, kx)
                            right = right + (2 * k + 1) * sj * cx * G[kx]
                    pref = ph / ((jv + 1) * (2 * k + 1))
                    S += pref * np.einsum('mp,em,mq->epq', left, W2, right)
    return 0.5 * (S + np.transpose(S, (0, 2, 1)))


# ------------------------------------------------------------------ Brueckner solver
class Brueckner:
    """(h_DHF + lambda Sigma(2)(E)) in the valence basis of a pure-DHF ``_Model``."""

    def __init__(self, model: _Model, lmax: int = 5, rmax: float = 40.0, n_int: int = 40):
        self.m = model
        self.lmax, self.rmax, self.n_int = lmax, rmax, n_int
        self._mb: MBPT | None = None
        self.sig: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}   # kappa -> (nodes, S, O)

    @property
    def mb(self) -> MBPT:
        if self._mb is None:
            self._mb = MBPT(self.m, lmax=self.lmax, rmax=self.rmax, n_int=self.n_int)
        return self._mb

    def build(self, kap: int, nodes) -> None:
        """Sigma at the nodes and the projector O[i, p] = <valence basis i | MBPT state p>."""
        mb, m = self.mb, self.m
        S = sigma2_matrices(mb, kap, nodes)
        Pv, Qv, _ = m.comps[kap]
        rows = [evaluate(mb.b, kap, o.coef, m.bv.r, m.c) for o in mb.excited(kap)]
        Pp = np.array([pq[0] for pq in rows])
        Qp = np.array([pq[1] for pq in rows])
        O = (Pv * m.bv.w) @ Pp.T + (Qv * m.bv.w) @ Qp.T
        self.sig[kap] = (np.asarray(nodes, float), S, O)

    def sigma_at(self, kap: int, E: float) -> np.ndarray:
        nodes, S, O = self.sig[kap]
        n = len(nodes)
        # outside the nodes (high Rydberg levels, the discretised continuum) Sigma is held at the
        # nearest node: it varies on the scale of core excitation energies, not of E - threshold
        E = float(np.clip(E, nodes.min(), nodes.max()))
        wts = np.array([np.prod([(E - nodes[j]) / (nodes[i] - nodes[j]) for j in range(n) if j != i])
                        for i in range(n)])
        return O @ np.tensordot(wts, S, axes=1) @ O.T

    def dhf_energies(self, kap: int, nlev: int) -> np.ndarray:
        m = self.m
        eps = eigh(m.F0[kap], m.S[kap], eigvals_only=True)
        k0 = int(np.sum(eps < -m.c ** 2)) + m.ncore(kap)
        return eps[k0:k0 + nlev]

    def energies(self, kap: int, nlev: int, lam: float = 1.0, iters: int = 12, tol: float = 1e-10) -> np.ndarray:
        """Self-consistent Brueckner energies (a.u.) of the lowest nlev valence states of kappa."""
        m = self.m
        H0, Sv = m.F0[kap], m.S[kap]
        eps0 = eigh(H0, Sv, eigvals_only=True)
        k0 = int(np.sum(eps0 < -m.c ** 2)) + m.ncore(kap)
        out = np.empty(nlev)
        for i in range(nlev):
            E = eps0[k0 + i]
            for _ in range(iters):
                E_new = eigh(H0 + lam * self.sigma_at(kap, E), Sv, eigvals_only=True)[k0 + i]
                done = abs(E_new - E) < tol
                E = E_new
                if done:
                    break
            out[i] = E
        return out

    def orbital(self, kap: int, i: int, E: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
        """(P, Q) on the valence grid of valence state i of kappa, from h_DHF + lam Sigma(E)."""
        m = self.m
        H = m.F0[kap] + lam * self.sigma_at(kap, E)
        eps, vec = eigh(H, m.S[kap])
        k0 = int(np.sum(eps < -m.c ** 2)) + m.ncore(kap)
        v = vec[:, k0 + i]
        P, Q, _ = m.comps[kap]
        return v @ P, v @ Q

    def spectrum(self, kap: int, E: float, lam: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Every valence-channel eigenstate of h_DHF + lam Sigma(E) (energies, P, Q): the
        quasi-complete set for sums over states."""
        m = self.m
        eps, vec = eigh(m.F0[kap] + lam * self.sigma_at(kap, E), m.S[kap])
        k0 = int(np.sum(eps < -m.c ** 2)) + m.ncore(kap)
        P, Q, _ = m.comps[kap]
        return eps[k0:], vec[:, k0:].T @ P, vec[:, k0:].T @ Q


# ------------------------------------------------------------------ results
@dataclass(frozen=True)
class Sigma2Level:
    label: str
    n: int
    l: int
    j: float
    kappa: int
    energy_cm: float          # above the ground level
    binding_eV: float
    n_star: float
    quantum_defect: float
    dhf_energy_cm: float      # frozen-core DHF, same reference (DHF ground)
    fitted: bool


@dataclass(frozen=True)
class Sigma2Result:
    Z: int
    charge: int
    valence: str
    scaling: str                          # 'fit' (lambda per kappa) or 'none' (lambda = 1)
    ionization_cm: float                  # Brueckner ionization energy (= input when s is fitted)
    scale_factors: dict[str, float]       # {'s1/2': lambda, ...}
    scale_sources: dict[str, str]         # {'s1/2': 'fit 6s1/2' | 'mean of d' | 'unscaled'}
    lmax: int
    nodes_au: tuple[float, ...]
    levels: tuple[Sigma2Level, ...]


def _nodes(lo: float, hi: float) -> list[float]:
    x = np.cos(np.linspace(0.0, np.pi, N_NODES))
    return [float(v) for v in x * (lo - hi) / 2 + (lo + hi) / 2]


def _borrow(kap: int, fitted: Mapping[int, float]) -> tuple[float, str]:
    """Mean lambda of the fitted kappas with the nearest l (ties: the higher l)."""
    l = kappa_lj(kap)[0]
    ls = sorted({kappa_lj(k)[0] for k in fitted}, key=lambda x: (abs(x - l), -x))
    if not ls:
        return 1.0, "unscaled"
    pick = [v for k, v in fitted.items() if kappa_lj(k)[0] == ls[0]]
    return float(np.mean(pick)), f"mean of {'spdfgh'[ls[0]]}"


@dataclass
class BruecknerSetup:
    """A solved Brueckner model: the DHF model, Sigma at the nodes and the scale factors."""
    model: _Model
    br: Brueckner
    nv: int
    nmax: int
    nlev: dict[int, int]                 # kappa -> number of reported levels
    lam: dict[int, float]
    src: dict[int, str]
    fitted_keys: set[tuple[int, int]]
    use_fit: bool
    nodes: list[float]
    ground_dhf: float

    def n0(self, kap: int) -> int:
        return self.model.ncore(kap) + kappa_lj(kap)[0] + 1


def setup_brueckner(Z: int, *, charge: int = 0, mass_number: float | None = None,
                    ionization_cm: float | None = None,
                    fit_levels_cm: Mapping[tuple[int, int], float] | None = None,
                    scale: bool = True, n_extra: int = 6, lmax: int = 5, n_int: int = 40,
                    kappas: tuple[int, ...] = VALENCE_KAPPAS,
                    cache: MutableMapping[str, np.ndarray] | None = None,
                    progress: Callable[[str], None] | None = None) -> BruecknerSetup:
    """Core SCF, Sigma(2) at the energy nodes (or from ``cache``) and the scale factors.
    Arguments as for ``compute_sigma2``."""
    say = progress or (lambda _msg: None)
    core_nl, nv = one_valence_configuration(Z, charge)
    m = _Model(Z, core_nl, nv, 0.0, 0.0, mass_number)     # pure frozen-core DHF
    m.scf_core()
    m.setup_valence()
    br = Brueckner(m, lmax=lmax, n_int=n_int)
    nmax = nv + n_extra

    def n0(kap):
        return m.ncore(kap) + kappa_lj(kap)[0] + 1

    # a valence kappa needs MBPT states of its own symmetry (l <= lmax)
    nlev = {kap: nmax - n0(kap) + 1 for kap in kappas if nmax >= n0(kap) and kappa_lj(kap)[0] <= lmax}
    ground_dhf = float(br.dhf_energies(-1, 1)[0])
    use_fit = scale and ionization_cm is not None
    lo = -1.15 * ionization_cm / HARTREE_TO_CM1 if (use_fit and ionization_cm is not None) else 1.3 * ground_dhf
    e_high = max(float(br.dhf_energies(kap, k)[-1]) for kap, k in nlev.items())
    nodes = _nodes(lo, max(-0.004, 0.8 * e_high))

    meta = np.array([SIGMA2_VERSION, Z, charge, mass_number or 0.0, lmax, n_int, *nodes])
    cached = cache is not None and 'meta' in cache and np.array_equal(np.asarray(cache['meta']), meta)
    for kap in nlev:
        if cached and cache is not None and f"{kap}:S" in cache:
            br.sig[kap] = (np.asarray(nodes), np.asarray(cache[f"{kap}:S"]), np.asarray(cache[f"{kap}:O"]))
            continue
        say(f"Sigma(2) {kappa_label(0, kap)[1:]} ...")
        br.build(kap, nodes)
        if cache is not None:
            if not cached:
                cache.clear()
                cache['meta'] = meta
                cached = True
            cache[f"{kap}:S"], cache[f"{kap}:O"] = br.sig[kap][1], br.sig[kap][2]

    lam: dict[int, float] = {}
    src: dict[int, str] = {}
    fitted_keys: set[tuple[int, int]] = set()
    if use_fit:
        assert ionization_cm is not None
        ip_au = ionization_cm / HARTREE_TO_CM1
        targets = {-1: (nv, -ip_au)}
        for (n, kap), e_cm in sorted((fit_levels_cm or {}).items()):
            if kap in nlev and kap != -1 and n >= n0(kap) and (kap not in targets or n < targets[kap][0]):
                targets[kap] = (n, e_cm / HARTREE_TO_CM1 - ip_au)
        for kap, (n, e_target) in targets.items():
            i = n - n0(kap)
            if i >= nlev[kap]:
                continue

            def miss(x, kap=kap, i=i, e_target=e_target):
                return br.energies(kap, i + 1, x)[i] - e_target
            try:
                lam[kap] = float(brentq(miss, *SCALE_RANGE, xtol=1e-7))
            except ValueError:
                continue                       # no root in range: borrow below
            src[kap] = f"fit {kappa_label(n, kap)}"
            fitted_keys.add((n, kap))
        fitted = dict(lam)
        for kap in nlev:
            if kap not in lam:
                lam[kap], src[kap] = _borrow(kap, fitted)
    else:
        lam = {kap: 1.0 for kap in nlev}
        src = {kap: "unscaled" for kap in nlev}
    return BruecknerSetup(m, br, nv, nmax, nlev, lam, src, fitted_keys, use_fit, nodes, ground_dhf)


def compute_sigma2(Z: int, *, charge: int = 0, mass_number: float | None = None,
                   ionization_cm: float | None = None,
                   fit_levels_cm: Mapping[tuple[int, int], float] | None = None,
                   scale: bool = True, n_extra: int = 6, lmax: int = 5, n_int: int = 40,
                   kappas: tuple[int, ...] = VALENCE_KAPPAS,
                   cache: MutableMapping[str, np.ndarray] | None = None,
                   progress: Callable[[str], None] | None = None) -> Sigma2Result:
    """Brueckner (DHF + Sigma(2)) levels of a one-valence atom or ion.

    ionization_cm   measured ionization energy: fixes lambda for s (scale=True)
    fit_levels_cm   {(n, kappa): energy above ground, cm-1}: the lowest entry of each kappa fixes
                    lambda for that kappa (needs ionization_cm, which sets the ground level)
    scale           False: lambda = 1 everywhere, measured inputs ignored (pure ab initio)
    n_extra         levels up to n_valence + n_extra
    lmax            highest l of the MBPT orbitals; valence kappas above it are skipped
    cache           mapping kept by the caller between runs; Sigma matrices are read from and
                    written to it (keys 'meta', '<kappa>:S', '<kappa>:O')
    progress        optional callback for one-line status messages
    """
    say = progress or (lambda _msg: None)
    st = setup_brueckner(Z, charge=charge, mass_number=mass_number, ionization_cm=ionization_cm,
                         fit_levels_cm=fit_levels_cm, scale=scale, n_extra=n_extra, lmax=lmax,
                         n_int=n_int, kappas=kappas, cache=cache, progress=progress)
    say("Brueckner levels ...")
    levels = brueckner_levels(st)
    g = -levels[0].binding_eV / HARTREE_TO_EV
    return Sigma2Result(
        Z=Z, charge=charge, valence=kappa_label(st.nv, -1), scaling="fit" if st.use_fit else "none",
        ionization_cm=float(-g * HARTREE_TO_CM1),
        scale_factors={kappa_label(0, k)[1:]: st.lam[k] for k in st.nlev},
        scale_sources={kappa_label(0, k)[1:]: st.src[k] for k in st.nlev},
        lmax=lmax, nodes_au=tuple(st.nodes), levels=tuple(levels))


def brueckner_levels(st: BruecknerSetup, nmax: int | None = None) -> list[Sigma2Level]:
    """Bound Brueckner levels up to n = nmax (default: the setup's), lowest first."""
    m, br = st.model, st.br
    nmax = st.nmax if nmax is None else nmax
    raw = []
    for kap in st.nlev:
        k = nmax - st.n0(kap) + 1
        if k <= 0:
            continue
        e = br.energies(kap, k, st.lam[kap])
        e0 = br.dhf_energies(kap, k)
        raw += [(st.n0(kap) + i, kap, float(e[i]), float(e0[i])) for i in range(k) if e[i] < 0]
    g = next(e for n, kap, e, _ in raw if kap == -1 and n == st.nv)
    levels = []
    for n, kap, e, e0 in sorted(raw, key=lambda t: t[2]):
        ns = m.z_ion / math.sqrt(-2 * e)
        l, j2 = kappa_lj(kap)
        levels.append(Sigma2Level(kappa_label(n, kap), n, l, j2 / 2, kap, float((e - g) * HARTREE_TO_CM1),
                                  float(-e * HARTREE_TO_EV), float(ns), float(n - ns),
                                  float((e0 - st.ground_dhf) * HARTREE_TO_CM1),
                                  (n, kap) in st.fitted_keys))
    return levels
