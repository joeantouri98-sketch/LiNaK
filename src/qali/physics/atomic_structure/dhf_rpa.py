"""Core polarisation of the E1 operator in the random-phase approximation (RPA / TDHF).

The closed core responds to the photon field; the valence electron sees the bare dipole plus
the induced field dV(omega).  In reduced form (checked against an m-scheme sum over magnetic
substates; every core/excited combination gives exactly (-1)^(jb-jn+1)/3):

    <i||dV(w)||j> = sum_{b n} (-1)^(jb-jn+1)/3 [ Z1(i b j n) D_nb / (e_b - e_n + w)
                                               + Z1(i n j b) D_bn / (e_b - e_n - w) ]

with D = d + dV solved self-consistently over core orbitals b and excited states n of the core
ion's DHF basis (the MBPT basis of ``dhf_sigma``), and
    Z_k(abcd) = X_k(abcd) + sum_k' (2k+1) {ja jc k; jb jd k'} X_k'(abdc),
    X_k(abcd) = (-1)^k <a||C^k||c> <b||C^k||d> R_k(abcd)   (Johnson, Atomic Structure Theory).

Checks: static core polarisabilities match relativistic RPA (Johnson, Kolb & Huang) to 4 digits
(Na+ 0.9456 vs 0.9457, Cs+ 15.816 vs 15.81); Cs DHF+RPA 6s-6p1/2 4.9743, 6p3/2 7.0132
(Roberts et al., arXiv:2211.11134: 4.9747, 7.0136).

Engine contract: no files, prints or plots.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from qali.physics.atomic_structure.angular import reduced_c
from qali.physics.atomic_structure.dhf_sigma import MBPT, Orb, _j2, _phase, _rho, sixj2
from qali.physics.atomic_structure.dirac_basis import evaluate, psi_k

_KS = range(9)              # exchange multipoles searched (selection rules pick the few that apply)


def _c(a: int, b: int, k: int = 1) -> float:
    return reduced_c(a, b, k)


class CoreRPA:
    """Self-consistent dressed core-excitation amplitudes D_nb, D_bn at a set of photon frequencies."""

    def __init__(self, mb: MBPT):
        self.mb = mb
        w, r = mb.w, mb.b.r
        pairs = [(b, n) for b in mb.core for kn in mb.kappas if _c(kn, b.kappa) != 0.0
                 for n in mb.excited(kn)]
        self.pairs = pairs
        N = self.N = len(pairs)
        self._yc: dict = {}
        self.d_nb = np.array([_c(n.kappa, b.kappa) * np.sum(w * _rho(n, b) * r) for b, n in pairs])
        self.d_bn = np.array([_c(b.kappa, n.kappa) * np.sum(w * _rho(b, n) * r) for b, n in pairs])
        self.dE = np.array([b.eps - n.eps for b, n in pairs])
        self.cp = np.array([_phase((_j2(b.kappa) - _j2(n.kappa)) // 2 + 1) / 3.0 for b, n in pairs])
        self.Zm = self._interaction(N)
        self.x = self.d_nb.copy()
        self.y = self.d_bn.copy()
        self._nodes: list[float] = []
        self._sols: list[tuple[np.ndarray, np.ndarray]] = []

    # ---- radial helpers
    def _Ys(self, b: Orb, orbs: Sequence[Orb], k: int) -> np.ndarray:
        key = (id(b), id(orbs[0]), len(orbs), k)
        if key not in self._yc:
            self._yc[key] = psi_k(self.mb.b, np.array([_rho(b, o) for o in orbs]), k)
        return self._yc[key]

    def _Ypair(self, a: Orb, b: Orb, k: int) -> np.ndarray:
        key = ('p', id(a), id(b), k)
        if key not in self._yc:
            self._yc[key] = psi_k(self.mb.b, _rho(a, b)[None, :], k)[0]
        return self._yc[key]

    def _interaction(self, N: int):
        """Z1 blocks for the four index patterns of the coupled equations."""
        w = self.mb.w
        A, B, C, D = (np.zeros((N, N)) for _ in range(4))
        groups: dict = {}
        for idx, (b, n) in enumerate(self.pairs):
            groups.setdefault((id(b), n.kappa), []).append(idx)
        glist = [(self.pairs[ix[0]][0], self.pairs[ix[0]][1].kappa, np.array(ix)) for ix in groups.values()]
        for b, kn, I in glist:
            Nn = [self.pairs[i][1] for i in I]
            jb, jn = _j2(b.kappa), _j2(kn)
            PN, QN = np.array([o.P for o in Nn]), np.array([o.Q for o in Nn])
            rho_nb = np.array([_rho(o, b) for o in Nn]) * w
            for b2, kn2, J in glist:
                Nn2 = [self.pairs[i][1] for i in J]
                jb2, jn2 = _j2(b2.kappa), _j2(kn2)
                PN2, QN2 = np.array([o.P for o in Nn2]), np.array([o.Q for o in Nn2])
                Dir = rho_nb @ self._Ys(b2, Nn2, 1).T                       # int rho_nb Y1_{b2 n2}
                a = -_c(kn, b.kappa) * _c(b2.kappa, kn2) * Dir              # Z1(n b2 b n2)
                bb = -_c(kn, b.kappa) * _c(kn2, b2.kappa) * Dir             # Z1(n n2 b b2)
                cc = -_c(b.kappa, kn) * _c(b2.kappa, kn2) * Dir             # Z1(b b2 n n2)
                dd = -_c(b.kappa, kn) * _c(kn2, b2.kappa) * Dir             # Z1(b n2 n b2)
                for k in _KS:
                    s = sixj2(jn, jb, 2, jb2, jn2, 2 * k)                   # X_k(n b2 n2 b)
                    f = _c(kn, kn2, k) * _c(b2.kappa, b.kappa, k)
                    if s and f:
                        Ybb = self._Ypair(b2, b, k) * w
                        a += 3 * s * _phase(k) * f * ((PN * Ybb) @ PN2.T + (QN * Ybb) @ QN2.T)
                    s = sixj2(jn, jb, 2, jn2, jb2, 2 * k)                   # X_k(n n2 b2 b)
                    f = _c(kn, b2.kappa, k) * _c(kn2, b.kappa, k)
                    if s and f:
                        rho_nb2 = np.array([_rho(o, b2) for o in Nn]) * w
                        bb += 3 * s * _phase(k) * f * (rho_nb2 @ self._Ys(b, Nn2, k).T)
                    s = sixj2(jb, jn, 2, jb2, jn2, 2 * k)                   # X_k(b b2 n2 n)
                    f = _c(b.kappa, kn2, k) * _c(b2.kappa, kn, k)
                    if s and f:
                        rho_bn2 = np.array([_rho(b, o) for o in Nn2]) * w
                        cc += 3 * s * _phase(k) * f * (self._Ys(b2, Nn, k) @ rho_bn2.T)
                    s = sixj2(jb, jn, 2, jn2, jb2, 2 * k)                   # X_k(b n2 b2 n)
                    f = _c(b.kappa, b2.kappa, k) * _c(kn2, kn, k)
                    if s and f:
                        Ybb = self._Ypair(b, b2, k) * w
                        dd += 3 * s * _phase(k) * f * ((PN * Ybb) @ PN2.T + (QN * Ybb) @ QN2.T)
                A[np.ix_(I, J)], B[np.ix_(I, J)], C[np.ix_(I, J)], D[np.ix_(I, J)] = a, bb, cc, dd
        return A, B, C, D

    # ---- solving
    def _solve(self, omega: float) -> tuple[np.ndarray, np.ndarray]:
        A, B, C, D = self.Zm
        N = self.N
        gx = self.cp / (self.dE + omega)
        gy = self.cp / (self.dE - omega)
        M = np.block([[np.eye(N) - A * gx[None], -B * gy[None]],
                      [-C * gx[None], np.eye(N) - D * gy[None]]])
        sol = np.linalg.solve(M, np.concatenate([self.d_nb, self.d_bn]))
        return sol[:N], sol[N:]

    def prepare(self, omega_max: float, n_nodes: int = 4) -> None:
        """Solve at n_nodes frequencies on [0, omega_max]; ``at`` then interpolates (the core
        responds smoothly well below its first excitation)."""
        self._nodes = [float(v) for v in np.linspace(0.0, max(omega_max, 1e-6), n_nodes)]
        self._sols = [self._solve(om) for om in self._nodes]

    def at(self, omega: float) -> None:
        """Set the amplitudes x = D_nb, y = D_bn for this frequency."""
        if not self._nodes:
            self.x, self.y = self._solve(omega)
            return
        nodes = self._nodes
        wts = [np.prod([(omega - nodes[j]) / (nodes[i] - nodes[j]) for j in range(len(nodes)) if j != i])
               for i in range(len(nodes))]
        self.x = sum(wt * s[0] for wt, s in zip(wts, self._sols))
        self.y = sum(wt * s[1] for wt, s in zip(wts, self._sols))

    def alpha_core(self, rpa: bool = True) -> float:
        """Static dipole polarisability of the closed core (a.u.): (2/3) sum |<n||d||b>|^2 / (e_n - e_b),
        one bare and one dressed dipole with RPA."""
        ph = np.array([_phase((_j2(b.kappa) - _j2(n.kappa)) // 2) for b, n in self.pairs])
        D = self._solve(0.0)[0] if rpa else self.d_nb
        return float(2 / 3 * np.sum(ph * self.d_bn * D / (-self.dE)))


class ValenceE1:
    """RPA-dressed reduced dipoles <w||D||v> between orbitals on the valence grid of a dhf_cp model."""

    def __init__(self, rpa: CoreRPA, bv, c_light: float):
        self.R, self.bv = rpa, bv
        mb = rpa.mb
        self.on_v = {}
        for o in mb.core + [n for kap in mb.kappas for n in mb.excited(kap)]:
            self.on_v[id(o)] = evaluate(mb.b, o.kappa, o.coef, bv.r, c_light)
        self._yc: dict = {}
        self.groups: dict = {}
        for i, (b, n) in enumerate(rpa.pairs):
            self.groups.setdefault((id(b), n.kappa), (b, n.kappa, []))[2].append(i)
        self._stack = {key: (np.array(idx),
                             np.array([self.on_v[id(rpa.pairs[i][1])][0] for i in idx]),
                             np.array([self.on_v[id(rpa.pairs[i][1])][1] for i in idx]))
                       for key, (_b, _kn, idx) in self.groups.items()}

    def _rho(self, a, b) -> np.ndarray:
        Pa, Qa = a if isinstance(a, tuple) else self.on_v[id(a)]
        Pb, Qb = b if isinstance(b, tuple) else self.on_v[id(b)]
        return Pa * Pb + Qa * Qb

    def _direct(self, omega: float) -> np.ndarray:
        R = self.R
        coef = R.cp * (np.array([_c(b.kappa, n.kappa) for b, n in R.pairs]) * R.x / (R.dE + omega)
                       + np.array([_c(n.kappa, b.kappa) for b, n in R.pairs]) * R.y / (R.dE - omega))
        dens = np.zeros_like(self.bv.r)
        for ci, (b, n) in zip(coef, R.pairs):
            dens += ci * self._rho(b, n)
        return psi_k(self.bv, dens[None, :], 1)[0]

    def _Y(self, tag, orb, core: Orb, k: int) -> np.ndarray:
        key = (tag, id(core), k)
        if key not in self._yc:
            self._yc[key] = psi_k(self.bv, self._rho(orb, core)[None, :], k)[0]
        return self._yc[key]

    def forget(self, tag) -> None:
        """Drop cached potentials of one orbital."""
        for key in [k for k in self._yc if k[0] == tag]:
            del self._yc[key]

    def element(self, kw: int, w: tuple, kv: int, v: tuple, omega: float, wtag, vtag,
                direct: np.ndarray | None = None) -> tuple[float, float]:
        """(bare, dressed) <w||D||v> for orbitals w, v = (P, Q) on the valence grid; omega = E_w - E_v.
        The caller sets the RPA amplitudes for omega (``CoreRPA.at``) first."""
        R, bv = self.R, self.bv
        cwv = _c(kw, kv)
        if cwv == 0.0:
            return 0.0, 0.0
        wq = bv.w
        rho_wv = w[0] * v[0] + w[1] * v[1]
        bare = cwv * float(np.sum(wq * rho_wv * bv.r))
        V1 = self._direct(omega) if direct is None else direct
        tot = bare - cwv * float(np.sum(wq * rho_wv * V1))
        jw, jv = _j2(kw), _j2(kv)
        for key, (b, kn, _idx) in self.groups.items():
            idx, PN, QN = self._stack[key]
            jb, jn = _j2(b.kappa), _j2(kn)
            gx = R.cp[idx] * R.x[idx] / (R.dE[idx] + omega)
            gy = R.cp[idx] * R.y[idx] / (R.dE[idx] - omega)
            for k in _KS:
                s = sixj2(jw, jv, 2, jb, jn, 2 * k)        # 3{jw jv 1; jb jn k} X_k(w b n v)
                f = _c(kw, kn, k) * _c(b.kappa, kv, k)
                if s and f:
                    Ybv = self._Y(vtag, v, b, k) * wq
                    tot += 3 * s * _phase(k) * f * float(gx @ (PN @ (w[0] * Ybv) + QN @ (w[1] * Ybv)))
                s = sixj2(jw, jv, 2, jn, jb, 2 * k)        # 3{jw jv 1; jn jb k} X_k(w n b v)
                f = _c(kw, b.kappa, k) * _c(kn, kv, k)
                if s and f:
                    Ywb = self._Y(wtag, w, b, k) * wq
                    tot += 3 * s * _phase(k) * f * float(gy @ (PN @ (v[0] * Ywb) + QN @ (v[1] * Ywb)))
        return bare, tot
