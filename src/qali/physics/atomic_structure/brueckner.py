"""Brueckner-orbital spectrum and E1 network of a one-valence atom or ion.

Levels: frozen-core DHF plus the second-order correlation potential Sigma(2)(E) (``dhf_sigma``),
solved self-consistently per level, optionally scaled per kappa to one measured level.
E1: reduced dipoles between Brueckner orbitals, dressed by the core's RPA response at the
transition frequency (``dhf_rpa``); the core polarisability is the same RPA, so alpha(0) has no
external input either.

Accuracy (scaled, against precision data): lowest-p lifetimes within 0.5% for Na-Fr, Ca+, Sr+,
Ba+ (Cs 6s-6p1/2 4.4993 vs 4.5057); p -> d branching fractions of Ca+, Ba+ within ~1.5%.
Not included: structural radiation and normalisation (third order, ~0.1-0.2% on resonance
lines, several % on weak lines such as Cs 6s-7p), the core-valence (Pauli) term of alpha,
Breit, QED.

Engine contract: inputs as arguments, results as frozen dataclasses; no files, prints or plots.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass

import numpy as np

from qali.core.constants import AU_TIME_S, HARTREE_TO_EV, HC_EV_NM
from qali.physics.atomic_structure.angular import kappa_label, kappa_lj, reduced_c
from qali.physics.atomic_structure.dhf_cp import HARTREE_TO_CM1, VALENCE_KAPPAS, Lifetime, Transition
from qali.physics.atomic_structure.dhf_rpa import CoreRPA, ValenceE1
from qali.physics.atomic_structure.dhf_sigma import Sigma2Level, brueckner_levels, setup_brueckner

ALPHA_FS = 1 / 137.035999084


@dataclass(frozen=True)
class BruecknerResult:
    Z: int
    charge: int
    valence: str
    scaling: str                          # 'fit' or 'none'
    ionization_cm: float                  # Brueckner ionization energy (= input when s is fitted)
    scale_factors: dict[str, float]
    scale_sources: dict[str, str]
    lmax: int
    nodes_au: tuple[float, ...]
    alpha_core_au: float                  # RPA, static
    alpha_core_dhf_au: float              # same sum without the RPA dressing
    alpha0_au: float
    alpha_dynamic_au: dict[str, float]
    levels: tuple[Sigma2Level, ...]
    transitions: tuple[Transition, ...]
    lifetimes: tuple[Lifetime, ...]
    quantum_defects: dict[str, tuple[tuple[int, float], ...]]


def compute_brueckner(Z: int, *, charge: int = 0, mass_number: float | None = None,
                      ionization_cm: float | None = None,
                      fit_levels_cm: Mapping[tuple[int, int], float] | None = None,
                      scale: bool = True, n_max: int | None = None, n_defects: int = 30, lmax: int = 5,
                      omegas_eV: Sequence[float] = (), e1: bool = True,
                      cache: MutableMapping[str, np.ndarray] | None = None,
                      progress: Callable[[str], None] | None = None) -> BruecknerResult:
    """Brueckner levels (up to n_max; None: n_valence + 6), E1 network, lifetimes, alpha and quantum
    defects (up to n_defects).  Inputs as for ``dhf_sigma.compute_sigma2``; e1=False skips the
    dipoles (levels and defects only)."""
    say = progress or (lambda _msg: None)
    st = setup_brueckner(Z, charge=charge, mass_number=mass_number, ionization_cm=ionization_cm,
                         fit_levels_cm=fit_levels_cm, scale=scale, n_max=n_max, lmax=lmax,
                         cache=cache, progress=progress)
    m, br = st.model, st.br
    say("Brueckner levels ...")
    levels = brueckner_levels(st)
    g_E = -levels[0].binding_eV / HARTREE_TO_EV

    defects: dict[str, tuple[tuple[int, float], ...]] = {}
    for kap in st.nlev:
        k = n_defects - st.n0(kap) + 1
        if k <= 0:
            continue
        e = br.energies(kap, k, st.lam[kap])
        defects[kappa_label(0, kap)[1:]] = tuple(
            (st.n0(kap) + i, float(st.n0(kap) + i - m.z_ion / math.sqrt(-2 * e[i])))
            for i in range(k) if e[i] < 0)

    transitions: list[Transition] = []
    lifetimes: list[Lifetime] = []
    alpha_core = alpha_core_dhf = alpha0 = 0.0
    alpha_dyn: dict[str, float] = {}
    if e1:
        say("RPA core response ...")
        rpa = CoreRPA(br.mb)
        alpha_core, alpha_core_dhf = rpa.alpha_core(True), rpa.alpha_core(False)
        V = ValenceE1(rpa, m.bv, m.c)
        orbs = {}
        for lv in levels:
            i = lv.n - st.n0(lv.kappa)
            E = -lv.binding_eV / HARTREE_TO_EV
            orbs[lv.label] = (lv, E, br.orbital(lv.kappa, i, E, st.lam[lv.kappa]))
        omega_max = max(max(E for _, E, _ in orbs.values()) - g_E, -g_E)
        rpa.prepare(omega_max)

        def dressed(kw, w, Ew, kv, v, Ev, wtag, vtag):
            om = min(abs(Ew - Ev), omega_max)
            rpa.at(om)
            return V.element(kw, w, kv, v, om, wtag, vtag)

        say("E1 network ...")
        A_tot = {lab: 0.0 for lab in orbs}
        report = sorted(orbs.values(), key=lambda t: t[1])
        for a, (lo, E_lo, s_lo) in enumerate(report):
            for up, E_up, s_up in report[a + 1:]:
                if reduced_c(up.kappa, lo.kappa) == 0.0:
                    continue
                _bare, d = dressed(up.kappa, s_up, E_up, lo.kappa, s_lo, E_lo, up.label, lo.label)
                w = E_up - E_lo
                f = 2.0 / 3.0 * w * d * d / (kappa_lj(lo.kappa)[1] + 1)
                A = 4.0 / 3.0 * ALPHA_FS ** 3 * w ** 3 * d * d / (kappa_lj(up.kappa)[1] + 1) / AU_TIME_S
                A_tot[up.label] += A
                rho = s_lo[0] * s_up[0] + s_lo[1] * s_up[1]
                scale_ = m.bv.integrate(np.abs(rho) * m.bv.r)
                canc = abs(m.bv.integrate(rho * m.bv.r)) / scale_ if scale_ > 0 else 1.0
                transitions.append(Transition(lo.label, up.label, float(HC_EV_NM / (w * HARTREE_TO_EV)),
                                              float(abs(d)), float(f), float(A), float(canc)))
        lifetimes = [Lifetime(lab, float(1e9 / A)) for lab, A in A_tot.items() if A > 0]

        # alpha of the ground level: reported levels with their own orbitals, the rest of the
        # (discretised) spectrum from Sigma held at the top node
        say("Polarisability ...")
        g_lv, _, g_orb = orbs[kappa_label(st.nv, -1)]
        terms = []                                          # (omega_s, |D|^2)
        for kap in VALENCE_KAPPAS:
            if reduced_c(kap, -1) == 0.0 or kap not in st.nlev:
                continue
            own = {lv.n - st.n0(kap) for lv, _, _ in orbs.values() if lv.kappa == kap}
            eps, P, Q = br.spectrum(kap, max(st.nodes), st.lam[kap])
            for i in range(len(eps)):
                if i in own:
                    lv, E, s = orbs[kappa_label(st.n0(kap) + i, kap)]
                else:
                    E, s = float(eps[i]), (P[i], Q[i])
                if E - g_E > 50.0:                          # far continuum: negligible
                    break
                _b, d = dressed(kap, s, E, -1, g_orb, g_E, ('a', kap, i), 'ground')
                V.forget(('a', kap, i))
                terms.append((E - g_E, d * d))
        jg = kappa_lj(-1)[1]

        def alpha_v(omega):
            return 2.0 / (3 * (jg + 1)) * sum(d2 * w / (w * w - omega * omega) for w, d2 in terms)
        alpha0 = alpha_v(0.0) + alpha_core
        alpha_dyn = {f"{w:g}": float(alpha_v(w / HARTREE_TO_EV) + alpha_core) for w in omegas_eV}

    return BruecknerResult(
        Z=Z, charge=charge, valence=kappa_label(st.nv, -1), scaling="fit" if st.use_fit else "none",
        ionization_cm=float(-g_E * HARTREE_TO_CM1),
        scale_factors={kappa_label(0, k)[1:]: st.lam[k] for k in st.nlev},
        scale_sources={kappa_label(0, k)[1:]: st.src[k] for k in st.nlev},
        lmax=lmax, nodes_au=tuple(st.nodes),
        alpha_core_au=float(alpha_core), alpha_core_dhf_au=float(alpha_core_dhf), alpha0_au=float(alpha0),
        alpha_dynamic_au=alpha_dyn, levels=tuple(levels), transitions=tuple(transitions),
        lifetimes=tuple(lifetimes), quantum_defects=defects)
