"""Build the bundled ion-core dynamic polarisability table, resources/core/core_polarizability.json.

For each alkali the closed core (M+) is solved in Dirac-Hartree-Fock with a finite nucleus, and
its dipole response in the random-phase approximation (``dhf_rpa.CoreRPA``) at imaginary
frequency:

    α_c(iξ) = (1/3) Σ_{b n} [ ph d_bn x_nb(iξ) / (Δ − iξ) + d_bn y_bn(iξ) / (Δ + iξ) ],  Δ = e_n − e_b

with x, y the RPA-dressed core-excitation amplitudes (bare amplitudes give the DHF sum over
states (2/3) Σ |d|² Δ/(Δ² + ξ²)). The curve is then written as a positive pseudo-spectrum,
α_c(iξ) = Σ_k f_k / (Ω_k² + ξ²), by non-negative least squares on a log-spaced grid of Ω (relative
fit error < 1e-5), so the Polarizability stage can add the core as lines like any other.

Checks printed for each element: static α_c against relativistic RPA (Johnson, Kolb & Huang;
Li+ 0.1894, Na+ 0.9457, Cs+ 15.81 a.u.), the Thomas-Reiche-Kuhn limit ξ² α_c(iξ) → N_core (RPA
restores it; DHF alone overshoots), the imaginary part of α_c(iξ) (zero up to round-off), and the
pseudo-spectrum at real ω against the RPA solved there.

Run once (minutes per element, Fr ~15 min):
    python tools/make_core_polarizability.py            # all six alkalis
    python tools/make_core_polarizability.py Cs Fr      # some
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

from linak.core.constants import default_isotope_A
from linak.physics.atomic_structure.dhf_cp import _Model, one_valence_configuration
from linak.physics.atomic_structure.dhf_rpa import CoreRPA
from linak.physics.atomic_structure.dhf_sigma import MBPT, _j2, _phase

Z_OF = {'Li': 3, 'Na': 11, 'K': 19, 'Rb': 37, 'Cs': 55, 'Fr': 87}
LMAX = {'Li': 3, 'Na': 3, 'K': 3, 'Rb': 3, 'Cs': 4, 'Fr': 5}        # core l + 1 is enough; one spare
import os
OUT = Path(os.environ.get('LINAK_CORE_OUT') or
           Path(__file__).resolve().parents[1] / 'src' / 'linak' / 'resources' / 'core' / 'core_polarizability.json')
XI = np.concatenate([[0.0], np.logspace(-2, 3, 61)])
OMEGA_GRID = np.logspace(np.log10(0.2), np.log10(5000.0), 60)


def core_response(el):
    Z = Z_OF[el]
    core, nv = one_valence_configuration(Z)
    m = _Model(Z, core, nv, 0.0, 0.0, default_isotope_A(el))
    m.scf_core()
    m.setup_valence()
    rpa = CoreRPA(MBPT(m, lmax=LMAX[el], n_int=40))
    ph = np.array([_phase((_j2(b.kappa) - _j2(n.kappa)) // 2) for b, n in rpa.pairs])
    D = -rpa.dE

    def alpha(om, dressed=True):
        x, y = rpa._solve(om) if dressed else (rpa.d_nb, rpa.d_bn)
        return (1 / 3) * np.sum(ph * rpa.d_bn * x / (D - om) + rpa.d_bn * y / (D + om))

    return m, rpa, alpha


def pseudo_spectrum(xi, a):
    M = 1.0 / (OMEGA_GRID[None, :] ** 2 + xi[:, None] ** 2) / a[:, None]
    f, _ = nnls(M, np.ones_like(a), maxiter=5000)
    keep = f > 0
    W, f = OMEGA_GRID[keep], f[keep]
    rel = (np.sum(f / (W[None, :] ** 2 + xi[:, None] ** 2), axis=1) - a) / a
    return W, f, float(np.max(np.abs(rel)))


def main(argv):
    els = argv or list(Z_OF)
    table = json.loads(OUT.read_text(encoding='utf-8')) if OUT.is_file() else {}
    table.setdefault('elements', {})
    for el in els:
        t0 = time.time()
        m, rpa, alpha = core_response(el)
        a = np.array([alpha(1j * x) for x in XI])
        a_bare = np.array([alpha(1j * x, False) for x in XI])
        W, f, err = pseudo_spectrum(XI, a.real)
        real_w = [0.05, 0.1, 0.2]
        real_chk = [(w, float(alpha(w).real), float(np.sum(f / (W ** 2 - w * w)))) for w in real_w if w < 0.8 * W.min()]
        rec = dict(
            Z=Z_OF[el], A=default_isotope_A(el), core_electrons=int(m.n_core_el), lmax=LMAX[el],
            alpha0_rpa_au=round(float(a[0].real), 6), alpha0_dhf_au=round(float(a_bare[0].real), 6),
            trk_at_xi_1000=round(float(XI[-1] ** 2 * a[-1].real), 3),
            max_imag=float(np.max(np.abs(a.imag))), fit_max_rel_error=err,
            omega_au=[round(float(w), 6) for w in W], f=[float(f"{x:.8g}") for x in f],
            real_frequency_check=[[w, round(r, 6), round(p, 6)] for w, r, p in real_chk],
        )
        table['elements'][el] = rec
        print(f"{el}: α_c RPA {rec['alpha0_rpa_au']:.5g} (DHF {rec['alpha0_dhf_au']:.5g}) a.u.; "
              f"ξ²α(iξ) at ξ=1000 {rec['trk_at_xi_1000']} (N_core {rec['core_electrons']}); "
              f"{len(W)} lines, fit error {err:.1e}; Im {rec['max_imag']:.1e}; "
              f"real ω check {rec['real_frequency_check']}; {time.time() - t0:.0f} s", flush=True)
    table['_source'] = ('Ion-core (M+) dipole polarisability at imaginary frequency: Dirac-Hartree-Fock '
                        'core (finite nucleus) with random-phase-approximation dressing (linak dhf_rpa.CoreRPA), '
                        'as a pseudo-spectrum alpha(i xi) = sum_k f_k/(omega_k^2 + xi^2) in atomic units. '
                        'The Polarizability stage rescales f so alpha(0) equals its core-polarisability table '
                        '(the shape comes from RPA). Built by tools/make_core_polarizability.py.')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table['elements'] = dict(sorted(table['elements'].items(), key=lambda kv: kv[1]['Z']))
    OUT.write_text(json.dumps(table, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f"wrote {OUT}")


if __name__ == '__main__':
    main(sys.argv[1:])
