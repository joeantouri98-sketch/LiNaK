"""The hyperfine page: levels, Breit–Rabi, transitions and constants on one tabbed page.

Written by the Hyperfine / Zeeman stage as ``plots/<El>/<El>_hyperfine.html``. Every number is
computed in Python (``physics/hyperfine_core``) and embedded; the page only selects, converts units
and interpolates on the stored field grid.
"""
from __future__ import annotations

import json
import os

import numpy as np

from linak.core.constants import MU_B_MHz_G, nuclide_label
from linak.physics.hyperfine_core import HyperfineResult, HyperfineState
from linak.report_templates import render, write_offline_report

TEMPLATE = 'hyperfine_page.html'
TABS = ('lv', 'br', 'tr', 'co')
COMPOSITION_POINTS = 101


def _r(x, nd=6):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def _arr(a, nd=6):
    return [None if not np.isfinite(v) else round(float(v), nd) for v in np.asarray(a, float)]


def _composition(st: HyperfineState, B_G: np.ndarray):
    """Per level: its m_F block basis [(m_J, m_I)] and |⟨m_J, m_I|level⟩|² on a coarse field grid."""
    zs = st.zeeman
    assert zs is not None
    step = max(1, len(B_G) // (COMPOSITION_POINTS - 1))
    idx = list(range(0, len(B_G), step))
    if idx[-1] != len(B_G) - 1:
        idx.append(len(B_G) - 1)
    rows = [zs.solve(float(B_G[k]), vectors=True)[1] for k in idx]
    out = []
    for n, (_F, mF) in enumerate(zs.labels):
        cols = [c for c, (mj, mi) in enumerate(zs.basis) if abs(mj + mi - mF) < 1e-9]
        out.append(dict(basis=[list(zs.basis[c]) for c in cols],
                        p=[[round(float(V[n, c] ** 2), 5) for c in cols] for V in rows]))
    return [round(float(B_G[k]), 6) for k in idx], out


def _state(st: HyperfineState, B_G: np.ndarray) -> dict:
    assert st.BR_slopes is not None and st.BR_high_field is not None
    cB, comp = _composition(st, B_G)
    return dict(
        label=st.label, n=st.n, l=st.l, J=st.J, gJ=st.gJ, gJ_src=st.gJ_source,
        A=st.A_MHz, B=st.B_MHz, C=st.C_kHz, scaled_from=st.extrapolated_from, missing=st.constants_missing,
        F=[dict(F=lv.F, E=_r(lv.E_MHz, 9), gF=_r(lv.gF, 9)) for lv in st.levels],
        lv=[dict(F=F, mF=mF, mJ=hf[0], mI=hf[1], E=_arr(st.BR_energies[:, n]), dE=_arr(st.BR_slopes[:, n], 7))
            for n, ((F, mF), hf) in enumerate(zip(st.BR_labels, st.BR_high_field))],
        cB=cB, comp=comp)


def build_payload(element: str, result: HyperfineResult, gI_source: str, opts: dict) -> dict:
    nuc = result.nucleus
    B_G = result.B_G
    mc, cy = result.clock, result.cycling
    return dict(
        element=element, nuclide=nuclide_label(element, nuc.A), A=nuc.A, I=nuc.I, mu_I=nuc.mu_I, Q=nuc.Q_barn,
        gI=nuc.gI, gI_src=gI_source, muB=MU_B_MHz_G,
        B=_arr(B_G),
        states=[_state(st, B_G) for st in result.states],
        clock=None if mc is None else dict(state=mc.state, Fu=mc.F_upper, Fl=mc.F_lower, MHz=mc.frequency_MHz,
                                           cm=mc.wavelength_cm),
        cycling=None if cy is None else dict(lower=cy.lower, upper=cy.upper, Fl=cy.F_lower, Fu=cy.F_upper),
        microwave=[dict(l=[m.F_lower, m.mF_lower], u=[m.F_upper, m.mF_upper], kind=m.kind, nu0=_r(m.nu0_MHz, 9),
                        lin=_r(m.linear_Hz_per_G, 6), quad=_r(m.quadratic_Hz_per_G2, 6),
                        nu=_arr(m.nu_MHz - m.nu0_MHz, 9),
                        stat=[[_r(p.B_G, 7), _r(p.nu_MHz, 9), _r(p.curvature_Hz_per_G2, 6)] for p in m.stationary])
                   for m in (result.microwave or [])],
        optical=[dict(lower=ol.lower, upper=ol.upper, B=_arr(ol.B_G),
                      S=[[F, Fp, _r(v, 9)] for (F, Fp), v in sorted(ol.strengths_F.items())],
                      c=[[c.lower, c.upper, c.q, _arr(c.detuning_MHz, 5), _arr(c.strength, 6)] for c in ol.components])
                 for ol in (result.optical or [])],
        opts=opts,
    )


def payload_js(payload: dict) -> str:
    """The payload as a JS literal that is safe inside an inline script element."""
    text = json.dumps(payload, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    return text.replace('</', '<\\/').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')


def write_page(element: str, result: HyperfineResult, out_dir: str, *, gI_source: str = 'bare',
               B_unit: str = 'G', f_unit: str = 'MHz', B_cursor: float | None = None) -> str:
    """Write ``<out_dir>/<element>_hyperfine.html`` and return its path."""
    B_max = float(result.B_G[-1]) if len(result.B_G) else 0.0
    opts = dict(tabs=list(TABS), Bu=B_unit, fu=f_unit,
                Bc=B_cursor if B_cursor is not None else round(min(B_max, max(B_max / 5, 1e-3)), 6))
    payload = build_payload(element, result, gI_source, opts)
    html = render(TEMPLATE, TITLE=f'{payload["nuclide"]} hyperfine structure', DATA_JS=payload_js(payload))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{element}_hyperfine.html')
    write_offline_report(path, html)
    print(f'  ✓ {path}')
    return path
