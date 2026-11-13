"""The Compare page: one tabbed page for the cross-element results.

Tabs: D lines (the comparison table), Laser cooling (scattering rate, molasses force, Doppler
temperature), Polarizability (α(λ) of every element with its tune-out wavelengths), C₆ (homo- and
heteronuclear dispersion coefficients, Casimir–Polder), Trends (any quantity against Z) and
Accuracy (computed values against precision measurements). Written by the Compare stage
(``compare_elements``) to ``plots/comparison.html``; ``scattering_rate`` opens it on the laser
cooling tab.

The reference values below are the measurements (or, for C₆ and Fr α(0), the recommended
calculations) the Accuracy tab compares with. Each carries its uncertainty and source.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict

from linak.core.constants import ALPHA0_EXPT
from linak.report_templates import load as load_template
from linak.report_templates import render, write_offline_report

TEMPLATE = 'compare_page.html'
TABS = ('dl', 'lc', 'pol', 'c6', 'tr', 'acc')
TAB_FLAGS = {'dl': 'dlines', 'lc': 'cooling', 'pol': 'polarizability', 'c6': 'c6', 'tr': 'trends', 'acc': 'accuracy'}

# --------------------------------------------------------------------------- reference values
# D-line lifetimes (ns): (value, σ, source)
REF_TAU_NS = {
    ('Li', 'D1'): (27.102, 0.009, 'McAlexander, Abraham & Hulet, PRA 54, R5 (1996) (⁶Li 2p)'),
    ('Li', 'D2'): (27.102, 0.009, 'McAlexander, Abraham & Hulet, PRA 54, R5 (1996) (⁶Li 2p)'),
    ('Na', 'D1'): (16.299, 0.021, 'Steck, Sodium D Line Data'),
    ('Na', 'D2'): (16.2492, 0.0077, 'Steck, Sodium D Line Data'),
    ('K', 'D1'): (26.72, 0.05, 'Tiecke, Properties of Potassium'),
    ('K', 'D2'): (26.37, 0.05, 'Tiecke, Properties of Potassium'),
    ('Rb', 'D1'): (27.679, 0.027, 'Steck, Rubidium 87 D Line Data'),
    ('Rb', 'D2'): (26.2348, 0.0077, 'Steck, Rubidium 87 D Line Data'),
    ('Cs', 'D1'): (34.934, 0.094, 'Rafac et al., PRA 50, R1976 (1994)'),
    ('Cs', 'D2'): (30.462, 0.046, 'Patterson et al., PRA 91, 012506 (2015)'),
    ('Fr', 'D1'): (29.45, 0.11, 'Simsarian et al., PRA 57, 2448 (1998)'),
    ('Fr', 'D2'): (21.02, 0.11, 'Simsarian et al., PRA 57, 2448 (1998)'),
}
# Static ground-state polarizability (a.u.)
REF_ALPHA0 = {
    'Li': (164.2, 1.1, 'Miffre et al., EPJD 38, 353 (2006)'),
    'Na': (162.7, 0.8, 'Ekstrom et al., PRA 51, 3883 (1995)'),
    'K': (290.58, 1.42, 'Holmgren et al., PRA 81, 053607 (2010)'),
    'Rb': (318.79, 1.42, 'Holmgren et al., PRA 81, 053607 (2010)'),
    'Cs': (401.0, 0.6, 'Amini & Gould, PRL 91, 153001 (2003)'),
    'Fr': (317.8, 2.4, 'Derevianko et al., PRL 82, 3589 (1999) (calculation)'),
}
# C6 (a.u.): Derevianko, Porsev & Babb, ADNDT 96, 323 (2010) [arXiv:0902.3929], Table B
_C6_REF_SRC = 'Derevianko, Porsev & Babb, ADNDT 96, 323 (2010), Table B'
REF_C6 = {
    ('Li', 'Li'): (1389, 2), ('Li', 'Na'): (1467, 2), ('Li', 'K'): (2322, 5), ('Li', 'Rb'): (2545, 7),
    ('Li', 'Cs'): (3065, 17), ('Li', 'Fr'): (2682, 23),
    ('Na', 'Na'): (1556, 4), ('Na', 'K'): (2447, 6), ('Na', 'Rb'): (2683, 7), ('Na', 'Cs'): (3227, 18),
    ('Na', 'Fr'): (2842, 24),
    ('K', 'K'): (3897, 15), ('K', 'Rb'): (4274, 13), ('K', 'Cs'): (5159, 30), ('K', 'Fr'): (4500, 39),
    ('Rb', 'Rb'): (4690, 23), ('Rb', 'Cs'): (5663, 34), ('Rb', 'Fr'): (4946, 44),
    ('Cs', 'Cs'): (6846, 74), ('Cs', 'Fr'): (5968, 60),
    ('Fr', 'Fr'): (5256, 89),
}
# D1-D2 tune-out wavelength (nm, vacuum)
REF_TUNE_OUT = {
    'K': (768.9712, 0.0015, 'Holmgren et al., PRA 86, 053607 (2012) (³⁹K)'),
    'Rb': (790.032388, 0.000032, 'Leonard et al., PRA 92, 052501 (2015) (⁸⁷Rb)'),
}

ORDER = ('H', 'Li', 'Na', 'K', 'Rb', 'Cs', 'Fr')


def _ref_c6(a, b):
    key = (a, b) if (a, b) in REF_C6 else (b, a)
    return REF_C6.get(key)


def _r(v, p=6):
    if v is None:
        return None
    v = float(v)
    return float(f'{v:.{p}g}') if math.isfinite(v) else None


def load_polarizabilities(elements, data_dir):
    """{element: polarizability JSON} for the elements whose Polarizability stage has run."""
    out = {}
    for el in elements:
        path = os.path.join(data_dir, f'{el}_polarizability.json')
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding='utf-8') as f:
                d = json.load(f)
        except (OSError, ValueError):
            continue
        if d.get('transitions_gs'):
            out[el] = d
    return out


def c6_matrix(pol):
    """{(A, B): C6} by Casimir–Polder for every pair (A ≤ B in the input order), with an
    uncertainty from the two homonuclear ones: σ_AB/C6_AB ≈ ½√((σ_A/C6_A)² + (σ_B/C6_B)²)."""
    from linak.pipelines.polarizability import compute_C6_pair
    els = list(pol)
    out = {}
    for i, a in enumerate(els):
        for b in els[i:]:
            c = compute_C6_pair(pol[a]['transitions_gs'], pol[b]['transitions_gs'])
            ra = (pol[a].get('C6_sigma_au') or 0.0) / (pol[a].get('C6_au') or 1.0)
            rb = (pol[b].get('C6_sigma_au') or 0.0) / (pol[b].get('C6_au') or 1.0)
            sig = c * (ra if a == b else 0.5 * math.hypot(ra, rb))
            out[(a, b)] = (c, sig)
    return out


def tune_outs(pol, wl_min=300.0, wl_max=2000.0):
    from linak.pipelines.polarizability import tune_out_wavelengths
    return {el: tune_out_wavelengths(d['transitions_gs'], wl_min, wl_max) for el, d in pol.items()}


def accuracy_rows(rows, pol, c6, tuneouts):
    """[{q, who, calc, sig, ref, rsig, src}] for every computed value that has a reference."""
    out = []
    for row in rows:
        el = row['element'] if isinstance(row, dict) else row.element
        rec = row if isinstance(row, dict) else asdict(row)
        for line in ('D1', 'D2'):
            d = rec.get(line)
            ref = REF_TAU_NS.get((el, line))
            if d and ref and d.get('tau_ns'):
                out.append(dict(q=f'τ {line}', who=el, calc=d['tau_ns'], sig=d.get('sigma_tau_ns') or 0.0,
                                ref=ref[0], rsig=ref[1], src=ref[2], unit='ns', csrc=d.get('source', '')))
    for el, d in pol.items():
        ref = REF_ALPHA0.get(el)
        if ref:
            out.append(dict(q='α(0)', who=el, calc=d['alpha0_au'], sig=d.get('alpha0_sigma_au') or 0.0,
                            ref=ref[0], rsig=ref[1], src=ref[2], unit='a.u.', csrc='Polarizability stage'))
    for (a, b), (c, s) in c6.items():
        ref = _ref_c6(a, b)
        if ref:
            out.append(dict(q='C₆' if a == b else 'C₆ pair', who=a if a == b else f'{a}–{b}', calc=c, sig=s,
                            ref=ref[0], rsig=ref[1], src=_C6_REF_SRC, unit='a.u.', csrc='Casimir–Polder'))
    for el, lst in tuneouts.items():
        ref = REF_TUNE_OUT.get(el)
        d1d2 = [t for t in lst if t[1].endswith('1/2') and t[2].endswith('3/2') and t[1][:-3] == t[2][:-3]]
        if ref and d1d2:
            lam = max(d1d2)[0]
            out.append(dict(q='tune-out', who=el, calc=lam, sig=0.0, ref=ref[0], rsig=ref[1], src=ref[2],
                            unit='nm', csrc='Polarizability stage'))
    for r in out:
        r['calc'], r['sig'] = _r(r['calc'], 9), _r(r['sig'], 3)
    return out


def build_payload(rows, data_dir, T_K, *, tabs=TABS, tab='dl', element=None, line='D2'):
    recs = [r if isinstance(r, dict) else asdict(r) for r in rows]
    recs.sort(key=lambda r: (r.get('Z') or 999))
    els = [r['element'] for r in recs]
    pol = load_polarizabilities(els, data_dir)
    c6 = c6_matrix(pol) if pol else {}
    to = tune_outs(pol) if pol else {}
    polp = {}
    for el, d in pol.items():
        tr = [[_r(t['delta_E_au'], 10), _r(t.get('sign', 1) * t['fosc'], 7)] for t in d['transitions_gs']]
        polp[el] = dict(a0=_r(d['alpha0_au']), a0s=_r(d.get('alpha0_sigma_au')), C6=_r(d.get('C6_au')),
                        C6s=_r(d.get('C6_sigma_au')), gs=d.get('gs_label', ''), tr=tr,
                        to=[[_r(t[0], 10), t[1], t[2]] for t in to.get(el, [])])
    c6p = [[a, b, _r(c, 7), _r(s, 3)] + (list(_ref_c6(a, b)) if _ref_c6(a, b) else [None, None])
           for (a, b), (c, s) in c6.items()]
    refs = dict(tau={f'{k[0]} {k[1]}': list(v) for k, v in REF_TAU_NS.items()},
                a0={k: list(v) for k, v in REF_ALPHA0.items()},
                a0lit={k: v for k, v in ALPHA0_EXPT.items() if k in els},
                c6src=_C6_REF_SRC)
    el = element if element in els else (els[0] if els else None)
    return dict(rows=recs, T=T_K, pol=polp, c6=c6p, acc=accuracy_rows(recs, pol, c6, to), refs=refs,
                opts=dict(tabs=[t for t in TABS if t in tabs], tab=tab if tab in tabs else tabs[0],
                          el=el, line=line))


def payload_js(payload):
    text = json.dumps(payload, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    return text.replace('</', '<\\/').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')


def write_page(payload, out_dir):
    html = render(TEMPLATE, TITLE='Compare elements', PHYS_JS=load_template('compare_physics.js'),
                  DATA_JS=payload_js(payload))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'comparison.html')
    write_offline_report(path, html)
    return path
