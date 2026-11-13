"""The blackbody report: BBR shifts, depopulation, photoionization and Rydberg shifts on one tabbed page.

The Blackbody stage calls this after it has saved its data, so one run writes both
``data/<species>_blackbody.json`` and ``plots/<species>/<species>_blackbody.html``. The page recomputes what depends
on temperature in the browser (static shift, depopulation by thermal photons, direct photoionization and field
ionization) with the same formulas as ``physics/blackbody_core.py``. Tabs are chosen on the command line
(``--no-tab-shifts`` and so on); a tab whose data the run skipped is left out with a message.
"""

from __future__ import annotations

import json
import math
import os
import sys

from qali.core.constants import E_FIELD_REF_T_K, E_FIELD_REF_V_M, EATOMIC_V_M, HARTREE_TO_HZ
from qali.core.errors import MissingDataError
from qali.physics.blackbody_core import (
    BETEROV_E,
    BETEROV_PREF,
    build_lifetime_graph,
    compute_shifts,
)
from qali.report_templates import render, write_offline_report

TEMPLATE = 'plotinteractive_bbr.html'
SPEED_OF_LIGHT_NM_HZ = 2.99792458e17      # c in nm/s
DEFAULT_FIELD_V_CM = 100.0
# (tab id, argparse destination that switches it off)
TABS = (('sh', 'no_tab_shifts'), ('rt', 'no_tab_depopulation'), ('pi', 'no_tab_photoionization'),
        ('ry', 'no_tab_rydberg'))


def _r(x, digits=6):
    """Round to *digits* significant figures; None for a missing or non-finite value."""
    if x is None:
        return None
    x = float(x)
    return float(f'{x:.{digits}g}') if math.isfinite(x) else None


def _shift_block(bb: dict, trans_gs, trans_ex, shifts=None) -> tuple[dict | None, str | None]:
    """Shift-versus-temperature data, or (None, why).  *shifts* is the run's own result when it has one."""
    if not trans_gs:
        return None, 'no ground-state transitions'
    ex_label = bb.get('excited_label')
    if shifts is None:
        shifts = compute_shifts(trans_gs, trans_ex or None, float(bb.get('T_K') or 300.0))
    has_ex = bool(trans_ex) and shifts.dyn_ex_curve_hz is not None
    block = dict(
        a0g=_r(shifts.alpha0_gs_au, 9), a0e=_r(shifts.alpha0_ex_au, 9) if has_ex else None,
        Tdyn=[_r(x) for x in shifts.T_dyn_K], dyn_gs=[_r(x, 9) for x in shifts.dyn_gs_curve_hz],
        dyn_ex=[_r(x, 9) for x in shifts.dyn_ex_curve_hz] if has_ex else None,
        E_ref=E_FIELD_REF_V_M, T_ref=E_FIELD_REF_T_K, E_au=EATOMIC_V_M, hz_per_ha=HARTREE_TO_HZ)
    line = next((t for t in trans_gs if t.get('other_label') == ex_label and t.get('wl_nm')), None)
    block['nu_hz'] = SPEED_OF_LIGHT_NM_HZ / line['wl_nm'] if line else None
    note = None if has_ex else (f"no excited-state data for {ex_label}" if ex_label else None)
    return block, note


def _rates_block(bb: dict, life: dict | None) -> tuple[dict | None, list[str], str | None]:
    """Depopulation block with each state's channels, plus the partner label table."""
    rates = bb.get('rates')
    if not rates or not rates.get('states'):
        return None, [], 'the run skipped the depopulation rates'
    if life is None:
        return None, [], 'no lifetimes data for the channels'
    graph = build_lifetime_graph(life)
    labels: list[str] = []
    index: dict[str, int] = {}
    states = []
    for row in rates['states']:
        label = row['state']
        if label not in graph.states:
            continue
        info = graph.states[label]
        edges = []
        for partner, A, g_u, g_l, wl, kind in graph.edges_from.get(label, []):
            if partner not in index:
                index[partner] = len(labels)
                labels.append(partner)
            weight = A * (g_u / max(g_l, 1e-30)) if kind == 'abs' else A
            edges.append([_r(wl, 7), _r(weight, 6), index[partner], 1 if kind == 'abs' else 0])
        states.append(dict(s=label, n=info['n'], l=info['l'], Grad=_r(row.get('Gamma_rad_s'), 6), e=edges))
    if not states:
        return None, [], 'no state in the blackbody data matches the lifetimes data'
    return dict(T=rates.get('T_K'), n_min=rates.get('n_min'), states=states), labels, None


def _pi_block(bb: dict) -> tuple[dict | None, str | None]:
    cont = bb.get('continuum')
    if not cont or not cont.get('states'):
        return None, 'the run skipped photoionization'
    return dict(
        T=cont.get('T_K'), field=cont.get('field_V_cm'), E=BETEROV_E, pref=BETEROV_PREF,
        states=[dict(s=x['state'], n=x['n'], l=x['l'], ns=_r(x['n_star'], 7), A=x['A_L'],
                     ph=_r(x['phase_weight'], 7), pi=_r(x.get('Gamma_PI_s'), 6),
                     mixpi=_r(x.get('Gamma_mix_PI_s'), 6), ion=_r(x.get('Gamma_ion_s'), 6),
                     bbr=_r(x.get('Gamma_BBR_s'), 6), rad=_r(x.get('Gamma_rad_s'), 6),
                     fi=bool(x.get('field_ionized')))
                for x in cont['states']]), None


def _ry_block(bb: dict) -> tuple[dict | None, str | None]:
    ry = bb.get('rydberg_shifts')
    if not ry or not ry.get('states'):
        return None, 'the run skipped the Rydberg shifts'
    return dict(
        T=ry.get('T_K'), fw=_r(ry.get('farley_wing_hz'), 8),
        states=[dict(s=x['state'], n=x['n'], l=x['l'], a0=_r(x['alpha0_au'], 6), st=_r(x['shift_stat_hz'], 6),
                     dy=_r(x.get('shift_dyn_hz'), 6), rel=bool(x.get('dyn_reliable', True)),
                     dE=_r(x.get('min_dE_au'), 4), nt=x.get('n_transitions'),
                     sv=x.get('static_valid'), sn=x.get('static_note') or '', cp=_r(x.get('max_coupling'), 3))
                for x in ry['states']]), None


def build_payload(label: str, bb: dict, trans_gs, trans_ex, life: dict | None, opts: dict,
                  shifts=None) -> tuple[dict, dict]:
    """(payload, {tab: why it is missing or reduced}).  Deterministic: no timestamps.

    *bb* is the blackbody JSON document; *trans_gs* / *trans_ex* the polarizability networks behind the shift
    curves; *life* the lifetimes data (channels behind each rate); *opts* the page options
    (``tabs``, ``units``, ``T``, ``t_min``, ``t_max``, ``field``).
    """
    missing: dict[str, str] = {}
    wanted = list(opts['tabs'])
    payload: dict = dict(element=label, gs=bb.get('gs_label', '?'), ex=bb.get('excited_label'), Tdyn=None,
                         dyn_ex=None, a0g=None, a0e=None, rates=None, labels=[], pi=None, ry=None)
    tabs = []
    if 'sh' in wanted:
        block, note = _shift_block(bb, trans_gs, trans_ex, shifts)
        if block is None:
            missing['sh'] = note
        else:
            payload.update(block)
            tabs.append('sh')
            if note:
                missing['ex'] = note
    if 'rt' in wanted:
        block, labels, why = _rates_block(bb, life)
        if block is None:
            missing['rt'] = why
        else:
            payload['rates'], payload['labels'] = block, labels
            tabs.append('rt')
    if 'pi' in wanted:
        block, why = _pi_block(bb)
        if block is None:
            missing['pi'] = why
        else:
            payload['pi'] = block
            tabs.append('pi')
    if 'ry' in wanted:
        block, why = _ry_block(bb)
        if block is None:
            missing['ry'] = why
        else:
            payload['ry'] = block
            tabs.append('ry')
    field = opts.get('field')
    if field is None:
        field = ((payload['pi'] or {}).get('field')) or DEFAULT_FIELD_V_CM
    n_all = [s['n'] for block in (payload['rates'], payload['pi'], payload['ry']) if block
             for s in block['states']]
    payload.update(
        tabs=tabs, nu_hz=payload.get('nu_hz'),
        nrange=[min(n_all), max(n_all)] if n_all else [5, 50],
        opts=dict(unit=opts['units'], T=float(opts.get('T') or bb.get('T_K') or 300.0),
                  tmin=opts['t_min'], tmax=opts['t_max'], field=field),
        sub=f"{bb.get('gs_label', '?')} ground" + (f" · {bb['excited_label']} excited" if bb.get('excited_label') else ''))
    return payload, missing


def payload_js(payload: dict) -> str:
    """The payload as a JS literal that is safe inside an inline script element."""
    text = json.dumps(payload, separators=(',', ':'), ensure_ascii=False)
    return text.replace('</', '<\\/').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')


def options_from_args(args) -> dict:
    """Page options from the Blackbody stage's parsed arguments."""
    return dict(
        tabs=[t for t, dest in TABS if not getattr(args, dest, False)],
        units=args.plot_units, T=args.T, t_min=args.plot_t_min, t_max=args.plot_t_max, field=args.field)


def write_report(label: str, element: str, bb: dict, trans_gs, trans_ex, life: dict | None, opts: dict,
                 out_dir: str, shifts=None) -> str | None:
    """Write ``<out_dir>/<element>_blackbody.html`` and return its path (None when no tab is wanted)."""
    if not opts['tabs']:
        print('  Plot: every tab is switched off, nothing drawn.')
        return None
    payload, missing = build_payload(label, bb, trans_gs, trans_ex, life, opts, shifts)
    for tab, why in missing.items():
        print(f'  Plot: {tab} tab left out or reduced: {why}', file=sys.stderr)
    if not payload['tabs']:
        raise MissingDataError('nothing to draw: ' + '; '.join(missing.values()))
    html = render(TEMPLATE, TITLE=f'{label} blackbody radiation', DATA_JS=payload_js(payload))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{element}_blackbody.html')
    write_offline_report(path, html)
    print(f'  Wrote {path}   tabs: {", ".join(payload["tabs"])}')
    return path
