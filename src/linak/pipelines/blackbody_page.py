"""The blackbody page: BBR shifts, depopulation, photoionization and Rydberg shifts on one tabbed page.

A render-only stage. It reads the Blackbody stage's data (``<species>_blackbody.json``), the polarizability data
(for the shift-versus-temperature curves) and the lifetimes data (for the channels behind each depopulation
rate), and writes ``plots/<species>/<species>_blackbody_page.html``. The page recomputes what depends on
temperature in the browser (static shift, depopulation by thermal photons, direct photoionization and field
ionization) with the same formulas as ``physics/blackbody_core.py``; the blackbody stage keeps writing its own
plots. No new physics runs here beyond the shift-versus-temperature sweep, which uses
``blackbody_core.compute_shifts``.

Usage: linak --pipeline blackbody_page Na [options]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

from linak.core.constants import E_FIELD_REF_T_K, E_FIELD_REF_V_M, EATOMIC_V_M, HARTREE_TO_HZ
from linak.core.data_inputs import resolve_data_file
from linak.core.errors import MissingDataError
from linak.core.species import resolve_species
from linak.core.workspace import Workspace
from linak.physics.blackbody_core import build_lifetime_graph, compute_shifts
from linak.report_templates import render, write_offline_report

TEMPLATE = 'plotinteractive_bbr.html'
SPEED_OF_LIGHT_NM_HZ = 2.99792458e17      # c in nm/s
DEFAULT_FIELD_V_CM = 100.0
TABS = (('sh', '--no-shifts'), ('rt', '--no-depopulation'), ('pi', '--no-photoionization'),
        ('ry', '--no-rydberg'))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='linak --pipeline blackbody_page',
        description='BBR shifts, depopulation, photoionization and Rydberg shifts on one tabbed page')
    parser.add_argument('element', nargs='?', default=None, help='Species id, for example Na')
    parser.add_argument('--data-dir', default=str(Workspace.current().data_json_dir))
    parser.add_argument('--output-dir', default=str(Workspace.current().plots_dir),
                        help='Plots folder; the page goes in <output-dir>/<species>/')
    parser.add_argument('--blackbody-file', default=None, metavar='FILE',
                        help='Blackbody data to read instead of <species>_blackbody.json '
                             '(a path, or a name under the data folder).')
    parser.add_argument('--polarizability-file', default=None, metavar='FILE',
                        help='Polarizability data to read instead of <species>_polarizability.json. '
                             'Used for the shift-versus-temperature curves.')
    parser.add_argument('--lifetimes-file', default=None, metavar='FILE',
                        help='Lifetimes data to read instead of <species>_lifetimes.json. '
                             'Used for the channels behind each depopulation rate.')
    parser.add_argument('--no-shifts', action='store_true', help='Leave out the Shifts vs T tab.')
    parser.add_argument('--no-depopulation', action='store_true', help='Leave out the Depopulation tab.')
    parser.add_argument('--no-photoionization', action='store_true', help='Leave out the Photoionization tab.')
    parser.add_argument('--no-rydberg', action='store_true', help='Leave out the Rydberg shifts tab.')
    parser.add_argument('--units', choices=['hz', 'frac'], default='hz',
                        help='Units for the shifts when the page opens (switchable on the page).')
    parser.add_argument('--temperature', type=float, default=None,
                        help='Temperature the cursor starts at (K). Default: the blackbody run temperature.')
    parser.add_argument('--t-min', type=float, default=10.0, help='Lowest temperature shown (K).')
    parser.add_argument('--t-max', type=float, default=1000.0, help='Highest temperature shown (K).')
    parser.add_argument('--field', type=float, default=None,
                        help='Static field for field ionization (V/cm). Default: the run field, else 100.')
    return parser


def _r(x, digits=6):
    """Round to *digits* significant figures; None for a missing or non-finite value."""
    if x is None:
        return None
    x = float(x)
    return float(f'{x:.{digits}g}') if math.isfinite(x) else None


def _load_json(path):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def _shift_block(bb: dict, pol: dict | None) -> tuple[dict | None, str | None]:
    """Shift-versus-temperature data, or (None, why)."""
    if pol is None:
        return None, 'no polarizability data'
    trans_gs = pol.get('transitions_gs')
    if not trans_gs:
        return None, 'the polarizability data has no ground-state transitions'
    ex_label = bb.get('excited_label')
    trans_ex = pol.get('transitions_ex') if pol.get('excited_label') == ex_label else None
    shifts = compute_shifts(trans_gs, trans_ex or None, float(bb.get('T_K') or 300.0))
    block = dict(
        a0g=_r(shifts.alpha0_gs_au, 9), a0e=_r(shifts.alpha0_ex_au, 9) if trans_ex else None,
        Tdyn=[_r(x) for x in shifts.T_dyn_K], dyn_gs=[_r(x, 9) for x in shifts.dyn_gs_curve_hz],
        dyn_ex=[_r(x, 9) for x in shifts.dyn_ex_curve_hz] if trans_ex else None,
        E_ref=E_FIELD_REF_V_M, T_ref=E_FIELD_REF_T_K, E_au=EATOMIC_V_M, hz_per_ha=HARTREE_TO_HZ)
    line = next((t for t in trans_gs if t.get('other_label') == ex_label and t.get('wl_nm')), None)
    block['nu_hz'] = SPEED_OF_LIGHT_NM_HZ / line['wl_nm'] if line else None
    note = None if trans_ex else (f"no excited-state data for {ex_label}" if ex_label else None)
    return block, note


def _rates_block(bb: dict, life: dict | None) -> tuple[dict | None, list[str], str | None]:
    """Depopulation block with each state's channels, plus the partner label table."""
    rates = bb.get('rates')
    if not rates or not rates.get('states'):
        return None, [], 'the blackbody data has no rates (run the Blackbody stage with rates on)'
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
    from linak.physics.blackbody_core import BETEROV_E, BETEROV_PREF

    cont = bb.get('continuum')
    if not cont or not cont.get('states'):
        return None, 'the blackbody data has no photoionization block (run it with photoionization on)'
    keys = ('Gamma_PI_s', 'Gamma_mix_PI_s', 'Gamma_ion_s', 'Gamma_BBR_s', 'Gamma_rad_s')
    return dict(
        T=cont.get('T_K'), field=cont.get('field_V_cm'), E=BETEROV_E, pref=BETEROV_PREF,
        states=[dict(s=x['state'], n=x['n'], l=x['l'], ns=_r(x['n_star'], 7), A=x['A_L'], ph=_r(x['phase_weight'], 7),
                     pi=_r(x.get(keys[0]), 6), mixpi=_r(x.get(keys[1]), 6), ion=_r(x.get(keys[2]), 6),
                     bbr=_r(x.get(keys[3]), 6), rad=_r(x.get(keys[4]), 6), fi=bool(x.get('field_ionized')))
                for x in cont['states']]), None


def _ry_block(bb: dict) -> tuple[dict | None, str | None]:
    ry = bb.get('rydberg_shifts')
    if not ry or not ry.get('states'):
        return None, 'the blackbody data has no Rydberg shifts (run it with Rydberg shifts on)'
    return dict(
        T=ry.get('T_K'), fw=_r(ry.get('farley_wing_hz'), 8),
        states=[dict(s=x['state'], n=x['n'], l=x['l'], a0=_r(x['alpha0_au'], 6), st=_r(x['shift_stat_hz'], 6),
                     dy=_r(x.get('shift_dyn_hz'), 6), rel=bool(x.get('dyn_reliable', True)),
                     dE=_r(x.get('min_dE_au'), 4), nt=x.get('n_transitions'),
                     sv=x.get('static_valid'), sn=x.get('static_note') or '', cp=_r(x.get('max_coupling'), 3)) for x in ry['states']]), None


def build_payload(label: str, bb: dict, pol: dict | None, life: dict | None, args) -> tuple[dict, dict]:
    """(payload, {tab: why it is missing}). Deterministic: no timestamps."""
    missing: dict[str, str] = {}
    wanted = [t for t, flag in TABS if not getattr(args, flag.lstrip('-').replace('-', '_'))]
    T0 = float(args.temperature if args.temperature is not None else (bb.get('T_K') or 300.0))
    payload: dict = dict(element=label, gs=bb.get('gs_label', '?'), ex=bb.get('excited_label'), Tdyn=None,
                         dyn_ex=None, a0g=None, a0e=None, rates=None, labels=[], pi=None, ry=None)
    tabs = []
    if 'sh' in wanted:
        block, note = _shift_block(bb, pol)
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
    field = args.field if args.field is not None else (((payload['pi'] or {}).get('field')) or DEFAULT_FIELD_V_CM)
    n_all = [s['n'] for s in (payload['rates'] or {}).get('states', [])] + \
            [s['n'] for s in (payload['pi'] or {}).get('states', [])] + [s['n'] for s in (payload['ry'] or {}).get('states', [])]
    payload.update(
        tabs=tabs, nu_hz=payload.get('nu_hz'),
        nrange=[min(n_all), max(n_all)] if n_all else [5, 50],
        opts=dict(unit=args.units, T=T0, tmin=args.t_min, tmax=args.t_max, field=field),
        sub=f"{bb.get('gs_label', '?')} ground" + (f" · {bb['excited_label']} excited" if bb.get('excited_label') else ''))
    return payload, missing


def payload_js(payload: dict) -> str:
    """The payload as a JS literal that is safe inside an inline script element."""
    text = json.dumps(payload, separators=(',', ':'), ensure_ascii=False)
    return text.replace('</', '<\\/').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')


def _optional(spec, element, data_dir, what, default_name):
    """The chosen file, else the default one when it exists, else None."""
    chosen = resolve_data_file(spec, element, data_dir, what)
    if chosen:
        return _load_json(chosen)
    path = os.path.join(data_dir, default_name)
    return _load_json(path) if os.path.isfile(path) else None


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.element is None:
        parser.error('element required, for example Na')
    try:
        sp = resolve_species(args.element.strip())
    except ValueError as exc:
        parser.error(str(exc))
    element = sp['species_id']
    if args.t_min >= args.t_max or args.t_min <= 0:
        parser.error('--t-min must be positive and smaller than --t-max')

    bb = _optional(args.blackbody_file, element, args.data_dir, 'blackbody', f'{element}_blackbody.json')
    if bb is None:
        raise MissingDataError(f'{element}_blackbody.json not found in {args.data_dir}: '
                               f'run the Blackbody stage first')
    pol = _optional(args.polarizability_file, element, args.data_dir, 'polarizability',
                    f'{element}_polarizability.json')
    life = _optional(args.lifetimes_file, element, args.data_dir, 'lifetimes', f'{element}_lifetimes.json')

    payload, missing = build_payload(sp['label'], bb, pol, life, args)
    for tab, why in missing.items():
        print(f'  {tab}: left out or reduced: {why}', file=sys.stderr)
    if not payload['tabs']:
        raise MissingDataError('nothing to draw: ' + '; '.join(missing.values() or ['every tab is switched off']))
    html = render(TEMPLATE, TITLE=f"{sp['label']} blackbody radiation", DATA_JS=payload_js(payload))
    out_dir = os.path.join(args.output_dir, element)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{element}_blackbody_page.html')
    write_offline_report(path, html)
    print(f' Saved: {path}')
    print(f"   tabs: {', '.join(payload['tabs'])}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
