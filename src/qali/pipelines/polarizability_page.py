"""The polarizability page: polarizability, magic wavelengths and optical tweezer on one tabbed page.

A render-only stage. It reads the polarizability data (``<species>_polarizability.json``) and the lifetimes
data, and writes ``plots/<species>/<species>_response.html``. The page computes alpha(omega) in the browser
from the stored transition list, so the units, wavelength range, trap intensity and cursor wavelength can all
be changed there. No new physics runs here; the tweezer curves come from the same ``tweezer_core`` code the
Optical tweezer stage uses. The Polarizability and Optical tweezer stages keep writing their own pages.

Usage: qali --pipeline polarizability_page Na [options]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys

from qali.core.data_inputs import resolve_data_file
from qali.core.errors import MissingDataError
from qali.core.species import resolve_species
from qali.core.workspace import Workspace
from qali.report_templates import render, write_offline_report

TEMPLATE = 'plotinteractive_response.html'
DEFAULTS = dict(wl_min=300.0, wl_max=1800.0, cursor_wl=1064.0, intensity=10.0, waist=1.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='qali --pipeline polarizability_page',
        description='Polarizability, magic wavelengths and optical tweezer on one tabbed page')
    parser.add_argument('element', nargs='?', default=None, help='Species id, for example Na')
    parser.add_argument('--data-dir', default=str(Workspace.current().data_json_dir))
    parser.add_argument('--output-dir', default=str(Workspace.current().plots_dir),
                        help='Plots folder; the page goes in <output-dir>/<species>/')
    parser.add_argument('--polarizability-file', default=None, metavar='FILE',
                        help='Polarizability data to read instead of <species>_polarizability.json '
                             '(a path, or a name under the data folder).')
    parser.add_argument('--lifetimes-file', default=None, metavar='FILE',
                        help='Lifetimes data to read instead of <species>_lifetimes.json '
                             '(a path, or a name under the data folder). Used for the tweezer tab.')
    parser.add_argument('--no-polarizability', action='store_true', help='Leave out the Polarizability tab.')
    parser.add_argument('--no-magic', action='store_true', help='Leave out the Magic wavelengths tab.')
    parser.add_argument('--no-tweezer', action='store_true',
                        help='Leave out the Optical tweezer tab (it is also left out without lifetimes data).')
    parser.add_argument('--units', choices=['au', 'hz', 'uk'], default='au',
                        help='Units shown when the page opens (switchable on the page).')
    parser.add_argument('--scale', choices=['asinh', 'lin'], default='asinh',
                        help='Vertical scale when the page opens (switchable on the page).')
    parser.add_argument('--wl-min', type=float, default=DEFAULTS['wl_min'], help='Shortest wavelength shown (nm).')
    parser.add_argument('--wl-max', type=float, default=DEFAULTS['wl_max'], help='Longest wavelength shown (nm).')
    parser.add_argument('--cursor-wl', type=float, default=DEFAULTS['cursor_wl'],
                        help='Wavelength the cursor starts at (nm).')
    parser.add_argument('--intensity', type=float, default=DEFAULTS['intensity'],
                        help='Trap intensity (kW/cm^2) for depths and tweezer rates.')
    parser.add_argument('--waist', type=float, default=DEFAULTS['waist'], help='Beam waist (um).')
    return parser


def _r(x, digits=6):
    """Round a float to *digits* significant figures; None for a non-finite value."""
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return None
    return float(f'{x:.{digits}g}')


def _rows(transitions) -> list[list]:
    """Transition list as [dE, f, sign, label, source, wavelength nm, sigma f] rows for the page."""
    rows = []
    for t in transitions or []:
        rows.append([_r(t.get('delta_E_au'), 9), _r(t.get('fosc'), 6), int(t.get('sign', 1)),
                     str(t.get('other_label', '?')), str(t.get('source') or '?'),
                     _r(t.get('wl_nm'), 7), _r(t.get('sigma_f') or 0.0, 5)])
    return rows


def _tweezer(element, data_dir, polarizability_file, lifetimes_file, opts):
    """The tweezer block of the payload, or (None, why) when it cannot be computed."""
    from qali.pipelines import tweezer

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            out = tweezer.run_element(
                element, wl_nm=opts['cursor_wl'], I_kWcm2=opts['intensity'], waist_um=opts['waist'],
                data_dir=data_dir, polarizability_file=polarizability_file, lifetimes_file=lifetimes_file)
    except (MissingDataError, OSError, ValueError) as exc:
        return None, str(exc)
    if out is None:
        return None, 'no transition data'
    _gs, result = out
    sp, pt = result.spectra, result.point
    if not result.n_with_gamma:
        return None, 'no upper-state decay rates in the lifetimes data'
    return dict(
        wl=[_r(x, 6) for x in sp.wl_nm], rsc=[_r(x, 5) for x in sp.R_sc_Hz],
        heat=[_r(x, 5) for x in sp.heating_uK_s], tau=[_r(x, 5) for x in sp.tau_heat_s],
        I=result.I_ref_kWcm2,
        point=dict(
            wl_nm=pt.wl_nm, I_kWcm2=pt.I_kWcm2, mass_amu=_r(pt.mass_amu),
            top_channels=[dict(label=c.label or '?', wl_nm=_r(c.wl_nm, 7), fosc=_r(c.fosc, 5),
                               Gamma_s=_r(c.Gamma_s, 5), R_sc_Hz=_r(c.R_sc_Hz, 5)) for c in pt.top_channels]),
    ), None


def build_payload(element: str, label: str, pol: dict, tw: dict | None, opts: dict) -> dict:
    """Everything the page draws, as a plain dict (deterministic, no timestamps)."""
    tabs = [t for t in opts['tabs'] if t != 'tw' or tw]
    return dict(
        element=label, gs=pol.get('gs_label', '?'), ex=pol.get('excited_label', '?'),
        alpha0=_r(pol.get('alpha0_au'), 8), alpha0e=_r(pol.get('alpha0_expt_au'), 8),
        alpha0s=_r(pol.get('alpha0_sigma_au'), 5), C6=_r(pol.get('C6_au'), 8),
        C6e=_r(pol.get('C6_expt_au'), 8), C6s=_r(pol.get('C6_sigma_au'), 5),
        gt=_rows(pol.get('transitions_gs')), et=_rows(pol.get('transitions_ex')),
        magic=[dict(wl_nm=m['wl_nm'], alpha_au=m['alpha_au'],
                    delta_alpha_slope_au_per_nm=m['delta_alpha_slope_au_per_nm'],
                    nearby_resonance_nm=m['nearby_resonance_nm'],
                    nearby_resonance_label=m['nearby_resonance_label'], robustness=m['robustness'])
               for m in pol.get('magic_wavelengths') or []],
        mass=_r((tw or {}).get('point', {}).get('mass_amu') or pol.get('mass_amu') or 0.0) or 1.0,
        tw=tw, tabs=tabs,
        opts=dict(unit=opts['units'], scale=opts['scale'], wmin=opts['wl_min'], wmax=opts['wl_max'],
                  lam=opts['cursor_wl'], I=opts['intensity'], waist=opts['waist']))


def payload_js(payload: dict) -> str:
    """The payload as a JS literal that is safe inside an inline script element."""
    text = json.dumps(payload, separators=(',', ':'), ensure_ascii=False)
    return text.replace('</', '<\\/').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')


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
    if args.wl_min >= args.wl_max:
        parser.error('--wl-min must be smaller than --wl-max')

    pol_path = resolve_data_file(args.polarizability_file, element, args.data_dir, 'polarizability') \
        or os.path.join(args.data_dir, f'{element}_polarizability.json')
    if not os.path.isfile(pol_path):
        raise MissingDataError(f'{os.path.basename(pol_path)} not found in {args.data_dir}: '
                               f'run the Polarizability stage first')
    with open(pol_path, encoding='utf-8') as fh:
        pol = json.load(fh)
    if not pol.get('transitions_gs'):
        raise MissingDataError(f'{os.path.basename(pol_path)} has no ground-state transitions')

    tabs = [t for t, off in (('pol', args.no_polarizability), ('mag', args.no_magic), ('tw', args.no_tweezer)) if not off]
    if not tabs:
        parser.error('every tab is switched off')
    opts = dict(tabs=tabs, units=args.units, scale=args.scale, wl_min=args.wl_min, wl_max=args.wl_max,
                cursor_wl=args.cursor_wl, intensity=args.intensity, waist=args.waist)
    tw = None
    if 'tw' in tabs:
        tw, why = _tweezer(element, args.data_dir, pol_path,
                           resolve_data_file(args.lifetimes_file, element, args.data_dir, 'lifetimes'), opts)
        if tw is None:
            print(f'  Tweezer tab left out: {why}', file=sys.stderr)

    payload = build_payload(element, sp['label'], pol, tw, opts)
    html = render(TEMPLATE, TITLE=f"{sp['label']} polarizability and tweezer", DATA_JS=payload_js(payload))
    out_dir = os.path.join(args.output_dir, element)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{element}_response.html')
    write_offline_report(path, html)
    print(f' Saved: {path}')
    print(f"   tabs: {', '.join(payload['tabs'])}; {len(payload['gt'])} + {len(payload['et'])} lines, "
          f"{len(payload['magic'])} magic wavelength(s)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
