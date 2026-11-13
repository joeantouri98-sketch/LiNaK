"""
spectra.py
----------
The Spectrum page: one tabbed page with the absorption, emission and D-line spectra of a species,
written to ``plots/<species>/<species>_spectrum.html``.

Tabs (each can be left out):
  Absorption   cross-section per atom, optical depth, transmission of a vapour cell or the
               refractive index n − 1, Voigt line shapes (Doppler, natural, instrument)
  Emission     optically thin emission, upper levels populated at an excitation temperature
  D lines      the hyperfine-resolved D1 / D2 lines of a vapour cell with the natural isotope mix,
               Doppler-broadened or saturated-absorption (Doppler-free)
  Curve of growth  equivalent width against column density for any line
  Strip        the spectrum as colour bands (a white continuum through the cell; the emission)
  Lines        the line list (air and vacuum wavelengths, f, A, widths), searchable, CSV

Data: <species>_lifetimes.json (lines with f, A and sources; level lifetimes) and, for the D-line
tab, <El>_hf_constants.json and nuclear_data.json. The physics is in physics/spectrum_core.py and
report_templates/spectrum_physics.js; the page recomputes every spectrum live.

Usage:
  qali --pipeline spectra Na
  qali --pipeline spectra Rb --T 320 --cell-length 7.5
  qali --pipeline spectra Na_c1 --wl-min 25 --wl-max 50
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, deque

from qali.core.constants import (
    ATOMIC_MASS_AMU,
    DEFAULT_ISOTOPE_NO_ABUNDANCE,
    ISOTOPE_MASS_AMU,
    NATURAL_ABUNDANCE_PCT,
    default_isotope_A,
)
from qali.core.data_inputs import resolve_data_file
from qali.core.errors import MissingDataError
from qali.core.species import resolve_species
from qali.core.workspace import Workspace
from qali.physics import spectrum_core as sc
from qali.physics.rydberg import label_to_pretty
from qali.report_templates import load as load_template
from qali.report_templates import render, write_offline_report

TEMPLATE = 'spectrum_page.html'
TABS = ('ab', 'em', 'dl', 'cg', 'st', 'li')
TAB_FLAGS = {'ab': 'absorption', 'em': 'emission', 'dl': 'dlines', 'cg': 'growth', 'st': 'strip', 'li': 'lines'}
UNITS = ('nm', 'air', 'cm', 'eV', 'THz')
MIN_DE_EV = 2e-6        # the lifetimes table rounds ΔE to 1e-6 eV: a smaller gap is rounding, not a line
MIN_ABUNDANCE = 1e-3    # isotopes below 0.1% (40K, 0.0117%) are left out of the D-line tab


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qali --pipeline spectra",
                                     description="Spectrum page: absorption, emission, D lines, curve of growth")
    parser.add_argument("element", nargs="?", default=None, help="Species id: Na or ion stem Na_c1")
    parser.add_argument("--data-dir", default=str(Workspace.current().data_json_dir))
    parser.add_argument("--lifetimes-file", default=None, metavar="FILE",
                        help="Lifetimes data to read instead of <species>_lifetimes.json "
                             "(a path, or a name under the data folder).")
    parser.add_argument("--hf-file", default=None, metavar="FILE",
                        help="Hyperfine constants for the D-line tab instead of <El>_hf_constants.json.")
    parser.add_argument("--transitions-file", default=None, help=argparse.SUPPRESS)   # no longer read
    parser.add_argument("--wl-min", type=float, default=None, help="Initial view: shortest vacuum wavelength (nm)")
    parser.add_argument("--wl-max", type=float, default=None, help="Initial view: longest vacuum wavelength (nm)")
    parser.add_argument("--units", choices=UNITS, default="nm",
                        help="Initial x axis: nm (vacuum), air (nm), cm (cm⁻¹), eV, THz")
    parser.add_argument("--T", type=float, default=None,
                        help="Gas temperature (K); default a typical cell temperature for the element")
    parser.add_argument("--cell-length", type=float, default=5.0, help="Cell length (cm), default 5")
    parser.add_argument("--density", type=float, default=None,
                        help="Atom density (cm⁻³); default the saturated vapour at T")
    parser.add_argument("--t-exc", default="5000",
                        help="Excitation temperature of the emission tab (K, or 'inf')")
    parser.add_argument("--instrument-pm", type=float, default=0.0,
                        help="Instrument resolution FWHM (pm), Gaussian; 0 = none")
    for key, name in TAB_FLAGS.items():
        parser.add_argument(f"--no-tab-{name}", action="store_true", help=f"Leave out the {name} tab")
    return parser


# ----------------------------------------------------------------- data
def _load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _ground_label(lines):
    """The level every other level lies above. Levels that are never an upper level are the
    candidates; metastable ones (Na⁺ 3s ³P°₂, which no E1 line reaches from below) are among them,
    so the one from which all energies come out non-negative is the ground."""
    uppers = {t['upper_label'] for t in lines}
    cands = Counter(t['lower_label'] for t in lines if t['lower_label'] not in uppers)
    if not cands:
        raise MissingDataError("no ground state in the transitions (every level is an upper level)")
    for c, _n in cands.most_common():
        if min(level_energies(lines, c).values()) >= -1e-9:
            return c
    return cands.most_common(1)[0][0]


def level_energies(lines, ground):
    """Level energies (eV above the ground) from the lines' ΔE, walked outwards from the ground."""
    adj: dict[str, list[tuple[str, float]]] = {}
    for t in lines:
        dE = t.get('delta_E_eV')
        if dE is None:
            continue
        adj.setdefault(t['lower_label'], []).append((t['upper_label'], dE))
        adj.setdefault(t['upper_label'], []).append((t['lower_label'], -dE))
    E = {ground: 0.0}
    queue = deque([ground])
    while queue:
        a = queue.popleft()
        for b, d in adj.get(a, ()):
            if b not in E:
                E[b] = E[a] + d
                queue.append(b)
    return E


def build_lines(lt: dict):
    """(levels, lines, ground label, number of lines dropped as ΔE rounding) from a lifetimes JSON."""
    raw = [t for t in lt.get('transitions', []) if (t.get('fosc') or 0) > 0 and t.get('wavelength_nm')]
    dropped = sum(1 for t in raw if abs(t.get('delta_E_eV') or 0.0) <= MIN_DE_EV)
    raw = [t for t in raw if abs(t.get('delta_E_eV') or 0.0) > MIN_DE_EV]
    if not raw:
        raise MissingDataError("the lifetimes file has no lines with an oscillator strength")
    gs = _ground_label(raw)
    E = level_energies(raw, gs)
    tau = {r['state']: r.get('lifetime_s') for r in lt.get('lifetimes', [])}
    g: dict[str, int] = {}
    for t in raw:
        g.setdefault(t['lower_label'], int(t.get('g_lower') or 0))
        g.setdefault(t['upper_label'], int(t.get('g_upper') or 0))
    names = sorted((k for k in E if g.get(k)), key=lambda k: (E[k], k))
    idx = {k: i for i, k in enumerate(names)}
    levels = [[k, label_to_pretty(k), round(E[k], 7), g[k], tau.get(k)] for k in names]
    lines = []
    for t in raw:
        lo, up = idx.get(t['lower_label']), idx.get(t['upper_label'])
        if lo is None or up is None:
            continue
        lines.append([lo, up, round(float(t['wavelength_nm']), 6), float(t['fosc']), float(t.get('A_s') or 0.0),
                      t.get('fosc_source') or 'other', t.get('fosc_acc') or '', t.get('fosc_delta')])
    lines.sort(key=lambda r: r[2])
    return levels, lines, gs, dropped


def _isotope_I(nuclear, element, A, block_text):
    for iso in nuclear.get(element, []):
        if iso['A'] == A:
            return float(iso['I'])
    m = re.search(r'I\s*=\s*(\d+)(?:/(\d+))?', block_text or '')
    if m:
        return int(m.group(1)) / (int(m.group(2)) if m.group(2) else 1)
    return None


def _hf_blocks(path, element):
    """Mass number -> (state -> HfConstants, block text) for every measured block of a constants file."""
    from qali.pipelines.hyperfine import _block_isotope_A, _parse_hf_block
    raw = _load_json(path)
    out = {}
    for blk in [raw] + [v for k, v in raw.items() if k.startswith('_') and isinstance(v, dict)]:
        A = _block_isotope_A(blk, element)
        hf = _parse_hf_block(blk)
        if A is not None and hf:
            out[A] = (hf, str(blk.get('_isotope', '')))
    return out


def build_dlines(element, levels, lines, gs, data_dir, hf_file=None):
    """The D1 and D2 lines with every isotope's hyperfine components, or [] if the data are missing.

    Only measured constants are used (an isotope with no block of its own is left out rather than
    rescaled), with the natural abundances and the measured isotope shifts of physics/spectrum_core.
    """
    from qali.physics.hyperfine_core import parse_state_label
    from qali.pipelines.hyperfine import load_nuclear_index

    m = re.match(r'(\d+)([a-z])', gs)
    if not m or m.group(2) != 's':
        return [], 'the ground state is not an s state'
    n = m.group(1)
    names = [lv[0] for lv in levels]
    gi = names.index(gs)
    path = hf_file or os.path.join(data_dir, f'{element}_hf_constants.json')
    if not os.path.isfile(path):
        return [], f'no {element}_hf_constants.json'
    blocks = _hf_blocks(path, element)
    nuclear_path = os.path.join(data_dir, 'nuclear_data.json')
    if not os.path.isfile(nuclear_path):
        from qali.core.paths import reference_path
        nuclear_path = str(reference_path('data_json', 'nuclear_data.json'))
    nuclear = load_nuclear_index(nuclear_path) if os.path.isfile(nuclear_path) else {}

    abund = {A: p / 100.0 for (el, A), p in NATURAL_ABUNDANCE_PCT.items() if el == element and p / 100 >= MIN_ABUNDANCE}
    if not abund:
        A0 = DEFAULT_ISOTOPE_NO_ABUNDANCE.get(element)
        if A0 is None:
            return [], 'no isotope data'
        abund = {A0: 1.0}
    ref_A = default_isotope_A(element) or max(abund, key=abund.get)

    out = []
    for name, J_up in (('D1', 0.5), ('D2', 1.5)):
        up_label = f'{n}p{"1/2" if J_up == 0.5 else "3/2"}'
        if up_label not in names:
            continue
        ui = names.index(up_label)
        ln = next((r for r in lines if r[0] == gi and r[1] == ui), None)
        if ln is None:
            continue
        isos, missing = [], []
        for A, ab in sorted(abund.items()):
            hf, text = blocks.get(A, (None, ''))
            I = _isotope_I(nuclear, element, A, text)
            g_key = next((k for k in (hf or {}) if (parse_state_label(k) or (0, ''))[:2] == (int(n), 's')), None)
            shift = 0.0 if A == ref_A else sc.ISOTOPE_SHIFT_MHZ.get((element, A), {}).get(name)
            if hf is None or I is None or g_key is None or up_label not in hf or shift is None:
                missing.append(A)
                continue
            g_c, e_c = hf[g_key], hf[up_label]
            comps = sc.hyperfine_components(I, 0.5, J_up, g_c.A_MHz, g_c.B_MHz, e_c.A_MHz, e_c.B_MHz, e_c.C_kHz,
                                            shift, ab)
            isos.append(dict(A=A, I=I, ab=ab, m=ISOTOPE_MASS_AMU.get((element, A), ATOMIC_MASS_AMU.get(element)),
                             shift=shift, ref=A == ref_A,
                             comps=[[c.F, c.Fp, round(c.detuning_MHz, 4), c.weight, round(c.S, 9)] for c in comps]))
        if isos:
            out.append(dict(name=name, lower=gs, upper=up_label, lam=ln[2], f=ln[3], tau=levels[ui][4],
                            iso=isos, missing=missing, shift_src=sc.ISOTOPE_SHIFT_SOURCE.get(element)))
    return out, '' if out else 'no hyperfine constants for the D lines'


def default_range(lines, gs_index):
    """Initial view: the ground-state lines with f above 10⁻⁴ of the strongest, padded by 3%."""
    gl = [r for r in lines if r[0] == gs_index]
    if not gl:
        gl = lines
    fmax = max(r[3] for r in gl)
    wl = [r[2] for r in gl if r[3] >= 1e-4 * fmax]
    lo, hi = min(wl), max(wl)
    return round(lo / 1.03, 3), round(hi * 1.03, 3)


# ----------------------------------------------------------------- page
def build_payload(element, species, lt, data_dir, args, hf_file=None) -> dict:
    levels, lines, gs, dropped = build_lines(lt)
    gi = [lv[0] for lv in levels].index(gs)
    symbol = species.get('symbol', element)
    vp = sc.vapour_record(element) if species['kind'] != 'ion' else None
    dlines, dl_note = build_dlines(element, levels, lines, gs, data_dir, hf_file) if species['kind'] != 'ion' else ([], 'ion')
    T = args.T if args.T is not None else sc.DEFAULT_CELL_T.get(element, 300.0)
    lo, hi = default_range(lines, gi)
    if args.wl_min is not None:
        lo = args.wl_min
    if args.wl_max is not None:
        hi = args.wl_max
    t_exc = math.inf if str(args.t_exc).strip().lower() in ('inf', 'infinity', '∞') else float(args.t_exc)
    tabs = [t for t in TABS if not getattr(args, f'no_tab_{TAB_FLAGS[t]}')]
    if not dlines and 'dl' in tabs:
        tabs.remove('dl')
    N_typed = args.density
    if N_typed is None and vp is None:
        N_typed = 1e10
    return dict(
        element=element, label=species['label'], is_ion=species['kind'] == 'ion',
        mass=ATOMIC_MASS_AMU.get(symbol, ATOMIC_MASS_AMU.get(element, 40.0)), gs=gs, gi=gi,
        IE=lt.get('ionization_energy_eV'),
        levels=levels, lines=lines, dropped=dropped,
        vapour=vp,
        dlines=dlines, dl_note=dl_note,
        opts=dict(tabs=tabs, T=T, L=args.cell_length, N=N_typed, unit=args.units, wl=[lo, hi],
                  texc=None if math.isinf(t_exc) else t_exc, inst=args.instrument_pm),
    )


def payload_js(payload: dict) -> str:
    text = json.dumps(payload, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    return text.replace('</', '<\\/').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')


def write_page(element: str, payload: dict, out_dir: str) -> str:
    html = render(TEMPLATE, TITLE=f"{payload['label']} spectrum", PHYS_JS=load_template('spectrum_physics.js'),
                  DATA_JS=payload_js(payload))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{element}_spectrum.html')
    write_offline_report(path, html)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.element is None:
        avail = []
        if os.path.isdir(args.data_dir):
            avail = sorted(f[:-len('_lifetimes.json')] for f in os.listdir(args.data_dir) if f.endswith('_lifetimes.json'))
        parser.error("element required; available: " + (", ".join(avail) or "none"))
    try:
        species = resolve_species(args.element.strip())
    except ValueError as exc:
        parser.error(str(exc))
    element = species["species_id"]
    if args.transitions_file:
        print("  Note: --transitions-file is no longer read; the lines come from the lifetimes data.")
    lt_path = resolve_data_file(args.lifetimes_file, element, args.data_dir, "lifetimes") \
        or os.path.join(args.data_dir, f"{element}_lifetimes.json")
    if not os.path.isfile(lt_path):
        raise MissingDataError(f"no lifetimes data for {element} ({lt_path}); run the Lifetimes stage first")
    hf_file = resolve_data_file(args.hf_file, species.get('symbol', element), args.data_dir, "hyperfine constants")
    lt = _load_json(lt_path)
    payload = build_payload(element, species, lt, args.data_dir, args, hf_file)
    out_dir = os.path.join(Workspace.current().plots_dir, element)
    path = write_page(element, payload, out_dir)

    n_gs = sum(1 for r in payload['lines'] if r[0] == payload['gi'])
    print(f"  Lines and lifetimes from {os.path.basename(lt_path)}"
          + (f"; hyperfine constants from {os.path.basename(hf_file)}" if hf_file else ""))
    print(f"  {species['label']}: {len(payload['lines'])} lines ({n_gs} from the ground state "
          f"{label_to_pretty(payload['gs'])}), {len(payload['levels'])} levels")
    if payload['dropped']:
        print(f"  Left out {payload['dropped']} line(s) with ΔE ≤ {MIN_DE_EV:g} eV (below the table's rounding).")
    if payload['dlines']:
        for d in payload['dlines']:
            print(f"  {d['name']} {label_to_pretty(d['lower'])} → {label_to_pretty(d['upper'])}: "
                  + ", ".join(f"{element}-{i['A']} ({100 * i['ab']:.2f}%, {len(i['comps'])} components)" for i in d['iso']))
    elif 'dl' not in payload['opts']['tabs'] and payload['dl_note'] and species['kind'] != 'ion':
        print(f"  D-line tab left out: {payload['dl_note']}.")
    vp = payload['vapour']
    T = payload['opts']['T']
    if vp:
        N = sc.vapour_density_cm3(element, T)
        print(f"  Vapour at {T:g} K: {N:.3g} atoms/cm³ (cell {payload['opts']['L']:g} cm)")
    print(f"  ✓ {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
