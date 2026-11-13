"""
The ``feshbach`` pipeline: magnetic Feshbach resonances for alkali atoms.
-----------------------------------------------------------------------------
Scattering length a(B) per entrance channel from published coupled-channel
tables and fits where they exist (⁶Li, Cs) and from the literature resonance
parameters otherwise, plus the van der Waals scales β₆ and ā. The page has five
tabs: a(B) (with 1/(k_F a) for a Fermi gas), thermal cross section and loss
rates, Feshbach-molecule binding energies, the ground-state Zeeman thresholds,
and the resonance table with s_res. The physics is in ``linak.physics.feshbach``;
this module loads its inputs, prints a summary, writes the JSON and the page.
Sign convention (Chin et al.): the zero crossing is at B₀ + Δ; for a_bg·Δ > 0,
a → +∞ just below B₀.

Inputs (in the data folder)
---------------------------
    <element>_feshbach.json        literature B₀, Δ, a_bg, δμ, s_res and the published
                                   a(B) curves of each channel (user-maintained)
    <element>_feshbach_table.json  a coupled-channel a(B) table a curve names (⁶Li)
    <element>_hf_constants.json    ground-state hyperfine constant (Thresholds tab)
    <element>_polarizability.json  C₆ (fallback: the C6_EXPT table)
    nuclear_data.json              isotope mass (fallback: tabulated masses)

Outputs
-------
    data/<element>_feshbach_out.json          computed scales + a(B) samples
    plots/<element>/<element>_feshbach.html   interactive report

Usage
-----
    linak --pipeline feshbach Na
    linak --pipeline feshbach Na --B-max 2500 --kT-uK 1.0
    linak --pipeline feshbach Li --density 1e12 --field-unit mT
    linak --pipeline feshbach --all
"""

from __future__ import annotations

__version__ = '0.2'

import argparse
import json
import math
import os
from dataclasses import dataclass

import numpy as np
from linak.core.fileio import write_json
from linak.core.workspace import Workspace

from linak.core.constants import (
    ALKALI_MASS_AMU as ATOMIC_MASS_AMU,
    ISOTOPE_MASS_AMU,
    C6_EXPT as C6_EXPT_FALLBACK,
    default_isotope_A,
    nuclide_label,
    nuclide_plain,
)
from linak.core.errors import MissingDataError
from linak.report_templates import load as load_template
from linak.report_templates import render, write_offline_report
from linak.physics.feshbach import (
    FeshbachResult,
    Resonance,
    _checked_C6,
    channel_curve_spec,
    channel_kets,
    compute,
    is_boson,
    resonance_strength,
    validate_feshbach_params,
)

TEMPLATE = 'feshbach_page.html'
TABS = ('a', 'cs', 'mo', 'th', 'rs')
TAB_FLAGS = {'a': 'a', 'cs': 'collisions', 'mo': 'molecules', 'th': 'thresholds', 'rs': 'resonances'}

# ============================================================================
# ARGPARSE
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="linak --pipeline feshbach",
        description='Magnetic Feshbach resonances (scattering length vs B)')
    parser.add_argument('element', nargs='?', default=None)
    parser.add_argument('--all', action='store_true',
                        help='Run every element that has *_feshbach.json')
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--no-plot', action='store_true',
                        help='Write the data (JSON) only; skip the interactive plot.')
    parser.add_argument('--output-dir', default=None,
                        help='Plot directory (default: plots/<element> under the output folder)')
    parser.add_argument('--B-max', type=float, default=None,
                        help='Max B field in Gauss (default: auto from resonances)')
    parser.add_argument('--B-points', type=int, default=2000,
                        help='Number of B-field sample points')
    parser.add_argument('--channel', default=None,
                        help="Entrance channel for the combined a(B) curve, e.g. '|1,+1> + |1,+1>' "
                             "(default: the channel with the most s-wave resonances)")
    parser.add_argument('--kT-uK', type=float, default=1.0,
                        help='Gas temperature [µK] for the thermal cross section and rates '
                             '(default: 1 µK)')
    parser.add_argument('--density', type=float, default=None,
                        help='Density [cm⁻³] for k_F, T_F and the collision rates; per spin state '
                             'for a Fermi gas (default: 1e12 for fermions, 1e13 for bosons)')
    parser.add_argument('--field-unit', choices=('G', 'mT'), default='G',
                        help='Field unit the page opens in (switchable on the page)')
    for t in TABS[1:]:
        parser.add_argument(f'--no-tab-{TAB_FLAGS[t]}', action='store_true',
                            help=f'Leave out the {TAB_FLAGS[t]} tab')
    parser.add_argument('--isotope', type=int, default=None,
                        help='Mass number A (default: from feshbach JSON)')
    return parser

# ============================================================================
# DATA LOADING
# ============================================================================

@dataclass(frozen=True)
class FeshbachInputs:
    """Everything the calculation needs for one element, with provenance."""

    element: str
    isotope_A: int | None
    resonances: list[Resonance]
    a_bg_default_a0: float
    C6_au: float
    C6_source: str
    mass_amu: float
    mass_source: str
    note: str
    published: dict | None = None    # channel -> TableCurve | FitCurve
    stale_hint: str | None = None


def get_available_elements(data_dir):
    """Neutrals that have a literature feshbach parameter file (ions excluded)."""
    from linak.core.species import is_ion_id
    out = []
    if not os.path.isdir(data_dir):
        return out
    for name in os.listdir(data_dir):
        if name.endswith('_feshbach.json') and not name.endswith('_feshbach_out.json'):
            el = name.replace('_feshbach.json', '')
            if el and el[0].isupper() and not is_ion_id(el):
                out.append(el)
    return sorted(out)


def load_feshbach_params(element, data_dir):
    path = os.path.join(data_dir, f'{element}_feshbach.json')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'No literature file {path}\n'
            f'Create it with resonances: B0_G, Delta_G, a_bg_a0 '
            f'(see data/Na_feshbach.json).')
    with open(path, encoding='utf-8') as f:
        params = json.load(f)
    validate_feshbach_params(params, path)
    return params


def load_C6_au(element, data_dir):
    """Prefer polarizability pipeline C₆; fall back to experimental table."""
    path = os.path.join(data_dir, f'{element}_polarizability.json')
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
        if 'C6_au' in d:
            return _checked_C6(d['C6_au'], 'polarizability.json'), 'polarizability.json'
        if 'C6_expt_au' in d and d['C6_expt_au']:
            return _checked_C6(d['C6_expt_au'], 'polarizability.json (expt)'), 'polarizability.json (expt)'
    if element in C6_EXPT_FALLBACK:
        return _checked_C6(C6_EXPT_FALLBACK[element], 'C6_EXPT table'), 'C6_EXPT table'
    raise ValueError(f'No C₆ available for {element}')


def load_mass_amu(element, isotope_A, data_dir):
    """Mass of the isotope actually used: explicit isotope_A, else the most abundant natural
    isotope (constants.default_isotope_A). Precise table first, then nuclear_data.json 'mass_amu'
    if it ever carries one, then the elemental fallback (flagged in the source string)."""
    A = int(isotope_A) if isotope_A is not None else default_isotope_A(element)
    if A is not None and (element, A) in ISOTOPE_MASS_AMU:
        return ISOTOPE_MASS_AMU[(element, A)], f'ISOTOPE_MASS_AMU (A={A})'
    path = os.path.join(data_dir, 'nuclear_data.json')
    if A is not None and os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            nd = json.load(f)
        for iso in nd.get('isotopes', []):
            if iso.get('symbol') == element and iso.get('A') == A and 'mass_amu' in iso:
                return float(iso['mass_amu']), f'nuclear_data A={A}'
    if element in ATOMIC_MASS_AMU:
        return ATOMIC_MASS_AMU[element], f'ATOMIC_MASS_AMU fallback (no mass for A={A})'
    raise ValueError(f'No mass for {element}')


def load_inputs(element, data_dir, isotope_A=None):
    """The literature table, C₆ and mass for *element*."""
    params = load_feshbach_params(element, data_dir)
    A = isotope_A or params.get('isotope_A') or default_isotope_A(element)
    json_A = params.get('isotope_A')
    if isotope_A and json_A and int(isotope_A) != int(json_A):
        print(f"  WARNING: {element}_feshbach.json holds a_bg / B0 / Delta for {element}-{json_A}, "
              f"but isotope {isotope_A} was requested. Resonance positions and widths are "
              f"isotope-specific (they change with the scattering lengths), so only the mass and "
              f"C6-derived a-bar change here -- the resonance parameters are NOT those of "
              f"{element}-{isotope_A}.")
    rows = params.get('resonances', [])
    if not rows:
        raise ValueError(f'{element}_feshbach.json has no resonances')
    C6_au, C6_src = load_C6_au(element, data_dir)
    mass_amu, mass_src = load_mass_amu(element, A, data_dir)
    resonances = [Resonance.from_json(row) for row in rows]
    published = load_published_curves(params, resonances, data_dir, element)
    return FeshbachInputs(
        element=element,
        isotope_A=A,
        resonances=resonances,
        a_bg_default_a0=float(params.get('a_bg_default_a0', rows[0].get('a_bg_a0', 100.0))),
        C6_au=C6_au,
        C6_source=C6_src,
        mass_amu=mass_amu,
        mass_source=mass_src,
        note=params.get('note', ''),
        published=published,
        stale_hint=_stale_copy_hint(element, data_dir, params),
    )


def load_published_curves(params, resonances, data_dir, element):
    """{channel: TableCurve | FitCurve} from the "curves" entry of the parameter file. A table file
    that is missing is reported and that channel falls back to the resonance formula."""
    out = {}
    tables = {}
    for ch, spec in (params.get('curves') or {}).items():
        rows = None
        if spec.get('kind') == 'table':
            name = spec.get('file') or f'{element}_feshbach_table.json'
            if name not in tables:
                path = os.path.join(data_dir, os.path.basename(name))
                if not os.path.isfile(path):
                    print(f"  WARNING: {name} (the a(B) table of {ch}) is missing from {data_dir}; "
                          f"that channel uses the resonance formula instead.")
                    tables[name] = None
                else:
                    with open(path, encoding='utf-8') as f:
                        tables[name] = json.load(f).get('rows')
            rows = tables[name]
            if rows is None:
                continue
        out[ch] = channel_curve_spec(spec, resonances, rows)
    return out


def _stale_copy_hint(element, data_dir, params):
    """A note when the bundled parameter file has published a(B) curves that this copy lacks."""
    if params.get('curves'):
        return None
    try:
        from linak.core.paths import reference_path
        bundled = reference_path('data_json', f'{element}_feshbach.json')
        if os.path.abspath(bundled) == os.path.abspath(os.path.join(data_dir, f'{element}_feshbach.json')):
            return None
        with open(bundled, encoding='utf-8') as f:
            if not json.load(f).get('curves'):
                return None
    except (OSError, ValueError):
        return None
    return (f"  Note: the bundled {element}_feshbach.json now has published a(B) curves (tables or fits) that "
            f"your copy in {data_dir} lacks; delete your copy to get them (it is re-seeded on the next start).")


# ============================================================================
# PAGE
# ============================================================================

def _r(v, p=6):
    """Finite float rounded to *p* significant digits, else None (JSON has no NaN)."""
    v = float(v)
    if not math.isfinite(v):
        return None
    return float(f'{v:.{p}g}')


def _runs(flags):
    """[[i0, i1], ...] index runs where *flags* is True."""
    out, start = [], None
    for k, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = k
        elif not f and start is not None:
            out.append([start, k - 1])
            start = None
    return out


def resonance_rows(inputs: FeshbachInputs, result: FeshbachResult):
    """The resonance table: literature values plus the zero crossing and s_res (from δμ, else published)."""
    rows = []
    for r in inputs.resonances:
        a_bg = r.background(inputs.a_bg_default_a0)
        if r.dmu_muB is not None and a_bg:
            s_res, src = resonance_strength(a_bg, r.Delta_G, r.dmu_muB, result.a_bar_a0, result.mass_amu), 'δμ'
        elif r.s_res is not None:
            s_res, src = r.s_res, 'published'
        else:
            s_res, src = None, None
        rows.append(dict(label=r.label, ch=r.channel, pw=r.partial_wave, B0=r.B0_G, D=r.Delta_G,
                         abg=_r(a_bg), dmu=r.dmu_muB, sres=None if s_res is None else _r(s_res, 4),
                         sresSrc=src, sw=r.is_s_wave, ref=r.ref))
    return rows


def _ket_index(ket, zs, high_field_rank):
    """Level index of a channel state: '|n>' is the n-th lowest level at high field (the ⁶Li labels
    |1>, |2>, |3>), '|F,mF>' matches the zero-field label."""
    body = ket.strip('|> ').replace(' ', '')
    if ',' not in body:
        n = int(body)
        return high_field_rank[n - 1] if 1 <= n <= len(high_field_rank) else None
    F, mF = (float(x) for x in body.split(','))
    for i, (Fl, ml) in enumerate(zs.labels):
        if abs(Fl - F) < 1e-9 and abs(ml - mF) < 1e-9:
            return i
    return None


def thresholds(inputs: FeshbachInputs, result: FeshbachResult, data_dir, B_max, n=400):
    """Ground-state Zeeman levels and the channel states for the Thresholds tab, or None if the
    isotope's ground hyperfine constant is not available."""
    from linak.physics.hyperfine_core import ZeemanSystem, lande_gJ
    from linak.physics.hyperfine_core import MU_B_MHz_G, MU_N_MHz_G
    from linak.pipelines.hyperfine import get_isotope, load_atomic_gI, load_hf_constants, load_nuclear_index
    el, A = inputs.element, inputs.isotope_A
    try:
        nuclear = load_nuclear_index(os.path.join(data_dir, 'nuclear_data.json'))
    except (OSError, ValueError):
        return None
    iso = get_isotope(nuclear, el, A)
    if iso is None:
        return None
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        hf, _src = load_hf_constants(el, data_dir, isotope_A=A, nuclear=nuclear)
    if not hf:
        return None
    ground = next(iter(hf))            # the file lists the ground state first
    c = hf[ground]
    if not ground.endswith('1/2') or not c.A_MHz:
        return None
    gJ = lande_gJ(0, 0.5, 0.5, gJ_exp=c.gJ_exp, mass_u=float(A) if A else None)
    gI_prime = load_atomic_gI(el, data_dir, A)
    if gI_prime is not None:
        gI, gI_src = -gI_prime / (MU_N_MHz_G / MU_B_MHz_G), 'atomic, shielded'
    else:
        gI, gI_src = float(iso.get('g_I') or 0.0), 'bare nucleus'
    zs = ZeemanSystem(float(iso['I']), 0.5, c.A_MHz, c.B_MHz, gJ, gI, c.C_kHz)
    B = np.linspace(0.0, B_max, n)
    E = zs.energies(B)                                   # (n, levels) MHz
    E_hi = zs.solve(max(B_max, 1000.0))
    rank = [int(i) for i in np.argsort(E_hi)]
    chs = {}
    for ch in result.channels:
        kets = channel_kets(ch.channel)
        if len(kets) == 2:
            ij = [_ket_index(k, zs, rank) for k in kets]
            if None not in ij:
                chs[ch.channel] = ij
    return dict(B=[_r(b) for b in B], E=[[_r(e, 9) for e in E[:, i]] for i in range(zs.dim)],
                lab=[list(l) for l in zs.labels], hf=[list(h) for h in zs.high_field], chs=chs,
                I=float(iso['I']), A=c.A_MHz, gJ=round(gJ, 8), gI=gI, gIsrc=gI_src)


def build_payload(inputs: FeshbachInputs, result: FeshbachResult, args, data_dir):
    el, A = inputs.element, inputs.isotope_A
    fermion = result.is_boson is False
    density = args.density if args.density is not None else (1e12 if fermion else 1e13)
    B = result.B_G
    chans = []
    for c in result.channels:
        chans.append(dict(name=c.channel, ident=bool(len(channel_kets(c.channel)) == 2
                                                    and channel_kets(c.channel)[0] == channel_kets(c.channel)[1]),
                          src=c.source, ref=c.ref, abg=_r(c.a_bg_a0),
                          a=[_r(v, 7) for v in np.asarray(c.a_a0, dtype=float)],
                          br=_runs(c.bridged) if c.bridged is not None else [],
                          poles=[_r(p, 9) for p in c.poles_G], zeros=[_r(z, 9) for z in c.zeros_G]))
    default = next(i for i, c in enumerate(result.channels) if c.channel == result.default_channel)
    tabs = [t for t in TABS if t == 'a' or not getattr(args, f'no_tab_{TAB_FLAGS[t]}', False)]
    thr = thresholds(inputs, result, data_dir, float(B[-1])) if 'th' in tabs else None
    a_scale = max([abs(c.a_bg_a0) for c in result.channels if math.isfinite(c.a_bg_a0)] + [100.0])
    return dict(
        element=el, A=A, label=nuclide_label(el, A), plain=nuclide_plain(el, A), boson=result.is_boson,
        mass=result.mass_amu, massSrc=inputs.mass_source, C6=result.C6_au, C6src=inputs.C6_source,
        beta6=result.beta6_a0, abar=result.a_bar_a0, B=[_r(b, 9) for b in B], ch=chans, **{'def': default},
        res=resonance_rows(inputs, result), thr=thr, note=inputs.note,
        opts=dict(tabs=tabs, tab='a', unit=args.field_unit, T=args.kT_uK, n=density,
                  Bmax=float(B[-1]), yclip=float(_r(min(max(4 * a_scale, 500.0), 2e4), 2))),
    )


def payload_js(payload: dict) -> str:
    text = json.dumps(payload, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    return text.replace('</', '<\\/').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')


def write_page(element, payload, out_dir):
    html = render(TEMPLATE, TITLE=f"{payload['plain']} Feshbach", PHYS_JS=load_template('feshbach_physics.js'),
                  DATA_JS=payload_js(payload))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{element}_feshbach.html')
    write_offline_report(path, html)
    print(f'  Page -> {path}')
    return path

# ============================================================================
# JSON OUTPUT
# ============================================================================

def write_output_json(inputs: FeshbachInputs, result: FeshbachResult, data_dir):
    el = inputs.element
    path = os.path.join(data_dir, f'{el}_feshbach_out.json')

    # Downsample B-grid for JSON size
    step = max(1, len(result.B_G) // 400)
    payload = {
        'element': el,
        'isotope_A': inputs.isotope_A,
        'version': __version__,
        'mass_amu': result.mass_amu,
        'mass_source': inputs.mass_source,
        'C6_au': round(result.C6_au, 4),
        'C6_source': inputs.C6_source,
        'beta6_a0': round(result.beta6_a0, 4),
        'a_bar_a0': round(result.a_bar_a0, 4),
        'a_bg_default_a0': result.a_bg_default_a0,
        'kT_uK': result.kT_uK,
        'k_a0inv': result.k_a0inv,
        'resonances': [
            {
                'label': c.resonance.label,
                'channel': c.resonance.channel,
                'partial_wave': c.resonance.partial_wave,
                'B0_G': c.resonance.B0_G,
                'Delta_G': c.resonance.Delta_G,
                'a_bg_a0': c.a_bg_a0,
                'ref': c.resonance.ref,
                'in_s_wave_a': c.a_a0 is not None,
                'dmu_muB': row['dmu'],
                's_res': row['sres'],
                's_res_source': row['sresSrc'],
            }
            for c, row in zip(result.curves, resonance_rows(inputs, result))
        ],
        'default_channel': result.default_channel,
        'is_boson': result.is_boson,
        'channels': [
            {
                'channel': c.channel,
                'labels': c.labels,
                'a_bg_a0': c.a_bg_a0,
                'identical_bosons': c.identical_bosons,
                'sigma_prefactor': '8*pi*a^2' if c.identical_bosons else '4*pi*a^2',
                'source': c.source,
                'ref': c.ref,
                'poles_G': [round(p, 4) for p in c.poles_G],
                'zero_crossings_G': [round(z, 4) for z in c.zeros_G],
                'a_a0': [None if (math.isnan(v) or math.isinf(v)) else round(float(v), 4)
                         for v in np.asarray(c.a_a0, dtype=float)[::step]],
            }
            for c in result.channels
        ],
        'B_G': result.B_G[::step].tolist(),
        'a_combined_a0': [
            (None if (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
             else round(float(v), 4))
            for v in result.a_combined_a0[::step]
        ],
        'note': inputs.note,
    }
    write_json(path, payload)
    print(f'  JSON -> {path}')
    return path

# ============================================================================
# MAIN
# ============================================================================

def run_one(element, args):
    print('=' * 60)
    print(f'Feshbach - {element}')
    print('=' * 60)
    inputs = load_inputs(element, args.data_dir, isotope_A=args.isotope)
    result = compute(
        inputs.resonances, inputs.a_bg_default_a0, inputs.C6_au, inputs.mass_amu,
        kT_uK=args.kT_uK, B_max=args.B_max, B_points=args.B_points,
        boson=is_boson(element, inputs.isotope_A), channel=args.channel,
        published=inputs.published, refine=True,
    )
    if inputs.stale_hint:
        print(inputs.stale_hint)
    if len(result.channels) > 1:
        print(f"  NOTE: {len(result.channels)} entrance channels in {element}_feshbach.json; the "
              f"combined curve is channel {result.default_channel!r} (use --channel to pick another).")
    print(f"  C6   = {result.C6_au:.2f} au  ({inputs.C6_source})")
    print(f"  beta6= {result.beta6_a0:.3f} a0")
    print(f"  a_bar= {result.a_bar_a0:.3f} a0")
    print(f"  a_bg = {result.a_bg_default_a0:.3f} a0")
    print(f"  mass = {result.mass_amu:.6f} amu  ({inputs.mass_source})")
    print(f"  resonances: {len(inputs.resonances)}")
    for r in inputs.resonances:
        print(f"    - {r.label:20s}  B0={r.B0_G:7.1f} G  "
              f"Delta={r.Delta_G:6.3f} G  [{r.channel}]")
    for c in result.channels:
        how = {'table': 'coupled-channel table', 'fit': 'published fits',
               'resonances': 'isolated-resonance formula'}[c.source]
        zs = ', '.join(f'{z:.2f}' for z in c.zeros_G) or 'none'
        print(f"  a(B) {c.channel}: {how}; zero crossings at {zs} G")

    out_dir = args.output_dir or os.path.join(Workspace.current().plots_dir, element)
    if not args.no_plot:
        write_page(element, build_payload(inputs, result, args, args.data_dir), out_dir)
    write_output_json(inputs, result, args.data_dir)
    return result


def main(argv=None) -> int:
    from linak.core.species import require_neutral_species

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.data_dir is None:
        args.data_dir = str(Workspace.current().data_json_dir)
    if args.all:
        els = get_available_elements(args.data_dir)
        if not els:
            raise MissingDataError(
                f"No *_feshbach.json files found in {args.data_dir}")
        failed = []
        for el in els:
            try:
                run_one(el, args)
            except (MissingDataError, KeyError, ValueError, OSError) as exc:
                print(f"  ⚠  {el}: {type(exc).__name__}: {exc}")
                failed.append(el)
        if failed:
            print(f"  Failed: {', '.join(failed)}")
        return 1 if failed else 0

    if not args.element:
        parser.error(
            "missing ELEMENT (or pass --all); available: "
            f"{', '.join(get_available_elements(args.data_dir)) or '(none)'}")

    el = require_neutral_species(args.element, "feshbach.py")["species_id"]
    run_one(el, args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
