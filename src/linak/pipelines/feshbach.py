"""
The ``feshbach`` pipeline: magnetic Feshbach resonances for alkali atoms.
-----------------------------------------------------------------------------
Scattering length a(B) and elastic cross section σ(B) from literature
resonance parameters, plus the van der Waals scales β₆ and ā. The physics is
in ``linak.physics.feshbach``; this module loads its inputs, prints a summary,
writes the JSON and draws the report. Sign convention (Chin et al.): the zero
crossing is at B₀ + Δ; for a_bg·Δ > 0, a → +∞ just below B₀.

Inputs (in the data folder)
---------------------------
    <element>_feshbach.json        literature B₀, Δ, a_bg (user-maintained)
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
    linak --pipeline feshbach --all
"""

from __future__ import annotations

__version__ = '0.1'

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
from linak.report_templates import (
    fullscreen_css,
    inject_head,
    inject_title,
    write_offline_report,
)
from linak.physics.feshbach import (
    FeshbachResult,
    Resonance,
    _checked_C6,
    compute,
    is_boson,
    validate_feshbach_params,
)

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
                        help='Collision energy as temperature [µK] for σ(B) '
                             '(default: 1 µK)')
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
    return FeshbachInputs(
        element=element,
        isotope_A=A,
        resonances=[Resonance.from_json(row) for row in rows],
        a_bg_default_a0=float(params.get('a_bg_default_a0', rows[0].get('a_bg_a0', 100.0))),
        C6_au=C6_au,
        C6_source=C6_src,
        mass_amu=mass_amu,
        mass_source=mass_src,
        note=params.get('note', ''),
    )


# ============================================================================
# PLOTTING
# ============================================================================

def plot_feshbach(inputs: FeshbachInputs, result: FeshbachResult, output_dir):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    el = inputs.element
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'{el}_feshbach.html')

    B = result.B_G
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            f'{el} — scattering length a(B)',
            f'elastic cross section σ_el (E = k_B · {result.kT_uK} µK)',
        ),
    )

    colors = ['#e8a838', '#38bdf8', '#f472b6', '#4ade80', '#a78bfa', '#fb7185']

    # Combined curve
    fig.add_trace(go.Scatter(
        x=B, y=result.a_combined_a0,
        mode='lines', name=f"a(B) {result.default_channel or 'combined'}",
        line=dict(color='#f0f4ff', width=2.2),
        hovertemplate='B=%{x:.2f} G<br>a=%{y:.4g} a₀<extra>combined</extra>',
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=B, y=result.sigma_combined_a02,
        mode='lines', name=f"σ {result.default_channel or 'combined'}",
        line=dict(color='#f0f4ff', width=2.2),
        hovertemplate='B=%{x:.2f} G<br>σ=%{y:.4g} a₀²<extra>combined</extra>',
        showlegend=False,
    ), row=2, col=1)

    # Other entrance channels (each its own product over that channel's s-wave poles)
    for ch in result.channels:
        if ch.channel == result.default_channel:
            continue
        fig.add_trace(go.Scatter(
            x=B, y=np.clip(np.asarray(ch.a_a0, dtype=float), -5000, 5000),
            mode='lines', name=f"a(B) {ch.channel}", visible='legendonly',
            line=dict(width=1.8),
            hovertemplate='B=%{x:.2f} G<br>a=%{y:.4g} a₀<extra>' + ch.channel + '</extra>',
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=B, y=ch.sigma_a02, mode='lines', name=f"σ {ch.channel}",
            visible='legendonly', line=dict(width=1.8), showlegend=False,
        ), row=2, col=1)

    # Individual resonances (clipped for readability)
    for i, curve in enumerate(result.curves):
        res = curve.resonance
        c = colors[i % len(colors)]
        if curve.a_a0 is None:     # d/g-wave pole: not a pole of the s-wave a(B); marker only
            for _row in (1, 2):
                fig.add_vline(x=res.B0_G, line=dict(color=c, width=1, dash='dot'), row=_row, col=1)
            continue
        a = np.array(curve.a_a0, dtype=float)
        # Clip display of poles for legibility
        a_plot = np.clip(a, -5000, 5000)
        fig.add_trace(go.Scatter(
            x=B, y=a_plot,
            mode='lines', name=res.label,
            line=dict(color=c, width=1.4, dash='dot'),
            hovertemplate=(
                f"{res.label}<br>"
                f"channel: {res.channel}<br>"
                'B=%{x:.2f} G<br>a=%{y:.4g} a₀<extra></extra>'
            ),
        ), row=1, col=1)

        # Vertical marker at B₀
        fig.add_vline(
            x=res.B0_G, line=dict(color=c, width=1, dash='dash'),
            row=1, col=1,
        )
        fig.add_vline(
            x=res.B0_G, line=dict(color=c, width=1, dash='dash'),
            row=2, col=1,
        )

    # ā and a_bg guides
    fig.add_hline(
        y=result.a_bar_a0,
        line=dict(color='#94a3b8', width=1, dash='dot'),
        annotation_text=f"ā = {result.a_bar_a0:.1f} a₀",
        annotation_position='bottom right',
        row=1, col=1,
    )
    fig.add_hline(
        y=result.a_bg_default_a0,
        line=dict(color='#64748b', width=1, dash='dot'),
        annotation_text=f"a_bg = {result.a_bg_default_a0:.2f} a₀",
        annotation_position='top right',
        row=1, col=1,
    )

    fig.update_yaxes(title_text='a / a₀', row=1, col=1,
                     range=[-500, 500], zeroline=True)
    fig.update_yaxes(title_text='σ_el / a₀²', type='log', row=2, col=1)
    fig.update_xaxes(title_text='B [Gauss]', row=2, col=1)

    subtitle = (
        f"C₆ = {result.C6_au:.1f} au ({inputs.C6_source}) · "
        f"β₆ = {result.beta6_a0:.2f} a₀ · "
        f"ā = {result.a_bar_a0:.2f} a₀ · "
        f"m = {result.mass_amu:.6f} amu ({inputs.mass_source})"
    )
    A = inputs.isotope_A
    nuc = nuclide_label(el, A)
    nuc_plain = nuclide_plain(el, A)
    fig.update_layout(
        title=dict(
            text=(f'<b>{nuc} Feshbach resonances</b>'
                  f'<br><sup>{nuc_plain}  ·  {subtitle}</sup>'),
            x=0.5,
        ),
        template='plotly_dark',
        autosize=True,
        legend=dict(orientation='h', y=1.08),
        margin=dict(t=100, b=50, l=70, r=30),
    )

    raw_html = fig.to_html(include_plotlyjs=False, full_html=True)
    # Browser tab + accessible document title, then the shared fullscreen
    # wrapper — both snippet insertions live in the renderer.
    raw_html = inject_title(raw_html, f'{nuc_plain} Feshbach')
    raw_html = inject_head(raw_html, fullscreen_css())
    write_offline_report(out_path, raw_html)
    print(f'  Plot -> {out_path}')
    return out_path

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
            }
            for c in result.curves
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
    )
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

    out_dir = args.output_dir or os.path.join(Workspace.current().plots_dir, element)
    if not args.no_plot:
        plot_feshbach(inputs, result, out_dir)
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
