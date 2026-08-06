#!/usr/bin/env python3
"""
feshbach.py — Magnetic Feshbach resonances for alkali atoms
-----------------------------------------------------------------------------
v1: single-channel magnetic Feshbach parametrization + van der Waals scales.

Physics
-------
Near an isolated magnetic Feshbach resonance the s-wave scattering length is

    a(B) = a_bg · (1 − Δ / (B − B₀))

where a_bg is the background scattering length, B₀ the resonance field, and
Δ the magnetic width (Chin, Grimm, Julienne & Tiesinga, Rev. Mod. Phys. 82,
1225 (2010)). Sign convention: Δ > 0 ⇒ a → +∞ just above B₀.

From the Casimir–Polder C₆ (already computed by polarizability.py):

    β₆ = (2 μ C₆ / ℏ²)^{1/4}          van der Waals length
    ā  ≈ 0.955978 β₆                  mean scattering length (Gribakin–Flambaum)

Elastic s-wave cross section at collision wave number k:

    σ_el = 4π a² / (1 + k² a²)

Data sources
------------
    data_json/<element>_feshbach.json       literature B₀, Δ, a_bg (user-maintained)
    data_json/<element>_polarizability.json C₆_au (fallback: alpha_core.C6_EXPT)
    data_json/nuclear_data.json             isotope mass (fallback: ATOMIC_MASS_AMU)

Outputs
-------
    data_json/<element>_feshbach_out.json   computed scales + a(B) samples
    plots/<element>/<element>_feshbach.html interactive Plotly

Usage
-----
    python feshbach.py Na
    python feshbach.py Na --B-max 2500 --kT-uK 1.0
    python feshbach.py --all
"""

__version__ = '0.1'

import argparse
import json
import math
import os
import sys

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("Plotly not installed. Run: pip install plotly")
    sys.exit(1)

from constants import (
    A0_M,
    HBAR_SI,
    AMU_KG,
    K_B,
    AU_TO_SI_C6,
    A_BAR_FACTOR,
    ALKALI_MASS_AMU as ATOMIC_MASS_AMU,
    ISOTOPE_MASS_AMU,
    C6_EXPT as C6_EXPT_FALLBACK,
    nuclide_label,
    nuclide_plain,
)

# ============================================================================
# ARGPARSE
# ============================================================================

parser = argparse.ArgumentParser(
    description='Magnetic Feshbach resonances (scattering length vs B)')
parser.add_argument('element', nargs='?', default=None)
parser.add_argument('--all', action='store_true',
                    help='Run every element that has *_feshbach.json')
parser.add_argument('--data-dir', default='data_json')
parser.add_argument('--output-dir', default=None,
                    help='Plot directory (default: plots/<element>)')
parser.add_argument('--B-max', type=float, default=None,
                    help='Max B field in Gauss (default: auto from resonances)')
parser.add_argument('--B-points', type=int, default=2000,
                    help='Number of B-field sample points')
parser.add_argument('--kT-uK', type=float, default=1.0,
                    help='Collision energy as temperature [µK] for σ(B) '
                         '(default: 1 µK)')
parser.add_argument('--isotope', type=int, default=None,
                    help='Mass number A (default: from feshbach JSON)')

# ============================================================================
# DATA LOADING
# ============================================================================

def get_available_elements(data_dir='data_json'):
    """Neutrals that have a literature feshbach parameter file (ions excluded)."""
    from species import is_ion_id
    out = []
    if not os.path.isdir(data_dir):
        return out
    for name in os.listdir(data_dir):
        if name.endswith('_feshbach.json') and not name.endswith('_feshbach_out.json'):
            el = name.replace('_feshbach.json', '')
            if el and el[0].isupper() and not is_ion_id(el):
                out.append(el)
    return sorted(out)


def load_feshbach_params(element, data_dir='data_json'):
    path = os.path.join(data_dir, f'{element}_feshbach.json')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'No literature file {path}\n'
            f'Create it with resonances: B0_G, Delta_G, a_bg_a0 '
            f'(see data_json/Na_feshbach.json).')
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def load_C6_au(element, data_dir='data_json'):
    """Prefer polarizability pipeline C₆; fall back to experimental table."""
    path = os.path.join(data_dir, f'{element}_polarizability.json')
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
        if 'C6_au' in d:
            return float(d['C6_au']), 'polarizability.json'
        if 'C6_expt_au' in d and d['C6_expt_au']:
            return float(d['C6_expt_au']), 'polarizability.json (expt)'
    if element in C6_EXPT_FALLBACK:
        return C6_EXPT_FALLBACK[element], 'C6_EXPT table'
    raise ValueError(f'No C₆ available for {element}')


def load_mass_amu(element, isotope_A=None, data_dir='data_json'):
    """Isotope mass from nuclear_data.json, else elemental fallback."""
    path = os.path.join(data_dir, 'nuclear_data.json')
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            nd = json.load(f)
        cands = [iso for iso in nd.get('isotopes', [])
                 if iso.get('symbol') == element]
        if isotope_A is not None:
            cands = [iso for iso in cands if iso.get('A') == isotope_A]
        else:
            # Prefer stable isotopes
            stable = [iso for iso in cands if iso.get('stable')]
            if stable:
                cands = stable
        if cands:
            # nuclear_data may not store mass — use A as approx if needed
            iso = cands[0]
            if 'mass_amu' in iso:
                return float(iso['mass_amu']), f"nuclear_data A={iso.get('A')}"
            if isotope_A or iso.get('A'):
                A = isotope_A or iso['A']
                key = (element, int(A))
                if key in ISOTOPE_MASS_AMU:
                    return ISOTOPE_MASS_AMU[key], f'ISOTOPE_MASS_AMU (A={A})'
                if element in ATOMIC_MASS_AMU:
                    return ATOMIC_MASS_AMU[element], f'ATOMIC_MASS_AMU (A={A})'
    if isotope_A is not None and (element, int(isotope_A)) in ISOTOPE_MASS_AMU:
        return ISOTOPE_MASS_AMU[(element, int(isotope_A))], f'ISOTOPE_MASS_AMU (A={isotope_A})'
    if element in ATOMIC_MASS_AMU:
        return ATOMIC_MASS_AMU[element], 'ATOMIC_MASS_AMU'
    raise ValueError(f'No mass for {element}')

# ============================================================================
# PHYSICS
# ============================================================================

def reduced_mass_kg(mass_amu):
    """Identical-atom reduced mass μ = m/2."""
    m = mass_amu * AMU_KG
    return 0.5 * m


def beta6_a0(C6_au, mass_amu):
    """
    van der Waals length β₆ = (2 μ C₆ / ℏ²)^{1/4} in units of a₀.

    Working in SI then converting to a₀ keeps units transparent.
    """
    mu = reduced_mass_kg(mass_amu)
    C6_SI = C6_au * AU_TO_SI_C6
    beta_m = (2.0 * mu * C6_SI / HBAR_SI**2) ** 0.25
    return beta_m / A0_M


def a_bar_a0(beta6):
    """Mean scattering length ā ≈ 0.956 β₆ (Gribakin–Flambaum)."""
    return A_BAR_FACTOR * beta6


def scattering_length(B, a_bg, B0, Delta):
    """
    a(B) = a_bg (1 − Δ/(B − B₀)).

    Returns NaN exactly at B = B₀ (pole).
    """
    B = np.asarray(B, dtype=float)
    denom = B - B0
    with np.errstate(divide='ignore', invalid='ignore'):
        a = a_bg * (1.0 - Delta / denom)
    a = np.where(np.abs(denom) < 1e-12, np.nan, a)
    return a


def wave_number_a0(kT_uK, mass_amu):
    """
    Thermal wave number k = √(2 μ k_B T) / ℏ, returned in a₀⁻¹.

    Uses E = k_B T (not 3/2 kT) — convention for quoting a single energy scale.
    """
    mu = reduced_mass_kg(mass_amu)
    E = K_B * (kT_uK * 1e-6)
    k_SI = math.sqrt(2.0 * mu * E) / HBAR_SI
    return k_SI * A0_M   # 1/m · m/a₀⁻¹ → a₀⁻¹


def sigma_el_a0(a_a0, k_a0inv):
    """σ_el / a₀² = 4π a² / (1 + k² a²)."""
    a = np.asarray(a_a0, dtype=float)
    ka2 = (k_a0inv * a) ** 2
    with np.errstate(invalid='ignore'):
        return 4.0 * np.pi * a**2 / (1.0 + ka2)


def multi_resonance_a(B, resonances, a_bg_default):
    """
    Product of isolated-resonance factors (valid when resonances are well
    separated compared to their widths):

        a(B)/a_bg = Π_i (1 − Δ_i / (B − B₀,i))

    Uses each resonance's own a_bg if given, else a_bg_default for the
    overall scale (standard when all share one open channel).
    """
    B = np.asarray(B, dtype=float)
    # Use the first resonance's a_bg (or default) as overall scale —
    # all Na |1,1> resonances share the same a_bg.
    a_bg = float(resonances[0].get('a_bg_a0', a_bg_default))
    factor = np.ones_like(B, dtype=float)
    for res in resonances:
        B0 = float(res['B0_G'])
        Delta = float(res['Delta_G'])
        denom = B - B0
        with np.errstate(divide='ignore', invalid='ignore'):
            f = 1.0 - Delta / denom
        f = np.where(np.abs(denom) < 1e-12, np.nan, f)
        factor = factor * f
    return a_bg * factor

# ============================================================================
# COMPUTE
# ============================================================================

def compute_element(element, data_dir='data_json', isotope_A=None,
                    B_max=None, B_points=2000, kT_uK=1.0):
    params = load_feshbach_params(element, data_dir)
    A = isotope_A or params.get('isotope_A')
    resonances = params.get('resonances', [])
    if not resonances:
        raise ValueError(f'{element}_feshbach.json has no resonances')

    a_bg_default = float(params.get('a_bg_default_a0',
                                    resonances[0].get('a_bg_a0', 100.0)))

    C6_au, C6_src = load_C6_au(element, data_dir)
    mass_amu, mass_src = load_mass_amu(element, A, data_dir)

    beta6 = beta6_a0(C6_au, mass_amu)
    abar = a_bar_a0(beta6)
    k_th = wave_number_a0(kT_uK, mass_amu)

    # B grid: cover from ~0 to a bit past the highest resonance
    B_hi = B_max
    if B_hi is None:
        B_hi = max(float(r['B0_G']) for r in resonances) * 1.15
        B_hi = max(B_hi, 500.0)
    B = np.linspace(0.0, B_hi, B_points)

    # Combined a(B) for all s-wave resonances in the file
    a_comb = multi_resonance_a(B, resonances, a_bg_default)
    sig_comb = sigma_el_a0(a_comb, k_th)

    # Per-resonance curves (isolated single-pole form)
    per_res = []
    for res in resonances:
        a_bg = float(res.get('a_bg_a0', a_bg_default))
        B0 = float(res['B0_G'])
        Delta = float(res['Delta_G'])
        a = scattering_length(B, a_bg, B0, Delta)
        per_res.append({
            'label': res.get('label', f"B0={B0}G"),
            'channel': res.get('channel', ''),
            'partial_wave': res.get('partial_wave', 's'),
            'B0_G': B0,
            'Delta_G': Delta,
            'a_bg_a0': a_bg,
            'ref': res.get('ref', ''),
            'a_a0': a,
            'sigma_a02': sigma_el_a0(a, k_th),
        })

    return {
        'element': element,
        'isotope_A': A,
        'mass_amu': mass_amu,
        'mass_source': mass_src,
        'C6_au': C6_au,
        'C6_source': C6_src,
        'beta6_a0': beta6,
        'a_bar_a0': abar,
        'a_bg_default_a0': a_bg_default,
        'kT_uK': kT_uK,
        'k_a0inv': k_th,
        'B_G': B,
        'a_combined_a0': a_comb,
        'sigma_combined_a02': sig_comb,
        'resonances': per_res,
        'params_note': params.get('note', ''),
    }

# ============================================================================
# PLOTTING
# ============================================================================

def plot_feshbach(result, output_dir):
    el = result['element']
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'{el}_feshbach.html')

    B = result['B_G']
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            f'{el} — scattering length a(B)',
            f'elastic cross section σ_el (E = k_B · {result["kT_uK"]} µK)',
        ),
    )

    colors = ['#e8a838', '#38bdf8', '#f472b6', '#4ade80', '#a78bfa', '#fb7185']

    # Combined curve
    fig.add_trace(go.Scatter(
        x=B, y=result['a_combined_a0'],
        mode='lines', name='combined a(B)',
        line=dict(color='#f0f4ff', width=2.2),
        hovertemplate='B=%{x:.2f} G<br>a=%{y:.4g} a₀<extra>combined</extra>',
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=B, y=result['sigma_combined_a02'],
        mode='lines', name='combined σ',
        line=dict(color='#f0f4ff', width=2.2),
        hovertemplate='B=%{x:.2f} G<br>σ=%{y:.4g} a₀²<extra>combined</extra>',
        showlegend=False,
    ), row=2, col=1)

    # Individual resonances (clipped for readability)
    for i, res in enumerate(result['resonances']):
        c = colors[i % len(colors)]
        a = np.array(res['a_a0'], dtype=float)
        # Clip display of poles for legibility
        a_plot = np.clip(a, -5000, 5000)
        fig.add_trace(go.Scatter(
            x=B, y=a_plot,
            mode='lines', name=res['label'],
            line=dict(color=c, width=1.4, dash='dot'),
            hovertemplate=(
                f"{res['label']}<br>"
                f"channel: {res['channel']}<br>"
                'B=%{x:.2f} G<br>a=%{y:.4g} a₀<extra></extra>'
            ),
        ), row=1, col=1)

        # Vertical marker at B₀
        fig.add_vline(
            x=res['B0_G'], line=dict(color=c, width=1, dash='dash'),
            row=1, col=1,
        )
        fig.add_vline(
            x=res['B0_G'], line=dict(color=c, width=1, dash='dash'),
            row=2, col=1,
        )

    # ā and a_bg guides
    fig.add_hline(
        y=result['a_bar_a0'],
        line=dict(color='#94a3b8', width=1, dash='dot'),
        annotation_text=f"ā = {result['a_bar_a0']:.1f} a₀",
        annotation_position='bottom right',
        row=1, col=1,
    )
    fig.add_hline(
        y=result['a_bg_default_a0'],
        line=dict(color='#64748b', width=1, dash='dot'),
        annotation_text=f"a_bg = {result['a_bg_default_a0']:.2f} a₀",
        annotation_position='top right',
        row=1, col=1,
    )

    fig.update_yaxes(title_text='a / a₀', row=1, col=1,
                     range=[-500, 500], zeroline=True)
    fig.update_yaxes(title_text='σ_el / a₀²', type='log', row=2, col=1)
    fig.update_xaxes(title_text='B [Gauss]', row=2, col=1)

    subtitle = (
        f"C₆ = {result['C6_au']:.1f} au ({result['C6_source']}) · "
        f"β₆ = {result['beta6_a0']:.2f} a₀ · "
        f"ā = {result['a_bar_a0']:.2f} a₀ · "
        f"m = {result['mass_amu']:.6f} amu ({result['mass_source']})"
    )
    A = result.get('isotope_A')
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

    raw_html = fig.to_html(include_plotlyjs='cdn', full_html=True)
    _fs_css = (
        '<style>'
        'html,body{margin:0;padding:0;width:100%;height:100%;'
        'overflow:hidden;background:#0a0d14;}'
        '.plotly-graph-div{width:100vw!important;height:100vh!important;}'
        '</style>'
    )
    # Browser tab + accessible document title
    if '<title>' in raw_html:
        raw_html = raw_html.replace(
            '<title></title>',
            f'<title>{nuc_plain} Feshbach</title>', 1)
        raw_html = raw_html.replace(
            '<title>Plotly</title>',
            f'<title>{nuc_plain} Feshbach</title>', 1)
    else:
        raw_html = raw_html.replace(
            '</head>',
            f'<title>{nuc_plain} Feshbach</title></head>', 1)
    raw_html = raw_html.replace('</head>', _fs_css + '</head>', 1)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(raw_html)
    print(f'  Plot -> {out_path}')
    return out_path

# ============================================================================
# JSON OUTPUT
# ============================================================================

def write_output_json(result, data_dir='data_json'):
    el = result['element']
    path = os.path.join(data_dir, f'{el}_feshbach_out.json')

    # Downsample B-grid for JSON size
    step = max(1, len(result['B_G']) // 400)
    payload = {
        'element': el,
        'isotope_A': result['isotope_A'],
        'version': __version__,
        'mass_amu': result['mass_amu'],
        'mass_source': result['mass_source'],
        'C6_au': round(result['C6_au'], 4),
        'C6_source': result['C6_source'],
        'beta6_a0': round(result['beta6_a0'], 4),
        'a_bar_a0': round(result['a_bar_a0'], 4),
        'a_bg_default_a0': result['a_bg_default_a0'],
        'kT_uK': result['kT_uK'],
        'k_a0inv': result['k_a0inv'],
        'resonances': [
            {
                'label': r['label'],
                'channel': r['channel'],
                'partial_wave': r['partial_wave'],
                'B0_G': r['B0_G'],
                'Delta_G': r['Delta_G'],
                'a_bg_a0': r['a_bg_a0'],
                'ref': r['ref'],
            }
            for r in result['resonances']
        ],
        'B_G': result['B_G'][::step].tolist(),
        'a_combined_a0': [
            (None if (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
             else round(float(v), 4))
            for v in result['a_combined_a0'][::step]
        ],
        'note': result.get('params_note', ''),
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print(f'  JSON -> {path}')
    return path

# ============================================================================
# MAIN
# ============================================================================

def run_one(element, args):
    print('=' * 60)
    print(f'Feshbach - {element}')
    print('=' * 60)
    result = compute_element(
        element,
        data_dir=args.data_dir,
        isotope_A=args.isotope,
        B_max=args.B_max,
        B_points=args.B_points,
        kT_uK=args.kT_uK,
    )
    print(f"  C6   = {result['C6_au']:.2f} au  ({result['C6_source']})")
    print(f"  beta6= {result['beta6_a0']:.3f} a0")
    print(f"  a_bar= {result['a_bar_a0']:.3f} a0")
    print(f"  a_bg = {result['a_bg_default_a0']:.3f} a0")
    print(f"  mass = {result['mass_amu']:.6f} amu  ({result['mass_source']})")
    print(f"  resonances: {len(result['resonances'])}")
    for r in result['resonances']:
        print(f"    - {r['label']:20s}  B0={r['B0_G']:7.1f} G  "
              f"Delta={r['Delta_G']:6.3f} G  [{r['channel']}]")

    out_dir = args.output_dir or os.path.join('plots', element)
    plot_feshbach(result, out_dir)
    write_output_json(result, args.data_dir)
    return result


def main(argv=None):
    from species import require_neutral_species

    args = parser.parse_args(argv)
    if args.all:
        els = get_available_elements(args.data_dir)
        if not els:
            print('No *_feshbach.json files found in', args.data_dir)
            sys.exit(1)
        for el in els:
            run_one(el, args)
        return

    if not args.element:
        parser.print_help()
        print('\nAvailable:', ', '.join(get_available_elements(args.data_dir)) or '(none)')
        sys.exit(1)

    el = require_neutral_species(args.element, "feshbach.py")["species_id"]
    run_one(el, args)


if __name__ == '__main__':
    main()
