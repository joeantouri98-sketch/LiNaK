"""
hyperfine.py  —  Hyperfine structure, Zeeman & Breit-Rabi for any atom
-----------------------------------------------------------------------
For each atomic state (n, l, J) of one isotope:
  - Hyperfine F-level energies (magnetic dipole + electric quadrupole + octupole)
  - g_F factors and the m_F sublevels (weak-field Zeeman)
  - Breit-Rabi levels from 0 to --B-max Gauss
  - the ground-state microwave clock interval

The physics is ``physics/hyperfine_core.py``; this module loads the inputs,
prints a summary and writes <El>_hyperfine.json, <El>_hyperfine.html and
<El>_zeeman.html.

Data sources (no hardcoding):
  Nuclear data:  data_json/nuclear_data.json        (built from IAEA CSVs)
  HF constants:  data_json/<element>_hf_constants.json  (user provides; a template
                 is written when it is missing)

Usage:
  linak --pipeline hyperfine Na
  linak --pipeline hyperfine Na --isotope 23 --states 3s1/2 3p1/2 3p3/2
  linak --pipeline hyperfine Na --build-nuclear-data  (rebuild nuclear_data.json from CSVs)
  linak --pipeline hyperfine Na --B-max 1000          (Breit-Rabi up to 1000 Gauss)
  linak --pipeline hyperfine --all                    (run all elements with hf_constants)
"""

from __future__ import annotations

__version__ = '1.1'   # + octupole C term in hyperfine_energy (C_kHz field)
                      # + experimental gJ override (gJ_exp field in hf_constants)
                      # + works for non-alkali atoms with supplied gJ_exp

import argparse
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import asdict, replace

from linak.core.constants import default_isotope_A, nuclide_label, nuclide_plain
from linak.core.errors import MissingDataError
from linak.core.fileio import write_json
from linak.core.paths import reference_path
from linak.core.workspace import Workspace
from linak.physics.hyperfine_core import (
    HfConstants,
    HyperfineResult,
    HyperfineState,
    MicrowaveClock,
    Nucleus,
    compute,
)
from linak.report_templates import (
    fullscreen_css_with,
    inject_body,
    inject_head,
    inject_title,
    load as load_template,
    render as render_template,
    style_wrap,
    write_offline_report,
)

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="linak --pipeline hyperfine",
        description='Hyperfine structure and Zeeman diagram')
    parser.add_argument('element', nargs='?', default=None)
    parser.add_argument('--isotope',   type=int,   default=None,
                        help='Mass number A (default: the most abundant natural isotope; Fr: 210)')
    parser.add_argument('--states',    nargs='+',  default=None,
                        help='States to include e.g. 3s1/2 3p1/2 3p3/2')
    parser.add_argument('--B-max',     type=float, default=500.0,
                        help='Max B field for Breit-Rabi diagram in Gauss (default: 500)')
    parser.add_argument('--B-points',  type=int,   default=500,
                        help='Number of B-field points (default: 500)')
    parser.add_argument('--data-dir',  default=str(Workspace.current().data_json_dir))
    parser.add_argument('--output-dir',default=None,
                        help='Output directory (default: .linak/plots/<element>)')
    parser.add_argument('--all',       action='store_true',
                        help='Run all elements with available hf_constants')
    parser.add_argument('--build-nuclear-data', action='store_true',
                        help='Rebuild nuclear_data.json from IAEA CSV files')
    parser.add_argument('--magn-csv',  default=str(reference_path('Moments', 'magn_mom_recomm.csv')))
    parser.add_argument('--elec-csv',  default=str(reference_path('Moments', 'elec_mom_recomm.csv')))
    return parser

# ============================================================================
# STEP 1 — BUILD nuclear_data.json FROM IAEA CSVs  (if requested or missing)
# ============================================================================

# Half-life field of the IAEA tables: "22.5 ms", "3.8 m", "7x105 y",
# "2.3 x 105 y", "1.4x10*17 y", "5.0 1010y" (the exponent's superscript is
# flattened, so "105" means 10^5), with optional ">", "(", "est" decorations.
# Longest units first: "ms" must not match as minutes.
_HALFLIFE_RE = re.compile(
    r'([\d.]+)\s*(?:x\s*)?(?:10[\^*]?(\d+))?\s*(ms|us|ns|ps|fs|s|m|h|d|y)\b')
_HALFLIFE_UNIT_S = {'fs': 1e-15, 'ps': 1e-12, 'ns': 1e-9, 'us': 1e-6, 'ms': 1e-3,
                    's': 1, 'm': 60, 'h': 3600, 'd': 86400, 'y': 3.156e7}


def _parse_spin(s):
    """Nuclear spin I from an IAEA spin/parity field (``3/2-``, ``(3/2)+``, ``1(-)``).

    ``None`` for ambiguous ("4 or 5") or unparsable assignments.
    """
    text = s.strip()
    if ' or ' in text:
        return None
    m = re.fullmatch(r'(\d+)(?:/(\d+))?', re.sub(r'[()+\-\s]', '', text))
    if not m:
        return None
    return int(m.group(1)) / int(m.group(2) or 1)


def _parse_float(s):
    """A moment value without its sign decoration and uncertainty: ``+0.579(1)``."""
    s = str(s).strip()
    s = re.sub(r'^\([\+\-]\)', '', s)
    s = re.sub(r'\(\d+\)', '', s)
    s = re.sub(r'\s', '', s)
    try:
        return float(s)
    except ValueError:
        return None


def _halflife_s(s):
    """Half-life in seconds; ``inf`` for stable, 0.0 if blank or unparsable (unknown)."""
    s = s.strip().lower()
    if s == 'stable':
        return float('inf')
    m = _HALFLIFE_RE.search(s)
    if not m:
        return 0.0
    mantissa, exponent, unit = m.groups()
    try:
        value = float(mantissa)
    except ValueError:
        return 0.0
    if exponent is not None:
        value *= 10.0 ** int(exponent)
    return value * _HALFLIFE_UNIT_S[unit]


def build_nuclear_data(magn_csv, elec_csv, out_path):
    """Parse IAEA magnetic + quadrupole CSV files into nuclear_data.json."""
    import csv

    magn = {}
    with open(magn_csv, encoding='utf-8', errors='ignore') as f:
        for r in csv.DictReader(f):
            try:
                z  = int(r['z'])
                A  = int(r['n.n+n.z'].strip())
                if float(r['energy [keV]'].strip() or '0') != 0:
                    continue
                I  = _parse_spin(r['spin'])
                mu = _parse_float(r['magnetic dipole [nm]'])
                hl = _halflife_s(r['halflife'])
                if I is None or I == 0:
                    continue
                key = (z, A)
                if key not in magn:
                    magn[key] = {
                        'Z': z, 'A': A,
                        'symbol': r['symbol'].strip(),
                        'I': I, 'mu_I': mu,
                        'halflife_s': hl if math.isfinite(hl) else -1,
                        'halflife_str': r['halflife'].strip(),
                        'stable': r['halflife'].strip().lower() == 'stable',
                    }
            except (KeyError, ValueError):
                continue  # malformed row (blank Z/A/energy)

    quad = {}
    with open(elec_csv, encoding='utf-8', errors='ignore') as f:
        for r in csv.DictReader(f):
            try:
                z = int(r['z'])
                A = int(r['n.n+n.z'].strip())
                if float(r['energy [keV]'].strip() or '0') != 0:
                    continue
                Q = _parse_float(r['electric quadrupole [b]'])
                if Q is not None:
                    quad[(z, A)] = Q
            except (KeyError, ValueError):
                continue  # malformed row (blank Z/A/energy)

    isotopes = []
    for (z, A), d in sorted(magn.items()):
        d['Q_barn'] = quad.get((z, A), 0.0) or 0.0
        I  = d['I']
        mu = d['mu_I']
        d['g_I'] = round(mu / I, 7) if (mu is not None and I > 0) else None
        isotopes.append(d)

    out = {
        'source': 'IAEA Nuclear Moments — Stone (2025 update)',
        'files':  [magn_csv, elec_csv],
        'n_isotopes': len(isotopes),
        'isotopes': isotopes,
    }
    write_json(out_path, out)
    print(f"✓ Built {out_path}: {len(isotopes)} isotopes ({sum(d['stable'] for d in isotopes)} stable)")
    return out


def load_nuclear_index(path):
    """Load nuclear_data.json into a symbol -> [isotope, ...] lookup (sorted by A)."""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    nuclear = defaultdict(list)
    for iso in data['isotopes']:
        nuclear[iso['symbol']].append(iso)
    for sym in nuclear:
        nuclear[sym].sort(key=lambda x: x['A'])
    return nuclear

# ============================================================================
# STEP 2 — LOAD  <element>_hf_constants.json (or write a template)
# ============================================================================

def write_hf_constants_template(out_path):
    """Write a ``<element>_hf_constants.json`` template for the user to fill in."""
    print(f"  Creating template file at {out_path}")
    template = {
        "_source": "Fill from Steck alkali data sheets (steck.us/alkalidata) or NIST ASD",
        "_format": {"A_MHz": "hyperfine A constant (MHz)", "B_MHz": "hyperfine B constant (MHz, 0 for J=1/2)"},
        "_example_Na23": {
            "3s1/2": {"A_MHz": 885.813064, "B_MHz": 0.0},
            "3p1/2": {"A_MHz":  94.44,     "B_MHz": 0.0},
            "3p3/2": {"A_MHz":  18.534,    "B_MHz": 2.724}
        }
    }
    write_json(out_path, template)
    print(f"  Template saved to {out_path}. Please fill in A_MHz and B_MHz values.")


def _parse_hf_block(block) -> dict[str, HfConstants]:
    """state_label -> HfConstants of one block (keys starting with '_' are metadata)."""
    return {key: HfConstants.from_json(val) for key, val in block.items()
            if not key.startswith('_') and isinstance(val, dict) and 'A_MHz' in val}


def _block_isotope_A(block, element):
    """Mass number named in a block's '_isotope' string ('Rb-87 (I=3/2)' -> 87), or None."""
    m = re.match(r'\s*' + re.escape(element) + r'-(\d+)', str(block.get('_isotope', '')))
    return int(m.group(1)) if m else None


def load_hf_constants(element, data_dir, isotope_A=None, nuclear=None):
    """
    Hyperfine constants of ONE isotope from data_json/<element>_hf_constants.json, as
    (state_label -> HfConstants in file order, provenance text); the constants are None if the
    file is missing (a template is written) or has no states.

    Isotope handling (A = isotope_A; None = whatever the main block is):
      1. main block's '_isotope' is element-A          -> use it
      2. an alt block ('_Rb87_D_line', ...) is element-A -> use that block
      3. otherwise rescale the main block from its reference isotope (nuclear moments from
         *nuclear*):  A_hf ∝ g_I  and  B_hf ∝ Q   (hyperfine anomaly neglected, ~0.1-1 %)
         with a printed warning; C (octupole) is dropped (no scaling law).
    """
    path = os.path.join(data_dir, f"{element}_hf_constants.json")
    if not os.path.exists(path):
        print(f"\n  No hf_constants file found for {element}.")
        write_hf_constants_template(path)
        return None, ''

    with open(path, encoding='utf-8') as f:
        raw = json.load(f)

    main_A = _block_isotope_A(raw, element)
    if isotope_A is None or main_A is None or isotope_A == main_A:
        return _parse_hf_block(raw) or None, f"main block ({raw.get('_isotope', 'isotope not stated')})"

    for key, val in raw.items():
        if key.startswith('_') and isinstance(val, dict) and _block_isotope_A(val, element) == isotope_A:
            return _parse_hf_block(val) or None, f"alt block {key} ({val.get('_isotope')})"

    # no block for this isotope: rescale from the reference isotope
    ref = get_isotope(nuclear or {}, element, main_A)
    tgt = get_isotope(nuclear or {}, element, isotope_A)
    base = _parse_hf_block(raw)
    if not base or ref is None or tgt is None or not ref.get('g_I'):
        print(f"  ERROR: no {element}-{isotope_A} hyperfine constants and cannot rescale "
              f"from {element}-{main_A}; add a block to {path}")
        return None, f"main block ({raw.get('_isotope', 'isotope not stated')})"
    fA = tgt['g_I'] / ref['g_I']
    fB = (tgt['Q_barn'] / ref['Q_barn']) if ref.get('Q_barn') else 0.0
    hf = {lbl: replace(c, A_MHz=c.A_MHz * fA, B_MHz=c.B_MHz * fB, C_kHz=0.0)
          for lbl, c in base.items()}
    print(f"  WARNING: no measured {element}-{isotope_A} hyperfine constants in {os.path.basename(path)}.\n"
          f"           Rescaled from {element}-{main_A}: A x {fA:.5f} (g_I), B x {fB:.5f} (Q); "
          f"hyperfine anomaly neglected (~0.1-1 %), octupole dropped.")
    return hf, (f"RESCALED from {element}-{main_A}: A x {fA:.5f} (g_I ratio), "
                f"B x {fB:.5f} (Q ratio)")

# ============================================================================
# STEP 3 — NUCLEAR DATA HELPERS
# ============================================================================

def get_isotope(nuclear, element, A=None):
    """
    Return the isotope dict for (element, A) from the `nuclear` lookup.
    If A is None the default is the most abundant NATURAL isotope
    (constants.default_isotope_A: Li-7, Na-23, K-39, Rb-85, Cs-133; Fr-210, the isotope the
    shipped constants file covers). An explicit A always overrides the default.
    Elements outside that table fall back to the highest-A stable isotope, else the longest-lived.
    """
    isos = nuclear.get(element, [])
    if not isos:
        return None

    if A is None:
        A = default_isotope_A(element)

    if A is not None:
        for iso in isos:
            if iso['A'] == A:
                return iso
        print(f"  Warning: {element}-{A} not found in nuclear_data.json")
        return None

    stable = [i for i in isos if i['stable']]
    if stable:
        return stable[-1]

    # No stable: return longest-lived
    return max(isos, key=lambda i: i['halflife_s'])


# ============================================================================
# STEP 4 — RUN ONE ELEMENT
# ============================================================================

def run_element(element, nuclear, isotope_A=None, state_labels=None,
                B_max=500.0, B_points=500, data_dir=None) -> HyperfineResult | None:
    if data_dir is None:
        data_dir = str(Workspace.current().data_json_dir)

    print(f"\n{'='*65}")
    print(f"  {element}  — Hyperfine Structure")
    print(f"{'='*65}")

    # ── Nuclear data ─────────────────────────────────────────────────────────
    iso = get_isotope(nuclear, element, isotope_A)
    if iso is None:
        print(f"  ERROR: No nuclear data found for {element}")
        return None

    nucleus = Nucleus.from_json(iso)
    print(f"  Isotope: {element}-{nucleus.A}")
    print(f"  I = {nucleus.I}  μ_I = {nucleus.mu_I:.5f} nm  Q = {nucleus.Q_barn:.5f} b"
          f"  g_I = {nucleus.gI:.5f}")

    if nucleus.I == 0:
        print("  I=0: no hyperfine structure. Exiting.")
        return None

    # ── HF constants ─────────────────────────────────────────────────────────
    hf_consts, hf_source = load_hf_constants(element, data_dir, nucleus.A, nuclear)
    print(f"  HF constants: {hf_source}")
    if hf_consts is None:
        print(f"  No hf_constants available. Fill in {data_dir}/{element}_hf_constants.json")
        return None

    # ── Quantum defects, to scale A to states without constants ─────────────
    ryd_file = os.path.join(data_dir, f"{element}_rydberg.json")
    ryd_data = {}
    if os.path.exists(ryd_file):
        with open(ryd_file, encoding='utf-8') as f:
            ryd_data = json.load(f)

    # If no states specified, use all states in hf_constants
    if state_labels is None:
        state_labels = list(hf_consts)

    print(f"  States: {', '.join(state_labels)}")

    result = compute(nucleus, hf_consts, state_labels, ryd_data.get('quantum_defects', {}),
                     B_max=B_max, B_points=B_points)
    for label in result.skipped:
        print(f"  Warning: could not parse state '{label}' — skipping")
    for st in result.states:
        _print_state(st)
    if not result.states:
        print("  No states processed.")
        return None
    if result.cycling is not None:
        cl = result.cycling
        print(f"\n  Cycling transition: {cl.lower} F={cl.F_lower:.0f} -> {cl.upper} F'={cl.F_upper:.0f}")
    if result.clock is not None:
        _print_clock(result.clock)
    return result


def _print_state(st: HyperfineState) -> None:
    if st.extrapolated_from:
        print(f"    {st.label}: A_hf extrapolated from {st.extrapolated_from} "
              f"× (n*/n*)³ = {st.A_MHz:.3f} MHz")
    if st.constants_missing:
        print(f"    {st.label}: A_hf = 0 (not in hf_constants) — F levels degenerate")

    print(f"\n  State {st.label}:  J={st.J}  g_J={st.gJ:.4f}"
          f"{'  [experimental]' if st.gJ_source == 'experimental' else '  [Landé]'}"
          f"  A={st.A_MHz:.4f} MHz  B={st.B_MHz:.4f} MHz"
          f"{f'  C={st.C_kHz:.3f} kHz' if st.C_kHz != 0 else ''}")
    print(f"    F levels: {[lv.F for lv in st.levels]}")
    for lv in st.levels:
        print(f"    F={lv.F:.0f}:  E={lv.E_MHz:+.3f} MHz  g_F={lv.gF:+.5f}  "
              f"m_F={[int(2*m) for m in lv.mF]} (in units of 1/2)")


def _wavelength_text(mc: MicrowaveClock) -> str:
    if mc.wavelength_cm is None:
        return 'λ: none, the F levels are degenerate'
    return f'λ ≈ {mc.wavelength_cm:.3f} cm'


def _print_clock(mc: MicrowaveClock) -> None:
    print(f"\n  Ground-state microwave / clock transition ({mc.state}):")
    print(f"    F={mc.F_upper:.0f} ↔ F={mc.F_lower:.0f}: "
          f"{mc.frequency_MHz:.6f} MHz "
          f"({mc.frequency_GHz:.9f} GHz)  "
          f"({_wavelength_text(mc)})")


# ============================================================================
# STEP 5 — PLOTS
# ============================================================================


# Colour palette for m_F sublevels: centre = white, ± red/blue
def mF_color(mF, F):
    """Map m_F in [-F, +F] to a colour."""
    if F < 1e-9:
        return '#ffffff'
    # -1 to +1; clamped because degenerate levels (A = 0) get |m_F| > F labels (P45)
    norm = max(-1.0, min(1.0, mF / F))
    # Diverging: blue (norm=-1) → white (0) → red (norm=+1)
    r = int(255 * max(0, norm))
    b = int(255 * max(0, -norm))
    g = int(255 * (1 - abs(norm)) * 0.7)
    return f'rgb({r},{g},{b})'


def plot_hyperfine(element: str, result: HyperfineResult, out_dir) -> str:
    """
    Hyperfine structure plot — NO Breit-Rabi (lives in zeeman.html).

    Layout:
      Row 1 (tall, ~68%): single shared-axis Grotrian-style level diagram —
             all states on one x-axis (one column each), F bars, m_F ticks,
             F/g_F labels, intra-state E1 connections. Shared y-axis (MHz)
             across all states. Ground-state microwave clock interval is
             annotated when the result has a clock interval.
      Row 2 (~32%): dropdown-toggled panel using native Plotly updatemenus
             (same pattern as the Breit-Rabi state selector in zeeman.html):
               - "Splittings" — log-scale bar chart of |dE(F,F+-1)| per state
                 (clock interval highlighted)
               - "Constants"  — table of A, B, C, g_J, g_F per state
               - "Clock"      — ground ΔF=1 microwave frequency card

    Scales to any number of states automatically — just add to hf_constants.json.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    el     = element
    nucleus = result.nucleus
    nuc    = nuclide_label(el, nucleus.A)
    nuc_p  = nuclide_plain(el, nucleus.A)
    I      = nucleus.I
    states = result.states
    mc     = result.clock
    mc_state = mc.state if mc else (states[0].label if states else '')

    STATE_PALETTE = [
        '#f0c54d', '#4d9ff0', '#4df0b0', '#f05a4d',
        '#c54df0', '#f04d9f', '#7af07a', '#f0a44d',
    ]
    state_cols = [STATE_PALETTE[i % len(STATE_PALETTE)] for i in range(len(states))]
    CLOCK_COL = '#f0c54d'

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.68, 0.32],
        vertical_spacing=0.10,
        subplot_titles=[
            f'Hyperfine Level Structure — {nuc}  (I = {I})',
            'Hyperfine Splittings  |\u0394E(F, F\u00b11)|  (log scale)',
        ],
    )

    # ---- ROW 1: shared-axis level diagram (single x/y for all states) ------
    COL_W  = 1.4
    COL_SP = 2.4
    MF_DY  = 0.035

    all_E_vals = [lv.E_MHz for st in states for lv in st.levels]
    global_span = max(max(all_E_vals) - min(all_E_vals), 1.0)

    for si, st in enumerate(states):
        x_ctr   = si * COL_SP
        E_hf    = {lv.F: lv.E_MHz for lv in st.levels}
        scol    = state_cols[si]
        E_vals  = list(E_hf.values())
        E_span  = max(E_vals) - min(E_vals) if len(E_vals) > 1 else global_span * 0.1
        mF_drop = max(global_span * MF_DY, E_span * 0.08)

        fig.add_annotation(
            x=x_ctr, y=max(E_vals) + global_span * 0.05,
            text=f'<b>{st.label}</b>',
            showarrow=False, font=dict(size=12, color=scol),
            xref='x', yref='y', xanchor='center', yanchor='bottom',
        )

        fig.add_trace(go.Scatter(
            x=[x_ctr, x_ctr],
            y=[min(E_vals) - mF_drop * 1.6, max(E_vals)],
            mode='lines',
            line=dict(color=f'rgba({int(scol[1:3],16)},'
                            f'{int(scol[3:5],16)},'
                            f'{int(scol[5:7],16)},0.16)', width=1),
            showlegend=False, hoverinfo='skip',
        ), row=1, col=1)

        for lv in st.levels:
            F, E, mFs, gF = lv.F, lv.E_MHz, lv.mF, lv.gF
            x0    = x_ctr - COL_W * 0.42
            x1    = x_ctr + COL_W * 0.42
            F_lbl = int(F) if F == int(F) else F
            n_mF  = len(mFs)
            is_clock_F = (
                mc is not None
                and st.label == mc.state
                and (abs(F - mc.F_upper) < 1e-6 or abs(F - mc.F_lower) < 1e-6)
            )
            line_w = 4.5 if is_clock_F else 3
            line_c = CLOCK_COL if is_clock_F else scol

            hover = (f'<b>{nuc}  {st.label}  F={F_lbl}</b><br>'
                     f'E = {E:+.4f} MHz<br>'
                     f'g<sub>F</sub> = {gF:+.5f}<br>'
                     f'A = {st.A_MHz:.4f} MHz<br>'
                     f'm<sub>F</sub> = {int(min(mFs))}\u2026{int(max(mFs))}<br>'
                     f'2F+1 = {n_mF} sublevels')
            if is_clock_F:
                hover += (f'<br><b>Clock manifold</b> '
                          f'({mc.frequency_MHz:.6f} MHz gs interval)')

            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[E, E],
                mode='lines',
                line=dict(color=line_c, width=line_w),
                name=f'{st.label} F={F_lbl}',
                legendgroup=f'st{si}',
                showlegend=(F == st.levels[0].F),
                hovertemplate=hover + '<extra></extra>',
            ), row=1, col=1)

            fig.add_annotation(
                x=x0 - 0.05, y=E, text=f'F={F_lbl}',
                showarrow=False, xanchor='right',
                font=dict(size=9, color=line_c), xref='x', yref='y',
            )
            fig.add_annotation(
                x=x1 + 0.05, y=E, text=f'g<sub>F</sub>={gF:+.3f}',
                showarrow=False, xanchor='left',
                font=dict(size=8, color='#7a9acd'), xref='x', yref='y',
            )

            y_tick  = E - mF_drop
            tick_hw = COL_W * 0.36 / max(n_mF - 1, 1) if n_mF > 1 else 0
            for k, mF in enumerate(sorted(mFs)):
                tx = (x_ctr - COL_W * 0.36 + k * tick_hw * 2
                      if n_mF > 1 else x_ctr)
                fig.add_trace(go.Scatter(
                    x=[tx, tx], y=[E, y_tick], mode='lines',
                    line=dict(color=mF_color(mF, F), width=1.5),
                    showlegend=False,
                    hovertemplate=f'm<sub>F</sub>={int(mF):+d}<extra></extra>',
                ), row=1, col=1)
                if mF == max(mFs) or mF == min(mFs) or mF == 0:
                    fig.add_annotation(
                        x=tx, y=y_tick - global_span * 0.014,
                        text=(f'{int(mF):+d}' if mF == int(mF) else f'{mF:+.1f}'),
                        showarrow=False, xanchor='center', yanchor='top',
                        font=dict(size=7, color='#5a6a8a'), xref='x', yref='y',
                    )

        # Ground-state microwave clock arrow (ΔF=1 interval)
        if (mc is not None and st.label == mc.state
                and mc.F_upper in E_hf and mc.F_lower in E_hf):
            mc_nu = mc.frequency_MHz
            E_hi = E_hf[mc.F_upper]
            E_lo = E_hf[mc.F_lower]
            x_clk = x_ctr + COL_W * 0.55
            Fu_i = int(mc.F_upper) if mc.F_upper == int(mc.F_upper) else mc.F_upper
            Fl_i = int(mc.F_lower) if mc.F_lower == int(mc.F_lower) else mc.F_lower
            fig.add_trace(go.Scatter(
                x=[x_clk, x_clk], y=[E_lo, E_hi],
                mode='lines+markers',
                line=dict(color=CLOCK_COL, width=2.5),
                marker=dict(symbol='line-ew', size=10, color=CLOCK_COL,
                            line=dict(width=2, color=CLOCK_COL)),
                name='Microwave clock',
                legendgroup='clock',
                showlegend=True,
                hovertemplate=(
                    f'<b>Microwave clock</b><br>'
                    f'{st.label}  F={Fu_i} \u2194 F={Fl_i}<br>'
                    f'\u03bd = {mc_nu:.6f} MHz<br>'
                    f'= {mc.frequency_GHz:.9f} GHz<br>'
                    f'{_wavelength_text(mc)}'
                    f'<extra></extra>'
                ),
            ), row=1, col=1)
            fig.add_annotation(
                x=x_clk + 0.12, y=0.5 * (E_lo + E_hi),
                text=(f'<b>\u03bd<sub>clock</sub></b><br>'
                      f'{mc_nu:.3f} MHz<br>'
                      f'({mc.frequency_GHz:.6f} GHz)'),
                showarrow=False, xanchor='left', yanchor='middle',
                font=dict(size=10, color=CLOCK_COL),
                xref='x', yref='y',
                bgcolor='rgba(10,13,20,0.75)',
                bordercolor=CLOCK_COL, borderwidth=1, borderpad=4,
            )

    # Inter-state E1 transition connectors
    for si in range(len(states) - 1):
        st_l = states[si]
        st_u = states[si + 1]
        x_l  = si * COL_SP + COL_W * 0.42
        x_u  = (si + 1) * COL_SP - COL_W * 0.42
        for lo in st_l.levels:
            for up in st_u.levels:
                F_l, E_l = lo.F, lo.E_MHz
                F_u, E_u = up.F, up.E_MHz
                dF = abs(F_u - F_l)
                if dF > 1.001 or (F_l < 0.001 and F_u < 0.001):
                    continue
                dE_hf = E_u - E_l
                F_l_i = int(F_l) if F_l == int(F_l) else F_l
                F_u_i = int(F_u) if F_u == int(F_u) else F_u
                fig.add_trace(go.Scatter(
                    x=[x_l, x_u], y=[E_l, E_u], mode='lines',
                    line=dict(color='rgba(200,200,200,0.18)', width=1, dash='dot'),
                    showlegend=False,
                    hovertemplate=(
                        f'<b>E1: {st_l.label} F={F_l_i} \u2192 '
                        f'{st_u.label} F={F_u_i}</b><br>'
                        f'\u0394F = {int(F_u - F_l):+d}<br>'
                        f'E({st_l.label} F={F_l_i}) = {E_l:+.4f} MHz<br>'
                        f'E({st_u.label} F={F_u_i}) = {E_u:+.4f} MHz<br>'
                        f'\u0394E_hf = {dE_hf:+.4f} MHz<extra></extra>'
                    ),
                ), row=1, col=1)

    x_lo = -COL_W * 0.8
    x_hi = (len(states) - 1) * COL_SP + COL_W * 0.8
    if mc is not None:
        x_hi += COL_W * 0.55
    fig.update_xaxes(range=[x_lo, x_hi], showticklabels=False,
                     showgrid=False, zeroline=False, row=1, col=1)
    fig.update_yaxes(title_text='E_hf (MHz, rel. centroid)',
                     gridcolor='#1a2540', zeroline=True,
                     zerolinecolor='rgba(255,255,255,0.1)', row=1, col=1)

    # ---- ROW 2 TRACE SET A: Splittings bar chart (log scale) ---------------
    bar_x, bar_y, bar_col, bar_hover, bar_line_w, bar_line_c = [], [], [], [], [], []
    for si, st in enumerate(states):
        scol = state_cols[si]
        for lo, hi in zip(st.levels, st.levels[1:]):
            F_lo, F_hi = lo.F, hi.F
            dE     = abs(hi.E_MHz - lo.E_MHz)
            F_lo_i = int(F_lo) if F_lo == int(F_lo) else F_lo
            F_hi_i = int(F_hi) if F_hi == int(F_hi) else F_hi
            lam_cm = 3e4 / dE if dE > 0 else 0
            is_clock_bar = (
                mc is not None
                and st.label == mc.state
                and ((abs(F_lo - mc.F_lower) < 1e-6 and abs(F_hi - mc.F_upper) < 1e-6)
                     or (abs(F_lo - mc.F_upper) < 1e-6 and abs(F_hi - mc.F_lower) < 1e-6))
            )
            bar_x.append(f'{st.label}<br>F={F_lo_i}\u2194{F_hi_i}')
            bar_y.append(max(dE, 1e-3))
            bar_col.append(CLOCK_COL if is_clock_bar else scol)
            bar_line_w.append(2 if is_clock_bar else 1)
            bar_line_c.append('#ffffff' if is_clock_bar else 'rgba(255,255,255,0.2)')
            hover = (
                f'<b>{st.label}  F={F_lo_i}\u2194F={F_hi_i}</b><br>'
                f'\u0394E = {dE:.5f} MHz<br>'
                f'\u03bb \u2248 {lam_cm:.2f} cm  ({lam_cm*10:.1f} mm)'
            )
            if is_clock_bar:
                hover += (f'<br><b>Ground microwave clock</b><br>'
                          f'\u03bd = {dE:.6f} MHz = {dE/1e3:.9f} GHz')
            bar_hover.append(hover)

    bar_trace_idx = len(fig.data)
    fig.add_trace(go.Bar(
        x=bar_x, y=bar_y,
        marker=dict(
            color=bar_col,
            line=dict(color=bar_line_c, width=bar_line_w),
        ),
        text=[f'{v:.4f}' for v in bar_y],
        textposition='outside',
        textfont=dict(size=9, color='#c8d0e0'),
        showlegend=False,
        customdata=bar_hover,
        hovertemplate='%{customdata}<extra></extra>',
        visible=True,
    ), row=2, col=1)

    fig.update_xaxes(gridcolor='#1a2540', tickfont=dict(size=9), row=2, col=1)
    fig.update_yaxes(title_text='|\u0394E| (MHz) (log scale)',
                     type='log', gridcolor='#1a2540', row=2, col=1)

    # ---- Constants table data — rendered as an HTML overlay panel below ----
    # (go.Table sharing a row with go.Bar via visible-toggling is unreliable:
    # Plotly's default template draws a residual background for hidden Table
    # traces. An HTML/JS overlay panel is robust and matches the dropdown
    # button styling used for the Bar/Table switch.)
    tbl_state, tbl_A, tbl_B, tbl_C = [], [], [], []
    tbl_gJ, tbl_gJ_src, tbl_F_gF  = [], [], []
    for st in states:
        tbl_state.append(st.label)
        tbl_A.append(f'{st.A_MHz:.5f}')
        tbl_B.append(f'{st.B_MHz:.4f}' if st.B_MHz != 0 else '\u2014')
        tbl_C.append(f'{st.C_kHz:.3f}'
                     if st.C_kHz != 0 else '\u2014')
        tbl_gJ.append(f'{st.gJ:.5f}')
        tbl_gJ_src.append(st.gJ_source)
        gF_lines = ', '.join(
            f'F={int(lv.F) if lv.F==int(lv.F) else lv.F}: {lv.gF:+.4f}'
            for lv in st.levels
        )
        tbl_F_gF.append(gF_lines)

    # ---- Overall layout ------------------------------------------------------
    mc_str = ''
    if mc is not None:
        Fu_i = int(mc.F_upper) if mc.F_upper == int(mc.F_upper) else mc.F_upper
        Fl_i = int(mc.F_lower) if mc.F_lower == int(mc.F_lower) else mc.F_lower
        mc_str = (f' \u00b7 Clock {mc.state} F={Fu_i}\u2194F={Fl_i}'
                  f' @ {mc.frequency_MHz:.6f} MHz'
                  f' ({mc.frequency_GHz:.6f} GHz)')

    fig.update_layout(
        title=dict(
            text=(f'<b>{nuc} Hyperfine Structure</b>'
                  f'<br><sup>{nuc_p}  ·  I={I}  '
                  f'\u03bc_I={nucleus.mu_I:.4f} nm  '
                  f'Q={nucleus.Q_barn:.5f} b'
                  f'{mc_str}</sup>'),
            x=0.5, xanchor='center',
            font=dict(size=18, color='#e8edf5'),
        ),
        plot_bgcolor='#0f1628',
        paper_bgcolor='#0a0d14',
        font=dict(color='#c8d0e0',
                  family="'Helvetica Neue',Arial,sans-serif"),
        hovermode='closest',
        autosize=True,
        margin=dict(t=130, r=40, b=40, l=80),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(10,13,20,0.88)',
            bordercolor='#1e2840', borderwidth=1,
            x=1.01, xanchor='left', y=0.99, yanchor='top',
            font=dict(size=10),
            title=dict(text='<b>State (F levels)</b>',
                       font=dict(size=10, color='#8a9ac0')),
        ),
    )

    for ann in fig.layout.annotations:
        if ann.text:
            ann.font.color = '#8a9ac0'
            ann.font.size  = 11

    out_file = os.path.join(out_dir, f"{el}_hyperfine.html")
    raw_html = fig.to_html(include_plotlyjs=False, full_html=True)
    raw_html = inject_title(raw_html, f'{nuc_p} Hyperfine')
    raw_html = inject_head(raw_html, fullscreen_css_with(
        load_template('hyperfine_fs_extra.css')))

    # ---- Inject dropdown toggle + HTML constants / clock panels --------------
    _tbl_rows_js = json.dumps([
        {'state': s, 'A': a, 'B': b, 'C': c, 'gJ': gj, 'src': src, 'gF': gf}
        for s, a, b, c, gj, src, gf in zip(
            tbl_state, tbl_A, tbl_B, tbl_C, tbl_gJ, tbl_gJ_src, tbl_F_gF)
    ])
    _clock_js = json.dumps({
        'present': mc is not None,
        'state': mc_state,
        'F_upper': (int(mc.F_upper) if mc.F_upper == int(mc.F_upper) else mc.F_upper) if mc else None,
        'F_lower': (int(mc.F_lower) if mc.F_lower == int(mc.F_lower) else mc.F_lower) if mc else None,
        'frequency_MHz': mc.frequency_MHz if mc else None,
        'frequency_GHz': mc.frequency_GHz if mc else None,
        'frequency_Hz':  mc.frequency_Hz if mc else None,
        'wavelength_cm': mc.wavelength_cm if mc else None,
        'wavelength_m':  mc.wavelength_m if mc else None,
        'note': mc.note if mc else '',
        'isotope': nuc_p,
        'I': I,
    }, allow_nan=False)
    _clock_opt = (
        '<option value="clock">Clock frequency</option>'
        if mc is not None else ''
    )
    _panel_html = render_template('hyperfine_panel.html', clock_opt=_clock_opt) + render_template(
        'hyperfine_panel.js',
        tbl_rows_js=_tbl_rows_js,
        clock_js=_clock_js,
        bar_trace_idx=str(bar_trace_idx),
    )
    raw_html = inject_body(raw_html, _panel_html)

    write_offline_report(out_file, raw_html)
    print(f"\n  \u2713 {out_file}")
    return out_file


def plot_zeeman(element: str, result: HyperfineResult, out_dir) -> str:
    """
    Standalone full-page Breit-Rabi / Zeeman diagram with dropdown
    to switch between states.
    """
    import plotly.graph_objects as go

    el     = element
    nucleus = result.nucleus
    nuc    = nuclide_label(el, nucleus.A)
    nuc_p  = nuclide_plain(el, nucleus.A)
    I      = nucleus.I
    states = result.states
    B_arr  = result.B_G

    fig = go.Figure()

    buttons = []
    traces_per_state = []

    for si, st in enumerate(states):
        BR_E   = st.BR_energies
        BR_lab = st.BR_labels
        dim    = BR_E.shape[1]
        t_indices = []

        for li in range(dim):
            F_approx, mF_approx = BR_lab[li]
            c  = mF_color(mF_approx, F_approx)
            ls = f'F={F_approx:.0f}, m<sub>F</sub>={mF_approx:+.0f}'
            fig.add_trace(go.Scatter(
                x=B_arr.tolist(), y=BR_E[:, li].tolist(),
                mode='lines', line=dict(color=c, width=2),
                name=ls, visible=(si == 0),
                hovertemplate=(f'<b>{st.label}</b> {ls}<br>'
                               f'B=%{{x:.1f}} G  E=%{{y:.3f}} MHz<extra></extra>'),
            ))
            t_indices.append(len(fig.data) - 1)

        traces_per_state.append(t_indices)

    # Build dropdown
    total_traces = len(fig.data)
    for si, st in enumerate(states):
        vis = [False] * total_traces
        for ti in traces_per_state[si]:
            vis[ti] = True
        buttons.append(dict(
            label=st.label,
            method='update',
            args=[{'visible': vis},
                  {'title.text': (f'<b>{nuc} Breit-Rabi: {st.label}</b><br>'
                                  f'<sup>{nuc_p}  ·  J={st.J}  g_J={st.gJ:.4f}  '
                                  f'A={st.A_MHz:.3f} MHz</sup>')}],
        ))

    fig.update_layout(
        title=dict(
            text=(f'<b>{nuc} Breit-Rabi Diagram: {states[0].label}</b><br>'
                  f'<sup>{nuc_p}  ·  J={states[0].J}  g_J={states[0].gJ:.4f}  '
                  f'A={states[0].A_MHz:.3f} MHz  |  '
                  f'I={I}  μ_I={nucleus.mu_I:.4f} nm</sup>'),
            x=0.5, xanchor='center', font=dict(size=18, color='#e8edf5'),
        ),
        xaxis=dict(title='B field (Gauss)', gridcolor='#1a2540',
                   tickfont=dict(color='#6b7a99')),
        yaxis=dict(title='Energy (MHz)', gridcolor='#1a2540',
                   zeroline=True, zerolinecolor='#2a3555',
                   tickfont=dict(color='#6b7a99')),
        plot_bgcolor='#0f1628', paper_bgcolor='#0a0d14',
        font=dict(color='#c8d0e0'),
        hovermode='x unified',
        autosize=True,
        margin=dict(t=110, r=220, b=60, l=80),
        legend=dict(
            bgcolor='rgba(10,13,20,0.88)',
            bordercolor='#1e2840', borderwidth=1,
            x=1.01, xanchor='left', y=1, yanchor='top',
            font=dict(size=11),
            title=dict(text='<b>|F, m_F⟩</b><br><sub>colour: m_F (blue→red)</sub>',
                       font=dict(size=10, color='#8a9ac0')),
        ),
        updatemenus=[dict(
            type='dropdown', direction='down',
            buttons=buttons,
            x=0.01, xanchor='left', y=1.08, yanchor='top',
            bgcolor='#111830', bordercolor='#2a3060',
            font=dict(color='#c8d0e0'),
        )],
    )

    out_file = os.path.join(out_dir, f"{el}_zeeman.html")
    raw_html = fig.to_html(include_plotlyjs=False, full_html=True)
    raw_html = inject_title(raw_html, f'{nuc_p} Zeeman')
    raw_html = inject_head(raw_html, style_wrap(load_template('hyperfine_zeeman_fs.css')))
    write_offline_report(out_file, raw_html)
    print(f"  ✓ {out_file}")
    return out_file


# ============================================================================
# STEP 6 — SAVE JSON
# ============================================================================

def _special_json(result: HyperfineResult) -> dict:
    """The cycling (F, F') pair, the ground-state clock and the cycling line."""
    special = {}
    if result.cycling is not None:
        special['cycling'] = [result.cycling.F_lower, result.cycling.F_upper]
    if result.clock is not None:
        special['microwave_clock'] = asdict(result.clock)
    if result.cycling is not None:
        special['cycling_line'] = asdict(result.cycling)
    return special


def save_json(element: str, result: HyperfineResult, data_dir) -> None:
    nucleus = result.nucleus
    out = {
        'element':    element,
        'isotope_A':  nucleus.A,
        'I':          nucleus.I,
        'mu_I':       nucleus.mu_I,
        'Q_barn':     nucleus.Q_barn,
        'gI':         nucleus.gI,
        'special':    _special_json(result),
        'states': [
            {
                'label':      st.label,
                'n': st.n, 'l': st.l, 'J': st.J,
                'gJ':         st.gJ,
                'gJ_source':  st.gJ_source,
                'A_MHz':      st.A_MHz,
                'B_MHz':      st.B_MHz,
                'C_kHz':      st.C_kHz,
                'F_levels':   [asdict(lv) for lv in st.levels],
            }
            for st in result.states
        ],
    }
    path = os.path.join(data_dir, f"{element}_hyperfine.json")
    write_json(path, out)
    print(f"  ✓ {path}")


# ============================================================================
# MAIN
# ============================================================================

def get_available_elements(data_dir):
    from linak.core.species import is_ion_id
    avail = []
    if os.path.isdir(data_dir):
        for fname in os.listdir(data_dir):
            m = re.match(r'^(\w+)_hf_constants\.json$', fname)
            if m:
                el = m.group(1)
                if not is_ion_id(el):
                    avail.append(el)
    return sorted(avail)


def main(argv: list[str] | None = None) -> int:
    from linak.core.species import require_neutral_species

    parser = build_parser()
    args = parser.parse_args(argv)

    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)
    nuclear_path = os.path.join(data_dir, 'nuclear_data.json')

    if args.build_nuclear_data:
        build_nuclear_data(args.magn_csv, args.elec_csv, nuclear_path)
        if args.element is None:
            return 0

    if not os.path.exists(nuclear_path):
        raise MissingDataError(
            f"nuclear_data.json not found at {nuclear_path}; run: linak --pipeline "
            f"hyperfine --build-nuclear-data")

    nuclear = load_nuclear_index(nuclear_path)

    if args.all:
        elements = get_available_elements(data_dir)
        if not elements:
            raise MissingDataError(f"No _hf_constants.json files found in {data_dir}.")
    else:
        if args.element is None:
            avail = get_available_elements(data_dir)
            parser.error("missing ELEMENT (or pass --all); available: "
                         + (', '.join(avail) if avail else 'none'))
        elements = [require_neutral_species(args.element, "hyperfine.py")["species_id"]]

    failed = []
    for el in elements:
        out_dir = args.output_dir or os.path.join(Workspace.current().plots_dir, el)
        os.makedirs(out_dir, exist_ok=True)

        result = run_element(
            el,
            nuclear,
            isotope_A    = args.isotope,
            state_labels = args.states,
            B_max        = args.B_max,
            B_points     = args.B_points,
            data_dir     = data_dir,
        )
        if result is None:
            failed.append(el)
            continue

        save_json(el, result, data_dir)
        plot_hyperfine(el, result, out_dir)
        plot_zeeman(el, result, out_dir)

    if failed:
        print(f"\n  Failed: {', '.join(failed)}")
        return 1
    print("\n✓ Done.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
