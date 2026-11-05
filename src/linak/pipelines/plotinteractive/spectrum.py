"""Plot 2: the absorption spectrum (broadened per source, sticks, filter panel).

Data source: the lifetimes JSON transitions table (NIST + literature + ORCA +
Numerov + Coulomb f), plus ORCA levels not already in it. X-axis: wavelength (nm).
Broadening: Doppler (T-dependent, element-aware mass), instrumental (σ),
natural (Lorentzian per state from its lifetime); see ``physics/spectrum_core.py``.
"""

from __future__ import annotations

import json

import numpy as np

from linak.core.constants import AMU_KG, ATOMIC_MASS_AMU, C_SI, EV_TO_J
from linak.physics.rydberg import label_to_pretty
from linak.physics.spectrum_core import (
    SIGMA_INST_DEFAULT,
    adaptive_grid,
    broadened_spectrum,
    doppler_sigma_nm,
    line_gamma_nm,
)
from linak.report_templates import (
    inject_body,
    inject_head,
    js_constant,
    load,
    render as render_template,
    write_offline_report,
)

from .common import orca_line_strength
from .inputs import PlotInputs
from .view import View

_SOURCE_GROUPS = {
    'NIST': 'NIST', 'precision': 'NIST',
    'NIST-extrap': 'NIST-extrap',
    'literature': 'literature',
    'ORCA': 'ORCA', 'EOM-CCSD': 'ORCA',
    'Numerov': 'Numerov',
    'Coulomb': 'Coulomb', 'Coulomb(approx)': 'Coulomb',
    # 'unknown' (and any source string not listed) gets its OWN group: lumping it into
    # Coulomb would present an unclassified f-value as an approximation of known provenance.
    'unknown': 'unclassified',
}

# ── Stick spectrum (individual line markers) ──────────────────────────────────
# Shown as vertical lines at each wavelength, height = fosc, coloured by source
_SRC_COLORS = {
    'NIST': '#1a6eb5', 'ORCA': '#c0392b', 'Numerov': '#0e7c6b',
    'NIST-extrap': '#5b9bd5',
    'literature': '#c47a00', 'Coulomb': '#27ae60',
    'unclassified': '#7f8c8d',
}
_BROAD_COLORS = {
    ('NIST',    'doppler'):      '#1a6eb5',
    ('NIST',    'instrumental'): '#5dade2',
    ('NIST',    'natural'):      '#1a5276',
    ('ORCA',    'doppler'):      '#c0392b',
    ('ORCA',    'instrumental'): '#e74c3c',
    ('ORCA',    'natural'):      '#922b21',
    ('Numerov', 'doppler'):      '#0e7c6b',
    ('Numerov', 'instrumental'): '#48c9b0',
    ('Numerov', 'natural'):      '#0b5345',
    ('literature', 'doppler'):   '#c47a00',
    ('literature', 'instrumental'): '#e0a040',
    ('literature', 'natural'):   '#8a5500',
    ('Coulomb', 'doppler'):      '#27ae60',
    ('Coulomb', 'instrumental'): '#58d68d',
    ('Coulomb', 'natural'):      '#1e8449',
    ('NIST-extrap', 'doppler'):      '#5b9bd5',
    ('NIST-extrap', 'instrumental'): '#85c1e9',
    ('NIST-extrap', 'natural'):      '#2e6a9e',
    ('unclassified', 'doppler'):      '#7f8c8d',
    ('unclassified', 'instrumental'): '#aab7b8',
    ('unclassified', 'natural'):      '#5f6a6a',
    ('All',     'doppler'):      '#8e44ad',
    ('All',     'instrumental'): '#bb8fce',
    ('All',     'natural'):      '#6c3483',
}

# ── Broadened spectrum traces (one per source × broadening) ──────────────────
_hover_spec = 'λ = %{x:.3f} nm<br>Intensity = %{y:.4f}<extra></extra>'

def _hex_to_rgba(hex_col, alpha=0.15):
    """Convert '#rrggbb' to 'rgba(r,g,b,alpha)' for Plotly fillcolor."""
    h = hex_col.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


def plot_spectrum(inputs: PlotInputs, view: View, output_dir) -> None:
    """Write <El>_spectrum_levels.html."""
    import plotly.graph_objects as go

    element = inputs.element
    _species = inputs.species
    excitations = inputs.excitations
    lifetimes_by_state = inputs.lifetimes_by_state
    species_label = inputs.species_label
    is_ion_species = inputs.is_ion
    _disp = inputs.disp
    IE_for_orca_filter = view.IE_for_orca_filter
    label_lookup = view.label_lookup

    print("\n📈 Creating spectrum plot...")

    # ── Atomic masses for element-correct Doppler broadening ─────────────────────
    # Ions use species_id (Na_c1); fall back to chemical symbol (Na).
    _mass_amu = ATOMIC_MASS_AMU.get(
        element, ATOMIC_MASS_AMU.get(_species.get("symbol", element), 40.0))
    _mc2_sp   = _mass_amu * AMU_KG * C_SI**2 / EV_TO_J   # rest energy eV

    # ── Collect spectral lines from lifetimes transitions ────────────────────────
    _gs_label_sp = inputs.gs_label
    # Two datasets:
    #   absorption: lower == ground state (ns → X)
    #   emission:   ALL transitions with fosc > 0 (used separately)
    _lifetimes_trans = inputs.lifetimes_transitions
    _abs_lines  = []   # ground-state absorption
    _spec = {}   # source × broadening -> spectrum

    for _t in _lifetimes_trans:
        _fosc = _t.get('fosc', 0)
        if _fosc is None or _fosc <= 0:
            continue
        _wl  = _t['wavelength_nm']
        _dE  = _t['delta_E_eV']
        _ul  = _t['upper_label']
        _ll  = _t['lower_label']
        _src = _SOURCE_GROUPS.get(_t.get('fosc_source', ''), 'unclassified')
        _reg = _t.get('region', '')
        _acc = _t.get('fosc_acc') or _t.get('fosc_source', '')
        _A   = _t.get('A_s', 0)
        # PHYSICS FIX 6: carry n_degenerate so spectrum amplitude = fosc * n_deg
        _n_deg = _t.get('n_degenerate', 1) or 1
        _rec = dict(wl=_wl, dE=_dE, fosc=_fosc, ul=_ul, ll=_ll,
                    src=_src, reg=_reg, acc=_acc, A=_A, n_deg=_n_deg)
        if _ll == _gs_label_sp:
            _abs_lines.append(_rec)

    # Also include ORCA excitations not already in lifetimes
    _lt_ul_set = set(r['ul'] for r in _abs_lines)
    for _exc in excitations:
        _fosc_o = orca_line_strength(_exc)
        if _fosc_o <= 0 or _exc.get('spin_forbidden'):
            continue
        _eV_o = _exc['energy_eV']
        if IE_for_orca_filter and _eV_o >= IE_for_orca_filter:
            continue
        _wl_o = 1239.84 / _eV_o if _eV_o > 0 else 0
        _lbl_o = label_lookup.get(_eV_o, f"{_exc.get('to','?')}")
        if _lbl_o in _lt_ul_set:
            continue   # already covered by lifetimes
        _rec_o = dict(wl=_wl_o, dE=_eV_o, fosc=_fosc_o, ul=_lbl_o,
                      ll=_gs_label_sp, src='ORCA', reg='', acc='ORCA', A=0,
                      n_deg=1)
        _abs_lines.append(_rec_o)

    _abs_lines.sort(key=lambda r: r['wl'])
    print(f"  Absorption lines (gs→X): {len(_abs_lines)} "
          f"[NIST:{sum(1 for r in _abs_lines if r['src']=='NIST')} "
          f"ORCA:{sum(1 for r in _abs_lines if r['src']=='ORCA')} "
          f"Numerov:{sum(1 for r in _abs_lines if r['src']=='Numerov')} "
          f"literature:{sum(1 for r in _abs_lines if r['src']=='literature')} "
          f"NIST-extrap:{sum(1 for r in _abs_lines if r['src']=='NIST-extrap')} "
          f"Coulomb:{sum(1 for r in _abs_lines if r['src']=='Coulomb')} "
          f"unclassified:{sum(1 for r in _abs_lines if r['src']=='unclassified')}]")

    # BUG FIX 6: build the adaptive grid with the SMALLEST sigma that will be used.
    # Doppler σ is much finer than σ_inst — grid around Doppler so peaks resolve.
    # Reference wavelength: median gs→X line (not a hardcoded alkali D-line).
    _wl_doppler_ref = float(np.median([r['wl'] for r in _abs_lines])) if _abs_lines else 500.0
    _sigma_doppler_ref = doppler_sigma_nm(_wl_doppler_ref, _mc2_sp, 300.0)   # σ at T=300K
    _sigma_grid = min(SIGMA_INST_DEFAULT, max(_sigma_doppler_ref, 1e-4))
    _wl_grid = adaptive_grid(_abs_lines, _sigma_grid)
    _wl_lo   = float(_wl_grid[0])
    _wl_hi   = float(_wl_grid[-1])
    _wl_lo_str = f"{_wl_lo:.3f}"     # panel-JS token (was a formatted f-string anchor)
    _wl_hi_str = f"{_wl_hi:.3f}"
    print(f"  Adaptive grid: {len(_wl_grid)} points over {_wl_lo:.1f}–{_wl_hi:.1f} nm")
    print(f"  Grid sigma: {_sigma_grid*1000:.4f} pm "
          f"(Doppler σ at {_wl_doppler_ref:.1f} nm: {_sigma_doppler_ref*1000:.4f} pm)")

    # Split by source (fosc provenance — Numerov is QD radial f, not level energies)
    _nist_lines   = [r for r in _abs_lines if r['src'] == 'NIST']
    _orca_lines   = [r for r in _abs_lines if r['src'] == 'ORCA']
    _num_lines    = [r for r in _abs_lines if r['src'] == 'Numerov']
    _nxt_lines    = [r for r in _abs_lines if r['src'] == 'NIST-extrap']
    _lit_lines    = [r for r in _abs_lines if r['src'] == 'literature']
    _coul_lines   = [r for r in _abs_lines if r['src'] == 'Coulomb']
    _unk_lines    = [r for r in _abs_lines if r['src'] == 'unclassified']

    _SRC_LINE_GROUPS = [
        ('NIST', _nist_lines),
        ('ORCA', _orca_lines),
        ('Numerov', _num_lines),
        ('NIST-extrap', _nxt_lines),
        ('literature', _lit_lines),
        ('Coulomb', _coul_lines),
        ('unclassified', _unk_lines),
        ('All', _abs_lines),
    ]
    for _src_key, _lines in _SRC_LINE_GROUPS:
        if not _lines:
            continue
        for _broad in ('doppler', 'instrumental', 'natural'):
            _spec[(_src_key, _broad)] = broadened_spectrum(_lines, _wl_grid, _broad,
                                                           _mc2_sp, lifetimes_by_state)
    _SRC_COLORS_JS = json.dumps(_SRC_COLORS)   # serialised for JS f-string injection

    fig2 = go.Figure()

    _broad_labels = {
        'doppler': (
            f'Doppler T=300K (σ≈{_sigma_doppler_ref*1000:.2f} pm '
            f'@ {_wl_doppler_ref:.1f} nm — zoom in)'
        ),
        'instrumental': f'Instrumental σ={SIGMA_INST_DEFAULT}nm (overview)',
        'natural':      'Natural linewidth (ħ/τ) — zoom in',
    }

    for (_src_key, _broad), _arr in sorted(_spec.items()):
        _col  = _BROAD_COLORS.get((_src_key, _broad), '#888888')
        _name = f'{_src_key} — {_broad_labels[_broad]}'
        # Default visible: All-sources Instrumental (overview) so all peaks show
        _vis  = True if (_src_key == 'All' and _broad == 'instrumental') else 'legendonly'
        fig2.add_trace(go.Scatter(
            x=_wl_grid, y=_arr,
            mode='lines',
            fill='tozeroy',
            fillcolor=_hex_to_rgba(_col, 0.15),
            line=dict(color=_col, width=1.5),
            name=_name,
            hovertemplate=_hover_spec,
            visible=_vis,
            legendgroup=f'broad_{_src_key}_{_broad}',
        ))

    # ── Stick lines (fosc as vertical bars) ──────────────────────────────────────
    # FIX 3: Use global max fosc across ALL sources for common normalization
    # so relative heights correctly reflect physical oscillator strengths.
    _global_fosc_max = max((r['fosc'] for r in _abs_lines), default=1) or 1

    for _src_key, _lines, _col in [
            ('NIST',       _nist_lines, _SRC_COLORS['NIST']),
            ('ORCA',       _orca_lines, _SRC_COLORS['ORCA']),
            ('Numerov',    _num_lines,  _SRC_COLORS['Numerov']),
            ('literature', _lit_lines,  _SRC_COLORS['literature']),
            ('NIST-extrap', _nxt_lines, _SRC_COLORS['NIST-extrap']),
            ('Coulomb',    _coul_lines, _SRC_COLORS['Coulomb']),
            ('unclassified', _unk_lines, _SRC_COLORS['unclassified']),
    ]:
        if not _lines:
            continue
        # Build stick arrays: [wl, wl, None] repeated for each line
        _sx, _sy = [], []
        for r in _lines:
            _sx += [r['wl'], r['wl'], None]
            _sy += [0, r['fosc'] / _global_fosc_max, None]
        fig2.add_trace(go.Scatter(
            x=_sx, y=_sy,
            mode='lines',
            line=dict(color=_col, width=1),
            name=f'{_src_key} stick (fosc)',
            hoverinfo='skip',
            visible='legendonly',
            legendgroup=f'stick_{_src_key}',
        ))

    # ── Individual line markers (hover shows details) ─────────────────────────────
    for _src_key, _lines, _col in [
            ('NIST',       _nist_lines, _SRC_COLORS['NIST']),
            ('ORCA',       _orca_lines, _SRC_COLORS['ORCA']),
            ('Numerov',    _num_lines,  _SRC_COLORS['Numerov']),
            ('literature', _lit_lines,  _SRC_COLORS['literature']),
            ('NIST-extrap', _nxt_lines, _SRC_COLORS['NIST-extrap']),
            ('Coulomb',    _coul_lines, _SRC_COLORS['Coulomb']),
            ('unclassified', _unk_lines, _SRC_COLORS['unclassified']),
    ]:
        if not _lines:
            continue
        fig2.add_trace(go.Scatter(
            x=[r['wl'] for r in _lines],
            y=[r['fosc'] / _global_fosc_max for r in _lines],
            mode='markers',
            marker=dict(color=_col, size=6, symbol='line-ns',
                        line=dict(color=_col, width=2)),
            name=f'{_src_key} lines',
            customdata=[[_disp(r['ul']), _disp(r['ll']), r['wl'], r['dE']*1000,
                         r['fosc'], r['acc']] for r in _lines],
            hovertemplate=(
                '<b>%{customdata[0]} → %{customdata[1]}</b><br>'
                'λ = %{customdata[2]:.3f} nm<br>'
                'ΔE = %{customdata[3]:.2f} meV<br>'
                'fosc = %{customdata[4]:.4~g}<br>'
                'Acc: %{customdata[5]}<extra></extra>'
            ),
            visible='legendonly',
            legendgroup=f'markers_{_src_key}',
        ))

    # ── Layout ────────────────────────────────────────────────────────────────────
    # Coulomb rarely appears in gs absorption (n* thresholds on both ends).
    # Numerov QD radial f DOES appear for high-n principal-series members.
    _gs_disp_sp = label_to_pretty(_gs_label_sp)
    _spec_title_species = species_label if is_ion_species else element
    fig2.update_layout(
        title=dict(
            text=f'<b>{_spec_title_species} — Absorption Spectrum</b>'
                 f'<br><sub>fosc sources: NIST / literature / ORCA / Numerov / Coulomb • '
                 f'Ground state: {_gs_disp_sp} • '
                 f'{len(_abs_lines)} lines '
                 f'({len(_nist_lines)} NIST, {len(_orca_lines)} ORCA, '
                 f'{len(_num_lines)} Numerov, {len(_lit_lines)} literature, '
                 f'{len(_coul_lines)} Coulomb) • '
                 f'Mass = {_mass_amu:.3f} amu</sub>',
            x=0.5, xanchor='center', font=dict(size=18)
        ),
        xaxis=dict(
            title=dict(text='<b>Wavelength (nm)</b>', font=dict(size=14)),
            showgrid=True, gridcolor='#eee', zeroline=False,
        ),
        yaxis=dict(
            title=dict(text='<b>Relative Intensity (log)</b>', font=dict(size=14)),
            showgrid=True, gridcolor='#eee',
            type='log', range=[-5, 0.1],   # default log so weak lines are visible
        ),
        hovermode='closest',   # x unified breaks for multi-line spectra
        plot_bgcolor='white',
        autosize=True,
        updatemenus=[
            dict(
                type='buttons', direction='left',
                x=0.0, xanchor='left', y=1.08, yanchor='top',
                buttons=[
                    dict(label='Linear scale',
                         method='relayout',
                         args=[{'yaxis.type': 'linear',
                                'yaxis.range': [-0.02, 1.05],
                                'yaxis.title.text': '<b>Relative Intensity</b>'}]),
                    dict(label='Log scale',
                         method='relayout',
                         args=[{'yaxis.type': 'log',
                                'yaxis.range': [-5, 0.1],
                                'yaxis.title.text': '<b>Relative Intensity (log)</b>'}]),
                ],
                pad=dict(r=10, t=5),
                showactive=True,
                bgcolor='#f0f0f0', bordercolor='#ccc',
                font=dict(size=11),
            )
        ],
        legend=dict(
            title='<b>Click to toggle</b>',
            bgcolor='rgba(255,255,255,0.92)',
            bordercolor='#ccc', borderwidth=1,
            x=1.01, xanchor='left', y=1, yanchor='top',
            font=dict(size=11),
        ),
        margin=dict(r=350, t=100),
        annotations=[
            dict(
                text=(
                    'Log scale active by default — switch to <b>Linear</b> to compare relative strengths • '
                    'Zoom in on a peak for Doppler/natural fine structure'
                    if is_ion_species else
                    'Log scale active by default — switch to <b>Linear</b> to see D-line dominance • '
                    'Zoom in on a peak for Doppler/natural fine structure'
                ),
                x=0.5, y=-0.12, xref='paper', yref='paper',
                showarrow=False, font=dict(size=11, color='#666'),
                align='center',
            )
        ]
    )

    # ── Inject interactive filter panel into the spectrum HTML ────────────────────
    html_file2 = f"{output_dir}/{element}_spectrum_levels.html"

    _spec_lines_js = [
        dict(wl=r['wl'], dE=round(r['dE']*1000, 3), fosc=round(r['fosc'], 6),
             ul=r['ul'], ll=r['ll'],
             ulp=_disp(r['ul']), llp=_disp(r['ll']),
             src=r['src'], reg=r['reg'], acc=r['acc'],
             n_deg=r.get('n_deg', 1),
             gamma_nm=round(line_gamma_nm(r, lifetimes_by_state), 9))
        for r in _abs_lines
    ]



    _html2 = fig2.to_html(include_plotlyjs=False, full_html=True)
    _html2 = inject_head(
        _html2,
        load('plotinteractive_spectrum_fs.css') + load('plotinteractive_spectrum_panel.css'))
    _html2 = inject_body(
        _html2,
        load('plotinteractive_spectrum_panel.html')
        + render_template(
            'plotinteractive_spectrum_panel.js',
            spec_lines_js=json.dumps(_spec_lines_js),
            mass_amu=str(_mass_amu),
            kb_eV=js_constant('KB_EV'),
            SRC_COLORS_JS=_SRC_COLORS_JS,
            wl_lo=_wl_lo_str,
            wl_hi=_wl_hi_str,
        ))
    write_offline_report(html_file2, _html2)

    print(f" Saved: {html_file2}")
