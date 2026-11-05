"""Plot 4: the 3D Grotrian diagram (x = orbital l, y = n, z = energy).

One sphere per (n, l, J) state, size ∝ log(lifetime); E1 edges from the
transitions JSON, all hidden at load and enabled from the filter panel.
"""

from __future__ import annotations

import json
import math

import numpy as np

from linak.report_templates import (
    inject_body,
    inject_head,
    load,
    render as render_template,
    write_offline_report,
)

from .common import _fmt_f, REGION_COLORS, _disp3d, _fnv_hash, _region_order_js, ionization_energy_eV
from .inputs import PlotInputs

_L_ORDER4  = ['s', 'p', 'd', 'f', 'g']
_L_INT4    = {l: i for i, l in enumerate(_L_ORDER4)}

_REGION_COLORS4 = REGION_COLORS   # one palette for the 2-D arrows and the 3-D diagram (P19)
_SRC_COLORS4 = {'NIST': '#2196F3', 'QDT model': '#FF9800'}

def _parse_j4(j_str):
    if j_str is None: return None
    s = str(j_str).strip().split(',')[0].strip()
    if '/' in s:
        try:
            a, b = s.split('/')
            return int(a) / int(b)
        except Exception: return None
    try:
        v = float(s)
        return v if 0 <= v <= 20 else None
    except Exception: return None

def _j_offset4(j_float):
    offsets = {0.5: -0.15, 1.5: 0.0, 2.5: 0.15,
               3.5: -0.15, 4.5: 0.15, 5.5: 0.0}
    if j_float is None: return 0.0
    return offsets.get(round(j_float * 2) / 2, (j_float - 1.0) * 0.12)

class _Nodes4:
    """The 3D Grotrian nodes, one per level, found by label or by (n, l, J)."""

    def __init__(self, tau_by_label):
        self.nodes = []
        self._by_key = {}
        self._by_label = {}
        self._tau_by_label = tau_by_label

    def add(self, n, l, j_str, label, E_abs, src, term=None, label_pretty=None):
        """Register one Grotrian node if not already present.

        Key by ASCII label (unique across multiplets). Fall back to (n,l,J,term)
        so 3s 3P° J=1 and 3s 1P° J=1 are both kept.
        """
        if l not in _L_INT4:
            return
        j_float = _parse_j4(j_str)
        term_key = (term or '')
        key = (n, l, str(j_str) if j_str is not None else '', term_key, label or '')
        if label and label in self._by_label:
            return
        if key in self._by_key:
            return
        idx = len(self.nodes)
        self._by_key[key] = idx
        if label:
            self._by_label[label] = idx
        # Slight x-jitter by term so multiplet members at same J don't stack
        # (a stable hash: built-in str hashing is salted per process).
        _term_jit = (_fnv_hash(term_key) % 7 - 3) * 0.02 if term_key else 0.0
        _show = label_pretty or (_disp3d(label) if label else '')
        self.nodes.append(dict(
            n=n, l=l, l_int=_L_INT4[l],
            J=j_str, J_float=j_float,
            term=term,
            label=label,
            label_pretty=_show,
            energy=float(E_abs),
            source=src,
            x=_L_INT4[l] + _j_offset4(j_float) + _term_jit,
            y=float(n),
            z=float(E_abs),
            tau_ns=self._tau_by_label.get(label, 0),
            color=_SRC_COLORS4.get(src, '#90CAF9'),
        ))

    def resolve(self, n, l_char, j_val, label):
        """Resolve a transition endpoint to a node index (label first — unique)."""
        if label and label in self._by_label:
            return self._by_label[label]
        cands = [i for i, nd in enumerate(self.nodes) if nd['n'] == n and nd['l'] == l_char]
        if len(cands) == 1:
            return cands[0]
        if j_val is not None:
            jf = _parse_j4(j_val)
            j_matches = [i for i in cands
                         if self.nodes[i].get('J_float') == jf
                         or str(self.nodes[i].get('J') or '') == str(j_val)]
            if len(j_matches) == 1:
                return j_matches[0]
            if j_matches:
                return j_matches[0]
        return None


def _edge_width4(fosc, max_fosc):
    if fosc <= 0: return 1.0
    return 1.0 + 4.0 * (fosc / max_fosc) ** 0.4

def _hex_rgba4(h, a):
    h = h.lstrip('#')
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f'rgba({r},{g},{b},{a:.2f})'
_N_GRID_LINES = 8   # lines in each direction 

# Node traces — markers+text (Scatter3d mode='text' alone often fails in WebGL)
# Label low-n NIST/QDT states; leave high-n unlabeled to reduce clutter.
_LABEL_N_MAX4 = 6

# Inject fullscreen CSS + filter panel


def plot_3d(inputs: PlotInputs, output_dir) -> None:
    """Write <El>_3d_visualization.html."""
    import plotly.graph_objects as go

    element = inputs.element
    rydberg_data = inputs.rydberg
    rydberg_excitations = inputs.rydberg_excitations
    transitions_data = inputs.transitions
    lifetimes_by_state = inputs.lifetimes_by_state
    fosc_by_transition = inputs.fosc_by_transition
    species_label = inputs.species_label
    is_ion_species = inputs.is_ion
    _edges4 = []
    _gx, _gy, _gz = [], [], []
    _n_edges_by_region4 = {}

    print("\n Creating 3D Grotrian diagram...")

    # IE for the 3D plot 
    _IE4 = ionization_energy_eV(rydberg_data)
    _gs_n4 = rydberg_data.get('n_start', 2) if rydberg_data else 2
    _n_max4 = max((ex.get('n', 0) for ex in rydberg_excitations), default=20)

    _tau_by_label4 = {lt['state']: lt.get('lifetime_ns', 0)
                      for lt in lifetimes_by_state.values()}
    _nodes4_table = _Nodes4(_tau_by_label4)
    _nodes4 = _nodes4_table.nodes

    # Ground state is required for transitions to the lowest level (e.g. np → 3s).
    # transitions.py injects it; plot 4 must do the same or entire regions (UV) vanish.
    _gs_meta4 = rydberg_data.get('ground_state') if rydberg_data else None
    if _gs_meta4:
        _gs_lab4 = _gs_meta4.get('label') or f"{_gs_meta4['n']}{_gs_meta4['l']}"
        _nodes4_table.add(_gs_meta4['n'], _gs_meta4['l'], _gs_meta4.get('J'),
                   _gs_lab4, -_IE4, 'NIST', term=_gs_meta4.get('term'),
                   label_pretty=_disp3d(_gs_lab4))
    else:
        _nodes4_table.add(_gs_n4, 's', '1/2', f'{_gs_n4}s', -_IE4, 'NIST',
                   label_pretty=_disp3d(f'{_gs_n4}s'))

    for ex in rydberg_excitations:
        n     = ex.get('n')
        l     = ex.get('l')
        j_str = ex.get('J')
        label = ex.get('label', f'{n}{l}')
        E_abs = ex.get('energy_eV')
        src   = ex.get('source', 'QDT model')
        if n is None or l is None or E_abs is None:
            continue
        _nodes4_table.add(n, l, j_str, label, E_abs, src, term=ex.get('term'),
                   label_pretty=_disp3d(label))

    # Node size ∝ log(tau)
    _max_tau4 = max((nd['tau_ns'] for nd in _nodes4 if nd['tau_ns'] > 0), default=0)
    for nd in _nodes4:
        if _max_tau4 > 0 and nd['tau_ns'] > 0:
            nd['size'] = 5 + 10 * math.log10(nd['tau_ns'] + 1) / math.log10(_max_tau4 + 1)
        else:
            nd['size'] = 7
    _max_fosc4 = max((v for v in fosc_by_transition.values()), default=1) or 1

    for t in transitions_data:
        un   = t.get('upper_n', 0)
        ln   = t.get('lower_n', 0)
        ul_l = t.get('upper_l', '')
        ll_l = t.get('lower_l', '')
        ul   = t.get('upper_label', '')
        ll   = t.get('lower_label', '')
        uj   = t.get('upper_J')
        lj   = t.get('lower_J')
        wl   = t.get('wavelength_nm', 0)
        dE   = t.get('delta_E_eV', 0)
        region = t.get('region', 'Visible')

        u_idx = _nodes4_table.resolve(un, ul_l, uj, ul)
        l_idx = _nodes4_table.resolve(ln, ll_l, lj, ll)
        if u_idx is None or l_idx is None:
            continue

        fosc = fosc_by_transition.get((ul, ll), 0)
        _edges4.append(dict(
            u_idx=u_idx, l_idx=l_idx,
            upper_label=ul, lower_label=ll,
            wl=wl, region=region, fosc=fosc, dE=dE,
            upper_source=t.get('upper_source', ''),
            lower_source=t.get('lower_source', ''),
        ))

    print(f"  3D nodes: {len(_nodes4)}  edges: {len(_edges4)}")

    # Build fig4
    fig4 = go.Figure()

    # Ionization limit drawn as a sparse grid of lines instead of a Surface.
    # A Surface trace captures every hover event across its entire area, blocking
    # node hovers when viewed from above. Scatter3d lines have no hit area between
    # segments, so hover passes through to nodes underneath.
    _x_grid_lo, _x_grid_hi = -0.5, float(len(_L_ORDER4) - 0.5)
    # Include true ground n (ions: n_start may be above closed-shell ground)
    _y_grid_lo = float(min(
        (_gs_meta4.get('n') if _gs_meta4 else _gs_n4), _gs_n4))
    _y_grid_hi = float(_n_max4)
    # Horizontal lines (along x at fixed y)
    for _yi in np.linspace(_y_grid_lo, _y_grid_hi, _N_GRID_LINES):
        _gx += [_x_grid_lo, _x_grid_hi, None]
        _gy += [_yi, _yi, None]
        _gz += [0.0, 0.0, None]
    # Vertical lines (along y at fixed x)
    for _xi in np.linspace(_x_grid_lo, _x_grid_hi, _N_GRID_LINES):
        _gx += [_xi, _xi, None]
        _gy += [_y_grid_lo, _y_grid_hi, None]
        _gz += [0.0, 0.0, None]

    fig4.add_trace(go.Scatter3d(
        x=_gx, y=_gy, z=_gz,
        mode='lines',
        line=dict(color='rgba(220,50,50,0.35)', width=1),
        name='Ionization limit',
        showlegend=True,
        hoverinfo='skip',   # never intercepts hover
    ))

    # Edge traces - All hidden by default (visible=False) for fast initial load
    _region_legend4 = set()
    for e in sorted(_edges4, key=lambda x: x['fosc']):   # weak first
        nd_u = _nodes4[e['u_idx']]
        nd_l = _nodes4[e['l_idx']]
        region = e['region']
        col    = _REGION_COLORS4.get(region, '#888')
        op     = 0.15 + 0.75 * (math.log10(e['fosc'] / _max_fosc4 + 1) / math.log10(2)
                                 if e['fosc'] > 0 else 0)
        w      = _edge_width4(e['fosc'], _max_fosc4)
        show_leg = region not in _region_legend4
        _region_legend4.add(region)
        wl_str   = f"{e['wl']:.1f}" if e['wl'] < 1e5 else f"{e['wl']/1e3:.1f}k"
        fosc_str = _fmt_f(e['fosc']) if e['fosc'] > 0 else '—'
        _ul_show = _disp3d(e["upper_label"])
        _ll_show = _disp3d(e["lower_label"])
        fig4.add_trace(go.Scatter3d(
            x=[nd_l['x'], nd_u['x']],
            y=[nd_l['y'], nd_u['y']],
            z=[nd_l['z'], nd_u['z']],
            mode='lines',
            line=dict(color=_hex_rgba4(col, op), width=w),
            name=region,
            legendgroup=f'g3d_region_{region}',
            showlegend=show_leg,
            visible=False,   # hidden by default for fast load
            hovertemplate=(
                f'<b>{_ul_show} → {_ll_show}</b><br>'
                f'λ = {wl_str} nm<br>'
                f'ΔE = {e["dE"]:.4f} eV<br>'
                f'fosc = {fosc_str}<br>'
                f'Region: {region}<extra></extra>'
            ),
        ))

    _n_edge_traces4 = len(_edges4)
    for src in ['NIST', 'QDT model']:
        grp = [nd for nd in _nodes4 if nd['source'] == src]
        if not grp:
            continue
        col = _SRC_COLORS4.get(src, '#90CAF9')
        texts = [
            (nd.get('label_pretty') or _disp3d(nd['label']))
            if nd['n'] <= _LABEL_N_MAX4 else ''
            for nd in grp
        ]
        fig4.add_trace(go.Scatter3d(
            x=[nd['x'] for nd in grp],
            y=[nd['y'] for nd in grp],
            z=[nd['z'] for nd in grp],
            mode='markers+text',
            marker=dict(
                size=[nd['size'] for nd in grp],
                color=col, opacity=0.90,
                line=dict(color='rgba(255,255,255,0.3)', width=1),
            ),
            text=texts,
            textfont=dict(size=11, color='#FFE082', family='Arial, sans-serif'),
            textposition='top center',
            name=src,
            legendgroup=f'g3d_src_{src}',
            showlegend=True,
            customdata=[[nd['label'],
                         nd.get('label_pretty') or _disp3d(nd['label']),
                         nd['n'], nd['l'],
                         nd['J'] or '—', nd['energy'], nd['tau_ns']] for nd in grp],
            hovertemplate=(
                '<b>%{customdata[1]}</b><br>'
                'n = %{customdata[2]}  l = %{customdata[3]}  J = %{customdata[4]}<br>'
                'E = %{customdata[5]:.5f} eV<br>'
                'τ = %{customdata[6]:.2f} ns<br>'
                f'[{src}]<extra></extra>'
            ),
        ))
    for e in _edges4:
        _n_edges_by_region4[e['region']] = _n_edges_by_region4.get(e['region'], 0) + 1
    _region_summary4 = '  '.join(
        f"{r}: {n}" for r, n in sorted(_n_edges_by_region4.items(), key=lambda x: -x[1])
    )

    _y_min4 = min((nd['y'] for nd in _nodes4), default=float(_gs_n4)) - 0.5
    _y_max4 = max((nd['y'] for nd in _nodes4), default=float(_n_max4)) + 0.5
    _gs_n_true4 = (_gs_meta4.get('n') if _gs_meta4 else _gs_n4)
    _gs_lab_show4 = (
        _disp3d(_gs_meta4.get('label')) if _gs_meta4 and _gs_meta4.get('label')
        else f'{_gs_n_true4}s'
    )
    _title_sp4 = species_label if is_ion_species else element

    fig4.update_layout(
        title=dict(
            text=(f'<b>{_title_sp4} — 3D Grotrian Diagram</b>'
                  f'<br><sup>{len(_nodes4)} states · {len(_edges4)} transitions · {_region_summary4}'
                  f' · Enable transitions in filter panel →</sup>'),
            x=0.5, xanchor='center',
            font=dict(size=17, color='#f0f0f0'),
        ),
        scene=dict(
            xaxis=dict(
                title=dict(text='Orbital  l', font=dict(color='#ccc', size=12)),
                tickvals=list(range(len(_L_ORDER4))),
                ticktext=[f'{i} ({_L_ORDER4[i]})' for i in range(len(_L_ORDER4))],
                gridcolor='#2a2a3a', backgroundcolor='#0d0d1a',
                zerolinecolor='#333', tickfont=dict(color='#aaa', size=10),
                range=[-0.6, len(_L_ORDER4) - 0.4],
            ),
            yaxis=dict(
                title=dict(text='Principal quantum number  n', font=dict(color='#ccc', size=12)),
                gridcolor='#2a2a3a', backgroundcolor='#0d0d1a',
                zerolinecolor='#333', tickfont=dict(color='#aaa', size=10),
                range=[_y_min4, _y_max4],
            ),
            zaxis=dict(
                title=dict(text='Energy (eV)', font=dict(color='#ccc', size=12)),
                gridcolor='#2a2a3a', backgroundcolor='#0d0d1a',
                zerolinecolor='rgba(220,50,50,0.6)', tickfont=dict(color='#aaa', size=10),
            ),
            bgcolor='#0d0d1a',
            camera=dict(eye=dict(x=1.8, y=-1.6, z=0.9), up=dict(x=0, y=0, z=1)),
            aspectmode='manual',
            aspectratio=dict(x=1.0, y=1.6, z=1.2),
            annotations=[
                dict(x=2.0, y=float(_n_max4), z=0.0, text='Ionization limit',
                     showarrow=False, font=dict(color='rgba(220,80,80,0.8)', size=11),
                     xanchor='center'),
                dict(x=_L_INT4.get((_gs_meta4 or {}).get('l', 's'), 0),
                     y=float(_gs_n_true4), z=-_IE4,
                     text=f'Ground ({_gs_lab_show4})',
                     showarrow=False, font=dict(color='rgba(100,200,255,0.9)', size=11),
                     xanchor='center'),
            ],
        ),
        paper_bgcolor='#0a0a14',
        font=dict(family="'Helvetica Neue', Arial, sans-serif", color='#ddd'),
        autosize=True,
        legend=dict(
            title=dict(text='<b>Region / Source</b>', font=dict(color='#ccc')),
            bgcolor='rgba(15,15,30,0.85)', bordercolor='#333', borderwidth=1,
            font=dict(color='#ccc', size=11),
            x=0.01, xanchor='left', y=0.99, yanchor='top',
            itemclick='toggle',
            itemdoubleclick=False,   # handled manually in JS below
        ),
        margin=dict(l=0, r=0, t=75, b=0),
    )

    html_file4 = f"{output_dir}/{element}_3d_visualization.html"

    _edges_js4 = json.dumps([
        dict(
            i=i+1,   # trace index (0 = surface, 1..N = edges)
            ul=e['upper_label'], ll=e['lower_label'],
            ulp=_disp3d(e['upper_label']), llp=_disp3d(e['lower_label']),
            wl=round(e['wl'], 2), dE=round(e['dE'], 5),
            fosc=round(e['fosc'], 6), region=e['region'],
            us=e.get('upper_source', ''), ls=e.get('lower_source', ''),
        )
        for i, e in enumerate(sorted(_edges4, key=lambda x: x['fosc']))
    ])
    _rc_js4       = json.dumps(_REGION_COLORS4)
    _gs_n4_str    = str(_gs_n4)
    _n_max4_str   = str(_n_max4)
    _n_edge4_str  = str(_n_edge_traces4)



    _html4 = fig4.to_html(include_plotlyjs=False, full_html=True, config={'responsive': True})
    _html4 = _html4.replace('<html>', '<html style="width:100%;height:100%;margin:0;padding:0;">', 1)
    _html4 = _html4.replace('<body>', '<body style="width:100%;height:100%;margin:0;padding:0;background:#0a0a14;">', 1)
    _html4 = inject_head(_html4, load('plotinteractive_levels4.css'))
    _html4 = inject_body(
        _html4,
        render_template(
            'plotinteractive_levels4.html',
            gs_n4_str=_gs_n4_str,
            n_max4_str=_n_max4_str,
        )
        + render_template(
            'plotinteractive_levels4.js',
            edges_js4=_edges_js4,
            rc_js4=_rc_js4,
            n_edge4_str=_n_edge4_str,
            gs_n4_str=_gs_n4_str,
            n_max4_str=_n_max4_str,
            region_order_js=_region_order_js(),
        ))
    write_offline_report(html_file4, _html4)

    print(f" Saved: {html_file4}")
