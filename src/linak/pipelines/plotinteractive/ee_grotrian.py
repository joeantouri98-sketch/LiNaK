"""Plot 3: the Grotrian diagram in orbital-l columns (ground + excited→excited lines).

Layout: x = orbital type (s|p|d|f|g columns), y = energy on the plot-1 axis.
Levels: NIST/Rydberg states in each l column (coloured by n, symbol by J).
Arrows: E1 transitions (upper → lower), width ∝ log f, hidden until the panel
applies a filter.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict

import numpy as np

from linak.report_templates import (
    inject_body,
    inject_head,
    load,
    render as render_template,
    write_offline_report,
)

from .common import REGION_COLORS, _region_order_js, ionization_energy_eV
from .inputs import PlotInputs
from .view import View


def plot_ee_grotrian(inputs: PlotInputs, view: View, plot3_choice, output_dir) -> None:
    """Write <El>_ee_grotrian.html (skipped without rydberg data)."""
    import plotly.express as px
    import plotly.graph_objects as go

    element = inputs.element
    rydberg_data = inputs.rydberg
    transitions_data = inputs.transitions
    lifetimes_by_state = inputs.lifetimes_by_state
    species_label = inputs.species_label
    is_ion_species = inputs.is_ion
    _disp = inputs.disp
    _gs_meta_sp = inputs.gs_meta
    _gs_label_sp = inputs.gs_label
    _n_start_sp = inputs.n_start
    energy_shift = view.energy_shift
    shift_label = view.shift_label

    print("\n Creating excited→excited Grotrian (l-column layout)...")

    if not rydberg_data:
        print("  ⚠  No Rydberg/NIST data — skipping Plot 3")
    else:
        IE_ryd = ionization_energy_eV(rydberg_data)

        _L_ORDER  = ['s', 'p', 'd', 'f', 'g']
        _L_LABELS = {'s': 's  (l=0)', 'p': 'p  (l=1)', 'd': 'd  (l=2)',
                     'f': 'f  (l=3)', 'g': 'g  (l=4)'}
        _L_COL_CTR = {l: i for i, l in enumerate(_L_ORDER)}
        _COL_W = 0.8

        # Build level table — key by unique ASCII label (not (n,l,J); multiplets collide)
        _nist_levels = {}   # label -> level dict
        _label_to_exc = {}  # label -> excitation energy (eV above ground)
        if rydberg_data:
            for ex in rydberg_data.get('excitations', []):
                n   = ex['n']
                l   = ex['l']
                J   = ex.get('J') or ''
                lbl = ex.get('label', f"{n}{l}")
                exc = IE_ryd + ex['energy_eV']
                if exc <= 0 or exc >= IE_ryd:
                    continue
                lt_entry = lifetimes_by_state.get(lbl)
                tau_ns  = lt_entry.get('lifetime_ns', 0) if lt_entry else 0
                tau_src = lt_entry.get('dominant_source', '') if lt_entry else ''
                _nist_levels[lbl] = dict(
                    exc=exc, label=lbl,
                    label_pretty=ex.get('label_pretty') or _disp(lbl),
                    term=ex.get('term'),
                    tau_ns=tau_ns, tau_src=tau_src,
                    J=J, n=n, l=l,
                )
                _label_to_exc[lbl] = exc

        # Ground state as a real level in its l column (ions: 2p, not ns)
        if _gs_meta_sp and _gs_label_sp:
            _gs_l_ee = _gs_meta_sp.get('l') or 's'
            _nist_levels[_gs_label_sp] = dict(
                exc=0.0, label=_gs_label_sp,
                label_pretty=(_gs_meta_sp.get('label_pretty')
                              or _disp(_gs_label_sp)),
                term=_gs_meta_sp.get('term'),
                tau_ns=0, tau_src='',
                J=_gs_meta_sp.get('J') or '',
                n=_gs_meta_sp.get('n', _n_start_sp),
                l=_gs_l_ee,
                is_ground=True,
            )
            _label_to_exc[_gs_label_sp] = 0.0

        # Horizontal jitter so multiplet members at same (n,l) do not stack
        _nl_groups = defaultdict(list)
        for _nv in _nist_levels.values():
            _nl_groups[(_nv['n'], _nv['l'])].append(_nv)
        for _grp in _nl_groups.values():
            _grp.sort(key=lambda v: (v['exc'], v['label']))
            _n_g = len(_grp)
            for _i, _nv in enumerate(_grp):
                _nv['x_off'] = (_i - (_n_g - 1) / 2.0) * 0.055

        # Collect transitions (gs + ee) 
        _ee_all = []
        if inputs.has_lifetimes:
            for _t in inputs.lifetimes_transitions:
                if _t.get('fosc') is None or _t.get('fosc', 0) <= 0:
                    continue
                _ee_all.append(_t)
            _n_gs = sum(1 for t in _ee_all if t.get('lower_label') == _gs_label_sp)
            print(f"  Loaded {len(_ee_all)} transitions from lifetimes.json "
                  f"({_n_gs} gs→X, {len(_ee_all) - _n_gs} ee)")
        elif transitions_data:
            for _t in transitions_data:
                _ee_all.append({
                    'upper_label': _t['upper_label'],
                    'lower_label': _t['lower_label'],
                    'upper_n': _t.get('upper_n', 0),
                    'lower_n': _t.get('lower_n', 0),
                    'upper_l': _t.get('upper_l', ''),
                    'lower_l': _t.get('lower_l', ''),
                    'wavelength_nm': _t.get('wavelength_nm', 0),
                    'delta_E_eV': _t.get('delta_E_eV', 0),
                    'region': _t.get('region', ''),
                    'fosc': 1.0,
                    'fosc_source': 'topology',
                    'fosc_acc': None,
                })
            _n_gs = sum(1 for t in _ee_all if t.get('lower_label') == _gs_label_sp)
            print(f"  Loaded {len(_ee_all)} transitions from transitions.json "
                  f"({_n_gs} gs→X, {len(_ee_all) - _n_gs} ee, no fosc)")

        _ee_nist   = [t for t in _ee_all if t.get('fosc_source') in ('NIST', 'precision')]
        _ee_lit    = [t for t in _ee_all if t.get('fosc_source') == 'literature']
        _ee_topo   = [t for t in _ee_all if t.get('fosc_source') == 'topology']
        _ee_num    = [t for t in _ee_all if t.get('fosc_source') == 'Numerov']
        _ee_coul   = [t for t in _ee_all if t.get('fosc_source') == 'Coulomb']
        _ee_coul_a = [t for t in _ee_all if t.get('fosc_source') == 'Coulomb(approx)']
        _ee_nxt    = [t for t in _ee_all if t.get('fosc_source') == 'NIST-extrap']
        _ee_unc    = [t for t in _ee_all if t.get('fosc_source') == 'unclassified']
        _gs_nist   = [t for t in _ee_nist if t.get('lower_label') == _gs_label_sp]
        _ee_only   = [t for t in _ee_nist if t.get('lower_label') != _gs_label_sp]

        print("\n" + "="*70)
        print("PLOT 3: GROTRIAN DISPLAY OPTIONS (ground + excited→excited)")
        print("="*70)
        print(f"  NIST gs→X transitions:        {len(_gs_nist)}")
        print(f"  NIST ee transitions:          {len(_ee_only)}")
        print(f"  literature:                   {len(_ee_lit)}")
        print(f"  NIST-extrap:                  {len(_ee_nxt)}")
        print(f"  topology (no fosc):           {len(_ee_topo)}")
        print(f"  Numerov ee/gs:                {len(_ee_num)}")
        print(f"  Coulomb(approx) ee:           {len(_ee_coul_a)}")
        print(f"  Coulomb ee:                   {len(_ee_coul)}")

        _ee_choice = plot3_choice

        def _ee_n_ok(t, n_cap):
            return max(t.get('upper_n', 0), t.get('lower_n', 0)) <= n_cap

        if _ee_choice == "2":
            _theo_pick = [t for t in _ee_num if _ee_n_ok(t, 8)]
            visible_ee = _ee_nist + _ee_lit + _ee_nxt + _ee_topo + _theo_pick
        elif _ee_choice == "3":
            _theo_pick = [t for t in _ee_num if _ee_n_ok(t, 12)]
            visible_ee = _ee_nist + _ee_lit + _ee_nxt + _ee_topo + _theo_pick
        elif _ee_choice == "4":
            _theo_all = sorted(_ee_num + _ee_coul + _ee_coul_a,
                               key=lambda t: t.get('fosc', 0), reverse=True)
            visible_ee = _ee_nist + _ee_lit + _ee_nxt + _ee_unc + _ee_topo + _theo_all[:300]
        else:
            visible_ee = list(_ee_nist) + list(_ee_lit) + list(_ee_topo)

        # Keep only transitions whose endpoints map to known levels / l columns
        _gs_l_char = (_gs_meta_sp.get('l') or 's') if _gs_meta_sp else 's'

        def _resolve_ee(t):
            ul, ll = t['upper_label'], t['lower_label']
            is_gs = (ll == _gs_label_sp)
            u_exc = _label_to_exc.get(ul)
            l_exc = 0.0 if is_gs else _label_to_exc.get(ll)
            if u_exc is None and l_exc is not None and not is_gs:
                u_exc = l_exc + t.get('delta_E_eV', 0)
            if l_exc is None and u_exc is not None and not is_gs:
                l_exc = u_exc - t.get('delta_E_eV', 0)
            if is_gs and u_exc is None:
                u_exc = t.get('delta_E_eV', 0)
            ul_char = t.get('upper_l') or ''
            ll_char = (t.get('lower_l') or _gs_l_char) if is_gs else (t.get('lower_l') or '')
            if not ul_char or not ll_char:
                return None
            if ul_char not in _L_COL_CTR or ll_char not in _L_COL_CTR:
                return None
            if u_exc is None or l_exc is None:
                return None
            if is_gs:
                if u_exc <= 0:
                    return None
            elif u_exc <= l_exc:
                return None
            return dict(t, u_exc=u_exc, l_exc=l_exc, ul_char=ul_char, ll_char=ll_char,
                          kind='gs' if is_gs else 'ee')

        _ee_draw = [_resolve_ee(t) for t in visible_ee]
        _ee_draw = [t for t in _ee_draw if t]
        _n_gs_draw = sum(1 for t in _ee_draw if t['kind'] == 'gs')
        print(f" Will register {len(_ee_draw)} transitions on Plot 3 "
              f"({_n_gs_draw} gs→X, {len(_ee_draw) - _n_gs_draw} ee)")
        print("="*70)

        # Colour scale: n quantum number 
        _all_ns = sorted(set(v['n'] for v in _nist_levels.values()))
        _n_min, _n_max = (min(_all_ns), max(_all_ns)) if _all_ns else (3, 20)
        _N_CMAP = px.colors.sequential.Viridis

        def _n_color(n):
            if _n_max == _n_min:
                return _N_CMAP[len(_N_CMAP) // 2]
            idx = int((n - _n_min) / (_n_max - _n_min) * (len(_N_CMAP) - 1))
            return _N_CMAP[max(0, min(idx, len(_N_CMAP) - 1))]

        # Half-integer (alkali) + integer J (closed-shell / multiplet ions)
        _J_SYMBOLS = {
            '0': 'circle-open', '1': 'circle', '2': 'square', '3': 'diamond',
            '4': 'triangle-up', '5': 'star', '6': 'hexagon',
            '1/2': 'circle', '3/2': 'square', '5/2': 'diamond',
            '7/2': 'triangle-up', '9/2': 'star', '': 'circle-open',
        }

        _EE_SRC_DASH = {
            'NIST': 'solid', 'precision': 'solid', 'literature': 'solid',
            'ORCA': 'solid', 'EOM-CCSD': 'solid',
            'Numerov': 'dash', 'NIST-extrap': 'dash', 'unclassified': 'dot',
            'Coulomb': 'dash', 'Coulomb(approx)': 'dot', 'topology': 'dot',
        }

        def _ee_plot_y(e_exc):
            """Map excitation energy onto the active energy axis (matches plot 1)."""
            return float(e_exc) + float(energy_shift)

        _y_ie3 = _ee_plot_y(IE_ryd)
        _y_gs3 = _ee_plot_y(0.0)

        fig3 = go.Figure()

        # Draw levels 
        _level_legend_l = set()
        for nv in sorted(_nist_levels.values(),
                         key=lambda x: (x['l'], x['n'], x['exc'], x['label'])):
            n, l, J = nv['n'], nv['l'], nv.get('J') or ''
            if l not in _L_COL_CTR:
                continue
            x_ctr = _L_COL_CTR[l] + nv.get('x_off', 0.0)
            y_pos = _ee_plot_y(nv['exc'])
            col   = '#222222' if nv.get('is_ground') else _n_color(n)
            sym   = _J_SYMBOLS.get(str(J), 'circle')
            show_leg = l not in _level_legend_l
            _level_legend_l.add(l)
            tau_str = (f"<br>τ = {nv['tau_ns']:.2f} ns [{nv['tau_src']}]"
                       if nv['tau_ns'] > 0 else '')
            term_str = f", term={nv['term']}" if nv.get('term') else ''
            hover = (f"<b>{nv.get('label_pretty') or _disp(nv['label'])}</b><br>"
                     f"{shift_label} E = {y_pos:.4f} eV<br>"
                     f"n={n}, l={l}, J={J or '?'}{term_str}{tau_str}<extra></extra>")
            _hw = 0.12 if nv.get('is_ground') else 0.16
            fig3.add_trace(go.Scatter(
                x=[x_ctr - _hw, x_ctr + _hw], y=[y_pos, y_pos],
                mode='lines+markers',
                line=dict(color=col, width=3.0 if nv.get('is_ground') else 2.5),
                marker=dict(symbol=[sym, sym], size=7 if nv.get('is_ground') else 6,
                            color=col),
                name=f'{l}-states', legendgroup=f'level_{l}',
                showlegend=show_leg, hovertemplate=hover,
                customdata=[[nv['label']]],
            ))

        # Register ee arrows (hidden by default) 
        _fosc_max = max((t['fosc'] for t in _ee_draw), default=1) or 1
        _fosc_min = min((t['fosc'] for t in _ee_draw if t['fosc'] > 0), default=1e-12)
        _log_mx = math.log10(max(_fosc_max, 1e-12))
        _log_mn = math.log10(max(_fosc_min, 1e-12))
        _log_rng = max(_log_mx - _log_mn, 1.0)

        _n_level_traces3 = len(fig3.data)
        _ee_edges_js = []
        np.random.seed(42)
        for _t in sorted(_ee_draw, key=lambda x: (x.get('region', ''), x['kind'], x['wavelength_nm'])):
            src = _t.get('fosc_source', 'topology')
            region = _t.get('region') or 'Unknown'
            color = REGION_COLORS.get(region, '#999999')
            dash = _EE_SRC_DASH.get(src, 'dot')
            fosc = _t['fosc']
            _log_f = math.log10(max(fosc, 1e-12))
            lw = max(0.8, min(0.8 + ((_log_f - _log_mn) / _log_rng) * 3.2, 4.0))

            # Prefer level x_off so arrows meet the correct multiplet member
            _u_nv = _nist_levels.get(_t['upper_label'], {})
            _l_nv = _nist_levels.get(_t['lower_label'], {})
            x_u = (_L_COL_CTR[_t['ul_char']] + _u_nv.get('x_off', 0.0)
                   + np.random.uniform(-0.02, 0.02))
            x_l = (_L_COL_CTR[_t['ll_char']] + _l_nv.get('x_off', 0.0)
                   + np.random.uniform(-0.02, 0.02))
            y_u, y_l = _ee_plot_y(_t['u_exc']), _ee_plot_y(_t['l_exc'])
            wl = _t.get('wavelength_nm', 0)
            acc = _t.get('fosc_acc') or '—'
            ltype = f"{_t['ll_char']}\u2192{_t['ul_char']}"
            kind = _t['kind']

            trace_i = len(fig3.data)
            hover = (
                f"<b>{_disp(_t['upper_label'])} → {_disp(_t['lower_label'])}</b><br>"
                f"λ = {wl:.2f} nm<br>"
                f"ΔE = {_t.get('delta_E_eV', _t['u_exc'] - _t['l_exc']):.4f} eV<br>"
                f"Region: {region}<br>"
                f"fosc = {fosc:.5f}  [{src}]<br>"
                f"Acc: {acc}<extra></extra>"
            )
            fig3.add_trace(go.Scatter(
                x=[x_u, x_l], y=[y_u, y_l],
                mode='lines+markers',
                line=dict(color=color, width=lw, dash=dash),
                marker=dict(symbol=['circle', 'arrow-down'], size=[0, 7], color=color),
                name=region,
                legendgroup=f'reg_{region}',
                showlegend=False,
                visible=False,
                hovertemplate=hover,
                customdata=[[_t['upper_label'], _t['lower_label']]],
            ))
            _ee_edges_js.append(dict(
                i=trace_i,
                ul=_t['upper_label'], ll=_t['lower_label'],
                ulp=_disp(_t['upper_label']), llp=_disp(_t['lower_label']),
                un=int(_t.get('upper_n') or 0), ln=int(_t.get('lower_n') or 0),
                ul_l=_t['ul_char'], ll_l=_t['ll_char'],
                ltype=ltype, kind=kind,
                wl=round(float(wl), 2),
                dE=round(float(_t.get('delta_E_eV', y_u - y_l)), 5),
                fosc=round(float(fosc), 8),
                reg=region,
                src=src,
            ))

        _ee_edge_start3 = _n_level_traces3
        _n_ee_edges3 = len(_ee_edges_js)
        print(f" Plot 3: {_n_ee_edges3} transitions registered (hidden until Apply in panel)")

        # Region legend swatches (one dummy trace per region present) 
        _reg_legend_done = set()
        for _t in _ee_draw:
            region = _t.get('region') or 'Unknown'
            if region in _reg_legend_done:
                continue
            _reg_legend_done.add(region)
            fig3.add_trace(go.Scatter(
                x=[None], y=[None], mode='lines',
                line=dict(color=REGION_COLORS.get(region, '#999'), width=3),
                name=region, legendgroup=f'reg_{region}', showlegend=True,
            ))

        # Column labels 
        for l, ctr in _L_COL_CTR.items():
            fig3.add_annotation(
                x=ctr, y=_y_ie3 + 0.15 * (1 if shift_label != 'Absolute' else -1),
                xref='x', yref='y',
                text=f'<b>{_L_LABELS[l]}</b>',
                showarrow=False, font=dict(size=13, color='#333'),
            )
            if ctr < len(_L_ORDER) - 1:
                fig3.add_vline(x=ctr + _COL_W / 2, line_dash='dot',
                               line_color='#ddd', line_width=1)

        _ie_ann3 = ('Ionization (0 eV)' if shift_label == 'Absolute'
                    else f'Ionization  {IE_ryd:.3f} eV')
        fig3.add_hline(y=_y_ie3, line_dash='dash', line_color='red', line_width=1.5,
                       annotation_text=_ie_ann3,
                       annotation_position='right',
                       annotation_font=dict(color='red', size=11))
        fig3.add_hline(y=_y_gs3, line_dash='solid', line_color='black', line_width=2,
                       annotation_text=f'Ground ({species_label} {_disp(_gs_label_sp)})',
                       annotation_position='right',
                       annotation_font=dict(color='black', size=11))

        # n colourbar 
        fig3.add_trace(go.Scatter(
            x=[None] * len(_all_ns), y=[None] * len(_all_ns),
            mode='markers',
            marker=dict(
                color=_all_ns, colorscale='Viridis',
                cmin=_n_min, cmax=_n_max, showscale=True,
                colorbar=dict(title=dict(text='n', side='right'),
                              len=0.6, y=0.5, x=1.02, thickness=12),
                size=0,
            ),
            showlegend=False, hoverinfo='skip',
        ))

        _x_range = [-0.55, len(_L_ORDER) - 0.45]
        _y_lo3 = min(_y_gs3, _y_ie3) - 0.3
        _y_hi3 = max(_y_gs3, _y_ie3) + 0.4
        _title_sp3 = species_label if is_ion_species else element
        fig3.update_layout(
            title=dict(
                text=(f'<b>{_title_sp3} — Grotrian (l columns)</b>'
                      f'<br><sub>'
                      f'Columns = orbital l • Levels coloured by n • '
                      f'Multiplet terms kept separate • '
                      f'Arrows coloured by spectral region • '
                      f'Use filter panel → Apply • '
                      f'solid = NIST/literature fosc, dashed = Numerov/Coulomb</sub>'),
                x=0.5, xanchor='center', font=dict(size=16),
            ),
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                       range=_x_range),
            yaxis=dict(
                title=dict(text=f'<b>{shift_label} Energy (eV)</b>', font=dict(size=14)),
                showgrid=True, gridcolor='#eee',
                range=[_y_lo3, _y_hi3],
            ),
            hovermode='closest',
            plot_bgcolor='white',
            autosize=True,
            margin=dict(r=180, t=120, b=40),
            legend=dict(
                title='<b>Series</b>',
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#ccc', borderwidth=1,
                x=1.08, xanchor='left', y=1, yanchor='top',
                font=dict(size=11), tracegroupgap=4,
            ),
        )

        html_file3 = f"{output_dir}/{element}_ee_grotrian.html"
        _raw_html3 = fig3.to_html(include_plotlyjs=False, full_html=True)

        _EE_EDGES_JS = json.dumps(_ee_edges_js)
        _EE_RC_JS = json.dumps(REGION_COLORS)
        _EE_SRC_COLORS_JS = json.dumps({
            'NIST': '#1a6eb5', 'precision': '#1a6eb5', 'literature': '#c47a00',
            'ORCA': '#c0392b', 'EOM-CCSD': '#c0392b',
            'Numerov': '#0e7c6b', 'NIST-extrap': '#5b9bd5', 'unclassified': '#7f8c8d',
            'Coulomb': '#27ae60', 'Coulomb(approx)': '#52be80',
            'topology': '#888888',
        })





        # The EE fullscreen strip is byte-identical to the grotrian one —
        # single template file (one place for each thing).
        _raw_html3 = inject_head(
            _raw_html3,
            load('plotinteractive_grot_fs.css') + load('plotinteractive_ee_panel.css'))
        _raw_html3 = inject_body(
            _raw_html3,
            load('plotinteractive_ee_panel.html')
            + render_template(
                'plotinteractive_ee_panel.js',
                EE_EDGES_JS=_EE_EDGES_JS,
                EE_RC_JS=_EE_RC_JS,
                EE_SRC_COLORS_JS=_EE_SRC_COLORS_JS,
                ee_edge_start3=str(_ee_edge_start3),
                n_ee_edges3=str(_n_ee_edges3),
                region_order_js=_region_order_js(),
            ))
        write_offline_report(html_file3, _raw_html3)
        print(f" Saved: {html_file3}")
