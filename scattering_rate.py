#!/usr/bin/env python3
"""
scattering_rate.py
------------------
Generates an interactive scattering rate and radiation pressure widget
for Group 1 alkali atoms using Plotly.

Physics:
    R_scatt(I, Δ) = (Γ/2) · s / (1 + s + (2Δ/Γ)²)
    where s = I/I_sat (saturation parameter)
    F_rad = ħk · R_scatt
    T_Doppler = ħΓ / (2k_B)    [Doppler cooling limit]
    v_capture ≈ Γ/k             [capture velocity]

Data source: data_json/comparison_table.json (from compare_elements.py)
Falls back to hardcoded values if JSON not found.

Output: plots/scattering_rate.html

Usage:
    python scattering_rate.py
    python scattering_rate.py --element Na --line D2
    python scattering_rate.py --output-dir plots
"""

import json
import os
import sys
import argparse
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.colors as pc
except ImportError:
    print("Plotly not installed. Run: pip install plotly")
    sys.exit(1)

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(description='Generate scattering rate widget')
parser.add_argument('--element',    default=None, help='Default element (Li/Na/K/Rb/Cs/Fr)')
parser.add_argument('--line',       default='D2', choices=['D1','D2'], help='D1 or D2 line')
parser.add_argument('--data-dir',   default='data_json', help='Data directory')
parser.add_argument('--output-dir', default='plots',     help='Output directory')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

from constants import (
    HBAR_SI as hbar,
    KB_SI as kB,
    AMU_KG as amu,
    C_SI as c,
)
pi = np.pi

# ============================================================================
# ELEMENT DATA
# ============================================================================

# Fallback hardcoded data (used if comparison_table.json not found)
FALLBACK = {
    'Li': {'mass':6.941,
           'D1':{'wl':670.776,'A':3.690e7,'Isat':7.630,'tau':27.10,'vrec':8.571,'Erec':3066.0},
           'D2':{'wl':670.776,'A':3.690e7,'Isat':3.815,'tau':27.10,'vrec':8.571,'Erec':3066.0}},
    'Na': {'mass':22.990,
           'D1':{'wl':589.755,'A':6.134e7,'Isat':3.960,'tau':16.302,'vrec':2.946,'Erec':1199.9},
           'D2':{'wl':589.158,'A':6.147e7,'Isat':3.980,'tau':16.269,'vrec':2.950,'Erec':1199.9}},
    'K':  {'mass':39.098,
           'D1':{'wl':770.107,'A':3.729e7,'Isat':5.180,'tau':26.814,'vrec':1.330,'Erec':416.6},
           'D2':{'wl':766.700,'A':3.763e7,'Isat':2.626,'tau':26.577,'vrec':1.330,'Erec':416.6}},
    'Rb': {'mass':85.468,
           'D1':{'wl':794.978,'A':3.607e7,'Isat':4.480,'tau':27.726,'vrec':0.600,'Erec':184.0},
           'D2':{'wl':780.240,'A':3.744e7,'Isat':2.502,'tau':26.707,'vrec':0.600,'Erec':184.0}},
    'Cs': {'mass':132.905,
           'D1':{'wl':894.591,'A':2.850e7,'Isat':2.500,'tau':35.090,'vrec':0.350,'Erec': 99.2},
           'D2':{'wl':852.346,'A':3.139e7,'Isat':1.649,'tau':31.854,'vrec':0.350,'Erec': 99.2}},
    'Fr': {'mass':223.000,
           'D1':{'wl':817.165,'A':3.286e7,'Isat':3.880,'tau':30.431,'vrec':0.220,'Erec': 83.3},
           'D2':{'wl':718.183,'A':4.254e7,'Isat':4.010,'tau':23.505,'vrec':0.250,'Erec': 83.3}},
}

# Try loading from comparison_table.json
ct_file = os.path.join(args.data_dir, 'comparison_table.json')
ELEMENTS = {}

if os.path.exists(ct_file):
    with open(ct_file) as f:
        ct = json.load(f)
    for row in ct.get('elements', []):
        el = row['element']
        ELEMENTS[el] = {'mass': row['mass_amu']}
        for line_key in ['D1', 'D2']:
            d = row.get(line_key)
            if d:
                ELEMENTS[el][line_key] = {
                    'wl':   d['wl_nm'],
                    'A':    d['A_s'],
                    'Isat': d['Isat_nat_mWcm2'],
                    'tau':  d['tau_ns'],
                    'vrec': d.get('v_rec_ms', 1.0),
                    'Erec': d.get('E_rec_nK', 100.0),
                }
    print(f"Loaded {len(ELEMENTS)} elements from {ct_file}")
else:
    ELEMENTS = FALLBACK
    print(f"comparison_table.json not found — using fallback data")

EL_NAMES = ['Li','Na','K','Rb','Cs','Fr']
EL_NAMES = [e for e in EL_NAMES if e in ELEMENTS]
DEFAULT_EL = args.element or 'Rb'
if DEFAULT_EL not in ELEMENTS:
    DEFAULT_EL = EL_NAMES[0]

EL_COLORS = {
    'Li':'#4df0b0', 'Na':'#f0c54d', 'K':'#4d9ff0',
    'Rb':'#f05a4d', 'Cs':'#c54df0', 'Fr':'#f04d9f'
}

# ============================================================================
# PHYSICS FUNCTIONS
# ============================================================================

def params(el, line):
    """Return (Gamma, k, Isat, mass_kg) for an element/line."""
    d = ELEMENTS[el][line]
    G   = d['A']                          # rad/s
    k   = 2*pi / (d['wl'] * 1e-9)        # m^-1
    Is  = d['Isat']                       # mW/cm²
    m   = ELEMENTS[el]['mass'] * amu      # kg
    return G, k, Is, m

def R_scatt(G, Is, I_over_Isat, delta_over_Gamma):
    """Photon scattering rate [s^-1]."""
    s = I_over_Isat
    x = delta_over_Gamma
    return (G/2) * s / (1 + s + 4*x*x)

def F_rad_aN(G, k, Is, I_over_Isat, delta_over_Gamma):
    return hbar * k * R_scatt(G, Is, I_over_Isat, delta_over_Gamma) * 1e18  # aN

def accel_km_s2(G, k, Is, m, I_over_Isat, delta_over_Gamma):
    return hbar * k * R_scatt(G, Is, I_over_Isat, delta_over_Gamma) / m / 1e3

def T_doppler_uK(G):
    return hbar * G / (2*kB) * 1e6

def v_capture(G, k):
    return G / k  # m/s

def recoil_vel(el, line):
    return ELEMENTS[el][line].get('vrec', 1.0)  # cm/s

# ============================================================================
# BUILD ALL TRACES FOR EACH ELEMENT × LINE COMBINATION
# ============================================================================
# Strategy: pre-compute ALL traces for all elements and I-values, then
# use Plotly updatemenus + sliders to show/hide the right ones.
# This makes the final HTML fully interactive without any server.

print("Computing traces...")

# Detuning axis: -10Γ to +5Γ, 500 points
N_det = 500
det_G = np.linspace(-10, 5, N_det)   # in Γ units

# Intensity values for multi-curve plot
I_vals = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
I_labels = ['0.1 Isat','0.5 Isat','1 Isat','2 Isat','5 Isat','10 Isat']
I_colors = ['#1a6eb5','#2980b9','#4df0b0','#f0c54d','#f05a4d','#c54df0']

# Velocity axis: ±4×v_capture (element-dependent, we use Rb as reference width)
N_vel = 400

# ============================================================================
# BUILD FIGURE
# ============================================================================

fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=[
        'R<sub>scatt</sub> vs Detuning (multiple I/I<sub>sat</sub>)',
        'R<sub>scatt</sub> vs Intensity',
        'Force vs Atom Velocity (counter-prop. beams)',
        'Doppler Cooling Force (Δ = −Γ/2, optimal)',
        'All Elements @ Δ=−Γ/2, I=I<sub>sat</sub>',
        'Key Parameters Table',
    ],
    specs=[
        [{'type':'scatter'},{'type':'scatter'},{'type':'scatter'}],
        [{'type':'scatter'},{'type':'scatter'},{'type':'table'}],
    ],
    vertical_spacing=0.14,
    horizontal_spacing=0.08,
)

# ── Colour palette per I-value ────────────────────────────────────────────
I_palette = pc.sample_colorscale('Plasma', len(I_vals))  # always exactly len(I_vals) colors

# ── Plot 1: R_scatt vs detuning, multi-I curves ──────────────────────────
# One set of traces per (element, line) combo, hidden by default except default
for el in EL_NAMES:
    for line in ['D1', 'D2']:
        if line not in ELEMENTS[el]:
            continue
        G, k, Is, m = params(el, line)
        wl_nm = ELEMENTS[el][line]['wl']
        visible = (el == DEFAULT_EL and line == args.line)

        for ii, (I_s, I_lbl, I_col) in enumerate(zip(I_vals, I_labels, I_palette)):
            R = R_scatt(G, Is, I_s, det_G) / 1e6  # Mphot/s
            fig.add_trace(go.Scatter(
                x=det_G, y=R,
                mode='lines',
                line=dict(color=I_col, width=2.5 if I_s==1.0 else 1.2),
                name=I_lbl,
                legendgroup=f'Rdet_{el}_{line}',
                showlegend=(visible and ii < len(I_vals)),
                visible=visible,
                hovertemplate=f'<b>{el} {line}</b> I={I_lbl}<br>Δ=%{{x:.2f}}Γ<br>R=%{{y:.3f}} Mphot/s<extra></extra>',
            ), row=1, col=1)

        # Vertical line at Δ=0 (just once per el/line)
        fig.add_trace(go.Scatter(
            x=[0,0], y=[0, G/2/1e6*1.05],
            mode='lines', line=dict(color='rgba(255,255,255,0.15)', width=1, dash='dot'),
            showlegend=False, visible=visible,
            hoverinfo='skip',
        ), row=1, col=1)

        # Vertical line at Δ=-Γ/2 (optimal detuning)
        fig.add_trace(go.Scatter(
            x=[-0.5,-0.5], y=[0, G/2/1e6*1.05],
            mode='lines', line=dict(color='rgba(77,240,176,0.3)', width=1, dash='dot'),
            showlegend=False, visible=visible,
            hoverinfo='skip',
        ), row=1, col=1)

# ── Plot 2: R_scatt vs I/Isat (saturation curve) ─────────────────────────
I_log = np.logspace(-2, 2, 400)  # 0.01 to 100 × Isat
det_values = [0, -0.5, -1.0, -2.0]
det_colors = ['#4df0b0','#4d9ff0','#f0c54d','#f05a4d']
det_labels = ['Δ=0','Δ=−Γ/2','Δ=−Γ','Δ=−2Γ']

for el in EL_NAMES:
    for line in ['D1','D2']:
        if line not in ELEMENTS[el]:
            continue
        G, k, Is, m = params(el, line)
        visible = (el == DEFAULT_EL and line == args.line)

        for dG, dcol, dlbl in zip(det_values, det_colors, det_labels):
            R = R_scatt(G, Is, I_log, dG) / 1e6
            fig.add_trace(go.Scatter(
                x=I_log * Is,   # mW/cm²
                y=R,
                mode='lines',
                line=dict(color=dcol, width=2 if dG==0 else 1.2,
                          dash='solid' if dG==0 else 'dash'),
                name=dlbl,
                legendgroup=f'Rsat_{el}_{line}',
                showlegend=False,
                visible=visible,
                hovertemplate=f'<b>{el} {line}</b> {dlbl}<br>I=%{{x:.3f}} mW/cm²<br>R=%{{y:.3f}} Mphot/s<extra></extra>',
            ), row=1, col=2)

        # I_sat marker
        fig.add_trace(go.Scatter(
            x=[Is, Is], y=[0, G/2/1e6],
            mode='lines', line=dict(color='rgba(77,240,176,0.25)', width=1, dash='dot'),
            showlegend=False, visible=visible, hoverinfo='skip',
        ), row=1, col=2)

# ── Plot 3: Net force vs velocity (counter-propagating beams) ─────────────
for el in EL_NAMES:
    for line in ['D1','D2']:
        if line not in ELEMENTS[el]:
            continue
        G, k, Is, m = params(el, line)
        vc  = v_capture(G, k)
        visible = (el == DEFAULT_EL and line == args.line)

        v_arr = np.linspace(-4*vc, 4*vc, N_vel)
        # Optimal detuning Δ = -Γ/2 for Doppler cooling
        for dG, dcol, dlbl in [(-0.5,'#4df0b0','Δ=−Γ/2'),
                                (-1.0,'#4d9ff0','Δ=−Γ'),
                                (-0.25,'#f0c54d','Δ=−Γ/4')]:
            # Counter-propagating: one beam from left, one from right
            d_plus  = dG - v_arr*k/G   # atom moving at v, laser from left (+x)
            d_minus = dG + v_arr*k/G   # laser from right (-x)
            F_net = (hbar*k*(R_scatt(G,Is,1.0,d_plus) - R_scatt(G,Is,1.0,d_minus)) * 1e18)

            fig.add_trace(go.Scatter(
                x=v_arr, y=F_net,
                mode='lines',
                line=dict(color=dcol, width=2 if dG==-0.5 else 1.2,
                          dash='solid' if dG==-0.5 else 'dash'),
                name=dlbl,
                legendgroup=f'Fv_{el}_{line}',
                showlegend=False,
                visible=visible,
                hovertemplate=f'<b>{el} {line}</b> {dlbl}<br>v=%{{x:.1f}} m/s<br>F=%{{y:.4f}} aN<extra></extra>',
            ), row=1, col=3)

        # Zero force line
        fig.add_trace(go.Scatter(
            x=[v_arr[0], v_arr[-1]], y=[0,0],
            mode='lines', line=dict(color='rgba(255,255,255,0.1)', width=1),
            showlegend=False, visible=visible, hoverinfo='skip',
        ), row=1, col=3)
        # Capture velocity markers
        for v_sign in [-vc, vc]:
            fig.add_trace(go.Scatter(
                x=[v_sign, v_sign], y=[F_net.min()*1.1, F_net.max()*1.1],
                mode='lines', line=dict(color='rgba(240,197,77,0.3)', width=1, dash='dot'),
                showlegend=False, visible=visible, hoverinfo='skip',
            ), row=1, col=3)

# ── Plot 4: Doppler cooling force profile (Δ fixed at -Γ/2) ──────────────
for el in EL_NAMES:
    for line in ['D1','D2']:
        if line not in ELEMENTS[el]:
            continue
        G, k, Is, m = params(el, line)
        vc  = v_capture(G, k)
        visible = (el == DEFAULT_EL and line == args.line)
        v_arr = np.linspace(-4*vc, 4*vc, N_vel)

        # Multiple I values at fixed Δ=-Γ/2
        for I_s, I_lbl, I_col in zip([0.5,1.0,2.0], ['0.5 Isat','1 Isat','2 Isat'],
                                      ['#4d9ff0','#4df0b0','#f05a4d']):
            d_plus  = -0.5 - v_arr*k/G
            d_minus = -0.5 + v_arr*k/G
            F_net = (hbar*k*(R_scatt(G,Is,I_s,d_plus) - R_scatt(G,Is,I_s,d_minus)) * 1e18)
            fig.add_trace(go.Scatter(
                x=v_arr, y=F_net,
                mode='lines',
                line=dict(color=I_col, width=2 if I_s==1.0 else 1.2,
                          dash='solid' if I_s==1.0 else 'dash'),
                name=I_lbl,
                legendgroup=f'Fdop_{el}_{line}',
                showlegend=False,
                visible=visible,
                hovertemplate=f'<b>{el} {line}</b> {I_lbl}<br>v=%{{x:.1f}} m/s<br>F=%{{y:.4f}} aN<extra></extra>',
            ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=[v_arr[0],v_arr[-1]], y=[0,0],
            mode='lines', line=dict(color='rgba(255,255,255,0.1)', width=1),
            showlegend=False, visible=visible, hoverinfo='skip',
        ), row=2, col=1)

# ── Plot 5: Bar chart — all elements @ Δ=-Γ/2, I=Isat ───────────────────
for line in ['D1','D2']:
    el_labels, R_vals, F_vals, Td_vals, vc_vals, colors_bar = [], [], [], [], [], []
    for el in EL_NAMES:
        if line not in ELEMENTS[el]:
            continue
        G, k, Is, m = params(el, line)
        R = R_scatt(G, Is, 1.0, -0.5) / 1e6   # Δ=-Γ/2, I=Isat
        F = hbar*k * R_scatt(G,Is,1.0,-0.5) * 1e18
        Td = T_doppler_uK(G)
        vc = v_capture(G, k)
        el_labels.append(el)
        R_vals.append(R)
        F_vals.append(F)
        Td_vals.append(Td)
        vc_vals.append(vc)
        colors_bar.append(EL_COLORS.get(el,'#888'))

    visible = (line == args.line)
    # R_scatt bars
    fig.add_trace(go.Bar(
        x=el_labels, y=R_vals,
        name=f'R_scatt {line}',
        marker_color=colors_bar,
        marker_line_color='rgba(255,255,255,0.3)',
        marker_line_width=1,
        showlegend=False,
        visible=visible,
        hovertemplate='<b>%{x}</b><br>R_scatt=%{y:.3f} Mphot/s<extra></extra>',
    ), row=2, col=2)

# ── Plot 6: Summary table ─────────────────────────────────────────────────
for line in ['D1','D2']:
    el_col, wl_col, G_col, Is_col, Rmax_col, Fmax_col, Td_col, vc_col, vr_col, Er_col = \
        [], [], [], [], [], [], [], [], [], []
    for el in EL_NAMES:
        if line not in ELEMENTS[el]:
            continue
        G, k, Is, m = params(el, line)
        d = ELEMENTS[el][line]
        Rmax = G/2/1e6
        Fmax = hbar*k*G/2*1e18
        Td   = T_doppler_uK(G)
        vc   = v_capture(G, k)
        el_col.append(el)
        wl_col.append(f"{d['wl']:.3f}")
        G_col.append(f"{G/(2*pi)/1e6:.3f}")
        Is_col.append(f"{Is:.3f}")
        Rmax_col.append(f"{Rmax:.3f}")
        Fmax_col.append(f"{Fmax:.4f}")
        Td_col.append(f"{Td:.1f}")
        vc_col.append(f"{vc:.2f}")
        vr_col.append(f"{d.get('vrec',0):.3f}")
        Er_col.append(f"{d.get('Erec',0):.1f}")

    visible = (line == args.line)
    fig.add_trace(go.Table(
        header=dict(
            values=['<b>El.</b>','<b>λ (nm)</b>','<b>Γ/2π (MHz)</b>',
                    '<b>I_sat (mW/cm²)</b>','<b>R_max (Mphot/s)</b>',
                    '<b>F_max (aN)</b>','<b>T_Dop (μK)</b>',
                    '<b>v_cap (m/s)</b>','<b>v_rec (cm/s)</b>','<b>E_rec (nK)</b>'],
            fill_color='#1a2540',
            font=dict(color='#e8edf5', size=11),
            align='center', height=28,
            line_color='#2a3555',
        ),
        cells=dict(
            values=[el_col,wl_col,G_col,Is_col,Rmax_col,Fmax_col,Td_col,vc_col,vr_col,Er_col],
            fill_color=[['#111a2f' if i%2==0 else '#0f1628' for i in range(len(el_col))]],
            font=dict(color=['#4df0b0'] + ['#c8d0e0']*9, size=11),
            align='center', height=24,
            line_color='#1e2840',
        ),
        visible=visible,
    ), row=2, col=3)

# ============================================================================
# AXIS LABELS
# ============================================================================

fig.update_xaxes(title_text='Detuning Δ (Γ units)',      row=1, col=1, gridcolor='#1a2540', zerolinecolor='#2a3555')
fig.update_yaxes(title_text='R<sub>scatt</sub> (10⁶/s)', row=1, col=1, gridcolor='#1a2540')
fig.update_xaxes(title_text='Intensity (mW/cm²)',          row=1, col=2, gridcolor='#1a2540', type='log')
fig.update_yaxes(title_text='R<sub>scatt</sub> (10⁶/s)', row=1, col=2, gridcolor='#1a2540')
fig.update_xaxes(title_text='Atom velocity (m/s)',         row=1, col=3, gridcolor='#1a2540', zerolinecolor='#2a3555')
fig.update_yaxes(title_text='F<sub>net</sub> (aN)',        row=1, col=3, gridcolor='#1a2540')
fig.update_xaxes(title_text='Atom velocity (m/s)',         row=2, col=1, gridcolor='#1a2540', zerolinecolor='#2a3555')
fig.update_yaxes(title_text='F<sub>net</sub> (aN)',        row=2, col=1, gridcolor='#1a2540')
fig.update_xaxes(title_text='Element',                     row=2, col=2, gridcolor='#1a2540')
fig.update_yaxes(title_text='R<sub>scatt</sub> (10⁶/s)',  row=2, col=2, gridcolor='#1a2540')

# ============================================================================
# DROPDOWN MENUS: element + line selector
# ============================================================================

# Build visibility masks
# Trace order per (el, line) per subplot:
# Plot1: len(I_vals)+2 traces per (el,line)
# Plot2: len(det_values)+1 traces per (el,line)
# Plot3: 3+1+2 = 6 traces per (el,line)
# Plot4: 3+1 = 4 traces per (el,line)
# Plot5: 1 trace per line (all elements combined into one bar)
# Plot6: 1 trace per line

n_el_line = len(EL_NAMES) * 2   # total (el,line) combos
P1 = len(I_vals) + 2
P2 = len(det_values) + 1
P3 = 3 + 1 + 2   # force curves + zero + 2×capture vel
P4 = 3 + 1
P5 = 1   # per line (2 total: D1 and D2)
P6 = 1

# Count traces: for each (el,line) pair, P1+P2+P3+P4 traces
# Then 2 traces for P5 (one per line) and 2 for P6
per_combo = P1 + P2 + P3 + P4

combos = []
for el in EL_NAMES:
    for line in ['D1','D2']:
        if line in ELEMENTS.get(el, {}):
            combos.append((el, line))

total_combo_traces = len(combos) * per_combo
# P5: 2 traces (D1 bar, D2 bar)
# P6: 2 traces (D1 table, D2 table)
total_traces = total_combo_traces + 2 + 2

def make_visibility(target_el, target_line):
    """
    Build a visibility mask aligned with the ACTUAL trace insertion order.

    Traces are added in subplot-loop order, NOT combo-blocked order:
      Loop P1 (all combos) → Loop P2 (all combos) → P3 → P4 → P5(D1,D2) → P6(D1,D2)

    We replicate that exact order here so vis[i] matches fig.data[i].
    """
    vis = []
    # P1: one trace per I value + 2 vlines per combo
    for el, line in combos:
        show = (el == target_el and line == target_line)
        vis.extend([show] * (len(I_palette) + 2))
    # P2: (len(det_values)+1) traces per combo
    for el, line in combos:
        show = (el == target_el and line == target_line)
        vis.extend([show] * (len(det_values) + 1))
    # P3: 6 traces per combo  (3 force curves + 1 zero line + 2 capture vel lines)
    for el, line in combos:
        show = (el == target_el and line == target_line)
        vis.extend([show] * 6)
    # P4: 4 traces per combo  (3 Doppler curves + 1 zero line)
    for el, line in combos:
        show = (el == target_el and line == target_line)
        vis.extend([show] * 4)
    # P5: 1 bar trace per line (D1 then D2)
    vis.append(target_line == 'D1')
    vis.append(target_line == 'D2')
    # P6: 1 table trace per line (D1 then D2)
    vis.append(target_line == 'D1')
    vis.append(target_line == 'D2')
    return vis

# Build dropdown buttons
buttons_el = []
for el in EL_NAMES:
    buttons_el.append(dict(
        label=f'{el} D₁',
        method='update',
        args=[{'visible': make_visibility(el, 'D1')},
              {'title': {'text': f'<b>{el} D₁ Line — Scattering Rate & Radiation Pressure</b>'}}]
    ))
    buttons_el.append(dict(
        label=f'{el} D₂',
        method='update',
        args=[{'visible': make_visibility(el, 'D2')},
              {'title': {'text': f'<b>{el} D₂ Line — Scattering Rate & Radiation Pressure</b>'}}]
    ))

# ============================================================================
# LAYOUT
# ============================================================================

# Print per-element key metrics for the title annotation
def metrics_str(el, line):
    if line not in ELEMENTS.get(el,{}):
        return ''
    G, k, Is, m = params(el, line)
    wl = ELEMENTS[el][line]['wl']
    Td = T_doppler_uK(G)
    vc = v_capture(G, k)
    return (f"λ={wl:.1f}nm · Γ/2π={G/(2*pi)/1e6:.2f}MHz · "
            f"I_sat={Is:.2f}mW/cm² · T_Dop={Td:.0f}μK · v_cap={vc:.1f}m/s")

fig.update_layout(
    title=dict(
        text=f'<b>{DEFAULT_EL} {args.line} Line — Scattering Rate & Radiation Pressure</b>',
        x=0.5, xanchor='center', font=dict(size=18, color='#e8edf5'),
    ),
    height=950,
    autosize=True,
    plot_bgcolor='#0f1628',
    paper_bgcolor='#0a0d14',
    font=dict(family='monospace', color='#c8d0e0', size=11),
    hovermode='closest',
    hoverlabel=dict(bgcolor='#111a2f', bordercolor='#2a3555', font_color='#e8edf5'),
    updatemenus=[
        dict(
            type='dropdown',
            direction='down',
            buttons=buttons_el,
            x=0.01, xanchor='left',
            y=1.06, yanchor='top',
            bgcolor='#111a2f',
            bordercolor='#2a3555',
            font=dict(color='#c8d0e0'),
            active=[i for i,(el,ln) in enumerate(
                [(el,ln) for el in EL_NAMES for ln in ['D1','D2'] if ln in ELEMENTS.get(el,{})]
            ) if el==DEFAULT_EL and ln==args.line][0],
        ),
    ],
    legend=dict(
        bgcolor='rgba(10,13,20,0.8)',
        bordercolor='#1e2840',
        borderwidth=1,
        font=dict(color='#c8d0e0', size=10),
        x=1.01, xanchor='left', y=1, yanchor='top',
    ),
    margin=dict(l=60, r=160, t=140, b=60),
)

# Style all subplot axes — use update_xaxes/update_yaxes so subplot
# domains set by make_subplots are not overwritten.
fig.update_xaxes(gridcolor='#1a2540', zerolinecolor='#2a3555',
                 tickfont=dict(color='#6b7a99'))
fig.update_yaxes(gridcolor='#1a2540', zerolinecolor='#2a3555',
                 tickfont=dict(color='#6b7a99'))

# ============================================================================
# ANNOTATIONS: formula box
# ============================================================================
fig.add_annotation(
    x=0.5, y=1.08,
    xref='paper', yref='paper',
    text=('<b style="color:#4df0b0">R</b><sub>scatt</sub> = (Γ/2)·s/(1+s+(2Δ/Γ)²) &nbsp;|&nbsp; '
          's = I/I<sub>sat</sub> &nbsp;|&nbsp; '
          'F = ℏk·R<sub>scatt</sub> &nbsp;|&nbsp; '
          'T<sub>Dop</sub> = ℏΓ/2k<sub>B</sub> &nbsp;|&nbsp; '
          'v<sub>cap</sub> ≈ Γ/k'),
    showarrow=False,
    font=dict(size=12, color='#9aaac4'),
    align='center',
    bgcolor='rgba(17,26,47,0.8)',
    bordercolor='#1e2840',
    borderwidth=1,
    borderpad=6,
)

# ============================================================================
# PRINT SUMMARY
# ============================================================================
print(f"\n{'='*65}")
print(f"  Scattering Rate Widget — Key Parameters")
print(f"{'='*65}")
print(f"  {'El':>4}  {'Line':>4}  {'λ(nm)':>8}  {'Γ/2π(MHz)':>10}  "
      f"{'I_sat':>7}  {'T_Dop(μK)':>10}  {'v_cap(m/s)':>11}")
print(f"  {'-'*65}")
for el in EL_NAMES:
    for line in ['D1','D2']:
        if line not in ELEMENTS.get(el,{}):
            continue
        G, k, Is, m = params(el, line)
        d = ELEMENTS[el][line]
        Td = T_doppler_uK(G)
        vc = v_capture(G, k)
        print(f"  {el:>4}  {line:>4}  {d['wl']:>8.3f}  {G/(2*pi)/1e6:>10.3f}  "
              f"{Is:>7.3f}  {Td:>10.1f}  {vc:>11.2f}")

# ============================================================================
# SAVE
# ============================================================================
out_file = os.path.join(args.output_dir, 'scattering_rate.html')
_raw_html = fig.to_html(include_plotlyjs='cdn', full_html=True)
_fs_css = ('<style>html,body{margin:0;padding:0;width:100%;height:100%;'
           'overflow:hidden;background:#0a0d14;}'
           '.plotly-graph-div{width:100vw!important;height:100vh!important;}'
           '</style>')
_raw_html = _raw_html.replace('</head>', _fs_css + '</head>', 1)
with open(out_file, 'w', encoding='utf-8') as _fh:
    _fh.write(_raw_html)
print(f"\n✓ Saved: {out_file}")
print(f"  Total traces: {total_traces}")
print(f"  Dropdown: {len(buttons_el)} buttons ({len(EL_NAMES)} elements × 2 lines)")
print(f"{'='*65}\n")
