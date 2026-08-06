#!/usr/bin/env python3
"""
tweezer.py — Optical-tweezer scattering & heating
-----------------------------------------------------------------------------
Computes the dissipative physics of a far-off-resonant optical dipole trap
(single-beam Gaussian tweezer): photon scattering rate, recoil heating, and
an estimate of how long an atom stays trapped before heating out.

This is the piece left after polarizability.py absorbed the conservative
AC Stark / trap-depth / ω_r,ω_z calculator.  Do NOT use this script for
near-resonant laser-cooling curves — that is scattering_rate.py.

Physics
-------
Complex dynamic polarizability with finite upper-state linewidths Γ
(atomic units; matches the real-α convention in alpha_core.alpha_real):

    α(ω) = Σ_k  s_k f_k / (ω_k² − ω² − i Γ_k ω)

    Re[α] → trap depth U = −(2π a₀³/c) · Re[α] · I     (Grimm et al. 2000)
    Im[α] → scattering  R_sc = (4π a₀³ / (ħ c)) · Im[α] · I

Far off resonance this recovers the two-level Grimm relation
R_sc ≈ (Γ/|Δ|) · |U|/ħ for each channel.

Γ_k comes from lifetimes.py (A_total of the upper state).  Core / continuum
channels (source == 'core') are dropped from Im[α] — they do not scatter
trap light at optical frequencies in this model.

Recoil heating (isotropic spontaneous emission + laser absorption):

    Ė = 2 E_r(λ_trap) · R_sc
    E_r = h² / (2 m λ²)

Trap-heating lifetime (order-of-magnitude: deposit one trap depth of energy):

    τ_heat ≈ |U| / Ė

Data sources:
    data_json/<element>_polarizability.json   (cached transitions_gs)
    data_json/<element>_lifetimes.json        (Γ = A_total per upper state)
    Fallback: rebuild network via alpha_core if polarizability cache missing

Outputs:
    data_json/<element>_tweezer.json
    plots/<element>/<element>_tweezer.html

Usage:
    python tweezer.py Na
    python tweezer.py Na --wl 1064 --intensity 50 --waist 0.8
    python tweezer.py --all
"""

__version__ = '1.0'

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

from alpha_core import (
    alpha_real,
    get_available_elements,
    load_cached_transitions,
    load_element,
    build_transitions_for_state,
    omega_au_from_nm,
    nm_from_omega_au,
)
from polarizability import (
    U_joules,
    U_to_uK,
    U_to_kHz_h,
    kWcm2_to_Wm2,
    trap_frequencies_Hz,
)
from constants import (
    ATOMIC_MASS_AMU,
    A0_SI,
    C_SI,
    H_SI,
    HBAR_SI,
    KB_SI,
    AMU_KG,
    AU_TIME_S,
)

# Windows consoles often default to a legacy code page; force UTF-8.
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ============================================================================
# CONSTANTS (derived)
# ============================================================================

# R_sc (s⁻¹) = R_SC_PREFACTOR * Im[α_au] * I[W/m²]
R_SC_PREFACTOR = 4.0 * math.pi * A0_SI**3 / (HBAR_SI * C_SI)

DEFAULT_WL_NM = {
    'Li': 1064.0, 'Na': 1064.0, 'K': 1064.0,
    'Rb': 1064.0, 'Cs': 1064.0, 'Fr': 1064.0,
}
DEFAULT_I_KWCM2 = 50.0
DEFAULT_WAIST_UM = 1.0

# ============================================================================
# LIFETIMES → Γ LOOKUP
# ============================================================================

def load_gamma_map(element, data_dir='data_json'):
    """
    Map state label → {A_s, lifetime_ns, E_rec_J, mass_amu} from lifetimes JSON.
    Returns (gamma_map, mass_amu or None).
    """
    path = os.path.join(data_dir, f"{element}_lifetimes.json")
    if not os.path.exists(path):
        print(f"  ! {path} not found — scattering needs upper-state Γ")
        return {}, None

    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    gmap = {}
    mass = None
    for row in data.get('lifetimes', []):
        st = row.get('state')
        if not st:
            continue
        A = row.get('A_total_s')
        if A is None or A <= 0:
            continue
        gmap[st] = dict(
            A_s=float(A),
            lifetime_ns=float(row.get('lifetime_ns') or 1e9 / A),
            E_rec_J=row.get('E_rec_J'),
            E_rec_nK=row.get('E_rec_nK'),
        )
        if mass is None and row.get('mass_amu'):
            mass = float(row['mass_amu'])
    return gmap, mass


def attach_gamma(transitions_gs, gamma_map):
    """
    Return a working copy of transitions with Gamma_au attached for Im[α].
    Drops source=='core'.  Channels without a lifetime Γ are kept for Re[α]
    but contribute zero to Im[α] (Gamma_au = 0).
    """
    out = []
    n_with = 0
    n_skip_core = 0
    for tr in transitions_gs:
        if tr.get('source') == 'core':
            n_skip_core += 1
            continue
        t = dict(tr)
        label = t.get('other_label', '')
        info = gamma_map.get(label)
        if info:
            t['Gamma_au'] = info['A_s'] * AU_TIME_S
            t['Gamma_s'] = info['A_s']
            n_with += 1
        else:
            t['Gamma_au'] = 0.0
            t['Gamma_s'] = 0.0
        out.append(t)
    return out, n_with, n_skip_core


# ============================================================================
# COMPLEX POLARIZABILITY / SCATTERING
# ============================================================================

def alpha_complex(omega_au, transitions_list):
    """
    Return (Re[α], Im[α]) in atomic units at real frequency ω.

    Uses the same f / (ω_k² − ω²) convention as alpha_real, with a
    Lorentzian damping −i Γ ω in the denominator.  Near a pole the
    real part is set to NaN (same spirit as alpha_real's guard band).
    """
    re = 0.0
    im = 0.0
    GUARD = 1e-4
    near_pole = False
    for tr in transitions_list:
        dE = tr['delta_E_au']
        f = tr['fosc']
        sgn = tr['sign']
        Gam = tr.get('Gamma_au', 0.0) or 0.0
        x = dE * dE - omega_au * omega_au
        y = Gam * omega_au
        if abs(x) < GUARD * abs(dE) and Gam <= 0:
            near_pole = True
            continue
        denom2 = x * x + y * y
        if denom2 <= 0:
            near_pole = True
            continue
        re += sgn * f * x / denom2
        im += sgn * f * y / denom2
    if near_pole and abs(re) == 0 and abs(im) == 0:
        return float('nan'), float('nan')
    return re, im


def scattering_rate_Hz(im_alpha_au, I_Wm2):
    """Photon scattering rate R_sc [s⁻¹] from Im[α]."""
    if not math.isfinite(im_alpha_au):
        return float('nan')
    return R_SC_PREFACTOR * im_alpha_au * I_Wm2


def recoil_energy_J(wl_nm, mass_amu):
    """E_r = h² / (2 m λ²) for photons at wavelength wl_nm."""
    lam = wl_nm * 1e-9
    m = mass_amu * AMU_KG
    return (H_SI ** 2) / (2.0 * m * lam * lam)


def channel_contributions(omega_au, transitions_list, I_Wm2, top_n=5):
    """Rank individual channels by their R_sc contribution."""
    rows = []
    for tr in transitions_list:
        Gam = tr.get('Gamma_au', 0.0) or 0.0
        if Gam <= 0:
            continue
        dE = tr['delta_E_au']
        f = tr['fosc']
        sgn = tr['sign']
        x = dE * dE - omega_au * omega_au
        y = Gam * omega_au
        denom2 = x * x + y * y
        if denom2 <= 0:
            continue
        im_k = sgn * f * y / denom2
        R_k = scattering_rate_Hz(im_k, I_Wm2)
        wl_res = tr.get('wl_nm') or nm_from_omega_au(abs(dE))
        rows.append(dict(
            label=tr.get('other_label'),
            wl_nm=round(wl_res, 4),
            fosc=f,
            Gamma_s=tr.get('Gamma_s', 0.0),
            R_sc_Hz=R_k,
            source=tr.get('source'),
        ))
    rows.sort(key=lambda r: abs(r['R_sc_Hz']), reverse=True)
    return rows[:top_n]


# ============================================================================
# POINT EVALUATION
# ============================================================================

def evaluate_point(transitions, wl_nm, I_kWcm2, waist_um, mass_amu):
    """Full tweezer point calculation at one (λ, I, w0)."""
    omega = omega_au_from_nm(wl_nm)
    I_Wm2 = kWcm2_to_Wm2(I_kWcm2)
    re_a, im_a = alpha_complex(omega, transitions)
    if not math.isfinite(re_a):
        re_a = alpha_real(omega, transitions)

    U_J = U_joules(re_a, I_Wm2) if math.isfinite(re_a) else float('nan')
    U_uK = U_to_uK(U_J) if math.isfinite(U_J) else float('nan')
    U_kHz = U_to_kHz_h(U_J) if math.isfinite(U_J) else float('nan')

    R_sc = scattering_rate_Hz(im_a, I_Wm2) if math.isfinite(im_a) else float('nan')
    E_r = recoil_energy_J(wl_nm, mass_amu)
    E_dot = 2.0 * E_r * R_sc if math.isfinite(R_sc) else float('nan')
    heat_uK_s = (E_dot / KB_SI) * 1e6 if math.isfinite(E_dot) else float('nan')

    tau_heat_s = abs(U_J) / E_dot if (math.isfinite(U_J) and math.isfinite(E_dot)
                                       and E_dot > 0 and abs(U_J) > 0) else float('nan')
    tau_sc_s = 1.0 / R_sc if (math.isfinite(R_sc) and R_sc > 0) else float('nan')

    nu_r = nu_z = zR = float('nan')
    if math.isfinite(U_J) and abs(U_J) > 0:
        nu_r, nu_z, zR = trap_frequencies_Hz(U_J, waist_um, wl_nm, mass_amu)

    top = channel_contributions(omega, transitions, I_Wm2)

    return dict(
        wl_nm=wl_nm,
        I_kWcm2=I_kWcm2,
        I_Wm2=I_Wm2,
        waist_um=waist_um,
        mass_amu=mass_amu,
        alpha_re_au=re_a,
        alpha_im_au=im_a,
        U_J=U_J,
        U_uK=U_uK,
        U_kHz_h=U_kHz,
        R_sc_Hz=R_sc,
        R_sc_per_s=R_sc,
        E_r_J=E_r,
        E_r_nK=(E_r / KB_SI) * 1e9,
        heating_J_s=E_dot,
        heating_uK_s=heat_uK_s,
        tau_heat_s=tau_heat_s,
        tau_scatter_s=tau_sc_s,
        nu_r_Hz=nu_r,
        nu_z_Hz=nu_z,
        zR_um=(zR * 1e6) if math.isfinite(zR) else float('nan'),
        top_channels=top,
    )


# ============================================================================
# SPECTRUM
# ============================================================================

def compute_spectra(transitions, wl_min, wl_max, n_grid, I_kWcm2, mass_amu):
    """Wavelength sweep of Re[α], Im[α], R_sc, heating, τ_heat at fixed I."""
    res = []
    for tr in transitions:
        dE = abs(tr['delta_E_au'])
        if dE > 0:
            wl = nm_from_omega_au(dE)
            if wl_min <= wl <= wl_max:
                res.append(wl)
    grids = [np.linspace(wl_min, wl_max, n_grid)]
    for wl_r in res:
        lo = max(wl_min, wl_r - 2.0)
        hi = min(wl_max, wl_r + 2.0)
        grids.append(np.linspace(lo, hi, 80))
    wl_arr = np.unique(np.concatenate(grids))
    wl_arr = wl_arr[(wl_arr >= wl_min) & (wl_arr <= wl_max)]

    I_Wm2 = kWcm2_to_Wm2(I_kWcm2)
    re_a = np.empty_like(wl_arr)
    im_a = np.empty_like(wl_arr)
    R_sc = np.empty_like(wl_arr)
    heat = np.empty_like(wl_arr)
    tau = np.empty_like(wl_arr)
    U_uK_arr = np.empty_like(wl_arr)

    for i, wl in enumerate(wl_arr):
        omega = omega_au_from_nm(float(wl))
        r, im = alpha_complex(omega, transitions)
        re_a[i] = r
        im_a[i] = im
        R = scattering_rate_Hz(im, I_Wm2) if math.isfinite(im) else float('nan')
        R_sc[i] = R
        Er = recoil_energy_J(float(wl), mass_amu)
        Ed = 2.0 * Er * R if math.isfinite(R) else float('nan')
        heat[i] = (Ed / KB_SI) * 1e6 if math.isfinite(Ed) else float('nan')
        Uj = U_joules(r, I_Wm2) if math.isfinite(r) else float('nan')
        U_uK_arr[i] = U_to_uK(Uj) if math.isfinite(Uj) else float('nan')
        if math.isfinite(Uj) and math.isfinite(Ed) and Ed > 0 and abs(Uj) > 0:
            tau[i] = abs(Uj) / Ed
        else:
            tau[i] = float('nan')

    return dict(
        wl_nm=wl_arr.tolist(),
        alpha_re_au=re_a.tolist(),
        alpha_im_au=im_a.tolist(),
        R_sc_Hz=R_sc.tolist(),
        heating_uK_s=heat.tolist(),
        tau_heat_s=tau.tolist(),
        U_uK=U_uK_arr.tolist(),
    )


# ============================================================================
# RUN ELEMENT
# ============================================================================

def run_element(element, wl_nm=None, I_kWcm2=DEFAULT_I_KWCM2,
                waist_um=DEFAULT_WAIST_UM, wl_min=400.0, wl_max=1600.0,
                n_grid=1500, data_dir='data_json'):
    print(f"\n{'='*65}")
    print(f"  {element}  —  Optical tweezer scattering & heating")
    print(f"{'='*65}")

    cached = load_cached_transitions(element, data_dir)
    if cached and cached.get('transitions_gs'):
        trans_gs = cached['transitions_gs']
        gs_label = cached.get('gs_label') or '?'
        print(f"  Ground {gs_label}: {len(trans_gs)} cached transitions "
              f"from _polarizability.json")
    else:
        data = load_element(element, data_dir)
        if data is None:
            return None
        gs_label = data['gs_label']
        trans_gs = build_transitions_for_state(
            gs_label, data, data['transitions'])
        print(f"  Ground {gs_label}: built {len(trans_gs)} transitions "
              f"(no polarizability cache)")

    gamma_map, mass_lt = load_gamma_map(element, data_dir)
    mass = ATOMIC_MASS_AMU.get(element, mass_lt or 1.0)
    if mass_lt:
        mass = mass_lt

    transitions, n_g, n_core = attach_gamma(trans_gs, gamma_map)
    print(f"  Gamma attached for {n_g}/{len(transitions)} channels "
          f"(skipped {n_core} core)")

    if wl_nm is None:
        wl_nm = DEFAULT_WL_NM.get(element, 1064.0)

    point = evaluate_point(transitions, wl_nm, I_kWcm2, waist_um, mass)

    def _fmt(x, fmt='.4g'):
        return f"{x:{fmt}}" if isinstance(x, float) and math.isfinite(x) else str(x)

    print(f"  Point @ lambda={wl_nm:.1f} nm, I={I_kWcm2:g} kW/cm^2, w0={waist_um:g} um")
    print(f"    Re[alpha] = {_fmt(point['alpha_re_au'], '.2f')} au   "
          f"Im[alpha] = {_fmt(point['alpha_im_au'], '.3e')} au")
    u_khz = abs(point['U_kHz_h']) if math.isfinite(point['U_kHz_h']) else point['U_kHz_h']
    print(f"    U/k_B = {_fmt(point['U_uK'], '.2f')} uK   "
          f"(|U|/h = {_fmt(u_khz, '.2f')} kHz)")
    print(f"    R_sc  = {_fmt(point['R_sc_Hz'], '.4g')} /s   "
          f"(<dt> = {_fmt(point['tau_scatter_s'], '.3g')} s)")
    print(f"    Heating = {_fmt(point['heating_uK_s'], '.4g')} uK/s   "
          f"tau_heat ~ {_fmt(point['tau_heat_s'], '.3g')} s")
    if math.isfinite(point['nu_r_Hz']):
        print(f"    nu_r = {_fmt(point['nu_r_Hz'], '.1f')} Hz   "
              f"nu_z = {_fmt(point['nu_z_Hz'], '.1f')} Hz")
    if point['top_channels']:
        print("    Top scattering channels:")
        for ch in point['top_channels'][:3]:
            print(f"      {ch['label']:10s} lambda={ch['wl_nm']:.1f} nm  "
                  f"R={ch['R_sc_Hz']:.3g} /s  [{ch.get('source')}]")

    print("  Computing wavelength spectrum...")
    spectra = compute_spectra(
        transitions, wl_min, wl_max, n_grid, I_kWcm2, mass)

    resonances = []
    for tr in transitions:
        if (tr.get('Gamma_au') or 0) <= 0:
            continue
        if abs(tr.get('fosc', 0)) < 1e-4:
            continue
        resonances.append(dict(
            label=tr['other_label'],
            wl_nm=tr.get('wl_nm') or nm_from_omega_au(abs(tr['delta_E_au'])),
            fosc=tr['fosc'],
        ))
    resonances.sort(key=lambda r: -abs(r['fosc']))
    resonances = resonances[:12]

    return dict(
        element=element,
        gs_label=gs_label,
        mass_amu=mass,
        n_transitions=len(transitions),
        n_with_gamma=n_g,
        wl_min=wl_min,
        wl_max=wl_max,
        I_ref_kWcm2=I_kWcm2,
        waist_um=waist_um,
        point=point,
        spectra=spectra,
        resonances=resonances,
        transitions=transitions,
    )


# ============================================================================
# PLOT
# ============================================================================

def plot_tweezer(result, out_dir):
    el = result['element']
    sp = result['spectra']
    pt = result['point']
    I_ref = result['I_ref_kWcm2']
    waist = result['waist_um']

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            f"Photon scattering rate  ·  I = {I_ref:g} kW/cm²",
            f"Recoil heating rate  ·  Ė/k_B",
            f"Trap-heating lifetime  τ_heat ≈ |U|/Ė",
        ),
    )

    fig.add_trace(go.Scatter(
        x=sp['wl_nm'], y=sp['R_sc_Hz'], mode='lines',
        name='R_sc', line=dict(color='#4ea1ff', width=1.6),
        hovertemplate='λ=%{x:.2f} nm<br>R_sc=%{y:.4g} s⁻¹<extra></extra>',
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=sp['wl_nm'], y=sp['heating_uK_s'], mode='lines',
        name='heating', line=dict(color='#ff8c42', width=1.6),
        hovertemplate='λ=%{x:.2f} nm<br>Ė/k_B=%{y:.4g} µK/s<extra></extra>',
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=sp['wl_nm'], y=sp['tau_heat_s'], mode='lines',
        name='τ_heat', line=dict(color='#7dce82', width=1.6),
        hovertemplate='λ=%{x:.2f} nm<br>τ_heat=%{y:.4g} s<extra></extra>',
    ), row=3, col=1)

    for res in result['resonances']:
        wl = res['wl_nm']
        if not (result['wl_min'] <= wl <= result['wl_max']):
            continue
        for row in (1, 2, 3):
            fig.add_vline(
                x=wl, line=dict(color='rgba(200,80,80,0.35)', width=1, dash='dot'),
                row=row, col=1)
        fig.add_annotation(
            x=wl, y=1.02, yref='y domain', xref='x',
            text=res['label'], showarrow=False, textangle=-90,
            font=dict(size=9, color='#c66'), row=1, col=1)

    if math.isfinite(pt['R_sc_Hz']):
        fig.add_trace(go.Scatter(
            x=[pt['wl_nm']], y=[pt['R_sc_Hz']], mode='markers',
            name='point', marker=dict(size=10, color='#fff',
                                      line=dict(color='#4ea1ff', width=2)),
            hovertemplate=f"point λ={pt['wl_nm']:.1f} nm<br>"
                          f"R_sc={pt['R_sc_Hz']:.4g} s⁻¹<extra></extra>",
            showlegend=False,
        ), row=1, col=1)

    fig.update_yaxes(type='log', title_text='R_sc (s⁻¹)', row=1, col=1)
    fig.update_yaxes(type='log', title_text='µK / s', row=2, col=1)
    fig.update_yaxes(type='log', title_text='τ_heat (s)', row=3, col=1)
    fig.update_xaxes(title_text='Wavelength (nm)', row=3, col=1)

    fig.update_layout(
        template='plotly_dark',
        autosize=True,
        margin=dict(l=70, r=40, t=80, b=50),
        title=dict(
            text=(f"<b>{el} — Optical tweezer scattering & heating</b><br>"
                  f"<sup>ground {result['gs_label']}  ·  "
                  f"m = {result['mass_amu']:.3f} amu (IUPAC avg)  ·  "
                  f"{result['n_with_gamma']} channels with Γ  ·  "
                  f"w₀ = {waist:g} µm</sup>"),
            x=0.02, xanchor='left',
        ),
        legend=dict(orientation='h', y=1.02, x=1, xanchor='right'),
        hovermode='x unified',
    )

    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{el}_tweezer.html")
    raw = fig.to_html(include_plotlyjs='cdn', full_html=True)
    _fs_css = (
        '<style>'
        'html,body{margin:0;padding:0;width:100%;height:100%;'
        'overflow:hidden;background:#0a0d14;}'
        '.plotly-graph-div{width:100vw!important;height:100vh!important;}'
        '</style>'
    )
    raw = raw.replace('</head>', _fs_css + '</head>', 1)
    raw = _inject_controls(raw, result)
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(raw)
    print(f"  ✓ {out_file}")
    return out_file


def _inject_controls(raw_html, result):
    """Floating point calculator. Live α(λ) from spectrum interpolation;
    R_sc, U, heating scale linearly with I in the far-detuned regime."""
    pt = result['point']
    sp = result['spectra']
    el = result['element']
    mass = result['mass_amu']
    waist = result['waist_um']

    step = max(1, len(sp['wl_nm']) // 800)
    wl_js = sp['wl_nm'][::step]
    re_js = sp['alpha_re_au'][::step]
    im_js = sp['alpha_im_au'][::step]

    css = """
<style>
#tw-ctrl {
  position:fixed; top:16px; right:16px; width:280px;
  background:rgba(12,14,28,0.92); border:1px solid #2a2a5a; border-radius:10px;
  padding:12px 14px; font-family:'Helvetica Neue',Arial,sans-serif; font-size:12px;
  color:#ccc; box-shadow:0 4px 20px rgba(0,0,0,0.65); z-index:9999;
  backdrop-filter:blur(6px);
}
#tw-ctrl h3 { margin:0 0 9px; font-size:13px; color:#e0e8ff;
  border-bottom:1px solid #2a2a5a; padding-bottom:5px; cursor:pointer; }
#tw-ctrl .st { font-weight:bold; color:#7a9eff; font-size:10px;
  text-transform:uppercase; letter-spacing:.6px; margin-bottom:4px; margin-top:9px; }
#tw-ctrl input[type=number] { width:90px; background:#111830; border:1px solid #2a3060;
  border-radius:4px; color:#ccc; font-size:11px; padding:3px 6px; }
#tw-ctrl button { padding:5px 10px; font-size:11px; border:1px solid #2a3060;
  border-radius:5px; background:#111830; color:#aac; cursor:pointer; margin-top:8px; }
#tw-ctrl button:hover { background:#1a2850; color:#e0e8ff; }
#tw-out { background:#0d0d20; border:1px solid #2a2a4a; border-radius:6px;
  padding:8px 10px; font-size:11px; color:#9ab; line-height:1.7; margin-top:8px; }
#tw-out b { color:#e0e8ff; }
#tw-body { display:block; }
#tw-ctrl.collapsed #tw-body { display:none; }
</style>"""

    html = f"""
<div id="tw-ctrl">
  <h3 onclick="twToggle()">Tweezer point <span id="tw-tog">▼</span></h3>
  <div id="tw-body">
    <div class="st">Wavelength (nm)</div>
    <input type="number" id="tw-wl" value="{pt['wl_nm']}" step="0.1">
    <div class="st">Intensity (kW/cm²)</div>
    <input type="number" id="tw-I" value="{pt['I_kWcm2']}" step="1" min="0">
    <div class="st">Waist w₀ (µm)</div>
    <input type="number" id="tw-w0" value="{waist}" step="0.1" min="0.1">
    <button onclick="twCalc()">Calculate</button>
    <div id="tw-out">Click Calculate for live estimate.</div>
  </div>
</div>"""

    U_PREF = 2.0 * math.pi * A0_SI**3 / C_SI
    R_PREF = R_SC_PREFACTOR
    js = f"""
<script>
(function(){{
  const EL = {json.dumps(el)};
  const WL = {json.dumps(wl_js)};
  const RE = {json.dumps(re_js)};
  const IM = {json.dumps(im_js)};
  const MASS = {mass};
  const U_PREF = {U_PREF};
  const R_PREF = {R_PREF};
  const H = {H_SI}, KB = {KB_SI}, AMU = {AMU_KG};

  function interp(xs, ys, x){{
    if (x <= xs[0]) return ys[0];
    if (x >= xs[xs.length-1]) return ys[ys.length-1];
    let lo=0, hi=xs.length-1;
    while (hi-lo>1){{ const mid=(lo+hi)>>1; if (xs[mid] <= x) lo=mid; else hi=mid; }}
    const t = (x-xs[lo])/(xs[hi]-xs[lo]);
    const a = ys[lo], b = ys[hi];
    if (!isFinite(a) || !isFinite(b)) return NaN;
    return a*(1-t)+b*t;
  }}
  function fmt(x, d){{
    if (!isFinite(x)) return '—';
    const ax = Math.abs(x);
    if (ax !== 0 && (ax >= 1e4 || ax < 1e-3)) return x.toExponential(d);
    return x.toPrecision(d+1);
  }}

  window.twToggle = function(){{
    const box = document.getElementById('tw-ctrl');
    box.classList.toggle('collapsed');
    document.getElementById('tw-tog').textContent =
      box.classList.contains('collapsed') ? '▶' : '▼';
  }};

  window.twCalc = function(){{
    const wl = parseFloat(document.getElementById('tw-wl').value);
    const Ik = parseFloat(document.getElementById('tw-I').value);
    const w0 = parseFloat(document.getElementById('tw-w0').value);
    const out = document.getElementById('tw-out');
    if (!(wl > 0) || !(Ik >= 0) || !(w0 > 0)) {{
      out.innerHTML = 'Enter valid λ, I, w₀.'; return;
    }}
    const re = interp(WL, RE, wl);
    const im = interp(WL, IM, wl);
    const I = Ik * 1e3 / 1e-4;
    const U = -U_PREF * re * I;
    const UuK = U / KB * 1e6;
    const R = R_PREF * im * I;
    const lam = wl * 1e-9;
    const m = MASS * AMU;
    const Er = (H*H) / (2 * m * lam * lam);
    const Edot = 2 * Er * R;
    const heat = Edot / KB * 1e6;
    const tauH = (Math.abs(U) > 0 && Edot > 0) ? Math.abs(U)/Edot : NaN;
    const tauS = (R > 0) ? 1/R : NaN;
    const zR = Math.PI * (w0*1e-6)*(w0*1e-6) / lam;
    const Ur = Math.abs(U);
    const nur = (Ur > 0) ? Math.sqrt(4*Ur/(m*(w0*1e-6)*(w0*1e-6)))/(2*Math.PI) : NaN;
    const nuz = (Ur > 0) ? Math.sqrt(2*Ur/(m*zR*zR))/(2*Math.PI) : NaN;

    out.innerHTML =
      '<b>' + EL + ' @ ' + wl.toFixed(1) + ' nm</b><br>' +
      'I = ' + Ik + ' kW/cm² · w₀ = ' + w0 + ' µm<br>' +
      'Re[α] = <b>' + fmt(re,3) + '</b> au · Im[α] = <b>' + fmt(im,3) + '</b> au<br>' +
      'U/k_B = <b>' + fmt(UuK,3) + '</b> µK<br>' +
      'R_sc = <b>' + fmt(R,3) + '</b> s⁻¹ · ⟨Δt⟩ = <b>' + fmt(tauS,3) + '</b> s<br>' +
      'Heating = <b>' + fmt(heat,3) + '</b> µK/s<br>' +
      'τ_heat ≈ <b>' + fmt(tauH,3) + '</b> s<br>' +
      'ν_r = <b>' + fmt(nur,2) + '</b> Hz · ν_z = <b>' + fmt(nuz,2) + '</b> Hz' +
      '<div style="margin-top:6px;color:#6a7a9a;font-size:10px">' +
      'Live estimate: α(λ) interpolated from spectrum; scales ∝ I. ' +
      'Near resonances prefer the Python point calc.</div>';
  }};
  twCalc();
}})();
</script>"""

    raw_html = raw_html.replace('</head>', css + '</head>', 1)
    raw_html = raw_html.replace('</body>', html + js + '</body>', 1)
    return raw_html


# ============================================================================
# SAVE JSON
# ============================================================================

def _clean(obj):
    """Replace non-JSON floats with None."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def save_json(result, data_dir='data_json'):
    el = result['element']
    pt = result['point']
    out = dict(
        element=el,
        gs_label=result['gs_label'],
        mass_amu=result['mass_amu'],
        n_transitions=result['n_transitions'],
        n_with_gamma=result['n_with_gamma'],
        I_ref_kWcm2=result['I_ref_kWcm2'],
        waist_um=result['waist_um'],
        wl_range_nm=[result['wl_min'], result['wl_max']],
        point=_clean(pt),
        note=(
            'R_sc from Im[alpha(omega)] with Gamma = A_total from lifetimes.json. '
            'Heating uses 2 E_r(lambda_trap) per scatter. '
            'tau_heat = |U|/E_dot is an order-of-magnitude trap lifetime. '
            'Conservative depth/freqs also in polarizability.py AC Stark panel.'
        ),
    )
    path = os.path.join(data_dir, f"{el}_tweezer.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"  ✓ {path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Optical-tweezer photon scattering & recoil heating')
    parser.add_argument('element', nargs='?', default=None)
    parser.add_argument('--wl', type=float, default=None,
                        help='Trap wavelength (nm); default 1064')
    parser.add_argument('--intensity', type=float, default=DEFAULT_I_KWCM2,
                        help='Peak intensity (kW/cm²); default 50')
    parser.add_argument('--waist', type=float, default=DEFAULT_WAIST_UM,
                        help='1/e² waist (µm); default 1')
    parser.add_argument('--wl-min', type=float, default=400.0)
    parser.add_argument('--wl-max', type=float, default=1600.0)
    parser.add_argument('--n-grid', type=int, default=1500)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--data-dir', default='data_json')
    args = parser.parse_args()

    data_dir = args.data_dir
    os.makedirs('plots', exist_ok=True)
    print(f"tweezer.py v{__version__}")

    from species import require_neutral_species

    if args.all:
        elements = []
        for el in get_available_elements(data_dir):
            pol = os.path.join(data_dir, f"{el}_polarizability.json")
            life = os.path.join(data_dir, f"{el}_lifetimes.json")
            if os.path.exists(pol) and os.path.exists(life):
                elements.append(el)
        if not elements:
            print(f"No elements with polarizability+lifetimes in {data_dir}.")
            sys.exit(1)
        print("Running:", ", ".join(elements))
    else:
        if args.element is None:
            available = get_available_elements(data_dir)
            print("Available elements:", ", ".join(available) if available else "none")
            args.element = input("Enter element symbol: ").strip()
        elements = [require_neutral_species(args.element, "tweezer.py")["species_id"]]

    for el in elements:
        el_out = os.path.join('plots', el)
        os.makedirs(el_out, exist_ok=True)
        result = run_element(
            el,
            wl_nm=args.wl,
            I_kWcm2=args.intensity,
            waist_um=args.waist,
            wl_min=args.wl_min,
            wl_max=args.wl_max,
            n_grid=args.n_grid,
            data_dir=data_dir,
        )
        if result is None:
            continue
        save_json(result, data_dir)
        plot_tweezer(result, el_out)
