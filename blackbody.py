#!/usr/bin/env python3
"""
blackbody.py - Blackbody radiation (BBR) shift + depopulation rates
-----------------------------------------------------------------------------
Computes:
  1. Static (Itano) and dynamic (Planck x alpha) BBR energy shifts
     for ground + one D-line state, plus high-n Rydberg alpha(0) and
     damped dynamic Planck integrals (IR resonances) from transitions.json
  2. Bound-bound BBR depopulation rates Gamma_BBR from lifetimes.json
     (stimulated emission + absorption via nbar(omega,T))
  3. Continuum BBR photoionization Gamma_PI (Beterov eq.27 with A_L and
     quantum-defect phases), optional SFI (--field), and multi-step mixing
     Gamma_mix (eq.35); quenching Gamma_quench = Gamma_BBR + Gamma_PI

Shift validation: Itano <E^2>_T to <0.02%; Farley-Wing high-n asymptotic
delta_nu = pi (kB T)^2 / (3 h c^3) = 2416.666 Hz at 300 K.

Data: rydberg / transitions / polarizability cache / lifetimes JSON
Outputs: data_json/<el>_blackbody.json, plots/<el>/<el>_blackbody.html

Usage:
    python blackbody.py Na
    python blackbody.py Na --excited 3p3/2 --T 300
    python blackbody.py Na --n-min 10
    python blackbody.py Na --shift-n-min 10 --shift-n-max 25
    python blackbody.py Na --state 20s,25s
    python blackbody.py Na --no-rates --no-rydberg-shifts --no-pi
    python blackbody.py --all
"""

__version__ = '2.8'   # TRK + continuum/tail completion for high-n dyn BBR.
                      # Dynamic damped Planck v2.7; mixing v2.6.


import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np

try:
    import plotly.graph_objects as go
except ImportError:
    print("Plotly not installed. Run: pip install plotly")
    sys.exit(1)

from alpha_core import (
    alpha0_static,
    alpha_real,
    alpha_real_vectorized,
    build_transitions_for_state,
    coulomb_fosc,
    default_excited_label,
    get_available_elements,
    load_cached_transitions,
    load_element,
    L_CHAR,
)

# Headroom: alpha(0) for target n needs upward partners above n; n near
# series n_max is truncated and unreliable.
N_PAD_DEFAULT = 5

from constants import (
    E_FIELD_REF_V_M,
    E_FIELD_REF_T_K,
    EATOMIC_V_M,
    KB_EV,
    HBAR_EV_S,
    HBAR_SI,
    C_SI,
    KB_SI,
    EPS0_SI,
    H_SI,
    A0_SI,
    AU_TO_SI_ALPHA,
    ATOMIC_FREQ_RAD_S,
    HARTREE_TO_HZ,
    HARTREE_TO_EV,
    AU_TIME_S,
)

# ============================================================================
# CONSTANTS (BBR-specific plot colours; α(ω) core in alpha_core)
# ============================================================================

ELEM_COLORS = {
    'Li': '#4df0b0', 'Na': '#f0c54d', 'K': '#4d9ff0',
    'Rb': '#f05a4d', 'Cs': '#c54df0', 'Fr': '#f04d9f',
}

# ============================================================================
# BLACKBODY RADIATION SHIFT
# ============================================================================

def mean_E_squared_au(T_K):
    """
    Mean-square thermal electric field <E^2>_T, in atomic units.
        <E^2>_T = (831.9 V/m)^2 (T / 300 K)^4     (Itano, Lewis & Wineland 1982)
    """
    E_rms_V_m = E_FIELD_REF_V_M * (T_K / E_FIELD_REF_T_K) ** 2
    E_rms_au = E_rms_V_m / EATOMIC_V_M
    return E_rms_au ** 2


def bbr_static_shift_hz(alpha0_au, T_K):
    """
    Static (leading-order) blackbody radiation shift, in Hz.
        ΔE = -(1/2) α(0) <E^2>_T                          [Hartree]
        Δν = ΔE * (Eh/h)                                    [Hz]
    Dominant contribution to the BBR shift (>98% typically) for alkali-like
    atoms at T ~ 300 K. See module docstring for the (deliberately omitted)
    dynamic correction.
    """
    E2_au = mean_E_squared_au(T_K)
    dE_au = -0.5 * alpha0_au * E2_au
    return dE_au * HARTREE_TO_HZ


def _u_planck_arr(omega_rad_s, T_K):
    """Planck spectral energy density u(omega,T), J / (m^3 * rad/s)."""
    x = np.clip(HBAR_SI * omega_rad_s / (KB_SI * T_K), 1e-300, 700.0)
    return (HBAR_SI * omega_rad_s**3) / (math.pi**2 * C_SI**3) / np.expm1(x)


def bbr_total_shift_hz(transitions_list, T_K, n_pts=4000, omega_max_factor=40):
    """
    Full (static + dynamic) blackbody shift, in Hz, via direct numerical
    integration of the Planck spectrum against the real dynamic
    polarizability alpha(omega):

        Delta nu (T) = -(1/(2 eps0 h)) integral_0^omega_max u(omega,T) alpha_SI(omega) d omega

    omega_max_factor * (kB T / hbar) sets the upper integration cutoff --
    the Planck factor suppresses the integrand exponentially well before
    this for any alkali D-line-scale resonance at T <~ 1000 K (see module
    docstring). n_pts=4000 was checked to be converged to 6 significant
    figures against n_pts=20000 on real Na data.
    """
    if not transitions_list:
        return 0.0
    kT_over_hbar = KB_SI * T_K / HBAR_SI
    omega_max = omega_max_factor * kT_over_hbar
    w_grid = np.linspace(1e-3 * kT_over_hbar, omega_max, n_pts)
    w_au = w_grid / ATOMIC_FREQ_RAD_S
    alpha_SI = alpha_real_vectorized(w_au, transitions_list) * AU_TO_SI_ALPHA
    integrand = _u_planck_arr(w_grid, T_K) * alpha_SI
    val_J = -0.5 / EPS0_SI * np.trapezoid(integrand, w_grid)
    return val_J / H_SI


def farley_wing_shift_hz(T_K):
    """
    Farley-Wing high-n asymptotic BBR Stark shift (PRA 23, 2397 (1981)):
        delta_E = pi (k_B T)^2 / (3 c^3)     [Hartree]
        delta_nu = delta_E * (E_h / h)       [Hz]
    At T = 300 K this is +2416.666 Hz (independent of n, l).
    """
    kb_au = KB_SI / (H_SI * HARTREE_TO_HZ)
    c_au = C_SI / (A0_SI * ATOMIC_FREQ_RAD_S)
    dE_au = math.pi * (kb_au * float(T_K)) ** 2 / (3.0 * c_au ** 3)
    return dE_au * HARTREE_TO_HZ


def alpha_real_damped_vectorized(omega_au_arr, transitions_list):
    """
    Re[alpha(omega)] with Lorentzian damping (atomic units).
    Denominator: (dE^2 - omega^2)^2 + (Gamma * omega)^2, as in tweezer.alpha_complex.
    Soft floor on Gamma regularizes IR poles for high-n BBR integrals.
    """
    if not transitions_list:
        return np.zeros_like(omega_au_arr, dtype=float)
    dE = np.array([tr['delta_E_au'] for tr in transitions_list], dtype=float)
    f = np.array([tr['fosc'] for tr in transitions_list], dtype=float)
    sgn = np.array([tr['sign'] for tr in transitions_list], dtype=float)
    Gam = np.array([
        max(float(tr.get('Gamma_au') or 0.0), 1e-14 * max(abs(tr['delta_E_au']), 1e-12))
        for tr in transitions_list
    ], dtype=float)
    w = omega_au_arr[:, None]
    x = dE[None, :] ** 2 - w ** 2
    y = Gam[None, :] * w
    denom2 = x * x + y * y
    return (sgn[None, :] * f[None, :] * x / np.where(denom2 > 0, denom2, np.inf)).sum(axis=1)


def bbr_total_shift_damped_hz(transitions_list, T_K, n_pts=2000, omega_max_factor=40):
    """
    Full BBR shift via Planck x Re[alpha_damped(omega)].
    Use for high-n Rydberg states where IR resonances sit in the thermal band.
    """
    if not transitions_list:
        return 0.0
    kT_over_hbar = KB_SI * T_K / HBAR_SI
    omega_max = omega_max_factor * kT_over_hbar
    w_grid = np.linspace(1e-3 * kT_over_hbar, omega_max, n_pts)
    w_au = w_grid / ATOMIC_FREQ_RAD_S
    alpha_SI = alpha_real_damped_vectorized(w_au, transitions_list) * AU_TO_SI_ALPHA
    integrand = _u_planck_arr(w_grid, T_K) * alpha_SI
    val_J = -0.5 / EPS0_SI * np.trapezoid(integrand, w_grid)
    return val_J / H_SI


def load_gamma_map(element, data_dir='data_json'):
    """Map state label -> A_total_s from lifetimes.json (for damped alpha)."""
    path = os.path.join(data_dir, f'{element}_lifetimes.json')
    if not os.path.isfile(path):
        return {}
    with open(path, encoding='utf-8') as fh:
        data = json.load(fh)
    gmap = {}
    for row in data.get('lifetimes', []):
        st = row.get('state')
        A = row.get('A_total_s')
        if st and A is not None and float(A) > 0:
            gmap[st] = float(A)
    return gmap


def attach_gamma_au(transitions_list, gamma_map):
    """Copy transitions with Gamma_au from partner A_total (drop core).
    Preserves Gamma_au already set (e.g. continuum bins).
    """
    out = []
    for tr in transitions_list:
        if tr.get('source') == 'core':
            continue
        t = dict(tr)
        if t.get('Gamma_au'):
            out.append(t)
            continue
        A = gamma_map.get(t.get('other_label'))
        t['Gamma_au'] = (A * AU_TIME_S) if A else 0.0
        out.append(t)
    return out


# Discrete Rydberg tail above series n_max + Kramers continuum, with TRK
# closure so signed oscillator strength sums to 1 (one valence electron).
# Coulomb same-n f values otherwise overshoot TRK (~3) and inflate dyn to
# ~3x Farley-Wing; completion restores the high-n asymptote.
N_TAIL_DEFAULT = 200
N_CONT_BINS_DEFAULT = 80
EPS_MAX_AU_DEFAULT = 0.05


def _qd_delta(qd, l_char):
    return float(qd.get(l_char, 0.0) or 0.0)


def _qdt_energy_au(n, l_char, qd):
    n_eff = int(n) - _qd_delta(qd, l_char)
    if n_eff <= 0:
        return None
    return -0.5 / (n_eff * n_eff)


def discrete_tail_partners(label, data, n_from, n_to):
    """
    Synthetic E1 partners n_from..n_to via QDT energies + Coulomb fosc.
    Upward only (higher Rydberg / toward continuum).
    """
    n0, l0 = parse_nl_label(label)
    if n0 is None or l0 is None:
        return []
    E0 = data['state_energy_au'].get(label)
    if E0 is None:
        return []
    L0 = L_CHAR.get(l0)
    if L0 is None:
        return []
    qd = data.get('qd') or {}
    out = []
    for n in range(int(n_from), int(n_to) + 1):
        for l_char, L in L_CHAR.items():
            if abs(L - L0) != 1:
                continue
            E = _qdt_energy_au(n, l_char, qd)
            if E is None or E <= E0:
                continue
            f = coulomb_fosc(
                n, L, n0, L0,
                _qd_delta(qd, l_char), _qd_delta(qd, l0),
            )
            if f <= 0:
                continue
            out.append(dict(
                delta_E_au=E - E0,
                fosc=f,
                sign=+1,
                other_label=f'{n}{l_char}',
                source='QDT-tail',
                wl_nm=0.0,
            ))
    return out


def kramers_continuum_partners(E_bind_au, f_cont, n_bins=N_CONT_BINS_DEFAULT,
                               eps_max_au=EPS_MAX_AU_DEFAULT):
    """
    Distribute residual TRK strength f_cont over continuum bins with
    Kramers weight (I / (I+eps))^3 above the ionization threshold.
    """
    if f_cont <= 0 or E_bind_au <= 0:
        return []
    I = float(E_bind_au)
    n_bins = max(int(n_bins), 8)
    eps_max = float(eps_max_au)
    de = eps_max / n_bins
    weights = []
    for i in range(n_bins):
        eps = (i + 0.5) * de
        weights.append((I / (I + eps)) ** 3)
    W = sum(weights) * de
    if W <= 0:
        return []
    out = []
    for i, w in enumerate(weights):
        eps = (i + 0.5) * de
        f = f_cont * (w * de) / W
        out.append(dict(
            delta_E_au=I + eps,
            fosc=f,
            sign=+1,
            other_label=f'cont:{eps:.5g}',
            source='continuum',
            wl_nm=0.0,
            Gamma_au=de,
        ))
    return out


def complete_transitions_for_dyn(label, data, valence, n_tail_max=N_TAIL_DEFAULT,
                                 n_cont_bins=N_CONT_BINS_DEFAULT,
                                 eps_max_au=EPS_MAX_AU_DEFAULT):
    """
    Bound valence + QDT discrete tail + TRK closure + Kramers continuum.

    Returns (channels, meta) where meta has trk_scale, f_cont, n_tail, n_cont.
    Static alpha(0) should keep using the uncompleted valence network.
    """
    series_n_max = int(data.get('ryd', {}).get('n_max') or 0)
    channels = [dict(t) for t in valence]
    n_tail = 0
    if n_tail_max and series_n_max and n_tail_max > series_n_max:
        tail = discrete_tail_partners(
            label, data, series_n_max + 1, n_tail_max)
        channels.extend(tail)
        n_tail = len(tail)

    S = sum(t['sign'] * t['fosc'] for t in channels)
    if S > 1.0 + 1e-12:
        scale = 1.0 / S
        for t in channels:
            t['fosc'] = t['fosc'] * scale
            src = str(t.get('source') or '')
            if 'TRK' not in src:
                t['source'] = src + '-TRK' if src else 'TRK'
    else:
        scale = 1.0

    S2 = sum(t['sign'] * t['fosc'] for t in channels)
    f_cont = max(0.0, 1.0 - S2)
    E0 = data['state_energy_au'].get(label)
    E_bind = -float(E0) if E0 is not None and E0 < 0 else 0.0
    cont = kramers_continuum_partners(
        E_bind, f_cont, n_bins=n_cont_bins, eps_max_au=eps_max_au)
    channels.extend(cont)

    meta = dict(
        trk_scale=round(scale, 6),
        f_cont=round(f_cont, 6),
        n_tail=n_tail,
        n_cont=len(cont),
        n_tail_max=n_tail_max,
        trk_sum=round(sum(t['sign'] * t['fosc'] for t in channels), 6),
    )
    return channels, meta


# ============================================================================
# BLACKBODY DEPOPULATION RATES (bound-bound)
# ============================================================================

def nbar_photon(omega_rad_s, T_K):
    """
    Mean thermal photon number nbar(omega, T) = 1 / (exp(hbar omega / kT) - 1).
    """
    if T_K <= 0 or omega_rad_s <= 0:
        return 0.0
    x = HBAR_SI * float(omega_rad_s) / (KB_SI * float(T_K))
    if x > 700.0:
        return 0.0
    if x < 1e-12:
        return 1.0 / max(x, 1e-300)  # Rayleigh-Jeans
    return 1.0 / math.expm1(x)


def omega_from_nm(wl_nm):
    """Angular frequency (rad/s) from vacuum wavelength in nm."""
    return 2.0 * math.pi * C_SI / (float(wl_nm) * 1e-9)


def label_degeneracy(label):
    """
    Statistical weight g = 2J+1 from a state label.
    Examples: '3p1/2' -> 2, '3p3/2' -> 4, '4s' -> 2 (alkali ns = 2S1/2).
    Unresolved non-s: 2*(2l+1).
    """
    m = re.match(r'^(\d+)([spdfgh])(.*)$', str(label).strip())
    if not m:
        return 2
    l_char = m.group(2)
    suffix = m.group(3).strip()
    if suffix:
        if '/' in suffix:
            try:
                num, den = (int(x) for x in suffix.split('/', 1))
                return 2 * num // den + 1
            except (ValueError, ZeroDivisionError):
                pass
        else:
            try:
                return 2 * int(suffix) + 1
            except ValueError:
                pass
    if l_char == 's':
        return 2
    Lmap = {'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5}
    L = Lmap.get(l_char, 0)
    return 2 * (2 * L + 1)


def load_lifetimes_json(element, data_dir='data_json'):
    path = os.path.join(data_dir, f'{element}_lifetimes.json')
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def build_lifetime_graph(life_data):
    """
    Bidirectional E1 graph from lifetimes.json.

    Returns dict:
      states: {label: {n, l, A_total_s, lifetime_s}}
      edges_from: {label: [(partner, A_ul, g_u, g_l, wl_nm, kind), ...]}
        kind = 'stim' (upper->lower stimulated emission) or 'abs' (lower->upper)
    """
    states = {}
    edges_from = {}

    def ensure_state(label, n=None, l=None, A_total=0.0, lifetime_s=None):
        if label not in states:
            if n is None or l is None:
                m = re.match(r'^(\d+)([spdfgh])', label)
                n = int(m.group(1)) if m else 0
                l = m.group(2) if m else '?'
            states[label] = dict(
                n=n, l=l,
                A_total_s=float(A_total or 0.0),
                lifetime_s=lifetime_s,
            )
            edges_from[label] = []
        elif A_total:
            states[label]['A_total_s'] = float(A_total)
            if lifetime_s is not None:
                states[label]['lifetime_s'] = lifetime_s

    for entry in life_data.get('lifetimes', []):
        upper = entry['state']
        ensure_state(
            upper, n=entry.get('n'), l=entry.get('l'),
            A_total=entry.get('A_total_s', 0.0),
            lifetime_s=entry.get('lifetime_s'),
        )
        g_u = label_degeneracy(upper)
        for ch in entry.get('channels', []):
            lower = ch['lower_label']
            A = float(ch.get('A_s') or 0.0)
            wl = float(ch.get('wavelength_nm') or 0.0)
            if A <= 0 or wl <= 0:
                continue
            ensure_state(lower)
            g_l = label_degeneracy(lower)
            # From upper: stimulated emission toward lower
            edges_from[upper].append((lower, A, g_u, g_l, wl, 'stim'))
            # From lower: absorption toward upper
            edges_from[lower].append((upper, A, g_u, g_l, wl, 'abs'))

    return dict(states=states, edges_from=edges_from,
                n_start=life_data.get('n_start'),
                element=life_data.get('element'))


def bbr_depopulation(state, graph, T_K, top_n=5):
    """
    Bound-bound BBR depopulation for one state.

    stim: Gamma = A * nbar
    abs:  Gamma = A * (g_u / g_l) * nbar
    """
    info = graph['states'].get(state)
    if info is None:
        return None
    Gamma_rad = float(info.get('A_total_s') or 0.0)
    channels = []
    Gamma_BBR = 0.0
    for partner, A, g_u, g_l, wl, kind in graph['edges_from'].get(state, []):
        omega = omega_from_nm(wl)
        nb = nbar_photon(omega, T_K)
        if kind == 'stim':
            rate = A * nb
        else:
            rate = A * (g_u / max(g_l, 1e-30)) * nb
        if rate <= 0:
            continue
        Gamma_BBR += rate
        channels.append(dict(
            partner=partner, kind=kind, wavelength_nm=round(wl, 4),
            A_s=A, nbar=nb, rate_s=rate,
        ))
    channels.sort(key=lambda c: -c['rate_s'])
    Gamma_tot = Gamma_rad + Gamma_BBR
    tau_rad = (1.0 / Gamma_rad) if Gamma_rad > 0 else None
    tau_eff = (1.0 / Gamma_tot) if Gamma_tot > 0 else None
    top = []
    for c in channels[:top_n]:
        top.append(dict(
            partner=c['partner'], kind=c['kind'],
            wavelength_nm=c['wavelength_nm'],
            rate_s=round(c['rate_s'], 6),
            nbar=float(f'{c["nbar"]:.6g}'),
        ))
    return dict(
        state=state,
        n=info['n'],
        l=info['l'],
        Gamma_rad_s=Gamma_rad,
        Gamma_BBR_s=Gamma_BBR,
        tau_rad_s=tau_rad,
        tau_eff_s=tau_eff,
        n_channels=len(channels),
        top_channels=top,
    )


def run_rates(element, T_K=300.0, n_min=None, data_dir='data_json'):
    """
    Compute Gamma_BBR / tau_eff for all states with n >= n_min.
    Default n_min = n_start + 2 (skip lowest optical manifold).
    """
    life = load_lifetimes_json(element, data_dir)
    if life is None:
        print(f"  Rates skipped: no {element}_lifetimes.json")
        return None

    graph = build_lifetime_graph(life)
    n_start = life.get('n_start')
    if n_start is None:
        n_start = min((s['n'] for s in graph['states'].values() if s['n']), default=1)
    if n_min is None:
        n_min = int(n_start) + 2

    results = []
    for label, info in graph['states'].items():
        if info['n'] < n_min:
            continue
        row = bbr_depopulation(label, graph, T_K)
        if row is not None:
            results.append(row)
    results.sort(key=lambda r: (r['n'], r['l'], r['state']))

    # Sanity: ground-like absorption-only states (n == n_start, often not in table)
    # and lowest p if present.
    sanity = []
    for label, info in graph['states'].items():
        if info['n'] != int(n_start):
            continue
        row = bbr_depopulation(label, graph, T_K)
        if row:
            ratio = (row['Gamma_BBR_s'] / row['Gamma_rad_s']
                     if row['Gamma_rad_s'] > 0 else None)
            sanity.append(dict(
                state=row['state'],
                Gamma_BBR_s=row['Gamma_BBR_s'],
                Gamma_rad_s=row['Gamma_rad_s'],
                Gamma_BBR_over_rad=ratio,
            ))

    # Representative T-sweep for a few s/p states
    by_series = {'s': [], 'p': [], 'd': []}
    for r in results:
        if r['l'] in by_series:
            by_series[r['l']].append(r)
    reps = []
    for l_char, rows in by_series.items():
        if not rows:
            continue
        rows_n = sorted(rows, key=lambda r: r['n'])
        picks = {rows_n[0]['state'], rows_n[len(rows_n)//2]['state'], rows_n[-1]['state']}
        reps.extend(sorted(picks))
    reps = list(dict.fromkeys(reps))  # unique, preserve order

    T_sweep = np.linspace(50, 500, 40)
    sweep = dict(T_K=T_sweep.tolist(), states={})
    for lab in reps:
        sweep['states'][lab] = [
            bbr_depopulation(lab, graph, float(T))['Gamma_BBR_s'] for T in T_sweep
        ]

    # High-n / low-n summary prints
    if results:
        lo = results[0]
        hi = max(results, key=lambda r: r['Gamma_BBR_s'])
        print(f"  Rates at T = {T_K:.1f} K (n >= {n_min}, {len(results)} states):")
        tau_lo = (f", tau_eff = {lo['tau_eff_s']:.4g} s"
                  if lo['tau_eff_s'] else '')
        print(f"    lowest-n sample {lo['state']}: "
              f"Gamma_BBR = {lo['Gamma_BBR_s']:.4g} /s{tau_lo}")
        print(f"    max Gamma_BBR: {hi['state']} = {hi['Gamma_BBR_s']:.4g} /s")
        if hi['tau_rad_s'] and hi['tau_eff_s']:
            print(f"      tau_rad = {hi['tau_rad_s']:.4g} s -> "
                  f"tau_eff = {hi['tau_eff_s']:.4g} s")
    for s in sanity[:4]:
        if s['Gamma_BBR_over_rad'] is not None:
            print(f"    sanity {s['state']}: Gamma_BBR/Gamma_rad = "
                  f"{s['Gamma_BBR_over_rad']:.3e} (expect << 1 at 300 K for optical)")
        else:
            print(f"    sanity {s['state']}: Gamma_BBR = {s['Gamma_BBR_s']:.4g} /s "
                  f"(Gamma_rad = 0)")

    return dict(
        T_K=T_K,
        n_min=n_min,
        n_start=int(n_start),
        n_states=len(results),
        note=('bound-bound BBR only; stimulated emission + absorption via '
              'Einstein A and nbar(omega,T). No continuum photoionization.'),
        sanity=sanity,
        states=[
            dict(
                state=r['state'], n=r['n'], l=r['l'],
                Gamma_rad_s=round(r['Gamma_rad_s'], 6),
                Gamma_BBR_s=round(r['Gamma_BBR_s'], 6),
                tau_rad_s=(None if r['tau_rad_s'] is None else round(r['tau_rad_s'], 8)),
                tau_eff_s=(None if r['tau_eff_s'] is None else round(r['tau_eff_s'], 8)),
                top_channels=r['top_channels'],
            )
            for r in results
        ],
        T_sweep=sweep,
    )


# ============================================================================
# HIGH-n RYDBERG alpha(0) / STATIC BBR SHIFTS
# ============================================================================

def parse_nl_label(label):
    """Return (n, l_char) from a state label, or (None, None)."""
    m = re.match(r'^(\d+)([spdfgh])', str(label).strip())
    if not m:
        return None, None
    return int(m.group(1)), m.group(2)


def rydberg_shift_for_state(label, data, T_K, gamma_map=None, do_dyn=True,
                            do_cont=True, n_tail_max=N_TAIL_DEFAULT):
    """
    Build E1 alpha network for label; return static Itano + optional damped
    dynamic Planck BBR shift. Dynamic uses Lorentzian widths from lifetimes
    and (by default) TRK + QDT-tail + Kramers continuum completion so high-n
    approaches the Farley-Wing asymptote.
    """
    all_transitions = data['transitions']
    if label not in data['state_energy_au']:
        return None
    trans = build_transitions_for_state(label, data, all_transitions)
    valence = [t for t in trans if t.get('other_label') != 'core']
    alpha0 = alpha0_static(trans)
    if not math.isfinite(alpha0):
        return None
    n, l_char = parse_nl_label(label)
    if n is None:
        n_l = data.get('state_nl', {}).get(label)
        if n_l:
            n, l_char = n_l[0], n_l[1]
    min_dE = None
    if valence:
        min_dE = min(abs(t['delta_E_au']) for t in valence)
    n_sources = {}
    for t in valence:
        src = t.get('source') or 'unknown'
        n_sources[src] = n_sources.get(src, 0) + 1
    shift_hz = bbr_static_shift_hz(alpha0, T_K)
    shift_dyn = None
    n_gamma = 0
    cont_meta = None
    if do_dyn:
        if do_cont:
            dyn_channels, cont_meta = complete_transitions_for_dyn(
                label, data, valence, n_tail_max=n_tail_max)
        else:
            dyn_channels = [dict(t) for t in valence]
            cont_meta = dict(trk_scale=1.0, f_cont=0.0, n_tail=0, n_cont=0,
                             n_tail_max=0, trk_sum=None)
        damped = attach_gamma_au(dyn_channels, gamma_map or {})
        n_gamma = sum(1 for t in damped if (t.get('Gamma_au') or 0) > 0)
        shift_dyn = bbr_total_shift_damped_hz(damped, T_K)
    fw = farley_wing_shift_hz(T_K)
    return dict(
        state=label,
        n=n,
        l=l_char,
        alpha0_au=alpha0,
        shift_stat_hz=shift_hz,
        shift_dyn_hz=shift_dyn,
        farley_wing_hz=fw,
        n_transitions=len(valence),
        n_gamma=n_gamma,
        min_dE_au=min_dE,
        fosc_sources=n_sources,
        cont_meta=cont_meta,
    )


def run_rydberg_shifts(element, data, T_K=300.0, n_min=None, n_max=None,
                       extra_states=None, n_pad=N_PAD_DEFAULT, table=True,
                       do_dyn=True, do_cont=True, n_tail_max=N_TAIL_DEFAULT,
                       data_dir='data_json'):
    """
    Static + optional damped-dynamic BBR shifts from on-the-fly alpha networks.

    Default n window: [gs_n+2, series_n_max - n_pad]. Near the series end,
    upward partners are missing and alpha(0) is truncated; raise n_max in
    rydberg.py / transitions.py before trusting those states.
    Set table=False to compute only extra_states (CLI --state with
    --no-rydberg-shifts).
    """
    gs_n = int(data.get('gs_n') or 1)
    series_n_max = int(data.get('ryd', {}).get('n_max') or gs_n)
    if n_min is None:
        n_min = gs_n + 2
    if n_max is None:
        n_max = max(n_min, series_n_max - int(n_pad))

    labels = set()
    if table:
        for lab, (n, _l) in data.get('state_nl', {}).items():
            if n is None:
                continue
            if n_min <= int(n) <= n_max:
                labels.add(lab)
    for lab in (extra_states or []):
        lab = str(lab).strip()
        if lab:
            labels.add(lab)

    if not labels:
        print("  Rydberg alpha shifts: no states selected")
        return None

    gamma_map = load_gamma_map(element, data_dir) if do_dyn else {}
    fw = farley_wing_shift_hz(T_K)

    rows = []
    missing = []
    for lab in sorted(labels, key=lambda s: (parse_nl_label(s)[0] or 0, s)):
        row = rydberg_shift_for_state(
            lab, data, T_K, gamma_map=gamma_map, do_dyn=do_dyn,
            do_cont=do_cont, n_tail_max=n_tail_max)
        if row is None:
            missing.append(lab)
            continue
        rows.append(row)
    rows.sort(key=lambda r: (r['n'] or 0, r['l'] or '', r['state']))

    window = (f"n = {n_min}..{n_max}" if table else "--state only")
    if do_dyn:
        cont_note = (f", TRK+cont+tail(n<={n_tail_max})"
                     if do_cont else ", bound-only (no TRK/cont)")
        dyn_note = f", damped dyn on{cont_note} (Farley-Wing = {fw:+.3f} Hz)"
    else:
        dyn_note = ", static only"
    print(f"  Rydberg alpha shifts at T = {T_K:.1f} K "
          f"({window}, pad={n_pad}, series n_max={series_n_max}{dyn_note}): "
          f"{len(rows)} states")
    if missing:
        print(f"    skipped (no energy / network): {', '.join(missing[:8])}"
              + (' ...' if len(missing) > 8 else ''))
    for probe in (f'{gs_n}s', f'{gs_n}p1/2', f'{gs_n}p', '10s', '15s', '20s', '25s', '40s'):
        hit = next((r for r in rows if r['state'] == probe), None)
        if hit is None and probe in data['state_energy_au']:
            hit = rydberg_shift_for_state(
                probe, data, T_K, gamma_map=gamma_map, do_dyn=do_dyn,
                do_cont=do_cont, n_tail_max=n_tail_max)
        if hit is None:
            continue
        extra = ''
        if hit.get('shift_dyn_hz') is not None:
            extra = (f", dnu_dyn = {hit['shift_dyn_hz']:+.4g} Hz"
                     f" (FW ratio = {hit['shift_dyn_hz']/fw:.3f})")
            cm = hit.get('cont_meta') or {}
            if do_cont and cm.get('trk_scale') is not None:
                extra += f" [TRK scale={cm['trk_scale']:g}]"
        print(f"    {hit['state']}: alpha(0) = {hit['alpha0_au']:.4g} au, "
              f"dnu_stat = {hit['shift_stat_hz']:+.4g} Hz"
              f"{extra} ({hit['n_transitions']} valence partners)")

    if table and series_n_max - n_max < n_pad:
        print(f"    note: shift_n_max={n_max} is close to series n_max={series_n_max}; "
              f"alpha may be truncated (prefer n_max >= target + {n_pad})")

    note = ('static Itano from on-the-fly alpha(0) SOS; '
            f'n_pad={n_pad}: avoid states within {n_pad} of series n_max.')
    if do_dyn:
        note += (' damped Planck integral Re[alpha(omega)] with Gamma=A_total; '
                 f'Farley-Wing asymptote {fw:+.4f} Hz at T={T_K:g} K.')
        if do_cont:
            note += (' Dyn network: QDT discrete tail + TRK closure '
                     '(scale signed f to 1) + Kramers continuum residual.')
        else:
            note += ' Dyn continuum/TRK completion skipped (--no-rydberg-cont).'
    else:
        note += ' dynamic Planck integral skipped (--no-rydberg-dyn).'
    if not table:
        note += ' (--state only; n-table skipped)'

    def _round_shift(v):
        if v is None:
            return None
        return (round(v, 6) if abs(v) < 1e6 else float(f'{v:.6g}'))

    return dict(
        T_K=T_K,
        n_min=n_min if table else None,
        n_max=n_max if table else None,
        n_pad=n_pad,
        series_n_max=series_n_max,
        n_states=len(rows),
        do_dyn=do_dyn,
        do_cont=do_cont,
        n_tail_max=n_tail_max if do_cont else 0,
        farley_wing_hz=round(fw, 6),
        note=note,
        states=[
            dict(
                state=r['state'], n=r['n'], l=r['l'],
                alpha0_au=(round(r['alpha0_au'], 4) if abs(r['alpha0_au']) < 1e8
                           else float(f'{r["alpha0_au"]:.6g}')),
                shift_stat_hz=_round_shift(r['shift_stat_hz']),
                shift_dyn_hz=_round_shift(r.get('shift_dyn_hz')),
                n_transitions=r['n_transitions'],
                n_gamma=r.get('n_gamma', 0),
                min_dE_au=(None if r['min_dE_au'] is None
                           else float(f'{r["min_dE_au"]:.6g}')),
                fosc_sources=r['fosc_sources'],
                trk_scale=(None if not r.get('cont_meta')
                           else r['cont_meta'].get('trk_scale')),
                f_cont=(None if not r.get('cont_meta')
                        else r['cont_meta'].get('f_cont')),
                n_tail=(None if not r.get('cont_meta')
                        else r['cont_meta'].get('n_tail')),
                n_cont=(None if not r.get('cont_meta')
                        else r['cont_meta'].get('n_cont')),
            )
            for r in rows
        ],
    )


# ============================================================================
# CONTINUUM BBR PHOTOIONIZATION / QUENCHING
# ============================================================================

# Beterov et al., New J. Phys. 11, 013052 (2009):
#   eq. (27) direct PI with quantum-defect phases + A_L (Table 2)
#   eq. (30) BBR-induced SFI into n' >= n_c
#   eq. (9)  E_c ≈ 3.2e8 / n_eff^4  (V/cm)
BETEROV_PREF = 11500.0
BETEROV_E = 157890.0
E_C_PREF_V_CM = 3.2e8
L_QUANTUM = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5}

# Table 1: mu_L - mu_{L+1} near n~20 (continuum-adjacent)
QD_DIFF = {
    'Li': {'s': 0.352417, 'p': 0.0451664, 'd': 0.00162407},
    'Na': {'s': 0.493519, 'p': 0.840023,  'd': 0.0148029},
    'K':  {'s': 0.466733, 'p': 1.43762,   'd': 0.264237},
    'Rb': {'s': 0.490134, 'p': 1.29456,   'd': 1.34636},
    'Cs': {'s': 0.458701, 'p': 1.12661,   'd': 2.43295},
    'Fr': {'s': 0.46,     'p': 1.13,      'd': 2.4},  # Cs-like fallback
}

# Table 2: A_L scaling in eq. (27)/(30)
A_L_TABLE = {
    'Li': {'s': 1.0,  'p': 1.0,  'd': 0.9},
    'Na': {'s': 1.0,  'p': 1.0,  'd': 1.1},
    'K':  {'s': 0.9,  'p': 0.45, 'd': 1.3},
    'Rb': {'s': 1.0,  'p': 1.0,  'd': 0.6},
    'Cs': {'s': 0.85, 'p': 1.1,  'd': 0.35},
    'Fr': {'s': 0.85, 'p': 1.1,  'd': 0.35},
}


def quantum_defect_for_l(qd_table, l_char):
    """Lookup delta_l from rydberg quantum_defects dict."""
    if not qd_table or not l_char:
        return 0.0
    return float(qd_table.get(l_char, 0.0) or 0.0)


def binding_energy_eV(label, data):
    """
    Binding energy below continuum (eV).
    Rydberg JSON stores absolute energies with continuum = 0 (negative E).
    Ground uses -IE from load_element.
    """
    E_au = data['state_energy_au'].get(label)
    if E_au is None:
        return None
    E_bind = -float(E_au) * HARTREE_TO_EV
    if E_bind <= 0:
        return None
    return E_bind


def beterov_A_L(element, l_char):
    tab = A_L_TABLE.get(element) or {}
    return float(tab.get(l_char, 1.0))


def beterov_phase_weight(element, l_char):
    """
    cos^2 phase factor(s) from eq. (27).
    nS: only L' = L+1 term.  nP/nD: both L'+1 and L'-1.
    """
    diffs = QD_DIFF.get(element) or QD_DIFF['Na']
    pi6 = math.pi / 6.0

    def dmu(lc):
        return float(diffs.get(lc, 0.0))

    if l_char == 's':
        d_plus = math.pi * dmu('s')  # mu_S - mu_P
        return math.cos(d_plus + pi6) ** 2
    if l_char == 'p':
        d_plus = math.pi * dmu('p')   # mu_P - mu_D
        d_minus = math.pi * dmu('s')  # mu_S - mu_P
        return math.cos(d_plus + pi6) ** 2 + math.cos(d_minus - pi6) ** 2
    if l_char == 'd':
        d_plus = math.pi * dmu('d')   # mu_D - mu_F
        d_minus = math.pi * dmu('p')  # mu_P - mu_D
        return math.cos(d_plus + pi6) ** 2 + math.cos(d_minus - pi6) ** 2
    # Higher L: fall back to both-channel form with small diffs -> ~2
    return 2.0


def beterov_ln_term(x):
    """-ln(1 - e^{-x}) with stable limits."""
    if x > 40.0:
        return math.exp(-x)
    emx = math.exp(-x)
    return -math.log1p(-emx)


def gamma_pi_beterov(element, n_star, l_char, T_K):
    """
    Direct BBR photoionization (1/s), Beterov eq. (27):

        W = A_L * 11500 T / n*^(7/3) * [phase cos^2 weights]
            * ln(1 / (1 - exp(-157890 /(T n*^2))))
    """
    if n_star is None or n_star <= 0.5 or T_K <= 0:
        return 0.0
    n_star = float(n_star)
    A_L = beterov_A_L(element, l_char)
    phase = beterov_phase_weight(element, l_char)
    x = BETEROV_E / (float(T_K) * n_star * n_star)
    gamma = (A_L * BETEROV_PREF * float(T_K) / (n_star ** (7.0 / 3.0))
             * phase * beterov_ln_term(x))
    if not math.isfinite(gamma) or gamma < 0:
        return 0.0
    return gamma


def n_c_from_field(E_V_cm):
    """Critical n_eff for SFI: E_c ≈ 3.2e8 / n_eff^4  (V/cm)."""
    if E_V_cm is None or E_V_cm <= 0:
        return None
    return (E_C_PREF_V_CM / float(E_V_cm)) ** 0.25


def gamma_sfi_beterov(element, n_star, l_char, T_K, n_c):
    """
    BBR-induced transfer to n' >= n_c then field-ionized (1/s), eq. (30).
    Returns (Gamma_SFI, field_ionized_initial) where the flag means the
    initial state itself sits above the SFI threshold.
    """
    if n_c is None or n_star is None or n_star <= 0.5 or T_K <= 0:
        return 0.0, False
    n_star = float(n_star)
    n_c = float(n_c)
    if n_star >= n_c:
        return 0.0, True
    A_L = beterov_A_L(element, l_char)
    phase = beterov_phase_weight(element, l_char)
    # ln(1/(1-exp(E/(T nc^2) - E/(T n^2)))) - ln(1/(1-exp(-E/(T n^2))))
    x_n = BETEROV_E / (float(T_K) * n_star * n_star)
    x_diff = BETEROV_E / float(T_K) * (1.0 / (n_c * n_c) - 1.0 / (n_star * n_star))
    # x_diff is negative since n_c > n_star; rewrite as exp form carefully
    # term1: -ln(1 - exp(x_nc - x_n)) with x_nc = E/(T nc^2), x_n = E/(T n^2)
    x_nc = BETEROV_E / (float(T_K) * n_c * n_c)
    # exp(x_nc - x_n) = exp(-|x_diff|) since x_nc < x_n
    delta = x_n - x_nc  # positive
    if delta > 40.0:
        # 1 - exp(x_nc - x_n) = 1 - exp(-delta) ≈ 1; -ln(...) ≈ 0? No:
        # ln(1/(1-exp(x_nc-x_n))) = -ln(1 - e^{-delta}) ≈ e^{-delta} for large delta
        ln1 = beterov_ln_term(delta)
    else:
        ln1 = -math.log1p(-math.exp(-delta))
    ln2 = beterov_ln_term(x_n)
    bracket = ln1 - ln2
    if bracket < 0:
        bracket = 0.0
    gamma = (A_L * BETEROV_PREF * float(T_K) / (n_star ** (7.0 / 3.0))
             * phase * bracket)
    if not math.isfinite(gamma) or gamma < 0:
        return 0.0, False
    return gamma, False


def continuum_row_for_state(label, data, T_K, rates_by_state=None,
                            element=None, n_c=None):
    """One continuum / quenching row for a state label (direct PI/SFI only)."""
    element = element or data.get('element')
    n, l_char = parse_nl_label(label)
    if n is None:
        n_l = data.get('state_nl', {}).get(label)
        if n_l:
            n, l_char = n_l[0], n_l[1]
    if n is None or not l_char:
        return None
    E_bind = binding_energy_eV(label, data)
    if E_bind is None:
        return None
    qd = data.get('qd') or data.get('ryd', {}).get('quantum_defects') or {}
    delta = quantum_defect_for_l(qd, l_char)
    n_star = float(n) - delta
    if n_star <= 0.5:
        return None
    L = L_QUANTUM.get(l_char, 0)
    A_L = beterov_A_L(element, l_char)
    phase = beterov_phase_weight(element, l_char)
    Gamma_PI = gamma_pi_beterov(element, n_star, l_char, T_K)
    Gamma_SFI = 0.0
    field_ionized = False
    if n_c is not None:
        Gamma_SFI, field_ionized = gamma_sfi_beterov(
            element, n_star, l_char, T_K, n_c)
    Gamma_rad = 0.0
    Gamma_BBR = 0.0
    if rates_by_state and label in rates_by_state:
        rr = rates_by_state[label]
        Gamma_rad = float(rr.get('Gamma_rad_s') or 0.0)
        Gamma_BBR = float(rr.get('Gamma_BBR_s') or 0.0)
    # Free-space quench: BBR + direct PI (SFI / mix are experiment-side)
    Gamma_quench = Gamma_BBR + Gamma_PI
    Gamma_ion = Gamma_PI + Gamma_SFI
    Gamma_tot = Gamma_rad + Gamma_quench
    tau_eff = (1.0 / Gamma_tot) if Gamma_tot > 0 else None
    return dict(
        state=label,
        n=n,
        l=l_char,
        n_star=n_star,
        L=L,
        A_L=A_L,
        phase_weight=phase,
        E_bind_eV=E_bind,
        Gamma_PI_s=Gamma_PI,
        Gamma_SFI_s=Gamma_SFI,
        field_ionized=field_ionized,
        Gamma_mix_PI_s=0.0,
        Gamma_mix_SFI_s=0.0,
        Gamma_ion_s=Gamma_ion,
        Gamma_BBR_s=Gamma_BBR,
        Gamma_rad_s=Gamma_rad,
        Gamma_quench_s=Gamma_quench,
        tau_eff_s=tau_eff,
        n_mix_partners=0,
    )


def _round_rate(v):
    if v is None:
        return None
    return round(v, 6) if abs(v) < 1e6 else float(f'{v:.6g}')


def t_eff_gate(tau_s, t1_s, t2_s):
    """
    Effective interaction time for gated ion collection:
        t_eff = tau * (exp(-t1/tau) - exp(-t2/tau))
    """
    if tau_s is None or tau_s <= 0 or t2_s <= t1_s:
        return 0.0
    return float(tau_s) * (math.exp(-t1_s / tau_s) - math.exp(-t2_s / tau_s))


def _l_delta_ok(l_a, l_b):
    La = L_QUANTUM.get(l_a)
    Lb = L_QUANTUM.get(l_b)
    if La is None or Lb is None:
        return False
    return abs(La - Lb) == 1


def transfer_channels_from_state(label, graph, T_K, dn_max=3):
    """
    Spontaneous + BBR transfer rates from label to E1 partners (Delta L = +/-1).
    Returns list of (partner, Gamma_transfer_s, n_p, l_p).
    """
    info = graph['states'].get(label)
    if not info:
        return []
    n0 = info['n']
    out = []
    for partner, A, g_u, g_l, wl, kind in graph['edges_from'].get(label, []):
        n_p, l_p = parse_nl_label(partner)
        if n_p is None:
            pinfo = graph['states'].get(partner)
            if not pinfo:
                continue
            n_p, l_p = pinfo['n'], pinfo['l']
        if not _l_delta_ok(info['l'], l_p):
            continue
        if dn_max is not None and abs(int(n_p) - int(n0)) > int(dn_max):
            continue
        if wl <= 0 or A <= 0:
            continue
        omega = omega_from_nm(wl)
        nb = nbar_photon(omega, T_K)
        if kind == 'stim':
            # label is upper: spontaneous A + stimulated emission
            G_tr = A * (1.0 + nb)
        else:
            # label is lower: absorption only
            G_tr = A * (g_u / max(g_l, 1e-30)) * nb
        if G_tr > 0:
            out.append((partner, G_tr, int(n_p), l_p))
    return out


def mix_ionization_rate(Gamma_transfer, W_ion_p, tau_s, tau_p, t1_s, t2_s):
    """
    Effective ionization rate of the *initial* state via population of a
    neighbor that then ionizes (rate-equation solution).

        W_mix = Gamma_tr * W_ion_p / (gamma_p - gamma_s)
                * (1 - t_eff_p / t_eff_s)

    with gamma = 1/tau, t_eff = tau*(e^{-t1/tau}-e^{-t2/tau}).
    Always returns a non-negative rate.
    """
    if (Gamma_transfer <= 0 or W_ion_p <= 0 or tau_s is None or tau_p is None
            or tau_s <= 0 or tau_p <= 0):
        return 0.0
    te_s = t_eff_gate(tau_s, t1_s, t2_s)
    te_p = t_eff_gate(tau_p, t1_s, t2_s)
    if te_s <= 0:
        return 0.0
    gamma_s = 1.0 / tau_s
    gamma_p = 1.0 / tau_p
    dg = gamma_p - gamma_s
    if abs(dg) < 1e-12 * max(gamma_s, gamma_p, 1.0):
        # Equal-lifetime limit: N_p(t) = Gamma_tr * N0 * t * e^{-gamma t}
        # ions = W_ion * Gamma_tr * N0 * I, W_mix = W_ion*Gamma_tr*I/te_s
        # I = int_{t1}^{t2} t e^{-g t} dt
        g = gamma_s
        def I_part(t):
            # int t e^{-g t} dt = -e^{-gt}(t/g + 1/g^2)
            return -math.exp(-g * t) * (t / g + 1.0 / (g * g))
        I = I_part(t2_s) - I_part(t1_s)
        if I < 0:
            I = 0.0
        return Gamma_transfer * W_ion_p * I / te_s
    # Derived from integrating N_p(t); (1 - te_p/te_s)/(gamma_p - gamma_s)
    # is positive when the rate-equation solution is physical.
    factor = (1.0 - te_p / te_s) / dg
    if factor < 0:
        # Swap-equivalent form if gate/lifetime ordering flips the sign
        factor = (1.0 - te_s / te_p) / (gamma_s - gamma_p) if te_p > 0 else 0.0
    if factor < 0:
        return 0.0
    return Gamma_transfer * W_ion_p * factor


def apply_beterov_mixing(rows, graph, T_K, t1_s, t2_s, dn_max=3,
                         do_sfi_mix=False):
    """
    Fill Gamma_mix_PI / Gamma_mix_SFI on continuum rows (Beterov eq. 35).
    Main weight from Delta n ~ 1 partners with Delta L = +/-1.
    """
    by_label = {r['state']: r for r in rows}
    for r in rows:
        transfers = transfer_channels_from_state(
            r['state'], graph, T_K, dn_max=dn_max)
        mix_pi = 0.0
        mix_sfi = 0.0
        n_used = 0
        for partner, G_tr, _n_p, _l_p in transfers:
            pr = by_label.get(partner)
            if pr is None:
                continue
            # Need tau for partner; if missing rates, skip
            if r['tau_eff_s'] is None or pr.get('tau_eff_s') is None:
                continue
            contrib = mix_ionization_rate(
                G_tr, pr['Gamma_PI_s'],
                r['tau_eff_s'], pr['tau_eff_s'], t1_s, t2_s)
            mix_pi += contrib
            # Mix-SFI only via neighbors that are not already field-ionized
            # (direct SFI covers n' >= n_c).
            if do_sfi_mix and not pr.get('field_ionized'):
                mix_sfi += mix_ionization_rate(
                    G_tr, pr.get('Gamma_SFI_s') or 0.0,
                    r['tau_eff_s'], pr['tau_eff_s'], t1_s, t2_s)
            n_used += 1
        r['Gamma_mix_PI_s'] = mix_pi
        r['Gamma_mix_SFI_s'] = mix_sfi if do_sfi_mix else 0.0
        r['n_mix_partners'] = n_used
        r['Gamma_ion_s'] = (
            r['Gamma_PI_s'] + r['Gamma_SFI_s']
            + r['Gamma_mix_PI_s'] + r['Gamma_mix_SFI_s']
        )
    return rows


def run_continuum(element, data, T_K=300.0, n_min=None, rates=None,
                  extra_states=None, field_V_cm=None,
                  do_mix=True, t1_s=0.3e-6, t2_s=2.1e-6, mix_dn_max=3,
                  data_dir='data_json'):
    """
    Continuum BBR photoionization (Beterov eq.27) + optional SFI (eq.30)
    + optional multi-step mixing (eq.35).
    Free-space quench / tau_eff use Gamma_BBR + Gamma_PI only.
    """
    gs_n = int(data.get('gs_n') or 1)
    if n_min is None:
        n_min = gs_n + 2
    n_c = n_c_from_field(field_V_cm)

    rates_by_state = {}
    if rates and rates.get('states'):
        for r in rates['states']:
            rates_by_state[r['state']] = r

    labels = set()
    for lab, (n, _l) in data.get('state_nl', {}).items():
        if n is not None and int(n) >= n_min:
            labels.add(lab)
    for lab in (extra_states or []):
        lab = str(lab).strip()
        if lab:
            labels.add(lab)

    # Also include rate states below n_min so mix partners have PI/tau
    for lab in rates_by_state:
        labels.add(lab)

    rows = []
    for lab in sorted(labels, key=lambda s: (parse_nl_label(s)[0] or 0, s)):
        row = continuum_row_for_state(
            lab, data, T_K, rates_by_state, element=element, n_c=n_c)
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda r: (r['n'], r['l'], r['state']))

    # Ensure tau_eff on continuum rows (for mix) even if rates skipped some
    for r in rows:
        if r['tau_eff_s'] is None:
            g = (r['Gamma_rad_s'] or 0.0) + (r['Gamma_BBR_s'] or 0.0) + r['Gamma_PI_s']
            r['tau_eff_s'] = (1.0 / g) if g > 0 else None

    graph = None
    if do_mix:
        life = load_lifetimes_json(element, data_dir)
        if life is None:
            print("  Mixing skipped: no lifetimes.json")
            do_mix = False
        else:
            graph = build_lifetime_graph(life)
            apply_beterov_mixing(
                rows, graph, T_K, t1_s, t2_s,
                dn_max=mix_dn_max, do_sfi_mix=(n_c is not None),
            )

    # Table states: respect n_min for reported list (partners may be below)
    table_rows = [r for r in rows if r['n'] >= n_min]
    table_rows.sort(key=lambda r: (r['n'], r['l'], r['state']))

    if rates and rates.get('states'):
        pi_map = {r['state']: r for r in rows}
        for r in rates['states']:
            crow = pi_map.get(r['state'])
            g_pi = crow['Gamma_PI_s'] if crow else 0.0
            r['Gamma_PI_s'] = _round_rate(g_pi)
            if crow:
                r['Gamma_mix_PI_s'] = _round_rate(crow['Gamma_mix_PI_s'])
                r['Gamma_ion_s'] = _round_rate(crow['Gamma_ion_s'])
            g_tot = (r.get('Gamma_rad_s') or 0.0) + (r.get('Gamma_BBR_s') or 0.0) + g_pi
            r['Gamma_quench_s'] = _round_rate(
                (r.get('Gamma_BBR_s') or 0.0) + g_pi)
            r['tau_eff_s'] = (round(1.0 / g_tot, 8) if g_tot > 0 else None)

    sanity = []
    for probe in (data.get('gs_label'), f'{gs_n}p1/2', f'{gs_n}p', '10s', '20s', '25s'):
        if not probe:
            continue
        hit = next((r for r in rows if r['state'] == probe), None)
        if hit is None:
            continue
        ratio = None
        if hit['Gamma_BBR_s'] > 0:
            ratio = hit['Gamma_PI_s'] / hit['Gamma_BBR_s']
        sanity.append(dict(
            state=hit['state'],
            E_bind_eV=round(hit['E_bind_eV'], 6),
            A_L=hit['A_L'],
            Gamma_PI_s=hit['Gamma_PI_s'],
            Gamma_SFI_s=hit['Gamma_SFI_s'],
            Gamma_mix_PI_s=hit['Gamma_mix_PI_s'],
            Gamma_mix_SFI_s=hit['Gamma_mix_SFI_s'],
            Gamma_ion_s=hit['Gamma_ion_s'],
            Gamma_BBR_s=hit['Gamma_BBR_s'],
            Gamma_PI_over_BBR=ratio,
            field_ionized=hit['field_ionized'],
        ))

    field_note = (f", E = {field_V_cm:g} V/cm, n_c = {n_c:.2f}"
                  if n_c is not None else ", no SFI (--field not set)")
    mix_note = (f", mix on (t1={t1_s*1e6:.2f} us, t2={t2_s*1e6:.2f} us, "
                f"|dn|<={mix_dn_max})" if do_mix else ", mix off")
    print(f"  Continuum PI at T = {T_K:.1f} K (n >= {n_min}, {len(table_rows)} states; "
          f"Beterov eq.27 + A_L{field_note}{mix_note})")
    for s in sanity:
        extra = ''
        if s['Gamma_PI_over_BBR'] is not None:
            extra += f", Gamma_PI/Gamma_BBR = {s['Gamma_PI_over_BBR']:.3e}"
        if n_c is not None:
            if s['field_ionized']:
                extra += ' [initial state field-ionized]'
            else:
                extra += f", Gamma_SFI = {s['Gamma_SFI_s']:.4g} /s"
        if do_mix:
            extra += (f", mix_PI = {s['Gamma_mix_PI_s']:.4g} /s"
                      f", ion_tot = {s['Gamma_ion_s']:.4g} /s")
        print(f"    {s['state']}: E_bind = {s['E_bind_eV']:.4g} eV, "
              f"A_L = {s['A_L']:g}, Gamma_PI = {s['Gamma_PI_s']:.4g} /s{extra}")

    method = ('Beterov et al. NJP 11, 013052 (2009) eq. (27) with Table-2 A_L '
              'and quantum-defect phase weights')
    if n_c is not None:
        method += f'; eq. (30) SFI at E={field_V_cm:g} V/cm (n_c={n_c:.3f})'
    if do_mix:
        method += (f'; eq. (35) mixing with gate t1={t1_s}s, t2={t2_s}s, '
                   f'|dn|<={mix_dn_max}')

    return dict(
        T_K=T_K,
        n_min=n_min,
        n_states=len(table_rows),
        field_V_cm=field_V_cm,
        n_c=None if n_c is None else round(n_c, 4),
        mix=do_mix,
        t1_s=t1_s,
        t2_s=t2_s,
        mix_dn_max=mix_dn_max,
        method=method,
        note=('Gamma_PI = direct BBR photoionization. '
              'Gamma_SFI = BBR to n>=n_c then field-ionized (--field). '
              'Gamma_mix_* = ionization via neighbors (eq.35; gated t1,t2). '
              'Gamma_ion = PI + SFI + mix_PI + mix_SFI. '
              'Gamma_quench / free-space tau_eff = Gamma_BBR + Gamma_PI only.'),
        sanity=[
            dict(
                state=s['state'],
                E_bind_eV=s['E_bind_eV'],
                A_L=s['A_L'],
                Gamma_PI_s=_round_rate(s['Gamma_PI_s']),
                Gamma_SFI_s=_round_rate(s['Gamma_SFI_s']),
                Gamma_mix_PI_s=_round_rate(s['Gamma_mix_PI_s']),
                Gamma_mix_SFI_s=_round_rate(s['Gamma_mix_SFI_s']),
                Gamma_ion_s=_round_rate(s['Gamma_ion_s']),
                Gamma_BBR_s=_round_rate(s['Gamma_BBR_s']),
                Gamma_PI_over_BBR=(None if s['Gamma_PI_over_BBR'] is None
                                   else float(f"{s['Gamma_PI_over_BBR']:.6g}")),
                field_ionized=s['field_ionized'],
            )
            for s in sanity
        ],
        states=[
            dict(
                state=r['state'], n=r['n'], l=r['l'], L=r['L'],
                n_star=round(r['n_star'], 6),
                A_L=r['A_L'],
                phase_weight=round(r['phase_weight'], 6),
                E_bind_eV=round(r['E_bind_eV'], 6),
                Gamma_PI_s=_round_rate(r['Gamma_PI_s']),
                Gamma_SFI_s=_round_rate(r['Gamma_SFI_s']),
                Gamma_mix_PI_s=_round_rate(r['Gamma_mix_PI_s']),
                Gamma_mix_SFI_s=_round_rate(r['Gamma_mix_SFI_s']),
                field_ionized=r['field_ionized'],
                Gamma_ion_s=_round_rate(r['Gamma_ion_s']),
                Gamma_BBR_s=_round_rate(r['Gamma_BBR_s']),
                Gamma_rad_s=_round_rate(r['Gamma_rad_s']),
                Gamma_quench_s=_round_rate(r['Gamma_quench_s']),
                tau_eff_s=(None if r['tau_eff_s'] is None
                           else round(r['tau_eff_s'], 8)),
                n_mix_partners=r['n_mix_partners'],
            )
            for r in table_rows
        ],
    )


# ============================================================================
# MAIN COMPUTATION PER ELEMENT
# ============================================================================

def run_element(element, excited_label=None, T_K=300.0, data_dir='data_json',
                do_rates=True, n_min=None,
                do_rydberg_shifts=True, shift_n_min=None, shift_n_max=None,
                extra_states=None, n_pad=N_PAD_DEFAULT, do_rydberg_dyn=True,
                do_rydberg_cont=True, n_tail_max=N_TAIL_DEFAULT,
                do_pi=True, pi_n_min=None, field_V_cm=None,
                do_mix=True, t1_s=0.3e-6, t2_s=2.1e-6, mix_dn_max=3):
    print(f"\n{'='*65}")
    print(f"  {element}  -  Blackbody Radiation (shifts + rates + PI)")
    print(f"{'='*65}")

    data = load_element(element, data_dir)
    if data is None:
        return None

    all_transitions = data['transitions']
    gs_label = data['gs_label']
    cached = load_cached_transitions(element, data_dir)

    if cached and cached.get('transitions_gs'):
        trans_gs = cached['transitions_gs']
        print(f"  Ground state {gs_label}: using {len(trans_gs)} cached transitions "
              f"from _polarizability.json")
    else:
        trans_gs = build_transitions_for_state(gs_label, data, all_transitions)
        print(f"  Ground state {gs_label}: α(0) network built from pipeline JSON "
              f"({len(trans_gs)} transitions)")

    alpha0_gs = alpha_real(0.0, trans_gs)
    print(f"  Ground state {gs_label}: α(0) = {alpha0_gs:.2f} au")

    if excited_label is None and cached:
        excited_label = cached.get('excited_label')
    if excited_label is None:
        excited_label = default_excited_label(data, all_transitions)

    alpha0_ex = None
    trans_ex = []
    if excited_label:
        use_cached_ex = (cached and cached.get('transitions_ex')
                         and excited_label == cached.get('excited_label'))
        if use_cached_ex:
            trans_ex = cached['transitions_ex']
            print(f"  Excited state {excited_label}: using {len(trans_ex)} cached "
                  f"transitions from _polarizability.json")
        else:
            trans_ex = build_transitions_for_state(excited_label, data, all_transitions)
            print(f"  Excited state {excited_label}: network built from pipeline JSON "
                  f"({len(trans_ex)} transitions)")
        alpha0_ex = alpha_real(0.0, trans_ex)
        print(f"  Excited state {excited_label}: α(0) = {alpha0_ex:.2f} au")

    # T-sweep (10 K to 1000 K): static term (cheap, analytic)
    T_arr = np.linspace(10, 1000, 400)
    shift_gs_arr = np.array([bbr_static_shift_hz(alpha0_gs, T) for T in T_arr])
    shift_ex_arr = (np.array([bbr_static_shift_hz(alpha0_ex, T) for T in T_arr])
                     if alpha0_ex is not None else None)
    shift_diff_arr = (shift_ex_arr - shift_gs_arr) if shift_ex_arr is not None else None

    # T-sweep: full dynamic-corrected term (numerical integral; coarser
    # grid than the static sweep since each point costs a full integration,
    # but n_pts=4000 per point converges to 6 s.f. so 60 T-points is ample)
    print(f"  Computing dynamic-corrected shift across T-sweep "
          f"(validated against Farley-Wing high-n limit and Itano <E^2>_T)...")
    T_dyn_arr = np.linspace(10, 1000, 60)
    dyn_gs_arr = np.array([bbr_total_shift_hz(trans_gs, T) for T in T_dyn_arr])
    dyn_ex_arr = (np.array([bbr_total_shift_hz(trans_ex, T) for T in T_dyn_arr])
                  if trans_ex else None)
    dyn_diff_arr = (dyn_ex_arr - dyn_gs_arr) if dyn_ex_arr is not None else None

    # Point values at requested T
    shift_gs_T = bbr_static_shift_hz(alpha0_gs, T_K)
    shift_ex_T = bbr_static_shift_hz(alpha0_ex, T_K) if alpha0_ex is not None else None
    shift_diff_T = (shift_ex_T - shift_gs_T) if shift_ex_T is not None else None

    dyn_gs_T = bbr_total_shift_hz(trans_gs, T_K)
    dyn_ex_T = bbr_total_shift_hz(trans_ex, T_K) if trans_ex else None
    dyn_diff_T = (dyn_ex_T - dyn_gs_T) if dyn_ex_T is not None else None

    corr_gs_T = dyn_gs_T - shift_gs_T
    corr_ex_T = (dyn_ex_T - shift_ex_T) if dyn_ex_T is not None else None
    corr_diff_T = (dyn_diff_T - shift_diff_T) if dyn_diff_T is not None else None

    print(f"\n  At T = {T_K:.1f} K:")
    print(f"    dnu_stat(ground)    = {shift_gs_T:+.4f} Hz")
    print(f"    dnu_total(ground)   = {dyn_gs_T:+.4f} Hz   "
          f"(dynamic correction: {corr_gs_T:+.5f} Hz, {corr_gs_T/shift_gs_T*100:+.3f}%)")
    if shift_ex_T is not None:
        print(f"    dnu_stat(excited)   = {shift_ex_T:+.4f} Hz")
        print(f"    dnu_total(excited)  = {dyn_ex_T:+.4f} Hz   "
              f"(dynamic correction: {corr_ex_T:+.5f} Hz, {corr_ex_T/shift_ex_T*100:+.3f}%)")
        print(f"    dnu_stat(diff)      = {shift_diff_T:+.4f} Hz   "
              f"(excited - ground; shift on the {gs_label}->{excited_label} transition)")
        print(f"    dnu_total(diff)     = {dyn_diff_T:+.4f} Hz")

    rates = None
    if do_rates:
        rates = run_rates(element, T_K=T_K, n_min=n_min, data_dir=data_dir)

    continuum = None
    if do_pi:
        continuum = run_continuum(
            element, data, T_K=T_K,
            n_min=pi_n_min if pi_n_min is not None else n_min,
            rates=rates, extra_states=extra_states,
            field_V_cm=field_V_cm,
            do_mix=do_mix, t1_s=t1_s, t2_s=t2_s, mix_dn_max=mix_dn_max,
            data_dir=data_dir,
        )

    rydberg_shifts = None
    if do_rydberg_shifts or extra_states:
        rydberg_shifts = run_rydberg_shifts(
            element, data, T_K=T_K,
            n_min=shift_n_min, n_max=shift_n_max,
            extra_states=extra_states, n_pad=n_pad,
            table=bool(do_rydberg_shifts),
            do_dyn=do_rydberg_dyn, do_cont=do_rydberg_cont,
            n_tail_max=n_tail_max, data_dir=data_dir,
        )

    return dict(
        element=element, gs_label=gs_label, excited_label=excited_label,
        alpha0_gs_au=round(alpha0_gs, 4),
        alpha0_ex_au=round(alpha0_ex, 4) if alpha0_ex is not None else None,
        T_K=T_K,
        shift_gs_hz_at_T=round(shift_gs_T, 6),
        shift_ex_hz_at_T=round(shift_ex_T, 6) if shift_ex_T is not None else None,
        shift_diff_hz_at_T=round(shift_diff_T, 6) if shift_diff_T is not None else None,
        dyn_gs_hz_at_T=round(dyn_gs_T, 6),
        dyn_ex_hz_at_T=round(dyn_ex_T, 6) if dyn_ex_T is not None else None,
        dyn_diff_hz_at_T=round(dyn_diff_T, 6) if dyn_diff_T is not None else None,
        T_arr=T_arr, shift_gs_arr=shift_gs_arr,
        shift_ex_arr=shift_ex_arr, shift_diff_arr=shift_diff_arr,
        T_dyn_arr=T_dyn_arr, dyn_gs_arr=dyn_gs_arr,
        dyn_ex_arr=dyn_ex_arr, dyn_diff_arr=dyn_diff_arr,
        rates=rates,
        continuum=continuum,
        rydberg_shifts=rydberg_shifts,
    )

# ============================================================================
# PLOTTING
# ============================================================================

SERIES_COLORS = {
    's': '#4ea1ff', 'p': '#f0c54d', 'd': '#4df0b0',
    'f': '#f05a4d', 'g': '#c54df0',
}

_FS_CSS = (
    '<style>html,body{margin:0;padding:0;width:100%;height:100%;'
    'overflow:hidden;background:#0a0d14;}'
    '.plotly-graph-div{width:100vw!important;height:100vh!important;}'
    '</style>'
)

_BASE_LAYOUT = dict(
    plot_bgcolor='#0f1628', paper_bgcolor='#0a0d14',
    font=dict(color='#c8d0e0', family="'Helvetica Neue', Arial, sans-serif"),
    hovermode='x unified', autosize=True,
    margin=dict(t=90, r=40, b=60, l=80),
    legend=dict(bgcolor='rgba(10,13,20,0.88)', bordercolor='#1e2840',
                borderwidth=1, font=dict(size=11)),
)


def _avg_by_n(rows, value_key, abs_val=False, skip_field_ionized=False):
    by_n = defaultdict(list)
    for r in rows:
        if skip_field_ionized and r.get('field_ionized'):
            continue
        v = r.get(value_key)
        if v is None:
            continue
        by_n[r['n']].append(abs(v) if abs_val else v)
    ns = sorted(by_n.keys())
    ys = [sum(by_n[n]) / len(by_n[n]) for n in ns]
    return ns, ys


def _fig_shifts(result):
    el = result['element']
    gs_lbl, ex_lbl = result['gs_label'], result['excited_label']
    col = ELEM_COLORS.get(el, '#7777ff')
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result['T_arr'].tolist(), y=result['shift_gs_arr'].tolist(),
        mode='lines', line=dict(color=col, width=2, dash='dot'),
        name=f'dnu_stat (ground, {gs_lbl})',
        hovertemplate='T = %{x:.0f} K<br>dnu_stat = %{y:.4f} Hz<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=result['T_dyn_arr'].tolist(), y=result['dyn_gs_arr'].tolist(),
        mode='lines+markers', marker=dict(size=4),
        line=dict(color=col, width=2.5),
        name=f'dnu_total (ground, {gs_lbl})',
        hovertemplate='T = %{x:.0f} K<br>dnu_total = %{y:.4f} Hz<extra></extra>',
    ))
    if result['shift_ex_arr'] is not None:
        fig.add_trace(go.Scatter(
            x=result['T_arr'].tolist(), y=result['shift_ex_arr'].tolist(),
            mode='lines', line=dict(color='#ff6b6b', width=2, dash='dot'),
            name=f'dnu_stat (excited, {ex_lbl})',
            hovertemplate='T = %{x:.0f} K<br>dnu_stat = %{y:.4f} Hz<extra></extra>',
        ))
        fig.add_trace(go.Scatter(
            x=result['T_dyn_arr'].tolist(), y=result['dyn_ex_arr'].tolist(),
            mode='lines+markers', marker=dict(size=4),
            line=dict(color='#ff6b6b', width=2, dash='dash'),
            name=f'dnu_total (excited, {ex_lbl})',
            hovertemplate='T = %{x:.0f} K<br>dnu_total = %{y:.4f} Hz<extra></extra>',
        ))
        fig.add_trace(go.Scatter(
            x=result['T_arr'].tolist(), y=result['shift_diff_arr'].tolist(),
            mode='lines', line=dict(color='#a78bfa', width=2, dash='dot'),
            name=f'dnu_stat (diff, {gs_lbl}->{ex_lbl})',
            hovertemplate='T = %{x:.0f} K<br>dnu_diff,stat = %{y:.4f} Hz<extra></extra>',
        ))
        fig.add_trace(go.Scatter(
            x=result['T_dyn_arr'].tolist(), y=result['dyn_diff_arr'].tolist(),
            mode='lines+markers', marker=dict(size=4),
            line=dict(color='#a78bfa', width=2.5),
            name=f'dnu_total (diff, {gs_lbl}->{ex_lbl})',
            hovertemplate='T = %{x:.0f} K<br>dnu_diff,total = %{y:.4f} Hz<extra></extra>',
        ))
    fig.add_vline(x=300, line_dash='dot', line_color='rgba(255,255,255,0.25)',
                  line_width=1)
    fig.add_annotation(x=300, y=1.0, xref='x', yref='paper', text='300 K',
                       showarrow=False, yanchor='bottom',
                       font=dict(size=10, color='rgba(220,220,220,0.6)'))
    fig.add_hline(y=0, line_dash='dot', line_color='rgba(255,255,255,0.15)',
                  line_width=1)
    a0g, a0e = result['alpha0_gs_au'], result['alpha0_ex_au']
    sub = f'alpha(0)_ground = {a0g:.1f} au'
    if a0e is not None:
        sub += f'  ·  alpha(0)_excited = {a0e:.1f} au'
    sub += '  ·  dotted = static  ·  solid = static + dynamic'
    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(
            text=f'<b>{el} - BBR energy shifts vs T</b><br><sup>{sub}</sup>',
            x=0.5, xanchor='center', font=dict(size=18, color='#e8edf5'),
        ),
    )
    fig.update_xaxes(title='Temperature (K)', gridcolor='#1a2540',
                     tickfont=dict(color='#6b7a99'))
    fig.update_yaxes(title='dnu (Hz)', gridcolor='#1a2540', zeroline=True,
                     zerolinecolor='#2a3555', tickfont=dict(color='#6b7a99'))
    return fig


def _fig_series_vs_n(el, title, subtitle, states, value_key, y_title,
                     log_y=True, abs_val=False, sfi_key=None,
                     mix_key=None, ion_key=None, dyn_key=None, fw_hz=None):
    fig = go.Figure()
    by_l = {}
    for st in states:
        by_l.setdefault(st['l'], []).append(st)
    for l_char, rows in sorted(by_l.items()):
        rows = sorted(rows, key=lambda r: r['n'])
        ns, ys = _avg_by_n(rows, value_key, abs_val=abs_val)
        if not ns:
            continue
        if log_y:
            ys = [max(y, 1e-30) for y in ys]
        fig.add_trace(go.Scatter(
            x=ns, y=ys, mode='lines+markers', marker=dict(size=6),
            line=dict(color=SERIES_COLORS.get(l_char, '#aaa'), width=1.8),
            name=f'{value_key} ({l_char})',
            hovertemplate='n=%{x}<br>%{y:.4g}<extra></extra>',
        ))
        if sfi_key:
            ns2, ys2 = _avg_by_n(rows, sfi_key, skip_field_ionized=True)
            if ns2:
                ys2 = [max(y, 1e-30) for y in ys2]
                fig.add_trace(go.Scatter(
                    x=ns2, y=ys2, mode='lines+markers',
                    marker=dict(size=5, symbol='diamond'),
                    line=dict(color=SERIES_COLORS.get(l_char, '#aaa'),
                              width=1.4, dash='dot'),
                    name=f'{sfi_key} ({l_char})',
                    hovertemplate='n=%{x}<br>%{y:.4g}<extra></extra>',
                ))
        if mix_key:
            ns3, ys3 = _avg_by_n(rows, mix_key)
            if ns3:
                ys3 = [max(y, 1e-30) for y in ys3]
                fig.add_trace(go.Scatter(
                    x=ns3, y=ys3, mode='lines+markers',
                    marker=dict(size=5, symbol='square'),
                    line=dict(color=SERIES_COLORS.get(l_char, '#aaa'),
                              width=1.4, dash='dash'),
                    name=f'{mix_key} ({l_char})',
                    hovertemplate='n=%{x}<br>%{y:.4g}<extra></extra>',
                ))
        if ion_key:
            ns4, ys4 = _avg_by_n(rows, ion_key)
            if ns4:
                ys4 = [max(y, 1e-30) for y in ys4]
                fig.add_trace(go.Scatter(
                    x=ns4, y=ys4, mode='lines+markers',
                    marker=dict(size=4),
                    line=dict(color=SERIES_COLORS.get(l_char, '#aaa'),
                              width=2.2),
                    name=f'{ion_key} ({l_char})',
                    hovertemplate='n=%{x}<br>%{y:.4g}<extra></extra>',
                ))
        if dyn_key:
            ns5, ys5 = _avg_by_n(rows, dyn_key, abs_val=abs_val)
            if ns5:
                if log_y:
                    ys5 = [max(abs(y), 1e-30) for y in ys5]
                fig.add_trace(go.Scatter(
                    x=ns5, y=ys5, mode='lines+markers',
                    marker=dict(size=5, symbol='x'),
                    line=dict(color=SERIES_COLORS.get(l_char, '#aaa'),
                              width=1.5, dash='dash'),
                    name=f'|dnu_dyn| ({l_char})',
                    hovertemplate='n=%{x}<br>%{y:.4g}<extra></extra>',
                ))
    if fw_hz is not None and abs(fw_hz) > 0:
        n_lo = min(st['n'] for st in states if st.get('n') is not None)
        n_hi = max(st['n'] for st in states if st.get('n') is not None)
        fig.add_trace(go.Scatter(
            x=[n_lo, n_hi], y=[abs(fw_hz), abs(fw_hz)],
            mode='lines',
            line=dict(color='#e8edf5', width=1.5, dash='dot'),
            name=f'|Farley-Wing| = {abs(fw_hz):.1f} Hz',
            hovertemplate='FW %{y:.4g} Hz<extra></extra>',
        ))
    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(
            text=f'<b>{el} - {title}</b><br><sup>{subtitle}</sup>',
            x=0.5, xanchor='center', font=dict(size=18, color='#e8edf5'),
        ),
    )
    fig.update_xaxes(title='Principal quantum number n', gridcolor='#1a2540',
                     tickfont=dict(color='#6b7a99'))
    ykw = dict(title=y_title, gridcolor='#1a2540',
               tickfont=dict(color='#6b7a99'))
    if log_y:
        ykw['type'] = 'log'
    fig.update_yaxes(**ykw)
    return fig


def _write_plot_html(fig, path, inject_calculator=None):
    raw = fig.to_html(include_plotlyjs='cdn', full_html=True)
    raw = raw.replace('</head>', _FS_CSS + '</head>', 1)
    if inject_calculator is not None:
        raw = _inject_point_calculator(raw, inject_calculator)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(raw)
    print(f"  Wrote {path}")
    return path


def _write_blackbody_hub(el, out_dir, views):
    """
    Hub page with a dropdown that loads one fullscreen plot (iframe).
    views: list of (key, label, filename)
    """
    options = '\n'.join(
        f'    <option value="{fname}">{label}</option>'
        for _key, label, fname in views
    )
    default = views[0][2]
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{el} - Blackbody Radiation</title>
<style>
  html, body {{ margin:0; padding:0; width:100%; height:100%; overflow:hidden;
    background:#0a0d14; color:#c8d0e0;
    font-family:'Helvetica Neue', Arial, sans-serif; }}
  #bb-bar {{
    position:fixed; top:0; left:0; right:0; height:44px; z-index:100;
    display:flex; align-items:center; gap:12px; padding:0 14px;
    background:rgba(10,13,20,0.96); border-bottom:1px solid #1e2840;
    box-sizing:border-box;
  }}
  #bb-bar label {{ font-size:12px; color:#7a9eff; letter-spacing:.4px;
    text-transform:uppercase; font-weight:600; }}
  #bb-view {{
    background:#111830; border:1px solid #2a3060; border-radius:6px;
    color:#e0e8ff; font-size:13px; padding:6px 10px; min-width:220px;
  }}
  #bb-title {{ margin-left:auto; font-size:13px; color:#8a9ac0; }}
  #bb-frame {{
    position:fixed; top:44px; left:0; right:0; bottom:0;
    width:100%; height:calc(100vh - 44px); border:0; background:#0a0d14;
  }}
</style>
</head>
<body>
  <div id="bb-bar">
    <label for="bb-view">Plot</label>
    <select id="bb-view">
{options}
    </select>
    <span id="bb-title">{el} blackbody</span>
  </div>
  <iframe id="bb-frame" src="{default}" title="{el} blackbody plot"></iframe>
  <script>
  (function(){{
    const sel = document.getElementById('bb-view');
    const frame = document.getElementById('bb-frame');
    sel.addEventListener('change', function(){{ frame.src = sel.value; }});
  }})();
  </script>
</body>
</html>
"""
    path = os.path.join(out_dir, f'{el}_blackbody.html')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(html)
    print(f"  Wrote {path} (plot chooser hub)")
    return path


def plot_blackbody(result, out_dir):
    """
    Write one fullscreen HTML per plot family, plus a hub page with a
    dropdown to pick which plot to show (avoids a crowded multi-row figure).
    """
    el = result['element']
    rates = result.get('rates')
    continuum = result.get('continuum')
    ryd_sh = result.get('rydberg_shifts')
    views = []

    fig = _fig_shifts(result)
    fname = f'{el}_blackbody_shifts.html'
    _write_plot_html(fig, os.path.join(out_dir, fname), inject_calculator=result)
    views.append(('shifts', 'Energy shifts vs T', fname))

    if rates and rates.get('states'):
        fig = _fig_series_vs_n(
            el,
            f"BBR depopulation Gamma_BBR vs n",
            f"T = {rates['T_K']:.0f} K  ·  n >= {rates['n_min']}  ·  bound-bound",
            rates['states'], 'Gamma_BBR_s', 'Gamma_BBR (1/s)',
        )
        fname = f'{el}_blackbody_rates.html'
        _write_plot_html(fig, os.path.join(out_dir, fname))
        views.append(('rates', 'Depopulation Gamma_BBR vs n', fname))

    if continuum and continuum.get('states'):
        field = continuum.get('field_V_cm')
        sub = f"T = {continuum['T_K']:.0f} K  ·  n >= {continuum['n_min']}  ·  Beterov eq.27"
        if field:
            sub += f"  ·  SFI at E = {field:g} V/cm (dotted)"
        if continuum.get('mix'):
            sub += "  ·  dashed = mix_PI  ·  thick = ion_tot"
        fig = _fig_series_vs_n(
            el, 'BBR photoionization vs n', sub,
            continuum['states'], 'Gamma_PI_s', 'Gamma (1/s)',
            sfi_key='Gamma_SFI_s' if field else None,
            mix_key='Gamma_mix_PI_s' if continuum.get('mix') else None,
            ion_key='Gamma_ion_s' if continuum.get('mix') else None,
        )
        fname = f'{el}_blackbody_pi.html'
        _write_plot_html(fig, os.path.join(out_dir, fname))
        views.append(('pi', 'Photoionization / SFI vs n', fname))

    if ryd_sh and ryd_sh.get('states'):
        nwin = ''
        if ryd_sh.get('n_min') is not None:
            nwin = f"  ·  n = {ryd_sh['n_min']}..{ryd_sh['n_max']}"
        has_dyn = ryd_sh.get('do_dyn') and any(
            r.get('shift_dyn_hz') is not None for r in ryd_sh['states'])
        sub = f"T = {ryd_sh['T_K']:.0f} K{nwin}  ·  static Itano"
        if has_dyn:
            fw = ryd_sh.get('farley_wing_hz')
            sub += (f"  ·  dashed = dnu_dyn  ·  FW = {fw:+.1f} Hz"
                    if fw is not None else "  ·  dashed = dnu_dyn")
        fig = _fig_series_vs_n(
            el, 'Rydberg BBR shift vs n',
            sub,
            ryd_sh['states'], 'shift_stat_hz', 'shift (Hz)',
            abs_val=True,
            dyn_key='shift_dyn_hz' if has_dyn else None,
            fw_hz=ryd_sh.get('farley_wing_hz') if has_dyn else None,
        )
        fname = f'{el}_blackbody_alpha.html'
        _write_plot_html(fig, os.path.join(out_dir, fname))
        views.append(('alpha', 'Rydberg alpha / BBR shift vs n', fname))

    hub = _write_blackbody_hub(el, out_dir, views)
    return hub


def _inject_point_calculator(raw_html, result):
    a0g = result['alpha0_gs_au']
    a0e = result['alpha0_ex_au']
    E_ref, T_ref, E_au = E_FIELD_REF_V_M, E_FIELD_REF_T_K, EATOMIC_V_M
    hz_per_hartree = HARTREE_TO_HZ
    T_dyn = result['T_dyn_arr'].tolist()
    dyn_gs = result['dyn_gs_arr'].tolist()
    dyn_ex = result['dyn_ex_arr'].tolist() if result['dyn_ex_arr'] is not None else None

    css = """
<style>
#bb-ctrl {
  position:fixed; top:12px; right:12px; width:270px;
  background:rgba(10,10,22,0.94); border:1px solid #2a2a4a; border-radius:10px;
  padding:0; font-family:'Helvetica Neue',Arial,sans-serif; font-size:12px;
  color:#ccc; box-shadow:0 4px 20px rgba(0,0,0,0.65); z-index:9999;
  backdrop-filter:blur(6px); overflow:hidden;
}
#bb-ctrl-toggle {
  display:flex; align-items:center; justify-content:space-between; gap:8px;
  width:100%; margin:0; padding:10px 14px; box-sizing:border-box;
  background:transparent; border:0; border-bottom:1px solid transparent;
  color:#e0e8ff; font:inherit; font-size:13px; font-weight:600;
  cursor:pointer; text-align:left;
}
#bb-ctrl-toggle:hover { background:rgba(30,40,70,0.45); }
#bb-ctrl:not(.collapsed) #bb-ctrl-toggle { border-bottom-color:#2a2a5a; }
#bb-ctrl-chevron {
  flex:0 0 auto; font-size:11px; color:#7a9eff;
  transition:transform .15s ease;
}
#bb-ctrl.collapsed #bb-ctrl-chevron { transform:rotate(-90deg); }
#bb-ctrl-body { padding:10px 14px 12px; }
#bb-ctrl.collapsed #bb-ctrl-body { display:none; }
#bb-ctrl .st { font-weight:bold; color:#7a9eff; font-size:10px;
  text-transform:uppercase; letter-spacing:.6px; margin-bottom:4px; margin-top:9px; }
#bb-ctrl .st:first-child { margin-top:0; }
#bb-ctrl input[type=number] { width:90px; background:#111830; border:1px solid #2a3060;
  border-radius:4px; color:#ccc; font-size:11px; padding:3px 6px; }
#bb-ctrl button.bb-go { padding:5px 10px; font-size:11px; border:1px solid #2a3060;
  border-radius:5px; background:#111830; color:#aac; cursor:pointer; margin-top:8px; }
#bb-ctrl button.bb-go:hover { background:#1a2850; color:#e0e8ff; }
#bb-out { background:#0d0d20; border:1px solid #2a2a4a; border-radius:6px;
  padding:8px 10px; font-size:11px; color:#9ab; line-height:1.7; margin-top:8px; }
#bb-out b { color:#e0e8ff; }
</style>"""

    html = f"""
<div id="bb-ctrl" class="collapsed">
  <button type="button" id="bb-ctrl-toggle" aria-expanded="false"
          aria-controls="bb-ctrl-body" title="Show or hide calculator">
    <span>Point calculator</span>
    <span id="bb-ctrl-chevron" aria-hidden="true">&#9660;</span>
  </button>
  <div id="bb-ctrl-body">
    <div class="st">Temperature (K)</div>
    <input type="number" id="bb-T" value="300" step="1" min="1">
    <button type="button" class="bb-go" onclick="bbCalc()">Calculate</button>
    <div id="bb-out">Enter T and click Calculate.</div>
  </div>
</div>"""

    js = f"""
<script>
(function(){{
  const A0_GS = {a0g};
  const A0_EX = {a0e if a0e is not None else 'null'};
  const E_REF = {E_ref}, T_REF = {T_ref}, E_AU = {E_au};
  const HZ_PER_HARTREE = {hz_per_hartree};
  // Precomputed dynamic-corrected sweep (from the validated Planck-spectrum
  // integral) -- interpolated here rather than recomputed live, since the
  // full integral needs the actual transition network, not a closed form.
  const T_DYN  = {json.dumps(T_dyn)};
  const DYN_GS = {json.dumps(dyn_gs)};
  const DYN_EX = {json.dumps(dyn_ex) if dyn_ex is not None else 'null'};

  const panel = document.getElementById('bb-ctrl');
  const toggle = document.getElementById('bb-ctrl-toggle');
  if (toggle && panel) {{
    toggle.addEventListener('click', function(){{
      panel.classList.toggle('collapsed');
      const expanded = !panel.classList.contains('collapsed');
      toggle.setAttribute('aria-expanded', expanded ? 'true' : 'false');
    }});
  }}

  function shiftStaticHz(alpha0, T){{
    const E_rms_Vm = E_REF * Math.pow(T/T_REF, 2);
    const E_rms_au = E_rms_Vm / E_AU;
    const E2_au = E_rms_au * E_rms_au;
    const dE_au = -0.5 * alpha0 * E2_au;
    return dE_au * HZ_PER_HARTREE;
  }}

  function interp(xs, ys, x){{
    if (x <= xs[0]) return ys[0];
    if (x >= xs[xs.length-1]) return ys[ys.length-1];
    let lo=0, hi=xs.length-1;
    while (hi-lo>1){{ const mid=(lo+hi)>>1; if (xs[mid] <= x) lo=mid; else hi=mid; }}
    const t = (x-xs[lo])/(xs[hi]-xs[lo]);
    return ys[lo]*(1-t) + ys[hi]*t;
  }}

  window.bbCalc = function(){{
    const T = parseFloat(document.getElementById('bb-T').value);
    const out = document.getElementById('bb-out');
    if (!isFinite(T) || T <= 0) {{ out.innerHTML = 'Enter a valid temperature.'; return; }}
    const dGs = shiftStaticHz(A0_GS, T);
    const tGs = interp(T_DYN, DYN_GS, T);
    let html = `<b>T = ${{T.toFixed(1)}} K</b><br>`
      + `dnu_stat(ground) = <b>${{dGs.toFixed(4)}} Hz</b><br>`
      + `dnu_total(ground) = <b>${{tGs.toFixed(4)}} Hz</b> `
      + `<span style="color:#8a9ac0;font-size:10px">(interp.)</span>`;
    if (A0_EX !== null && DYN_EX !== null) {{
      const dEx = shiftStaticHz(A0_EX, T);
      const tEx = interp(T_DYN, DYN_EX, T);
      html += `<br>dnu_stat(excited) = <b>${{dEx.toFixed(4)}} Hz</b><br>`
            + `dnu_total(excited) = <b>${{tEx.toFixed(4)}} Hz</b><br>`
            + `dnu_stat(diff) = <b>${{(dEx-dGs).toFixed(4)}} Hz</b><br>`
            + `dnu_total(diff) = <b>${{(tEx-tGs).toFixed(4)}} Hz</b>`;
    }}
    out.innerHTML = html;
  }};
}})();
</script>"""

    raw_html = raw_html.replace('</head>', css + '</head>', 1)
    raw_html = raw_html.replace('</body>', html + js + '</body>', 1)
    return raw_html

# ============================================================================
# SAVE JSON
# ============================================================================

def save_json(result, data_dir):
    el = result['element']
    out = dict(
        element=el, gs_label=result['gs_label'], excited_label=result['excited_label'],
        alpha0_gs_au=result['alpha0_gs_au'], alpha0_ex_au=result['alpha0_ex_au'],
        T_K=result['T_K'],
        shift_gs_hz_at_T=result['shift_gs_hz_at_T'],
        shift_ex_hz_at_T=result['shift_ex_hz_at_T'],
        shift_diff_hz_at_T=result['shift_diff_hz_at_T'],
        dyn_gs_hz_at_T=result['dyn_gs_hz_at_T'],
        dyn_ex_hz_at_T=result['dyn_ex_hz_at_T'],
        dyn_diff_hz_at_T=result['dyn_diff_hz_at_T'],
        note=('shift_*_hz_at_T = static (leading-order) term only. '
              'dyn_*_hz_at_T = full static+dynamic term via Planck-spectrum '
              'integral of alpha(omega), validated against the Itano <E^2>_T '
              'formula and the Farley-Wing high-n closed form (Beloy et al. '
              '2025, arXiv:2507.00948, Sec. II). '
              'rates = bound-bound BBR depopulation from lifetimes.json; '
              'tau_eff includes Gamma_PI when continuum block is present. '
              'continuum = Beterov eq.27 PI (A_L + QD phases) + optional SFI '
              '(--field) + multi-step mixing (eq.35; --no-mix to skip); '
              'Gamma_quench = Gamma_BBR + Gamma_PI. '
              'rydberg_shifts = static Itano + damped Planck dyn with TRK/tail/'
              'Kramers continuum completion (Farley-Wing asymptote); '
              'for high-n labels (no dynamic integral).'),
    )
    if result.get('rates') is not None:
        out['rates'] = result['rates']
    if result.get('continuum') is not None:
        out['continuum'] = result['continuum']
    if result.get('rydberg_shifts') is not None:
        out['rydberg_shifts'] = result['rydberg_shifts']
    path = os.path.join(data_dir, f"{el}_blackbody.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"  Wrote {path}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Blackbody radiation shifts + bound-bound depopulation rates')
    parser.add_argument('element', nargs='?', default=None)
    parser.add_argument('--excited', default=None,
                        help='Excited state label e.g. 3p3/2')
    parser.add_argument('--T', type=float, default=300.0,
                        help='Temperature for the point value (K, default: 300)')
    parser.add_argument('--n-min', type=int, default=None,
                        help='Min n for rates table (default: ground n + 2)')
    parser.add_argument('--no-rates', action='store_true',
                        help='Skip BBR depopulation rates (shifts only)')
    parser.add_argument('--shift-n-min', type=int, default=None,
                        help='Min n for Rydberg alpha/static-shift table')
    parser.add_argument('--shift-n-max', type=int, default=None,
                        help='Max n for Rydberg alpha table (default: series n_max - pad)')
    parser.add_argument('--state', default=None,
                        help='Comma-separated extra labels e.g. 20s,25p')
    parser.add_argument('--no-rydberg-shifts', action='store_true',
                        help='Skip high-n Rydberg alpha(0) shift table')
    parser.add_argument('--no-rydberg-dyn', action='store_true',
                        help='Skip damped dynamic Planck integral for Rydberg shifts')
    parser.add_argument('--no-rydberg-cont', action='store_true',
                        help='Skip TRK/tail/continuum completion on Rydberg dyn network')
    parser.add_argument('--n-tail-max', type=int, default=N_TAIL_DEFAULT,
                        help=f'QDT discrete tail max n for dyn (default: {N_TAIL_DEFAULT})')
    parser.add_argument('--n-pad', type=int, default=N_PAD_DEFAULT,
                        help=f'Headroom below series n_max (default: {N_PAD_DEFAULT})')
    parser.add_argument('--pi-n-min', type=int, default=None,
                        help='Min n for continuum PI table (default: same as --n-min)')
    parser.add_argument('--no-pi', action='store_true',
                        help='Skip continuum BBR photoionization / quenching')
    parser.add_argument('--field', type=float, default=None,
                        help='Extraction field (V/cm) for Beterov SFI rates; omit to skip SFI')
    parser.add_argument('--no-mix', action='store_true',
                        help='Skip multi-step BBR mixing (W_mix)')
    parser.add_argument('--t1', type=float, default=0.3e-6,
                        help='Mix gate start time in seconds (default: 0.3 us)')
    parser.add_argument('--t2', type=float, default=2.1e-6,
                        help='Mix gate end time in seconds (default: 2.1 us)')
    parser.add_argument('--mix-dn-max', type=int, default=3,
                        help='Max |dn| for mix partners (default: 3)')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--data-dir', default='data_json')
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    os.makedirs('plots', exist_ok=True)
    print(f"blackbody.py v{__version__}")

    extra_states = None
    if args.state:
        extra_states = [s.strip() for s in args.state.split(',') if s.strip()]

    from species import require_neutral_species

    if args.all:
        elements = get_available_elements(DATA_DIR)
        if not elements:
            print(f"No elements found in {DATA_DIR}.")
            sys.exit(1)
    else:
        if args.element is None:
            available = get_available_elements(DATA_DIR)
            print("Available elements:", ', '.join(available) if available else 'none')
            args.element = input("Enter element symbol: ").strip()
        elements = [require_neutral_species(args.element, "blackbody.py")["species_id"]]

    for el in elements:
        el_out_dir = os.path.join('plots', el)
        os.makedirs(el_out_dir, exist_ok=True)
        result = run_element(
            el, excited_label=args.excited, T_K=args.T,
            data_dir=DATA_DIR, do_rates=not args.no_rates, n_min=args.n_min,
            do_rydberg_shifts=not args.no_rydberg_shifts,
            shift_n_min=args.shift_n_min, shift_n_max=args.shift_n_max,
            extra_states=extra_states, n_pad=args.n_pad,
            do_rydberg_dyn=not args.no_rydberg_dyn,
            do_rydberg_cont=not args.no_rydberg_cont,
            n_tail_max=args.n_tail_max,
            do_pi=not args.no_pi, pi_n_min=args.pi_n_min,
            field_V_cm=args.field,
            do_mix=not args.no_mix, t1_s=args.t1, t2_s=args.t2,
            mix_dn_max=args.mix_dn_max,
        )
        if result is None:
            continue
        save_json(result, DATA_DIR)
        plot_blackbody(result, el_out_dir)

    print(f"\n{'='*65}\n")
