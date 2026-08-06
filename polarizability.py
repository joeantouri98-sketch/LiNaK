#!/usr/bin/env python3
"""
polarizability.py
-----------------
Computes dynamic electric dipole polarizability α(ω), van der Waals C₆
coefficients, magic wavelengths, and AC Stark shift / optical dipole trap
depth for any element in the pipeline. One script, one plot per element —
this used to be two scripts (polarizability.py + acstark.py) producing two
near-duplicate plots; merged into one, since U(ω) is just α(ω) rescaled by
a constant at fixed intensity, not a second independent calculation.

Physics summary
---------------
Dynamic polarizability (atomic units):
    α_n(ω) = (2/3) Σ_k  f_nk · (E_k - E_n) / [(E_k - E_n)² - ω²]

where f_nk is the absorption oscillator strength, energies in Hartree,
ω in Hartree (= atomic units of frequency).  Positive terms: transitions
upward (k above n).  Negative terms: transitions downward (k below n).

Static polarizability: α(0)  [a₀³ = atomic units of volume]

C₆ coefficient via Casimir-Polder integral:
    C₆^AA = (3/π) ∫₀^∞ [α_A(iξ)]² dξ
    C₆^AB = (3/π) ∫₀^∞ α_A(iξ) · α_B(iξ) dξ

where α(iξ) is the polarizability at imaginary frequency, which is smooth
and positive everywhere (no resonances), computed by replacing ω → iξ:
    α_n(iξ) = (2/3) Σ_k  f_nk · (E_k - E_n) / [(E_k - E_n)² + ξ²]

Magic wavelengths: roots of  α_ground(ω) - α_excited(ω) = 0
Found by scanning ω grid and bracketing sign changes, then bisecting.

AC Stark shift / optical dipole trap depth (Grimm, Weidemüller &
Ovchinnikov, Adv. At. Mol. Opt. Phys. 42, 95 (2000), eq. 2):
    U(ω) = -(1/(2·ε0·c))·Re[α_SI(ω)]·I = -(2π·a0³/c)·α_au(ω)·I   [J]
Trap frequencies (Gaussian-beam harmonic approximation):
    z_R = π·w0²/λ,   ω_r = √(4|U0|/(m·w0²)),   ω_z = √(2|U0|/(m·z_R²))

Oscillator strength sources (priority order):
  1. NIST f-values from f_values/<element>_lines.txt
  2. Coulomb approximation (Bates & Damgaard 1949) using QDT n*

Data sources:
  data_json/<element>_rydberg.json     → state energies, quantum defects, IE
  data_json/<element>_transitions.json → complete E1 transition network
  data_json/<element>_lifetimes.json   → precomputed A coefficients (optional)
  f_values/<element>_lines.txt         → NIST experimental f-values

Outputs:
  data_json/<element>_polarizability.json
  plots/<element>/_polarizability.html         α(ω) + magic λ + AC Stark upgrade
                                                (dual μK axis, intensity slider,
                                                 point calculator with trap freq)
  plots/<element>/_magic_wavelengths.html      Grotrian-style magic λ atlas
  plots/C6_matrix.html                         cross-element C₆ heatmap
  plots/alpha_validation.html                  α(0) computed vs experiment

Usage:
  python polarizability.py Na
  python polarizability.py Na --excited 3p3/2
  python polarizability.py Na --wl-min 400 --wl-max 1500
  python polarizability.py Na --wl 1064 --intensity 10   (AC Stark point calc, kW/cm^2)
  python polarizability.py Na --waist 1.0                (beam waist, micron)
  python polarizability.py --all
  python polarizability.py --all --heteronuclear
"""

__version__ = '2.1'   # Shared α(ω) core extracted to alpha_core.py; CLI only under __main__.


import argparse
import csv
import io
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
    from plotly.subplots import make_subplots
    import plotly.colors as pc
except ImportError:
    print("Plotly not installed. Run: pip install plotly")
    sys.exit(1)

try:
    from scipy.integrate import quad
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("  scipy not found - C6 will use trapezoidal integration (less accurate)")

from alpha_core import (
    L_CHAR,
    alpha0_static,
    alpha_imag_freq,
    alpha_real,
    build_transitions_for_state,
    eV_to_au,
    get_available_elements,
    load_element,
    nm_from_omega_au,
    omega_au_from_nm,
)
from constants import (
    ALPHA0_EXPT,
    C6_EXPT,
    EV_TO_HARTREE,
    HARTREE_TO_EV,
    NM_TO_AU_OMEGA,
    AU_TO_ANGSTROM3,
    AU_TO_SI_ALPHA,
    AU_TO_SI_C6,
    HC_EV_NM,
    EPS0_SI,
    C_SI,
    A0_SI,
    H_SI,
    KB_SI,
    AMU_KG,
    ATOMIC_MASS_AMU,
)

# ============================================================================
# AC STARK / TRAP (derived)
# ============================================================================
# U = -(2*pi*a0^3/c) * alpha_au * I   [J], I in W/m^2
# (Grimm et al., Adv. At. Mol. Opt. Phys. 42, 95 (2000))

NM_TO_HARTREE = NM_TO_AU_OMEGA  # alias used in older plot code
U_PREFACTOR = 2.0 * math.pi * A0_SI**3 / C_SI


def U_joules(alpha_au, I_Wm2):
    """AC Stark shift / trap depth in Joules. Positive alpha -> negative U
    (attractive trap, red-detuned); negative alpha -> positive U (repulsive,
    blue-detuned)."""
    return -U_PREFACTOR * alpha_au * I_Wm2

def U_to_uK(U_J):
    return U_J / KB_SI * 1e6

def U_to_kHz_h(U_J):
    """U/h in kHz — the shift expressed as a frequency."""
    return U_J / H_SI / 1e3

def alpha_to_uK_factor(I_kWcm2):
    """uK per unit alpha_au, at the given intensity. U_uK = alpha_au * this."""
    I_Wm2 = I_kWcm2 * 1e3 / 1e-4
    return -U_PREFACTOR * I_Wm2 / KB_SI * 1e6

def kWcm2_to_Wm2(I_kWcm2):
    return I_kWcm2 * 1e3 / 1e-4

def trap_frequencies_Hz(U0_J, waist_um, wl_nm, mass_amu):
    """
    Radial and axial trap frequencies for a single-beam Gaussian tweezer,
    harmonic approximation near the focus. U0_J must be the (possibly signed)
    trap depth; only its magnitude is used.
    """
    w0 = waist_um * 1e-6           # m
    wl = wl_nm * 1e-9              # m
    m  = mass_amu * AMU_KG
    zR = math.pi * w0**2 / wl      # Rayleigh range, m
    U0 = abs(U0_J)
    omega_r = math.sqrt(4.0 * U0 / (m * w0**2))
    omega_z = math.sqrt(2.0 * U0 / (m * zR**2))
    return omega_r / (2 * math.pi), omega_z / (2 * math.pi), zR


def compute_alpha_spectrum(transitions_list, wl_min_nm, wl_max_nm, n_grid):
    """
    Compute α(ω) on a wavelength grid, handling divergences.
    Returns (wl_array, alpha_array) with NaN at resonances.
    """
    # Build adaptive grid: dense near each resonance, coarse elsewhere
    resonance_wls = []
    for tr in transitions_list:
        dE_au = abs(tr['delta_E_au'])
        if dE_au > 0:
            wl_res = nm_from_omega_au(dE_au)
            if wl_min_nm <= wl_res <= wl_max_nm:
                resonance_wls.append(wl_res)

    # Coarse background grid
    wl_bg = np.linspace(wl_min_nm, wl_max_nm, n_grid)

    # Fine grid near each resonance (±0.5 nm, 200 pts each)
    grids = [wl_bg]
    for wl_r in resonance_wls:
        lo = max(wl_min_nm, wl_r - 0.5)
        hi = min(wl_max_nm, wl_r + 0.5)
        grids.append(np.linspace(lo, hi, 200))

    wl_arr = np.unique(np.concatenate(grids))
    wl_arr = wl_arr[(wl_arr >= wl_min_nm) & (wl_arr <= wl_max_nm)]

    alpha_arr = np.array([
        alpha_real(omega_au_from_nm(wl), transitions_list)
        for wl in wl_arr
    ])

    return wl_arr, alpha_arr


def compute_C6(transitions_list, n_xi=2000):
    """
    C₆ = (3/π) ∫₀^∞ [α(iξ)]² dξ   (atomic units)

    Integration via Gauss-Legendre quadrature on a log-mapped grid.
    The integrand peaks near ξ ~ ΔE_min and decays as ξ⁻⁴ at large ξ.
    Log mapping x = log(1 + ξ/ξ₀) with ξ₀ = smallest transition energy
    captures the peak and tail efficiently.
    """
    if not transitions_list:
        return 0.0

    dE_min = min(abs(tr['delta_E_au']) for tr in transitions_list if tr['delta_E_au'] != 0)
    xi0    = dE_min * 0.1   # scale parameter

    def integrand_mapped(x):
        # x in [0, x_max], ξ = xi0*(exp(x) - 1)
        xi  = xi0 * (math.exp(x) - 1.0)
        dxi = xi0 * math.exp(x)   # Jacobian dξ/dx
        a   = alpha_imag_freq(xi, transitions_list)
        return a**2 * dxi

    # x_max: ξ_max = 10 × IE (well beyond where integrand is negligible)
    if transitions_list:
        dE_max = max(abs(tr['delta_E_au']) for tr in transitions_list)
    else:
        dE_max = 1.0
    x_max = math.log(1.0 + 10.0 * dE_max / xi0)

    if SCIPY_AVAILABLE:
        val, err = quad(integrand_mapped, 0, x_max, limit=200, epsrel=1e-6)
    else:
        # Trapezoidal fallback
        x_arr = np.linspace(0, x_max, n_xi)
        y_arr = np.array([integrand_mapped(x) for x in x_arr])
        val   = np.trapezoid(y_arr, x_arr)

    return (3.0 / math.pi) * val


def C6_combining_rule(C6_A, C6_B, alpha0_A, alpha0_B):
    """
    London combining rule for heteronuclear C₆:
    C₆^AB = 2·α_A·α_B·C₆_AA·C₆_BB / (α_B·C₆_AA + α_A·C₆_BB)
    All quantities in atomic units.
    Returns 0 if any input is zero.
    """
    denom = alpha0_B * C6_A + alpha0_A * C6_B
    if denom == 0 or C6_A == 0 or C6_B == 0:
        return 0.0
    return 2.0 * alpha0_A * alpha0_B * C6_A * C6_B / denom

# ============================================================================
# MAGIC WAVELENGTH FINDER
# ============================================================================

def find_magic_wavelengths(trans_ground, trans_excited,
                            wl_min_nm, wl_max_nm, n_scan=5000):
    """
    Find magic wavelengths where α_ground(ω) = α_excited(ω).

    Strategy:
      1. Scan Δα = α_ground - α_excited on a dense grid.
      2. Find sign changes (NaN-safe).
      3. Bisect each bracket to find the root.
      4. Record the slope dΔα/dλ at the root (robustness metric).

    Returns list of dicts with keys:
        wl_nm, alpha_au, delta_alpha_slope_au_per_nm,
        nearby_resonance_nm, nearby_resonance_label, source
    """
    # Collect all resonance wavelengths (avoid plotting through them)
    all_res = []
    for tr in trans_ground + trans_excited:
        dE = abs(tr['delta_E_au'])
        if dE > 0:
            wl_r = nm_from_omega_au(dE)
            if wl_min_nm < wl_r < wl_max_nm:
                all_res.append(wl_r)

    # Dense scan grid
    wl_scan = np.linspace(wl_min_nm, wl_max_nm, n_scan)

    def delta_alpha(wl):
        om = omega_au_from_nm(wl)
        ag = alpha_real(om, trans_ground)
        ae = alpha_real(om, trans_excited)
        if math.isnan(ag) or math.isnan(ae):
            return float('nan')
        return ag - ae

    da_arr = np.array([delta_alpha(wl) for wl in wl_scan])

    magic = []

    for i in range(len(wl_scan) - 1):
        a0, a1 = da_arr[i], da_arr[i+1]
        if math.isnan(a0) or math.isnan(a1):
            continue
        if a0 * a1 >= 0:
            continue  # no sign change

        # Bisect
        lo, hi = wl_scan[i], wl_scan[i+1]
        for _ in range(60):
            mid   = 0.5*(lo + hi)
            a_mid = delta_alpha(mid)
            if math.isnan(a_mid):
                break
            if a_mid == 0.0:
                lo = hi = mid
                break
            if a0 * a_mid < 0:
                hi = mid
            else:
                lo = mid
                a0 = a_mid
        else:
            mid = 0.5*(lo + hi)

        wl_magic = mid

        # Evaluate α at magic wavelength
        om_magic = omega_au_from_nm(wl_magic)
        ag_magic = alpha_real(om_magic, trans_ground)
        ae_magic = alpha_real(om_magic, trans_excited)
        alpha_at_magic = 0.5*(ag_magic + ae_magic) if not (math.isnan(ag_magic) or math.isnan(ae_magic)) else float('nan')

        # Slope dΔα/dλ (numerical) — steeper = more laser-noise sensitive
        dlam   = 0.01   # nm
        slope  = (delta_alpha(wl_magic + dlam) - delta_alpha(wl_magic - dlam)) / (2*dlam)

        # Nearest resonance
        if all_res:
            nearest_res = min(all_res, key=lambda r: abs(r - wl_magic))
            nearest_res_tr = min(
                trans_ground + trans_excited,
                key=lambda tr: abs(nm_from_omega_au(abs(tr['delta_E_au'])) - nearest_res)
                if abs(tr['delta_E_au']) > 0 else float('inf')
            )
            nearest_label = nearest_res_tr.get('other_label', '?')
        else:
            nearest_res   = float('nan')
            nearest_label = '?'

        magic.append(dict(
            wl_nm                     = round(wl_magic, 4),
            alpha_au                  = round(alpha_at_magic, 4) if not math.isnan(alpha_at_magic) else None,
            delta_alpha_slope_au_per_nm = round(slope, 4) if not math.isnan(slope) else None,
            nearby_resonance_nm       = round(nearest_res, 3) if not math.isnan(nearest_res) else None,
            nearby_resonance_label    = nearest_label,
            robustness                = 'robust' if slope is not None and abs(slope) < 50 else
                                        'sensitive' if abs(slope) < 200 else 'very sensitive',
        ))

    # Deduplicate (sometimes the scan finds the same root twice)
    deduped = []
    for m in sorted(magic, key=lambda x: x['wl_nm']):
        if not deduped or abs(m['wl_nm'] - deduped[-1]['wl_nm']) > 0.5:
            deduped.append(m)

    return deduped

# ============================================================================
# MAIN COMPUTATION PER ELEMENT
# ============================================================================

def run_element(element, excited_label=None,
                wl_min=200.0, wl_max=2000.0, n_grid=2000,
                I_ref_kWcm2=10.0, waist_um=1.0, wl_point=None,
                data_dir='data_json'):
    """
    Full polarizability pipeline for one element.
    Returns result dict or None on failure.
    """
    print(f"\n{'='*65}")
    print(f"  {element}")
    print(f"{'='*65}")

    data = load_element(element, data_dir=data_dir)
    if data is None:
        return None

    all_transitions = data['transitions']
    gs_label        = data['gs_label']

    # ── Ground state transitions ────────────────────────────────────────────
    trans_gs = build_transitions_for_state(gs_label, data, all_transitions)
    print(f"  Ground state {gs_label}: {len(trans_gs)} transitions with f>0")

    n_nist    = sum(1 for t in trans_gs if t['source'] in ('NIST', 'precision'))
    n_prec    = sum(1 for t in trans_gs if t['source'] == 'precision')
    n_coulomb = sum(1 for t in trans_gs if t['source'] == 'Coulomb')
    _prec = f"  (precision D-lines: {n_prec})" if n_prec else ""
    print(f"    NIST: {n_nist}{_prec}   Coulomb: {n_coulomb}")

    # ── Static polarizability ───────────────────────────────────────────────
    alpha0 = alpha_real(0.0, trans_gs)
    alpha0_expt = ALPHA0_EXPT.get(element)

    if alpha0_expt:
        err_pct = (alpha0 - alpha0_expt) / alpha0_expt * 100
        status  = '✅' if abs(err_pct) < 10 else '⚠'
        print(f"  α(0) = {alpha0:.2f} au   expt: {alpha0_expt:.2f} au  "
              f"({err_pct:+.1f}%) {status}")
    else:
        print(f"  α(0) = {alpha0:.2f} au   (no experimental reference)")

    # ── C₆ coefficient ──────────────────────────────────────────────────────
    print(f"  Computing C₆ (Casimir-Polder integral)...")
    C6 = compute_C6(trans_gs)
    C6_expt = C6_EXPT.get(element)

    if C6_expt:
        err_pct_c6 = (C6 - C6_expt) / C6_expt * 100
        status_c6  = '✅' if abs(err_pct_c6) < 20 else '⚠'
        print(f"  C₆   = {C6:.1f} au   expt: {C6_expt:.1f} au  "
              f"({err_pct_c6:+.1f}%) {status_c6}")
    else:
        print(f"  C₆   = {C6:.1f} au   (no experimental reference)")

    # ── Excited state transitions & magic wavelengths ──────────────────────
    magic_results = []
    excited_used  = None
    trans_ex      = []

    # Determine which excited state to use
    if excited_label is None:
        # Find the lowest excited state with l = gs_l ± 1 (first dipole-allowed)
        gs_l_int = L_CHAR.get(data['gs_l'], 0)
        candidates = []
        for t in all_transitions:
            if t['lower_label'] == gs_label:
                candidates.append(t['upper_label'])
        if candidates:
            # Pick the one with lowest energy
            def sort_key(lbl):
                E = data['state_energy_au'].get(lbl, -999)
                return E
            candidates.sort(key=sort_key)
            excited_label = candidates[0]

    if excited_label:
        excited_used = excited_label
        trans_ex = build_transitions_for_state(excited_label, data, all_transitions)
        n_ex_nist    = sum(1 for t in trans_ex if t['source'] == 'NIST')
        n_ex_coulomb = sum(1 for t in trans_ex if t['source'] == 'Coulomb')
        print(f"\n  Excited state {excited_label}: {len(trans_ex)} transitions")
        print(f"    NIST: {n_ex_nist}   Coulomb: {n_ex_coulomb}")

        print(f"  Searching for magic wavelengths ({wl_min:.0f}–{wl_max:.0f} nm)...")
        magic_results = find_magic_wavelengths(
            trans_gs, trans_ex, wl_min, wl_max)
        print(f"  Found {len(magic_results)} magic wavelengths")
        for m in magic_results:
            slope_str = f"{m['delta_alpha_slope_au_per_nm']:+.1f}" \
                        if m['delta_alpha_slope_au_per_nm'] is not None else '?'
            print(f"    λ = {m['wl_nm']:.3f} nm   α = {m['alpha_au']} au   "
                  f"slope = {slope_str} au/nm   [{m['robustness']}]   "
                  f"near {m['nearby_resonance_label']}")

    # ── Compute α(ω) spectra ────────────────────────────────────────────────
    print(f"  Computing α(ω) spectrum ({n_grid} points)...")
    wl_gs, alpha_gs = compute_alpha_spectrum(trans_gs, wl_min, wl_max, n_grid)
    if trans_ex:
        wl_ex, alpha_ex = compute_alpha_spectrum(trans_ex, wl_min, wl_max, n_grid)
    else:
        wl_ex, alpha_ex = wl_gs, None

    # ── AC Stark shift / trap depth ─────────────────────────────────────────
    # Reuses trans_gs/trans_ex directly — no JSON round-trip, since this used
    # to be a separate script (acstark.py) that re-loaded polarizability.py's
    # saved output. Merged into one process: same physics, one fewer file.
    I_ref_Wm2 = kWcm2_to_Wm2(I_ref_kWcm2)
    mass_amu  = ATOMIC_MASS_AMU.get(element)
    if mass_amu is None:
        print(f"  ⚠  Mass for {element} unknown — trap frequencies will be skipped")

    if trans_ex and magic_results:
        print(f"  AC Stark cross-check at each magic wavelength "
              f"(U_ground should equal U_excited, independent of intensity):")
        for m in magic_results:
            wl_m = m['wl_nm']
            om   = omega_au_from_nm(wl_m)
            a_gs = alpha_real(om, trans_gs)
            a_ex = alpha_real(om, trans_ex)
            if not (math.isnan(a_gs) or math.isnan(a_ex)):
                U_g = U_to_uK(U_joules(a_gs, I_ref_Wm2))
                U_e = U_to_uK(U_joules(a_ex, I_ref_Wm2))
                match = '✅' if abs(U_g - U_e) < 1e-6 * max(abs(U_g), 1) else '⚠'
                print(f"    λ = {wl_m:8.3f} nm   U_gs = {U_g:+10.3f} μK   "
                      f"U_ex = {U_e:+10.3f} μK   {match}")

    point_result = None
    if wl_point is not None and trans_ex:
        om = omega_au_from_nm(wl_point)
        a_gs = alpha_real(om, trans_gs)
        a_ex = alpha_real(om, trans_ex)
        if math.isnan(a_gs) or math.isnan(a_ex):
            print(f"\n  ⚠  λ = {wl_point} nm sits inside a resonance guard band.")
        else:
            U_gs_J = U_joules(a_gs, I_ref_Wm2)
            U_ex_J = U_joules(a_ex, I_ref_Wm2)
            print(f"\n  AC Stark point calculation @ λ = {wl_point} nm, "
                  f"I = {I_ref_kWcm2} kW/cm²:")
            print(f"    alpha_gs = {a_gs:+.3f} au   U_gs = {U_to_uK(U_gs_J):+.4f} μK "
                  f"= {U_to_kHz_h(U_gs_J):+.4f} kHz (h)")
            print(f"    alpha_ex = {a_ex:+.3f} au   U_ex = {U_to_uK(U_ex_J):+.4f} μK "
                  f"= {U_to_kHz_h(U_ex_J):+.4f} kHz (h)")
            point_result = dict(wl_nm=wl_point, alpha_gs_au=a_gs, alpha_ex_au=a_ex,
                                U_gs_uK=U_to_uK(U_gs_J), U_ex_uK=U_to_uK(U_ex_J))
            if mass_amu is not None and U_gs_J < 0:
                fr, fz, zR = trap_frequencies_Hz(U_gs_J, waist_um, wl_point, mass_amu)
                print(f"    Ground-state trap (waist={waist_um} μm): "
                      f"ω_r/2π = {fr/1e3:.2f} kHz   ω_z/2π = {fz/1e3:.3f} kHz")
                point_result['trap_ground'] = dict(
                    omega_r_Hz=fr, omega_z_Hz=fz, zR_um=zR * 1e6, waist_um=waist_um)
            if mass_amu is not None and U_ex_J < 0:
                fr, fz, zR = trap_frequencies_Hz(U_ex_J, waist_um, wl_point, mass_amu)
                print(f"    Excited-state trap (waist={waist_um} μm): "
                      f"ω_r/2π = {fr/1e3:.2f} kHz   ω_z/2π = {fz/1e3:.3f} kHz")
                point_result['trap_excited'] = dict(
                    omega_r_Hz=fr, omega_z_Hz=fz, zR_um=zR * 1e6, waist_um=waist_um)

    # ── Assemble result dict ────────────────────────────────────────────────
    result = dict(
        element          = element,
        gs_label         = gs_label,
        excited_label    = excited_used,
        alpha0_au        = round(alpha0, 4),
        alpha0_expt_au   = alpha0_expt,
        C6_au            = round(C6, 2),
        C6_expt_au       = C6_expt,
        magic_wavelengths = magic_results,
        transitions_gs   = trans_gs,
        transitions_ex   = trans_ex,
        wl_gs            = wl_gs,
        alpha_gs         = alpha_gs,
        wl_ex            = wl_ex,
        alpha_ex         = alpha_ex,
        wl_min           = wl_min,
        wl_max           = wl_max,
        I_ref_kWcm2      = I_ref_kWcm2,
        mass_amu         = mass_amu,
        waist_um         = waist_um,
        point_result     = point_result,
    )
    return result

# ============================================================================
# PLOTTING — INDIVIDUAL ELEMENT
# ============================================================================

ELEM_COLORS = {
    'Li':'#4df0b0', 'Na':'#f0c54d', 'K':'#4d9ff0',
    'Rb':'#f05a4d', 'Cs':'#c54df0', 'Fr':'#f04d9f',
}

def plot_polarizability(result, out_dir):
    """
    Two-panel dynamic polarizability plot with a live AC Stark / trap-depth
    upgrade built in: a secondary uK axis (right side), an intensity slider,
    and a point calculator with Gaussian-tweezer trap frequencies. This used
    to be two separate scripts (polarizability.py + acstark.py) producing
    two near-duplicate plots — merged into one, since U(omega) is just
    alpha(omega) rescaled by a constant at fixed intensity.
    """
    el       = result['element']
    gs_lbl   = result['gs_label']
    ex_lbl   = result['excited_label']
    magic    = result['magic_wavelengths']
    wl_gs    = result['wl_gs']
    alpha_gs = result['alpha_gs']
    wl_ex    = result['wl_ex']
    alpha_ex = result['alpha_ex']
    alpha0   = result['alpha0_au']
    alpha0_expt = result.get('alpha0_expt_au')
    C6       = result['C6_au']
    C6_ex    = result.get('C6_expt_au')
    wl_min   = result['wl_min']
    wl_max   = result['wl_max']
    trans_gs = result['transitions_gs']
    I_ref    = result['I_ref_kWcm2']
    mass_amu = result.get('mass_amu')

    col = ELEM_COLORS.get(el, '#7777ff')
    has_excited = alpha_ex is not None and ex_lbl is not None

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.10,
        subplot_titles=[
            f'α(ω) — ground state ({gs_lbl})' +
            (f' and excited state ({ex_lbl})' if has_excited else ''),
            'Differential polarizability Δα = α_ground − α_excited'
            if has_excited else '',
        ],
    )

    y_clip = max(abs(alpha0) * 5, 400)

    fig.add_trace(go.Scatter(
        x=wl_gs.tolist(), y=alpha_gs.tolist(), mode='lines',
        line=dict(color=col, width=2.5), name=f'α_ground ({gs_lbl})',
        connectgaps=False,
        hovertemplate='λ = %{x:.2f} nm<br>α_ground = %{y:.2f} au<extra></extra>',
    ), row=1, col=1)

    if has_excited:
        fig.add_trace(go.Scatter(
            x=wl_ex.tolist(), y=alpha_ex.tolist(), mode='lines',
            line=dict(color='#ff6b6b', width=2, dash='dash'),
            name=f'α_excited ({ex_lbl})', connectgaps=False,
            hovertemplate='λ = %{x:.2f} nm<br>α_excited = %{y:.2f} au<extra></extra>',
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=[wl_min, wl_max], y=[alpha0, alpha0], mode='lines',
        line=dict(color=col, width=1.5, dash='dot'),
        name=f'α(0) = {alpha0:.1f} au',
        hovertemplate=f'Static α(0) = {alpha0:.2f} au<extra></extra>', showlegend=True,
    ), row=1, col=1)
    if abs(alpha0) < y_clip:
        fig.add_annotation(
            x=wl_max * 0.99, y=alpha0, xref='x', yref='y',
            text=f'  α(0) = {alpha0:.1f} au', showarrow=False, xanchor='right',
            font=dict(size=11, color=col), bgcolor='rgba(10,13,20,0.75)', borderpad=3,
        )

    if alpha0_expt and abs(alpha0_expt) < y_clip:
        fig.add_trace(go.Scatter(
            x=[wl_min, wl_max], y=[alpha0_expt, alpha0_expt], mode='lines',
            line=dict(color='rgba(200,200,200,0.35)', width=1, dash='dash'),
            name=f'Expt α(0) = {alpha0_expt:.1f} au',
            hovertemplate=f'Expt α(0) = {alpha0_expt:.2f} au<extra></extra>', showlegend=True,
        ), row=1, col=1)

    fig.add_hline(y=0, line_dash='dot', line_color='rgba(255,255,255,0.15)',
                  line_width=1, row=1, col=1)

    # ── Resonance vertical lines (top 8 by fosc, deduplicated) ─────────────
    res_wls, res_seen = [], set()
    for tr in sorted(trans_gs, key=lambda t: t['fosc'], reverse=True):
        dE = abs(tr['delta_E_au'])
        if dE == 0:
            continue
        wl_r = nm_from_omega_au(dE)
        key = round(wl_r, 0)
        if wl_min <= wl_r <= wl_max and key not in res_seen:
            res_seen.add(key)
            res_wls.append((wl_r, tr['other_label']))
        if len(res_wls) >= 8:
            break
    for idx, (wl_r, lbl) in enumerate(sorted(res_wls)):
        fig.add_vline(x=wl_r, line_dash='dot', line_color='rgba(160,160,160,0.18)',
                      line_width=1, row=1, col=1)
        too_close = any(abs(wl_r - wl2) < 40 for wl2, _ in res_wls[:idx])
        if not too_close:
            fig.add_annotation(
                x=wl_r, y=1.0, xref='x', yref='y domain', text=lbl, showarrow=False,
                font=dict(size=9, color='rgba(180,180,180,0.7)'), textangle=0,
                xanchor='center', yanchor='bottom', bgcolor='rgba(10,13,20,0.6)',
                borderpad=2, row=1, col=1,
            )

    # ── Magic wavelength markers (top panel) ────────────────────────────────
    label_positions = []
    for m in magic:
        wl_m = m['wl_nm']
        al_m = m.get('alpha_au') or 0
        al_m_display = max(-y_clip * 0.88, min(y_clip * 0.88, al_m))
        rob = m.get('robustness', 'robust')
        color_m = '#00ff88' if rob == 'robust' else \
                  '#ffcc00' if rob == 'sensitive' else '#ff6644'
        fig.add_vline(x=wl_m, line_dash='dot', line_color=color_m, line_width=1.5,
                      opacity=0.65, row=1, col=1)
        too_close = any(abs(wl_m - px) < 100 for px in label_positions)
        y_label = 0.78 if too_close else 0.87
        label_positions.append(wl_m)
        fig.add_annotation(
            x=wl_m, y=y_label, xref='x', yref='y domain',
            text=f'<b>{wl_m:.1f} nm</b>', showarrow=True, arrowhead=0, arrowwidth=1,
            arrowcolor=color_m, ax=0, ay=20, xanchor='center', yanchor='bottom',
            font=dict(size=10, color=color_m), bgcolor='rgba(10,13,20,0.8)',
            borderpad=3, row=1, col=1,
        )
        fig.add_trace(go.Scatter(
            x=[wl_m], y=[al_m_display], mode='markers',
            marker=dict(color=color_m, size=13, symbol='star',
                        line=dict(color='white', width=1)),
            name=f'★ {wl_m:.1f} nm [{rob}]',
            hovertemplate=(f'<b>Magic wavelength</b><br>λ = {wl_m:.3f} nm<br>'
                           f'α = {al_m:.2f} au<br>Robustness: {rob}<extra></extra>'),
            showlegend=True,
        ), row=1, col=1)

    # ── Bottom panel: Δα ─────────────────────────────────────────────────────
    if has_excited:
        alpha_ex_interp = np.interp(wl_gs, wl_ex, alpha_ex,
                                     left=float('nan'), right=float('nan'))
        delta_alpha = alpha_gs - alpha_ex_interp
        fig.add_trace(go.Scatter(
            x=wl_gs.tolist(), y=delta_alpha.tolist(), mode='lines',
            line=dict(color='#a78bfa', width=2), name='Δα = α_ground − α_excited',
            connectgaps=False,
            hovertemplate='λ = %{x:.2f} nm<br>Δα = %{y:.2f} au<extra></extra>',
        ), row=2, col=1)
        fig.add_hline(y=0, line_dash='dash', line_color='rgba(255,255,255,0.3)',
                      line_width=1.5, row=2, col=1)
        for m in magic:
            wl_m = m['wl_nm']
            rob = m.get('robustness', 'robust')
            color_m = '#00ff88' if rob == 'robust' else \
                      '#ffcc00' if rob == 'sensitive' else '#ff6644'
            fig.add_vline(x=wl_m, line_dash='dot', line_color=color_m, line_width=1.5,
                          opacity=0.7, row=2, col=1)
            fig.add_trace(go.Scatter(
                x=[wl_m], y=[0], mode='markers',
                marker=dict(color=color_m, size=11, symbol='star',
                            line=dict(color='white', width=1)),
                showlegend=False,
                hovertemplate=f'Magic λ = {wl_m:.3f} nm [{rob}]<extra></extra>',
            ), row=2, col=1)

    # ── Layout ──────────────────────────────────────────────────────────────
    c6_str = f"C₆ = {C6:.0f} au" if C6 is not None else ""
    if C6 is not None and C6_ex:
        err = (C6 - C6_ex) / C6_ex * 100
        c6_str += f" (expt: {C6_ex:.0f}, {err:+.1f}%)"
    a0_str = f"α(0) = {alpha0:.1f} au"
    if alpha0_expt:
        err0 = (alpha0 - alpha0_expt) / alpha0_expt * 100
        a0_str += f" (expt: {alpha0_expt:.1f}, {err0:+.1f}%)"
    mass_str = (f'm = {mass_amu:.3f} amu (IUPAC avg) &nbsp;·&nbsp; '
                if mass_amu else '')

    fig.update_layout(
        title=dict(
            text=f'<b>{el} — Dynamic Polarizability α(ω)  ·  AC Stark / Trap Depth</b>'
                 f'<br><sup>{a0_str} &nbsp;·&nbsp; {c6_str} &nbsp;·&nbsp; '
                 f'{mass_str}'
                 f'{len(magic)} magic wavelength(s) &nbsp;·&nbsp; '
                 f'right axis: U(μK) at I = '
                 f'<span id="title-I">{I_ref:.2f}</span> kW/cm²</sup>',
            x=0.5, xanchor='center', font=dict(size=19, color='#e8edf5'),
        ),
        plot_bgcolor='#0f1628', paper_bgcolor='#0a0d14',
        font=dict(color='#c8d0e0', family="'Helvetica Neue', Arial, sans-serif"),
        hovermode='x unified', autosize=True,
        margin=dict(t=120, r=300, b=60, l=90),
        legend=dict(
            bgcolor='rgba(10,13,20,0.88)', bordercolor='#1e2840', borderwidth=1,
            x=1.01, xanchor='left', y=1, yanchor='top', font=dict(size=11),
            title=dict(
                text=('<b>Robustness key</b><br>'
                      '<span style="color:#00ff88">★ Green</span> robust<br>'
                      '<span style="color:#ffcc00">★ Yellow</span> sensitive<br>'
                      '<span style="color:#ff6644">★ Red</span> very sensitive'),
                font=dict(size=10, color='#8a9ac0'),
            ),
        ),
    )

    for row_i in (1, 2):
        fig.update_xaxes(range=[wl_min, wl_max], gridcolor='#1a2540',
                         tickfont=dict(color='#6b7a99'),
                         title_text='Wavelength (nm)' if row_i == 2 else '',
                         row=row_i, col=1)

    fig.update_yaxes(range=[-y_clip, y_clip], gridcolor='#1a2540',
                     zeroline=True, zerolinecolor='#2a3555',
                     tickfont=dict(color='#6b7a99'), title_text='α(ω)  (a₀³)',
                     row=1, col=1)
    da_clip = max(abs(alpha0) * 3, 200)
    fig.update_yaxes(range=[-da_clip, da_clip], gridcolor='#1a2540',
                     zeroline=True, zerolinecolor='#2a3555',
                     tickfont=dict(color='#6b7a99'), title_text='Δα  (a₀³)',
                     row=2, col=1)

    for ann in fig.layout.annotations:
        if hasattr(ann, 'font') and ann.font:
            ann.font.color = '#8a9ac0'
            ann.font.size = 12

    # ── Secondary uK axis (right side), row 1 only — a pure linear rescale of
    #    the alpha axis, so it needs no VISIBLE traces: only its own tick range
    #    is a function of I, updated live by the slider via Plotly.relayout.
    #    BUT Plotly.js never draws an axis that has zero traces referencing
    #    it — the axis object exists in the layout spec and responds to
    #    relayout, but is invisible in the DOM. A single invisible anchor
    #    trace forces Plotly to actually render (and keep responsive) the
    #    axis line, ticks, and title. ──────────────────────────────────────
    factor0 = alpha_to_uK_factor(I_ref)   # uK per unit alpha, at I_ref
    fig.add_trace(go.Scatter(
        x=[wl_min, wl_max], y=[y_clip * factor0, -y_clip * factor0],
        xaxis='x', yaxis='y3', mode='lines', line=dict(width=0),
        showlegend=False, hoverinfo='skip',
    ))   # NOTE: no row=/col= here — make_subplots silently overwrites any
         # explicit xaxis/yaxis on a trace added via row=/col=, which is
         # exactly what discarded yaxis='y3' the first time this was tried.
    fig.update_layout(
        yaxis3=dict(
            overlaying='y', side='right', range=[y_clip * factor0, -y_clip * factor0],
            title=dict(text='U (μK) at I(t)', font=dict(color='#e8b84d', size=12)),
            tickfont=dict(color='#e8b84d'), showgrid=False, anchor='x',
        )
    )
    # Anchor yaxis3 to the top subplot's x-axis / domain explicitly.
    fig.layout.yaxis3.update(domain=fig.layout.yaxis.domain)

    out_file = os.path.join(out_dir, f"{el}_polarizability.html")
    raw_html = fig.to_html(include_plotlyjs='cdn', full_html=True)
    _fs_css = ('<style>html,body{margin:0;padding:0;width:100%;height:100%;'
               'overflow:hidden;background:#0a0d14;}'
               '.plotly-graph-div{width:100vw!important;height:100vh!important;}'
               '</style>')
    raw_html = raw_html.replace('</head>', _fs_css + '</head>', 1)
    raw_html = _inject_control_panel(raw_html, result, factor0)
    with open(out_file, 'w', encoding='utf-8') as fh:
        fh.write(raw_html)
    print(f"  ✓ {out_file}")
    return out_file


def _inject_control_panel(raw_html, result, factor0):
    """
    Intensity slider (rescales only the yaxis3 uK-axis range/title — the
    alpha(omega) traces underneath never change) + point calculator with
    trap frequencies.
    """
    el       = result['element']
    I_base   = result['I_ref_kWcm2']
    wl_gs    = result['wl_gs'].tolist()
    alpha_gs = result['alpha_gs'].tolist()
    wl_ex    = result['wl_ex'].tolist()
    alpha_ex = result['alpha_ex'].tolist()
    mass_amu = result['mass_amu']
    waist0   = result['waist_um']

    css = """
<style>
#as-ctrl {
  position:fixed; top:12px; right:12px; width:290px;
  max-height:calc(100vh - 24px); overflow-y:auto;
  background:rgba(10,10,22,0.94); border:1px solid #2a2a4a; border-radius:10px;
  padding:12px 14px; font-family:'Helvetica Neue',Arial,sans-serif; font-size:12px;
  color:#ccc; box-shadow:0 4px 20px rgba(0,0,0,0.65); z-index:9999;
  backdrop-filter:blur(6px);
}
#as-ctrl.collapsed { width:auto; padding:8px 12px; max-height:none; }
#as-ctrl.collapsed>*:not(h3) { display:none; }
#as-ctrl h3 { margin:0; font-size:13px; color:#e0e8ff;
  border-bottom:1px solid #2a2a5a; padding-bottom:5px;
  display:flex; align-items:center; gap:8px; cursor:pointer; }
#as-ctrl.collapsed h3 { border-bottom:none; padding-bottom:0; margin:0; }
#as-ctrl .tog { margin-left:auto; font-size:11px; color:#667; user-select:none; }
#as-ctrl .sec { margin-top:9px; margin-bottom:10px; }
#as-ctrl .st { font-weight:bold; color:#7a9eff; font-size:10px;
  text-transform:uppercase; letter-spacing:.6px; margin-bottom:4px; }
#as-ctrl input[type=range] { width:100%; accent-color:#4a7ec7; }
#as-ctrl input[type=number] { width:70px; background:#111830; border:1px solid #2a3060;
  border-radius:4px; color:#ccc; font-size:11px; padding:3px 6px; }
#as-ctrl button { padding:5px 10px; font-size:11px; border:1px solid #2a3060;
  border-radius:5px; background:#111830; color:#aac; cursor:pointer; margin-top:6px; }
#as-ctrl button:hover { background:#1a2850; color:#e0e8ff; }
#as-out { background:#0d0d20; border:1px solid #2a2a4a; border-radius:6px;
  padding:8px 10px; font-size:11px; color:#9ab; line-height:1.7; margin-top:6px; }
#as-out b { color:#e0e8ff; }
</style>"""

    html = f"""
<div id="as-ctrl" class="collapsed">
  <h3 onclick="asToggle()">☀ AC Stark <span class="tog" id="as-tog">▼</span></h3>
  <div class="sec">
    <div class="st">Trap intensity &nbsp;<span id="as-ival">{I_base:.2f}</span> kW/cm²</div>
    <input type="range" id="as-Islider" min="-2" max="3" value="{math.log10(I_base):.4f}" step="0.02"
           oninput="asOnIntensity(this.value)">
    <div style="display:flex;justify-content:space-between;font-size:10px;color:#556">
      <span>0.01</span><span>1000 kW/cm²</span>
    </div>
    <div style="font-size:10px;color:#667;margin-top:4px;">
      Right-hand μK axis rescales live — the α(ω) curves themselves never change.
    </div>
  </div>
  <div class="sec" style="border-top:1px solid #1a1a3a;padding-top:8px;">
    <div class="st">Point calculator</div>
    <div style="display:flex;gap:6px;align-items:center;margin-bottom:5px;">
      <span style="width:60px;color:#89a">λ (nm)</span>
      <input type="number" id="as-wl" value="{(wl_gs[0]+wl_gs[-1])/2:.1f}" step="1">
    </div>
    <div style="display:flex;gap:6px;align-items:center;margin-bottom:5px;">
      <span style="width:60px;color:#89a">waist (μm)</span>
      <input type="number" id="as-waist" value="{waist0}" step="0.1">
    </div>
    <button onclick="asCalc()">▶ Calculate</button>
    <div id="as-out">Enter a wavelength and click Calculate.</div>
  </div>
</div>"""

    js = f"""
<script>
(function(){{
  const WL_GS = {json.dumps(wl_gs)};
  const A_GS  = {json.dumps(alpha_gs)};
  const WL_EX = {json.dumps(wl_ex)};
  const A_EX  = {json.dumps(alpha_ex)};
  const MASS_AMU = {mass_amu if mass_amu else 'null'};
  const A0 = 5.29177210903e-11, C_SI = 2.99792458e8, KB_SI = 1.380649e-23,
        H_SI = 6.62607015e-34, AMU_KG = 1.66053906660e-27;
  const U_PREF = 2*Math.PI*A0**3/C_SI;

  function gd(){{ return document.querySelector('.plotly-graph-div'); }}

  window.asToggle = function(){{
    const el  = document.getElementById('as-ctrl');
    const tog = document.getElementById('as-tog');
    el.classList.toggle('collapsed');
    tog.textContent = el.classList.contains('collapsed') ? '▼' : '▲';
  }};

  function alphaToUKFactor(I_kWcm2){{
    const I_Wm2 = I_kWcm2*1e3/1e-4;
    return -U_PREF*I_Wm2/KB_SI*1e6;
  }}

  window.asOnIntensity = function(logI){{
    const I = Math.pow(10, parseFloat(logI));
    document.getElementById('as-ival').textContent = I.toFixed(2);
    const titleI = document.getElementById('title-I');
    if (titleI) titleI.textContent = I.toFixed(2);
    const g = gd(); if(!g||!g.layout) return;
    const yr = g.layout.yaxis.range;   // alpha axis range never changes
    const factor = alphaToUKFactor(I);
    Plotly.relayout(g, {{'yaxis3.range': [yr[1]*factor, yr[0]*factor]}});
  }};

  function asInterp(xs, ys, x){{
    if (x <= xs[0]) return ys[0];
    if (x >= xs[xs.length-1]) return ys[ys.length-1];
    let lo=0, hi=xs.length-1;
    while (hi-lo>1){{ const mid=(lo+hi)>>1; if (xs[mid] <= x) lo=mid; else hi=mid; }}
    const t = (x-xs[lo])/(xs[hi]-xs[lo]);
    return ys[lo]*(1-t) + ys[hi]*t;
  }}

  window.asCalc = function(){{
    const wl = parseFloat(document.getElementById('as-wl').value);
    const waist_um = parseFloat(document.getElementById('as-waist').value);
    const logI = parseFloat(document.getElementById('as-Islider').value);
    const I_kWcm2 = Math.pow(10, logI);
    const out = document.getElementById('as-out');
    if (!isFinite(wl)) {{ out.innerHTML = 'Enter a valid wavelength.'; return; }}

    const a_gs = asInterp(WL_GS, A_GS, wl);
    const a_ex = asInterp(WL_EX, A_EX, wl);
    const factor = alphaToUKFactor(I_kWcm2);
    const U_gs = a_gs * factor, U_ex = a_ex * factor;

    let html = `<b>λ = ${{wl.toFixed(2)}} nm</b>, I = ${{I_kWcm2.toFixed(2)}} kW/cm²<br>`
      + `α_gs = ${{a_gs.toFixed(2)}} au &nbsp; U_gs = <b>${{U_gs.toFixed(3)}} μK</b><br>`
      + `α_ex = ${{a_ex.toFixed(2)}} au &nbsp; U_ex = <b>${{U_ex.toFixed(3)}} μK</b>`;

    if (MASS_AMU) {{
      const w0 = waist_um*1e-6, wl_m = wl*1e-9, m = MASS_AMU*AMU_KG;
      const zR = Math.PI*w0*w0/wl_m;
      function trapFreq(U_uK){{
        const U0_J = Math.abs(U_uK)*1e-6*KB_SI;
        const wr = Math.sqrt(4*U0_J/(m*w0*w0));
        const wz = Math.sqrt(2*U0_J/(m*zR*zR));
        return [wr/(2*Math.PI), wz/(2*Math.PI)];
      }}
      if (U_gs < 0) {{
        const [fr,fz] = trapFreq(U_gs);
        html += `<br>Ground trap (w0=${{waist_um}}μm): ω_r/2π=${{(fr/1e3).toFixed(2)}} kHz, `
              + `ω_z/2π=${{(fz/1e3).toFixed(3)}} kHz`;
      }}
      if (U_ex < 0) {{
        const [fr,fz] = trapFreq(U_ex);
        html += `<br>Excited trap (w0=${{waist_um}}μm): ω_r/2π=${{(fr/1e3).toFixed(2)}} kHz, `
              + `ω_z/2π=${{(fz/1e3).toFixed(3)}} kHz`;
      }}
    }} else {{
      html += `<br><span style="color:#667">mass unknown — no trap freq.</span>`;
    }}
    out.innerHTML = html;
  }};
}})();
</script>"""

    raw_html = raw_html.replace('</head>', css + '</head>', 1)
    raw_html = raw_html.replace('</body>', html + js + '</body>', 1)
    return raw_html

# ============================================================================
# SAVE JSON  (small side file — does not touch <el>_polarizability.json)
# ============================================================================


def plot_magic_wavelengths(result, out_dir):
    """
    Plot 2: Grotrian-style magic wavelength atlas.
    x = wavelength (nm), y = polarizability.
    Shows where magic wavelengths fall relative to atomic resonances.
    """
    el    = result['element']
    magic = result['magic_wavelengths']
    if not magic:
        return None

    gs_lbl = result['gs_label']
    ex_lbl = result['excited_label'] or '?'
    col    = ELEM_COLORS.get(el, '#7777ff')

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.65, 0.35],
        subplot_titles=[
            f'Differential Polarizability Δα(ω) = α_ground − α_excited ({gs_lbl} → {ex_lbl})',
            'Magic Wavelength Atlas — dΔα/dλ slope (robustness indicator)',
        ],
        vertical_spacing=0.12,
    )

    wl_gs    = result['wl_gs']
    alpha_gs = result['alpha_gs']
    wl_ex    = result['wl_ex']
    alpha_ex = result['alpha_ex']

    # Compute differential α on the shared grid
    if alpha_ex is not None:
        # Interpolate ex onto gs grid
        alpha_ex_interp = np.interp(wl_gs, wl_ex, alpha_ex,
                                     left=float('nan'), right=float('nan'))
        delta_alpha = alpha_gs - alpha_ex_interp
    else:
        delta_alpha = alpha_gs * 0

    # Panel 1: Δα vs wavelength
    fig.add_trace(go.Scatter(
        x=wl_gs.tolist(), y=delta_alpha.tolist(),
        mode='lines', line=dict(color=col, width=2),
        name='Δα(ω)', connectgaps=False,
        hovertemplate='λ = %{x:.2f} nm<br>Δα = %{y:.2f} au<extra></extra>',
    ), row=1, col=1)

    fig.add_hline(y=0, line_dash='dash',
                  line_color='rgba(255,255,255,0.3)', line_width=1.5,
                  row=1, col=1)

    # Magic wavelength markers on panel 1
    for m in magic:
        wl_m  = m['wl_nm']
        rob   = m['robustness']
        color_m = '#00ff88' if rob == 'robust' else \
                  '#ffcc00' if rob == 'sensitive' else '#ff6644'
        fig.add_vline(x=wl_m, line_dash='dot',
                      line_color=color_m, line_width=1.5, opacity=0.7,
                      row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[wl_m], y=[0],
            mode='markers+text',
            marker=dict(color=color_m, size=12, symbol='star',
                        line=dict(color='white', width=1)),
            text=[f"{wl_m:.1f}"],
            textposition='top center',
            textfont=dict(size=9, color=color_m),
            showlegend=False,
            hovertemplate=(
                f'λ = {wl_m:.3f} nm<br>'
                f'α = {m["alpha_au"]} au<br>'
                f'[{rob}]<extra></extra>'
            ),
        ), row=1, col=1)

    # Panel 2: slope bar chart
    wls    = [m['wl_nm'] for m in magic]
    slopes = [abs(m['delta_alpha_slope_au_per_nm'] or 0) for m in magic]
    robs   = [m['robustness'] for m in magic]
    colors_bar = ['#00ff88' if r == 'robust' else
                  '#ffcc00' if r == 'sensitive' else '#ff6644'
                  for r in robs]

    fig.add_trace(go.Bar(
        x=wls, y=slopes,
        marker_color=colors_bar,
        marker_line_color='rgba(255,255,255,0.2)',
        marker_line_width=1,
        name='|dΔα/dλ| (au/nm)',
        hovertemplate='λ = %{x:.2f} nm<br>|slope| = %{y:.1f} au/nm<extra></extra>',
        text=[f"{s:.0f}" for s in slopes],
        textposition='outside',
        textfont=dict(color='#c8d0e0', size=9),
    ), row=2, col=1)

    # Threshold lines
    for thresh, label, color_t in [(50, 'robust threshold', '#00ff88'),
                                    (200, 'sensitive threshold', '#ffcc00')]:
        fig.add_hline(y=thresh, line_dash='dot',
                      line_color=color_t, line_width=1, opacity=0.5,
                      row=2, col=1)
        fig.add_annotation(
            x=result['wl_max'], y=thresh, xref='x2', yref='y2',
            text=label, showarrow=False,
            font=dict(size=9, color=color_t),
            xanchor='right',
        )

    fig.update_layout(
        title=dict(
            text=f'<b>{el} — Magic Wavelength Atlas</b>'
                 f'<br><sup>{len(magic)} magic wavelength(s) · '
                 f'{gs_lbl} → {ex_lbl} transition</sup>',
            x=0.5, xanchor='center', font=dict(size=18, color='#e8edf5'),
        ),
        plot_bgcolor='#0f1628', paper_bgcolor='#0a0d14',
        font=dict(color='#c8d0e0'),
        autosize=True,
        margin=dict(t=100, r=80),
        showlegend=False,
    )
    fig.update_xaxes(gridcolor='#1a2540', tickfont=dict(color='#6b7a99'))
    fig.update_yaxes(gridcolor='#1a2540', tickfont=dict(color='#6b7a99'))
    fig.update_xaxes(title_text='Wavelength (nm)', row=2, col=1)
    fig.update_yaxes(title_text='Δα (au)', row=1, col=1)
    fig.update_yaxes(title_text='|dΔα/dλ| (au/nm)', row=2, col=1)

    out_file = os.path.join(out_dir, f"{el}_magic_wavelengths.html")
    raw_html = fig.to_html(include_plotlyjs='cdn', full_html=True)
    _fs_css = ('<style>html,body{margin:0;padding:0;width:100%;height:100%;'
               'overflow:hidden;background:#0a0d14;}'
               '.plotly-graph-div{width:100vw!important;height:100vh!important;}'
               '</style>')
    raw_html = raw_html.replace('</head>', _fs_css + '</head>', 1)
    with open(out_file, 'w', encoding='utf-8') as fh:
        fh.write(raw_html)
    print(f"  ✓ {out_file}")
    return out_file

# ============================================================================
# PLOTTING — CROSS-ELEMENT (saved to plots/ directly)
# ============================================================================

def plot_alpha_validation(results, out_dir):
    """
    Bar chart: computed α(0) vs experimental for all elements processed.
    """
    els      = [r['element'] for r in results]
    computed = [r['alpha0_au'] for r in results]
    expt     = [r.get('alpha0_expt_au') for r in results]
    errors   = []
    for c, e in zip(computed, expt):
        if e:
            errors.append(abs(c - e) / e * 100)
        else:
            errors.append(None)

    fig = go.Figure()

    # Computed bars
    fig.add_trace(go.Bar(
        x=els, y=computed,
        name='Computed (this pipeline)',
        marker_color=[ELEM_COLORS.get(el, '#7777ff') for el in els],
        marker_line_color='rgba(255,255,255,0.3)',
        marker_line_width=1,
        hovertemplate='<b>%{x}</b><br>Computed α(0) = %{y:.2f} au<extra></extra>',
    ))

    # Experimental markers
    expt_clean = [e if e else 0 for e in expt]
    fig.add_trace(go.Scatter(
        x=els, y=expt_clean,
        mode='markers',
        marker=dict(symbol='diamond', size=12, color='white',
                    line=dict(color='#4df0b0', width=2)),
        name='Experimental',
        hovertemplate='<b>%{x}</b><br>Expt α(0) = %{y:.2f} au<extra></extra>',
    ))

    # Error annotation
    for el, err in zip(els, errors):
        if err is not None:
            fig.add_annotation(
                x=el, y=0, xref='x', yref='paper',
                text=f'{err:.1f}%',
                showarrow=False,
                font=dict(size=10,
                          color='#4df0b0' if err < 5 else
                          '#f0c54d' if err < 15 else '#f05a4d'),
                yshift=8,
            )

    fig.update_layout(
        title=dict(
            text='<b>Static Polarizability α(0) — Computed vs Experimental</b>'
                 '<br><sup>Error % shown below bars · '
                 '◆ = experimental value · bars = this pipeline</sup>',
            x=0.5, xanchor='center', font=dict(size=18, color='#e8edf5'),
        ),
        xaxis=dict(title='Element', gridcolor='#1a2540',
                   tickfont=dict(color='#c8d0e0')),
        yaxis=dict(title='α(0) (atomic units a₀³)', gridcolor='#1a2540',
                   tickfont=dict(color='#6b7a99')),
        plot_bgcolor='#0f1628', paper_bgcolor='#0a0d14',
        font=dict(color='#c8d0e0'),
        barmode='group',
        autosize=True,
        legend=dict(bgcolor='rgba(10,13,20,0.8)', bordercolor='#1e2840'),
        margin=dict(t=100),
    )

    out_file = os.path.join(out_dir, 'alpha_validation.html')
    raw_html = fig.to_html(include_plotlyjs='cdn', full_html=True)
    _fs_css = ('<style>html,body{margin:0;padding:0;width:100%;height:100%;'
               'overflow:hidden;background:#0a0d14;}'
               '.plotly-graph-div{width:100vw!important;height:100vh!important;}'
               '</style>')
    raw_html = raw_html.replace('</head>', _fs_css + '</head>', 1)
    with open(out_file, 'w', encoding='utf-8') as fh:
        fh.write(raw_html)
    print(f"\n  ✓ {out_file}")
    return out_file


def plot_C6_matrix(results, out_dir, heteronuclear=False):
    """
    Heatmap of C₆ coefficients.
    Diagonal: homonuclear (computed directly).
    Off-diagonal: heteronuclear via combining rule (if --heteronuclear).
    """
    els      = [r['element'] for r in results]
    alpha0s  = {r['element']: r['alpha0_au'] for r in results}
    C6s      = {r['element']: r['C6_au'] for r in results}

    n = len(els)
    matrix = np.zeros((n, n))
    text   = [[''] * n for _ in range(n)]

    for i, el_i in enumerate(els):
        for j, el_j in enumerate(els):
            if i == j:
                val = C6s[el_i]
            elif heteronuclear and i < j:
                val = C6_combining_rule(
                    C6s[el_i], C6s[el_j],
                    alpha0s[el_i], alpha0s[el_j]
                )
            else:
                val = C6_combining_rule(
                    C6s[el_i], C6s[el_j],
                    alpha0s[el_i], alpha0s[el_j]
                ) if heteronuclear else 0.0
            matrix[i, j] = val
            if val > 0:
                expt_str = ''
                if i == j and el_i in C6_EXPT:
                    expt_str = f'<br>expt: {C6_EXPT[el_i]:.0f}'
                text[i][j] = f'{val:.0f}{expt_str}'
            else:
                text[i][j] = '—'

    # Use log scale for colour (C₆ spans orders of magnitude across elements)
    log_matrix = np.where(matrix > 0, np.log10(matrix), 0)

    fig = go.Figure(go.Heatmap(
        z=log_matrix,
        x=els, y=els,
        text=text,
        texttemplate='%{text}',
        textfont=dict(size=11, color='white'),
        colorscale='Viridis',
        colorbar=dict(
            title='log₁₀(C₆) [au]',
            tickvals=[3, 3.5, 4, 4.5],
            ticktext=['10³', '10³·⁵', '10⁴', '10⁴·⁵'],
            tickfont=dict(color='#c8d0e0'),
            title_font=dict(color='#c8d0e0'),
        ),
        hovertemplate='<b>%{y}–%{x}</b><br>C₆ = %{text} au<extra></extra>',
    ))

    # Diagonal border to distinguish homo from heteronuclear
    for i in range(n):
        fig.add_shape(type='rect',
                      x0=i-0.5, x1=i+0.5, y0=i-0.5, y1=i+0.5,
                      line=dict(color='rgba(77,240,176,0.6)', width=2))

    title_str = ('Full cross-element C₆ matrix (off-diagonal: London combining rule)'
                 if heteronuclear else
                 'Homonuclear C₆ coefficients (diagonal only)')

    fig.update_layout(
        title=dict(
            text=f'<b>van der Waals C₆ Coefficients (atomic units)</b>'
                 f'<br><sup>{title_str}</sup>',
            x=0.5, xanchor='center', font=dict(size=18, color='#e8edf5'),
        ),
        xaxis=dict(tickfont=dict(color='#c8d0e0'), title='Element B'),
        yaxis=dict(tickfont=dict(color='#c8d0e0'), title='Element A',
                   autorange='reversed'),
        plot_bgcolor='#0f1628', paper_bgcolor='#0a0d14',
        font=dict(color='#c8d0e0'),
        autosize=True,
        margin=dict(t=100),
    )

    out_file = os.path.join(out_dir, 'C6_matrix.html')
    raw_html = fig.to_html(include_plotlyjs='cdn', full_html=True)
    _fs_css = ('<style>html,body{margin:0;padding:0;width:100%;height:100%;'
               'overflow:hidden;background:#0a0d14;}'
               '.plotly-graph-div{width:100vw!important;height:100vh!important;}'
               '</style>')
    raw_html = raw_html.replace('</head>', _fs_css + '</head>', 1)
    with open(out_file, 'w', encoding='utf-8') as fh:
        fh.write(raw_html)
    print(f"  ✓ {out_file}")
    return out_file

# ============================================================================
# SAVE JSON
# ============================================================================

def _trim_transitions(trans_list):
    """
    Keep only the fields needed to recompute alpha(omega) / alpha(i*xi) at any
    frequency: delta_E_au, fosc, sign, other_label, source, wl_nm. These are
    exactly the fields alpha_real()/alpha_imag_freq() read — nothing else.
    """
    return [
        dict(delta_E_au=t['delta_E_au'], fosc=t['fosc'], sign=t['sign'],
             other_label=t['other_label'], source=t['source'], wl_nm=t['wl_nm'])
        for t in trans_list
    ]



# ============================================================================
# MAIN
# ============================================================================

def save_json(result, data_dir='data_json'):
    """Save polarizability results to data_json/<element>_polarizability.json."""
    el = result['element']
    out = dict(
        element=el,
        gs_label=result['gs_label'],
        excited_label=result['excited_label'],
        alpha0_au=result['alpha0_au'],
        alpha0_expt_au=result.get('alpha0_expt_au'),
        C6_au=result['C6_au'],
        C6_expt_au=result.get('C6_expt_au'),
        magic_wavelengths=result['magic_wavelengths'],
        n_gs_transitions=len(result['transitions_gs']),
        n_ex_transitions=len(result['transitions_ex']),
        wl_range_nm=[result['wl_min'], result['wl_max']],
        I_ref_kWcm2=result.get('I_ref_kWcm2'),
        point_result=result.get('point_result'),
        transitions_gs=_trim_transitions(result['transitions_gs']),
        transitions_ex=_trim_transitions(result['transitions_ex']),
    )
    path = os.path.join(data_dir, f"{el}_polarizability.json")
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  ✓ {path}")


def main():
    parser = argparse.ArgumentParser(
        description='Dynamic polarizability, C₆, magic wavelengths, and AC Stark / trap depth')
    parser.add_argument('element', nargs='?', default=None)
    parser.add_argument('--excited', default=None)
    parser.add_argument('--wl-min', type=float, default=200.0)
    parser.add_argument('--wl-max', type=float, default=2000.0)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--heteronuclear', action='store_true')
    parser.add_argument('--data-dir', default='data_json')
    parser.add_argument('--n-grid', type=int, default=2000)
    parser.add_argument('--intensity', type=float, default=10.0)
    parser.add_argument('--waist', type=float, default=1.0)
    parser.add_argument('--wl', type=float, default=None)
    parser.add_argument('--state', default=None,
                        help='Print alpha(0) for one label and exit (e.g. 20s)')
    args = parser.parse_args()

    data_dir = args.data_dir
    os.makedirs('plots', exist_ok=True)
    print(f"polarizability.py v{__version__}")

    from species import require_neutral_species

    if args.state:
        if args.element is None:
            print("Provide an element with --state, e.g. python polarizability.py Na --state 20s")
            sys.exit(1)
        el = require_neutral_species(args.element, "polarizability.py")["species_id"]
        data = load_element(el, data_dir)
        if data is None:
            sys.exit(1)
        lab = args.state.strip()
        if lab not in data['state_energy_au']:
            print(f"  State '{lab}' not in {el} rydberg set")
            sys.exit(1)
        trans = build_transitions_for_state(lab, data, data['transitions'])
        a0 = alpha0_static(trans)
        valence = [t for t in trans if t.get('other_label') != 'core']
        print(f"  {el} {lab}: alpha(0) = {a0:.6g} au  "
              f"({len(valence)} valence + core partners)")
        sys.exit(0)

    if args.all:
        elements = get_available_elements(data_dir)
        if not elements:
            print(f"No elements found in {data_dir}. Run rydberg.py + transitions.py first.")
            sys.exit(1)
        print("Running all elements:", ", ".join(elements))
    else:
        if args.element is None:
            available = get_available_elements(data_dir)
            print("Available elements:", ", ".join(available) if available else "none")
            args.element = input("Enter element symbol: ").strip()
        elements = [require_neutral_species(args.element, "polarizability.py")["species_id"]]

    all_results = []
    for el in elements:
        el_out_dir = os.path.join("plots", el)
        os.makedirs(el_out_dir, exist_ok=True)
        result = run_element(
            el,
            excited_label=args.excited,
            wl_min=args.wl_min,
            wl_max=args.wl_max,
            n_grid=args.n_grid,
            I_ref_kWcm2=args.intensity,
            waist_um=args.waist,
            wl_point=args.wl,
            data_dir=data_dir,
        )
        if result is None:
            continue
        save_json(result, data_dir)
        plot_polarizability(result, el_out_dir)
        if result["magic_wavelengths"]:
            plot_magic_wavelengths(result, el_out_dir)
        all_results.append(result)

    if len(all_results) > 1:
        print("\n" + "=" * 65)
        print("  Generating cross-element plots → plots/")
        print("=" * 65)
        plot_alpha_validation(all_results, "plots")
        plot_C6_matrix(all_results, "plots", heteronuclear=args.heteronuclear)
    elif len(all_results) == 1:
        plot_alpha_validation(all_results, "plots")

    print("\n" + "=" * 65)
    print("  Summary")
    print("=" * 65)
    for r in all_results:
        el = r["element"]
        print(f"    plots/{el}/{el}_polarizability.html")
        print(f"    data_json/{el}_polarizability.json")
    print("\n" + "=" * 65 + "\n")


if __name__ == "__main__":
    main()
