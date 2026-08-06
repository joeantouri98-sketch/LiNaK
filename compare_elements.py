#!/usr/bin/env python3
"""
compare_elements.py
-------------------
Generates a cross-element comparison table for Group 1 alkali metals:
Li, Na, K, Rb, Cs, Fr

For each element extracts from data_json/{element}_lifetimes.json:
  - D1 (np_1/2) and D2 (np_3/2) wavelengths, lifetimes, A coefficients
  - Natural linewidth Δν_nat, σ_peak(nat), I_sat(nat)
  - Doppler linewidth Δν_D (T=300K), σ_peak(D), I_sat(D)
  - SOC fine-structure splitting ΔE_SOC
  - Quantum defects δ_s, δ_p, δ_d from data_json/{element}_rydberg.json
  - Maximum photon scattering rate R_scatt = A/2 (at I→∞ on resonance)
  - Recoil velocity v_rec = ħk/m and recoil energy E_rec

Falls back to hardcoded experimental literature values for elements
whose pipeline has not been run yet.

Outputs:
  data_json/comparison_table.json    — machine-readable summary
  plots/comparison_table.html        — interactive sortable HTML table

Usage:
  python compare_elements.py
  python compare_elements.py --T 500      (Doppler at 500 K)
  python compare_elements.py --elements Na K Rb
"""

import json
import os
import sys
import math
import argparse

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(description='Cross-element alkali comparison table')
parser.add_argument('--T',        type=float, default=300.0,
                    help='Temperature for Doppler broadening in K (default: 300)')
parser.add_argument('--elements', nargs='+',
                    default=['Li', 'Na', 'K', 'Rb', 'Cs', 'Fr'],
                    help='Elements to include (default: all 6 alkalis)')
parser.add_argument('--data-dir', default='data_json',
                    help='Directory containing JSON data files (default: data_json)')
parser.add_argument('--output-dir', default='plots',
                    help='Output directory for HTML table (default: plots)')
args = parser.parse_args()

from species import require_neutral_species

T_K      = args.T
elements = [
    require_neutral_species(el, "compare_elements.py")["species_id"]
    for el in args.elements
]
DATA_DIR = args.data_dir
OUT_DIR  = args.output_dir
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

from constants import (
    KB_EV as kB,
    AMU_KG as amu,
    C_SI as c,
    EV_TO_J as eV_J,
    H_SI as h,
    HBAR_SI as hbar,
    HC_EV_NM as HC_nm,
)
from element_data import ELEMENTS

pi = math.pi
_Z_BY_SYMBOL = {sym: Z for Z, (sym, _q, _m) in ELEMENTS.items()}

# ============================================================================
# LITERATURE / FALLBACK VALUES
# ============================================================================
# Used when pipeline JSON is not available.
# Sources: Steck D-line data sheets, NIST ASD, Sansonetti 2011.
# fosc_* are J-resolved absorption oscillator strengths (not multiplet totals).

LITERATURE = {
    'Li': {
        'Z': 3, 'mass_amu': 6.941, 'gs': '2s', 'np': '2p',
        'IE_eV': 5.391714,
        'D1_nm': 670.976,   'D2_nm': 670.962,
        'tau_D1_ns': 27.10,  'tau_D2_ns': 27.10,
        'fosc_D1': 0.2490,   'fosc_D2': 0.4980,
        'g_upper_D1': 2, 'g_lower_D1': 2,
        'g_upper_D2': 4, 'g_lower_D2': 2,
        'delta_s': 0.3995,   'delta_p': 0.0472,
        'soc_meV': 0.043,
        'source': 'literature',
    },
    'Na': {
        'Z': 11, 'mass_amu': 22.990, 'gs': '3s', 'np': '3p',
        'IE_eV': 5.139076,
        'D1_nm': 589.7558, 'D2_nm': 589.1583,
        'tau_D1_ns': 16.299, 'tau_D2_ns': 16.249,
        'fosc_D1': 0.3200,   'fosc_D2': 0.6410,
        'g_upper_D1': 2, 'g_lower_D1': 2,
        'g_upper_D2': 4, 'g_lower_D2': 2,
        'delta_s': 1.3478,   'delta_p': 0.8551,
        'soc_meV': 2.19,
        'source': 'literature',
    },
    'K': {
        'Z': 19, 'mass_amu': 39.098, 'gs': '4s', 'np': '4p',
        'IE_eV': 4.340664,
        'D1_nm': 770.1083, 'D2_nm': 766.7000,
        'tau_D1_ns': 26.72,  'tau_D2_ns': 26.37,
        'fosc_D1': 0.3320,   'fosc_D2': 0.6660,
        'g_upper_D1': 2, 'g_lower_D1': 2,
        'g_upper_D2': 4, 'g_lower_D2': 2,
        'delta_s': 2.1800,   'delta_p': 1.7130,
        'soc_meV': 7.23,
        'source': 'literature',
    },
    'Rb': {
        'Z': 37, 'mass_amu': 85.468, 'gs': '5s', 'np': '5p',
        'IE_eV': 4.177128,
        'D1_nm': 794.9788, 'D2_nm': 780.2414,
        'tau_D1_ns': 27.70,  'tau_D2_ns': 26.24,
        'fosc_D1': 0.3424,   'fosc_D2': 0.6958,
        'g_upper_D1': 2, 'g_lower_D1': 2,
        'g_upper_D2': 4, 'g_lower_D2': 2,
        'delta_s': 3.1311,   'delta_p': 2.6549,
        'soc_meV': 29.43,
        'source': 'literature',
    },
    'Cs': {
        'Z': 55, 'mass_amu': 132.905, 'gs': '6s', 'np': '6p',
        'IE_eV': 3.893905,
        'D1_nm': 894.5930, 'D2_nm': 852.3471,
        'tau_D1_ns': 34.894, 'tau_D2_ns': 30.462,
        'fosc_D1': 0.3435,   'fosc_D2': 0.7142,
        'g_upper_D1': 2, 'g_lower_D1': 2,
        'g_upper_D2': 4, 'g_lower_D2': 2,
        'delta_s': 4.0493,   'delta_p': 3.5915,
        'soc_meV': 68.70,   # 554.04 cm^-1 NIST ASD
        'source': 'literature',
    },
    'Fr': {
        'Z': 87, 'mass_amu': 223.000, 'gs': '7s', 'np': '7p',
        'IE_eV': 4.072741,
        'D1_nm': 817.2,    'D2_nm': 718.2,
        'tau_D1_ns': 29.45, 'tau_D2_ns': 21.02,   # Simsarian 1999
        'fosc_D1': 0.340,   'fosc_D2': 0.736,
        'g_upper_D1': 2, 'g_lower_D1': 2,
        'g_upper_D2': 4, 'g_lower_D2': 2,
        'delta_s': 4.9731,  'delta_p': 4.4965,
        'soc_meV': 209.10,  # 1686.6 cm^-1 NIST ASD / Sansonetti 2007
        'source': 'literature',
    },
}

# ============================================================================
# PHYSICS FUNCTIONS
# ============================================================================

def doppler_fwhm_Hz(wl_nm, T_K, mass_amu):
    """Doppler FWHM in Hz (Gaussian profile)."""
    nu0   = c / (wl_nm * 1e-9)
    mc2   = mass_amu * amu * c**2 / eV_J   # eV
    return nu0 * math.sqrt(8 * math.log(2) * kB * T_K / mc2)

def sigma_peak_nat(wl_nm, g_upper, g_lower):
    """
    Peak absorption cross-section for natural (Lorentzian) broadening [m²].

    Matches lifetimes.py with Δν_nat = A/(2π):
      σ = (λ²/8π)·(g_u/g_l)·(A/Δν_nat) = (λ²/4)·(g_u/g_l)
    Isotropic / m_J-averaged light.
    """
    return (wl_nm * 1e-9)**2 / 4.0 * (g_upper / g_lower)

def sigma_peak_doppler(wl_nm, A_s, T_K, mass_amu, g_upper, g_lower,
                       sig_nat_m2=None, dnu_nat_Hz=None):
    """
    Peak absorption cross-section for Doppler (Gaussian) broadening [m²].

    Scales the natural peak by linewidths (consistent with the natural σ shown
    in the table):
      σ_D = σ_nat · (Δν_nat / Δν_D) · √(ln 2 / π)
    with Δν_nat = A/(2π) [Hz] and Δν_D = Doppler FWHM [Hz].
    """
    if sig_nat_m2 is None:
        sig_nat_m2 = sigma_peak_nat(wl_nm, g_upper, g_lower)
    if dnu_nat_Hz is None:
        dnu_nat_Hz = A_s / (2 * pi)
    dnu_D = doppler_fwhm_Hz(wl_nm, T_K, mass_amu)
    if dnu_D <= 0:
        return sig_nat_m2
    return sig_nat_m2 * (dnu_nat_Hz / dnu_D) * math.sqrt(math.log(2) / pi)

def isat_from_sigma(wl_nm, sigma_m2, tau_s):
    """
    Saturation intensity [W/m²] from cross-section and lifetime.
    I_sat = hν / (2 σ τ)
    """
    E_J = h * c / (wl_nm * 1e-9)
    return E_J / (2 * sigma_m2 * tau_s)

def recoil_velocity(wl_nm, mass_amu):
    """Single-photon recoil velocity [m/s]."""
    return h / (wl_nm * 1e-9 * mass_amu * amu)

def recoil_energy_nK(wl_nm, mass_amu):
    """Recoil energy in nK (E_rec/k_B).  Uses SI kB to avoid eV/J confusion."""
    v_rec = recoil_velocity(wl_nm, mass_amu)
    E_rec = 0.5 * mass_amu * amu * v_rec**2   # J
    kB_J  = 1.380649e-23                        # J/K
    return E_rec / kB_J * 1e9                   # nK

def max_scattering_rate(A_s):
    """Maximum photon scattering rate [photons/s] at I→∞: R = A/2."""
    return A_s / 2

def rad_pressure_force(wl_nm, A_s):
    """Maximum radiation pressure force [aN] (attonewtons, 10⁻¹⁸ N) at I→∞."""
    p_photon = h / (wl_nm * 1e-9)   # kg·m/s
    return p_photon * max_scattering_rate(A_s) * 1e18   # → aN

# ============================================================================
# LOAD PIPELINE DATA
# ============================================================================

def load_element_data(element, data_dir):
    """
    Load lifetime and rydberg data from pipeline JSONs.
    Returns a dict with all fields needed for the table,
    or None if the file doesn't exist.
    """
    lt_file = os.path.join(data_dir, f'{element}_lifetimes.json')
    ry_file = os.path.join(data_dir, f'{element}_rydberg.json')

    if not os.path.exists(lt_file):
        return None
    if not os.path.exists(ry_file):
        print(f"    Lifetimes JSON present but missing {os.path.basename(ry_file)}")
        return None

    with open(lt_file, encoding='utf-8') as f:
        lt = json.load(f)
    with open(ry_file, encoding='utf-8') as f:
        ry = json.load(f)

    IE  = ry['ionization_energy_eV']
    n_s = ry.get('n_start', 2)
    qd  = ry.get('quantum_defects', {})

    lifetimes = lt.get('lifetimes', [])
    lt_by_state = {x['state']: x for x in lifetimes}

    # Find lowest np state (D-line). For Na: ground=3s, first p = 3p (same n).
    # Cannot assume n_p = n_s+1. Search directly in lifetimes for lowest p state.
    p_states = sorted(
        [x for x in lifetimes if x['l'] == 'p'],
        key=lambda x: (x['n'], x.get('state', ''))
    )
    if not p_states:
        print(f"    No p-state lifetimes found for {element}")
        return None

    # ── Find D1 and D2 among lowest-n p-states ──────────────────────────────
    n_p = p_states[0]['n']
    lowest_p = sorted(
        [x for x in p_states if x['n'] == n_p],
        key=lambda x: x['state']
    )

    d1, d2 = None, None
    d1_label, d2_label = '', ''

    # First try: find by J label
    for s in lowest_p:
        st = s['state']
        if d1 is None and ('1/2' in st or len(lowest_p) == 1):
            d1 = s; d1_label = st
        elif d2 is None and '3/2' in st:
            d2 = s; d2_label = st

    # Second try: if J labels not present, use first two distinct states
    if d1 is None and lowest_p:
        d1 = lowest_p[0]; d1_label = d1['state']
    if d2 is None:
        if len(lowest_p) > 1:
            d2 = lowest_p[1]; d2_label = d2['state']
        else:
            d2 = d1; d2_label = d1_label  # degenerate (Li, or no J-split)

    if d1 is None:
        print(f"    No p-state lifetime found for {element}")
        return None

    lit = LITERATURE.get(element, {})

    # Mass priority: lifetimes JSON → literature dict → rydberg JSON → fallback
    # Use explicit > 0 check: lifetimes.py writes 0.0 (not None) for unknown elements
    mass_amu = None
    _lt_mass = lifetimes[0].get('mass_amu') if lifetimes else None
    if _lt_mass and _lt_mass > 0:
        mass_amu = _lt_mass
    if not mass_amu:
        _lit_mass = lit.get('mass_amu')
        if _lit_mass and _lit_mass > 0:
            mass_amu = _lit_mass
    if not mass_amu:
        _ry_mass = ry.get('mass_amu')
        if _ry_mass and _ry_mass > 0:
            mass_amu = _ry_mass
    if not mass_amu:
        print(f"    WARNING: mass_amu not found for {element} — Doppler quantities unreliable")
        mass_amu = 40.0  # loud fallback rather than silently wrong

    def _entry_data(entry, label, g_upper, g_lower):
        if entry is None:
            return None
        wl    = entry['dominant_wl_nm']
        tau_s = entry['lifetime_s']
        A_s   = entry['A_total_s']

        # Natural — prefer lifetimes.py precomputed values (same Lorentzian convention)
        sig_nat_cm2 = (entry.get('sigma_peak_nat_cm2')
                       or entry.get('sigma_peak_cm2') or 0)
        if not sig_nat_cm2:
            sig_nat_cm2 = sigma_peak_nat(wl, g_upper, g_lower) * 1e4
        sig_nat_1e9 = sig_nat_cm2 * 1e9
        Isat_nat = (entry.get('I_sat_nat_mWcm2')
                    or entry.get('I_sat_mW_cm2') or 0)
        if not Isat_nat:
            Isat_nat = isat_from_sigma(wl, sig_nat_cm2 * 1e-4, tau_s) * 0.1

        dnu_nat = entry.get('delta_nu_nat_MHz', A_s / (2 * math.pi) / 1e6)

        # Doppler — always recompute at CLI --T so JSON matches the HTML slider default
        sig_D_m2 = sigma_peak_doppler(
            wl, A_s, T_K, mass_amu, g_upper, g_lower,
            sig_nat_m2=sig_nat_cm2 * 1e-4,
            dnu_nat_Hz=dnu_nat * 1e6,
        )
        sig_D_cm2 = sig_D_m2 * 1e4
        Isat_D = isat_from_sigma(wl, sig_D_m2, tau_s) * 0.1
        dnu_D = doppler_fwhm_Hz(wl, T_K, mass_amu) / 1e6
        sig_D_1e12 = sig_D_cm2 * 1e12

        v_rec_ms = entry.get('v_rec_ms')
        if v_rec_ms is None:
            v_rec_ms = recoil_velocity(wl, mass_amu)
        v_rec_cms = entry.get('v_rec_cms')
        if v_rec_cms is None:
            v_rec_cms = v_rec_ms * 100
        E_rec_nK = (entry.get('E_rec_nK')
                    or recoil_energy_nK(wl, mass_amu))

        return dict(
            label=label,
            wl_nm=round(wl, 4),
            tau_ns=round(entry['lifetime_ns'], 4),
            sigma_tau_ns=round(entry.get('sigma_tau_ns', 0), 4),
            A_s=round(A_s, 3),
            fosc=round(entry.get('dominant_fosc', 0), 6),
            acc=entry.get('quality_acc', '?'),
            dnu_nat_MHz=round(dnu_nat, 4),
            dnu_D_MHz=round(dnu_D, 4),
            sig_nat_1e9cm2=round(sig_nat_1e9, 4),
            sig_D_1e12cm2=round(sig_D_1e12, 4),
            Isat_nat_mWcm2=round(Isat_nat, 4),
            Isat_D_mWcm2=round(Isat_D, 2),
            R_scatt_max=round(max_scattering_rate(A_s) / 1e6, 4),
            F_rad_aN=round(rad_pressure_force(wl, A_s), 4),
            v_rec_ms=round(v_rec_ms, 6),
            v_rec_cms=round(v_rec_cms, 4),
            E_rec_nK=round(E_rec_nK, 4),
            source=entry.get('dominant_source', '?'),
        )

    d1_data = _entry_data(d1, d1_label, g_upper=2, g_lower=2)
    d2_data = _entry_data(d2, d2_label, g_upper=4, g_lower=2)

    # SOC splitting
    soc_meV = None
    if d1_data and d2_data and d1_data['wl_nm'] != d2_data['wl_nm']:
        E1 = HC_nm / d1_data['wl_nm']
        E2 = HC_nm / d2_data['wl_nm']
        soc_meV = round(abs((E2 - E1) * 1000), 3)

    return dict(
        element=element,
        Z=_Z_BY_SYMBOL.get(element, lit.get('Z', 0)),
        mass_amu=mass_amu,
        gs=f"{n_s}s",
        IE_eV=round(IE, 6),
        delta_s=round(qd.get('s', 0), 6),
        delta_p=round(qd.get('p', 0), 6),
        delta_d=round(qd.get('d', 0), 6),
        soc_meV=soc_meV,
        D1=d1_data,
        D2=d2_data,
        source='pipeline',
        T_K=T_K,
    )


def build_from_literature(element, T_K):
    """Build table entry from hardcoded literature values."""
    lit = LITERATURE.get(element)
    if lit is None:
        return None
    mass = lit['mass_amu']

    def _lit_entry(line, wl_key, tau_key, fosc_key, g_upper, g_lower):
        wl   = lit.get(wl_key)
        tau  = lit.get(tau_key)
        fosc = lit.get(fosc_key)
        if wl is None or tau is None:
            return None
        tau_s = tau * 1e-9
        A_s   = 1.0 / tau_s
        sig_nat_m2 = sigma_peak_nat(wl, g_upper, g_lower)
        dnu_nat_Hz = A_s / (2 * pi)
        sig_D_m2 = sigma_peak_doppler(
            wl, A_s, T_K, mass, g_upper, g_lower,
            sig_nat_m2=sig_nat_m2, dnu_nat_Hz=dnu_nat_Hz,
        )
        Isat_nat = isat_from_sigma(wl, sig_nat_m2, tau_s) * 0.1
        Isat_D   = isat_from_sigma(wl, sig_D_m2,   tau_s) * 0.1
        dnu_nat  = dnu_nat_Hz / 1e6
        dnu_D    = doppler_fwhm_Hz(wl, T_K, mass) / 1e6
        v_ms     = recoil_velocity(wl, mass)
        return dict(
            label=line,
            wl_nm=round(wl, 4),
            tau_ns=round(tau, 4),
            sigma_tau_ns=round(tau * 0.01, 4),
            A_s=round(A_s, 3),
            fosc=round(fosc, 6) if fosc is not None else None,
            acc='lit',
            dnu_nat_MHz=round(dnu_nat, 4),
            dnu_D_MHz=round(dnu_D, 4),
            sig_nat_1e9cm2=round(sig_nat_m2 * 1e4 * 1e9, 4),
            sig_D_1e12cm2=round(sig_D_m2 * 1e4 * 1e12, 4),
            Isat_nat_mWcm2=round(Isat_nat, 4),
            Isat_D_mWcm2=round(Isat_D, 2),
            R_scatt_max=round(max_scattering_rate(A_s) / 1e6, 4),
            F_rad_aN=round(rad_pressure_force(wl, A_s), 4),
            v_rec_ms=round(v_ms, 6),
            v_rec_cms=round(v_ms * 100, 4),
            E_rec_nK=round(recoil_energy_nK(wl, mass), 4),
            source='literature',
        )

    np_label = lit['np']
    D1 = _lit_entry(f"{np_label}_1/2", 'D1_nm', 'tau_D1_ns', 'fosc_D1',
                    lit['g_upper_D1'], lit['g_lower_D1'])
    D2 = _lit_entry(f"{np_label}_3/2", 'D2_nm', 'tau_D2_ns', 'fosc_D2',
                    lit['g_upper_D2'], lit['g_lower_D2'])

    return dict(
        element=element,
        Z=_Z_BY_SYMBOL.get(element, lit.get('Z', 0)),
        mass_amu=mass,
        gs=lit['gs'],
        IE_eV=lit['IE_eV'],
        delta_s=lit.get('delta_s', 0),
        delta_p=lit.get('delta_p', 0),
        delta_d=0,
        soc_meV=lit.get('soc_meV'),
        D1=D1,
        D2=D2,
        source='literature',
        T_K=T_K,
    )

# ============================================================================
# BUILD TABLE
# ============================================================================

print(f"\n{'='*70}")
print(f"  Cross-Element Alkali Comparison  (T = {T_K:.0f} K)")
print(f"{'='*70}")

table_data = []
for el in elements:
    print(f"\n  {el}:")
    row = load_element_data(el, DATA_DIR)
    if row is None:
        print(f"    No pipeline data — using literature values")
        row = build_from_literature(el, T_K)
    if row is None:
        print(f"    No data available — skipping")
        continue
    print(f"    Source: {row['source']}")
    if row['D1']:
        d = row['D1']
        print(f"    D1: wl={d['wl_nm']:.3f}nm  tau={d['tau_ns']:.3f}ns  "
              f"I_sat(nat)={d['Isat_nat_mWcm2']:.2f}mW/cm2")
    if row['D2']:
        d = row['D2']
        print(f"    D2: wl={d['wl_nm']:.3f}nm  tau={d['tau_ns']:.3f}ns  "
              f"I_sat(nat)={d['Isat_nat_mWcm2']:.2f}mW/cm2")
    if row['soc_meV']:
        print(f"    SOC: {row['soc_meV']:.2f} meV")
    table_data.append(row)

# Save JSON under data_json/ (HTML stays under --output-dir, default plots/)
json_out = os.path.join(DATA_DIR, 'comparison_table.json')
with open(json_out, 'w', encoding='utf-8') as f:
    json.dump({'T_K': T_K, 'elements': table_data}, f, indent=2)
print(f"\nSaved JSON: {json_out}")

# ============================================================================
# BUILD INTERACTIVE HTML TABLE
# ============================================================================

import json as _json

_TABLE_DATA_JS = _json.dumps(table_data)
_T_K = T_K

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Alkali Metal Comparison Table</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: 'Segoe UI', Arial, sans-serif;
  background: #f5f7fa;
  color: #333;
  padding: 20px;
}}
h1 {{ font-size: 22px; color: #2c3e50; margin-bottom: 4px; }}
.subtitle {{ color: #666; font-size: 13px; margin-bottom: 16px; }}

.controls {{
  display: flex; gap: 10px; flex-wrap: wrap;
  align-items: center; margin-bottom: 14px;
}}
.controls label {{ font-size: 12px; color: #555; font-weight: bold; }}
.controls select, .controls input {{
  font-size: 12px; padding: 4px 8px;
  border: 1px solid #ccc; border-radius: 4px;
  background: #fff;
}}
button {{
  padding: 5px 12px; font-size: 12px;
  border: 1px solid #ccc; border-radius: 4px;
  background: #fff; cursor: pointer;
}}
button:hover {{ background: #e8eeff; }}
button.active {{ background: #4a7ec7; color: #fff; border-color: #3a6ab0; }}
.export-btn {{
  margin-left: auto; background: #27ae60; color: #fff;
  border-color: #219a52;
}}
.export-btn:hover {{ background: #219a52; }}

.table-wrap {{ overflow-x: auto; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
table {{
  border-collapse: collapse;
  width: 100%;
  background: #fff;
  font-size: 12px;
}}
thead tr:first-child th {{
  background: #2c3e50;
  color: #fff;
  padding: 8px 10px;
  text-align: center;
  font-size: 13px;
  letter-spacing: 0.3px;
  white-space: nowrap;
}}
thead tr:nth-child(2) th {{
  background: #34495e;
  color: #ecf0f1;
  padding: 6px 8px;
  font-size: 11px;
  cursor: pointer;
  user-select: none;
  white-space: nowrap;
  position: relative;
}}
thead tr:nth-child(2) th:hover {{ background: #4a6fa5; }}
thead tr:nth-child(2) th.sorted-asc::after  {{ content: ' ▲'; font-size: 9px; }}
thead tr:nth-child(2) th.sorted-desc::after {{ content: ' ▼'; font-size: 9px; }}

tbody tr {{ border-bottom: 1px solid #eee; }}
tbody tr:hover {{ background: #f0f6ff; }}
tbody tr.D2-row {{ background: #fafafa; }}
tbody tr.D2-row:hover {{ background: #e8f4ff; }}

td {{
  padding: 6px 8px;
  text-align: right;
  white-space: nowrap;
}}
td.element-cell {{
  text-align: left;
  font-weight: bold;
  font-size: 14px;
  color: #2c3e50;
  border-right: 2px solid #ddd;
  min-width: 60px;
}}
td.line-cell {{
  font-size: 11px;
  color: #777;
  text-align: left;
  border-right: 1px solid #eee;
}}
td.separator {{ border-right: 2px solid #ddd; }}

.src-pipeline {{ color: #27ae60; font-weight: bold; }}
.src-literature {{ color: #e67e22; }}
.src-NIST {{ color: #2980b9; }}
.src-ORCA {{ color: #c0392b; }}
.src-precision {{ color: #00838f; font-weight: bold; }}
.src-acc {{ color: #555; font-weight: 600; }}

.unit {{ font-size: 10px; color: #999; font-weight: normal; }}

.footnotes {{
  margin-top: 16px;
  font-size: 11px;
  color: #888;
  line-height: 1.8;
}}
</style>
</head>
<body>

<h1>Alkali Metal D-line Spectroscopic Parameters</h1>
<div class="subtitle">
  Group 1 atoms Li → Fr &nbsp;|&nbsp;
  Doppler broadening at T = <span id="T-display">{_T_K:.0f}</span> K &nbsp;|&nbsp;
  <span class="src-NIST">■ NIST</span> &nbsp;
  <span class="src-pipeline">■ pipeline</span> &nbsp;
  <span class="src-ORCA">■ ORCA</span> &nbsp;
  <span class="src-literature">■ literature</span> &nbsp;
  <span class="src-precision">■ precision</span>
</div>

<div class="controls">
  <label>Temperature (Doppler):</label>
  <input type="range" id="T-slider" min="1" max="2000" value="{_T_K:.0f}" step="1"
         style="width:140px" oninput="syncTFromSlider(this.value)">
  <input type="number" id="T-input" value="{_T_K:.0f}" min="1" max="5000" step="1"
         style="width:65px" oninput="syncTFromInput(this.value)">
  <span style="font-size:12px;color:#666">K</span>

  <label style="margin-left:16px">Highlight:</label>
  <select id="heat-col" onchange="applyHeatmap()">
    <option value="">None</option>
    <option value="tau">τ (lifetime)</option>
    <option value="Isat_nat">I_sat natural</option>
    <option value="Isat_D">I_sat Doppler</option>
    <option value="soc">SOC splitting</option>
    <option value="delta_p">δ_p</option>
    <option value="R_scatt">R_scatt</option>
    <option value="E_rec">E_rec</option>
  </select>

  <button class="export-btn" onclick="exportCSV()">⬇ Export CSV</button>
</div>

<div class="table-wrap">
<table id="main-table">
<thead>
<tr>
  <th colspan="6">Atom</th>
  <th colspan="5">D₁ line  (np₁/₂, J=½)</th>
  <th colspan="5">D₂ line  (np₃/₂, J=³/₂)</th>
  <th colspan="3">Natural linewidth (D₂)</th>
  <th colspan="3">Doppler T=<span id="T-head">{_T_K:.0f}</span>K (D₂)</th>
  <th colspan="4">Laser cooling (D₂)</th>
</tr>
<tr>
  <!-- Atom -->
  <th onclick="sortBy('element')" data-col="element">Atom</th>
  <th onclick="sortBy('Z')" data-col="Z">Z</th>
  <th onclick="sortBy('IE_eV')" data-col="IE_eV">IE <span class="unit">(eV)</span></th>
  <th onclick="sortBy('delta_s')" data-col="delta_s">δ_s</th>
  <th onclick="sortBy('delta_p')" data-col="delta_p">δ_p</th>
  <th onclick="sortBy('delta_d')" data-col="delta_d" class="separator">δ_d</th>
  <!-- D1 -->
  <th onclick="sortBy('D1_wl')" data-col="D1_wl">λ <span class="unit">(nm)</span></th>
  <th onclick="sortBy('D1_tau')" data-col="D1_tau">τ <span class="unit">(ns)</span></th>
  <th onclick="sortBy('D1_A')" data-col="D1_A">A <span class="unit">(10⁷/s)</span></th>
  <th onclick="sortBy('D1_fosc')" data-col="D1_fosc">fosc</th>
  <th onclick="sortBy('soc')" data-col="soc" class="separator">ΔE_SOC <span class="unit">(meV)</span></th>
  <!-- D2 -->
  <th onclick="sortBy('D2_wl')" data-col="D2_wl">λ <span class="unit">(nm)</span></th>
  <th onclick="sortBy('D2_tau')" data-col="D2_tau">τ <span class="unit">(ns)</span></th>
  <th onclick="sortBy('D2_A')" data-col="D2_A">A <span class="unit">(10⁷/s)</span></th>
  <th onclick="sortBy('D2_fosc')" data-col="D2_fosc">fosc</th>
  <th onclick="sortBy('D2_acc')" data-col="D2_acc" class="separator">Acc</th>
  <!-- Natural (D2) -->
  <th onclick="sortBy('dnu_nat')" data-col="dnu_nat">Δν_nat <span class="unit">(MHz)</span></th>
  <th onclick="sortBy('sig_nat')" data-col="sig_nat">σ_peak <span class="unit">(10⁻⁹cm²)</span></th>
  <th onclick="sortBy('Isat_nat')" data-col="Isat_nat" class="separator">I_sat <span class="unit">(mW/cm²)</span></th>
  <!-- Doppler (D2) -->
  <th onclick="sortBy('dnu_D')" data-col="dnu_D">Δν_D <span class="unit">(GHz)</span></th>
  <th onclick="sortBy('sig_D')" data-col="sig_D">σ_D <span class="unit">(10⁻¹²cm²)</span></th>
  <th onclick="sortBy('Isat_D')" data-col="Isat_D" class="separator">I_sat_D <span class="unit">(mW/cm²)</span></th>
  <!-- Laser cooling (D2) -->
  <th onclick="sortBy('R_scatt')" data-col="R_scatt">R_scatt <span class="unit">(10⁶/s)</span></th>
  <th onclick="sortBy('F_rad')" data-col="F_rad">F_rad <span class="unit">(aN)</span></th>
  <th onclick="sortBy('v_rec')" data-col="v_rec">v_rec <span class="unit">(cm/s)</span></th>
  <th onclick="sortBy('E_rec')" data-col="E_rec">E_rec <span class="unit">(nK)</span></th>
</tr>
</thead>
<tbody id="table-body">
</tbody>
</table>
</div>

<div class="footnotes">
  <b>Notes:</b>
  Natural σ_peak matches lifetimes.py Lorentzian peak
  (σ ∝ λ²/4 × g_u/g_l for Δν_nat = A/2π); isotropic / m_J-averaged light.
  Doppler σ_D = σ_nat × (Δν_nat/Δν_D) × √(ln2/π), recomputed live with the T slider.
  I_sat = hν/(2σ_peak τ) — two-level approximation (cycling transition I_sat lower by ~3).
  R_scatt(max) = A/2 — maximum scattering rate at I→∞ on resonance.
  F_rad = ħk · R_scatt — radiation pressure force in aN (10⁻¹⁸ N).
  v_rec = ħk/m — single-photon recoil velocity.
  SOC = |E(np₃/₂) − E(np₁/₂)| — fine-structure splitting.
  Acc = NIST accuracy tag on the D₂ f/A (AAA→E), or lit / precision.
  <br>
  Sources: <span class="src-NIST">NIST ASD</span>,
  <span class="src-ORCA">ORCA TD-DFT</span>,
  <span class="src-literature">literature fallback (Steck / NIST)</span>,
  <span class="src-precision">precision (curated Fr)</span>.
</div>

<script>
const RAW_DATA = {_TABLE_DATA_JS};

// Physical constants (for client-side Doppler recomputation)
const kB_eV = 8.617333e-5;
const amu_kg = 1.66053906660e-27;
const c_ms   = 2.99792458e8;
const eV_J   = 1.602176634e-19;
const h_J    = 6.62607015e-34;
const pi     = Math.PI;

function dopplerFWHM_MHz(wl_nm, T_K, mass_amu) {{
  const nu0 = c_ms / (wl_nm * 1e-9);
  const mc2_eV = mass_amu * amu_kg * c_ms * c_ms / eV_J;
  return nu0 * Math.sqrt(8 * Math.log(2) * kB_eV * T_K / mc2_eV) / 1e6;
}}

function isatFromSigma_mWcm2(wl_nm, sig_cm2, tau_s) {{
  const E_J = h_J * c_ms / (wl_nm * 1e-9);
  const sig_m2 = sig_cm2 * 1e-4;
  return E_J / (2 * sig_m2 * tau_s) * 0.1;   // W/m² → mW/cm²
}}

// Scale natural σ to Doppler using FWHM ratio (matches Python / table σ_nat)
function sigmaDopplerFromNat_cm2(sig_nat_cm2, dnu_nat_MHz, dnu_D_MHz) {{
  if (!sig_nat_cm2 || !dnu_nat_MHz || !dnu_D_MHz) return 0;
  return sig_nat_cm2 * (dnu_nat_MHz / dnu_D_MHz) * Math.sqrt(Math.log(2) / pi);
}}

// State
let currentT     = {_T_K:.0f};
let sortCol      = 'Z';
let sortAsc      = true;
let tableRows    = [];   // built from RAW_DATA
let displayRows  = [];   // current sort order (matches DOM)

function buildRows(T_K) {{
  tableRows = [];
  RAW_DATA.forEach(row => {{
    const d1 = row.D1, d2 = row.D2 || row.D1;
    if (!d1) return;

    const dnu_D2_MHz = dopplerFWHM_MHz(d2.wl_nm, T_K, row.mass_amu);
    const sig_nat_cm2 = (d2.sig_nat_1e9cm2 || 0) * 1e-9;
    const sig_D2_cm2 = sigmaDopplerFromNat_cm2(
      sig_nat_cm2, d2.dnu_nat_MHz, dnu_D2_MHz);
    const Isat_D2 = isatFromSigma_mWcm2(d2.wl_nm, sig_D2_cm2, d2.tau_ns * 1e-9);

    tableRows.push({{
      element: row.element,
      Z: row.Z,
      IE_eV: row.IE_eV,
      delta_s: row.delta_s,
      delta_p: row.delta_p,
      delta_d: row.delta_d,
      soc: row.soc_meV,
      source: row.source,
      mass: row.mass_amu,
      D1_wl:   d1.wl_nm,
      D1_tau:  d1.tau_ns,
      D1_sig:  d1.sigma_tau_ns,
      D1_A:    d1.A_s / 1e7,
      D1_fosc: d1.fosc,
      D1_acc:  d1.acc,
      D1_src:  d1.source,
      D2_wl:   d2.wl_nm,
      D2_tau:  d2.tau_ns,
      D2_sig:  d2.sigma_tau_ns,
      D2_A:    d2.A_s / 1e7,
      D2_fosc: d2.fosc,
      D2_acc:  d2.acc,
      D2_src:  d2.source,
      dnu_nat:  d2.dnu_nat_MHz,
      sig_nat:  d2.sig_nat_1e9cm2,
      Isat_nat: d2.Isat_nat_mWcm2,
      dnu_D:   dnu_D2_MHz / 1e3,
      sig_D:   sig_D2_cm2 * 1e12,
      Isat_D:  Isat_D2,
      R_scatt: d2.R_scatt_max,
      F_rad:   (d2.F_rad_aN != null ? d2.F_rad_aN : d2.F_rad_pN),
      v_rec:   d2.v_rec_cms,
      E_rec:   d2.E_rec_nK,
    }});
  }});
}}

function fmt(v, digits=3) {{
  if (v === null || v === undefined) return '—';
  if (typeof v === 'string') return v;
  const n = Number(v);
  if (isNaN(n)) return '—';
  return n.toFixed(digits);
}}

function srcClass(src) {{
  if (!src) return '';
  if (src === 'NIST') return 'src-NIST';
  if (src === 'ORCA') return 'src-ORCA';
  if (src === 'pipeline') return 'src-pipeline';
  if (src === 'precision') return 'src-precision';
  if (src === 'literature' || src === 'lit') return 'src-literature';
  return '';
}}

function accClass(acc) {{
  if (!acc) return 'src-acc';
  if (acc === 'precision') return 'src-precision';
  if (acc === 'lit' || acc === 'literature') return 'src-literature';
  return 'src-acc';
}}

function renderTable() {{
  const tbody = document.getElementById('table-body');
  tbody.innerHTML = '';

  const col = sortCol, asc = sortAsc;
  displayRows = [...tableRows].sort((a, b) => {{
    let av = a[col], bv = b[col];
    if (av == null) av = asc ? Infinity : -Infinity;
    if (bv == null) bv = asc ? Infinity : -Infinity;
    if (typeof av === 'string') return asc ? av.localeCompare(bv) : bv.localeCompare(av);
    return asc ? av - bv : bv - av;
  }});

  displayRows.forEach(r => {{
    const tr = document.createElement('tr');

    tr.innerHTML =
      `<td class="element-cell" style="color:#2c3e50">${{r.element}}</td>` +
      `<td style="text-align:center">${{r.Z}}</td>` +
      `<td>${{fmt(r.IE_eV,4)}}</td>` +
      `<td>${{fmt(r.delta_s,4)}}</td>` +
      `<td>${{fmt(r.delta_p,4)}}</td>` +
      `<td class="separator">${{fmt(r.delta_d,4)}}</td>`;

    tr.innerHTML +=
      `<td>${{fmt(r.D1_wl,3)}}</td>` +
      `<td class="${{srcClass(r.D1_src)}}">${{fmt(r.D1_tau,3)}} <span style="font-size:10px;color:#bbb">±${{fmt(r.D1_sig,2)}}</span></td>` +
      `<td>${{fmt(r.D1_A,3)}}</td>` +
      `<td>${{r.D1_fosc != null ? fmt(r.D1_fosc,4) : '—'}}</td>` +
      `<td class="separator" style="font-weight:bold;color:#8e44ad">${{r.soc != null ? fmt(r.soc,2) : '—'}}</td>`;

    tr.innerHTML +=
      `<td>${{fmt(r.D2_wl,3)}}</td>` +
      `<td class="${{srcClass(r.D2_src)}}">${{fmt(r.D2_tau,3)}} <span style="font-size:10px;color:#bbb">±${{fmt(r.D2_sig,2)}}</span></td>` +
      `<td>${{fmt(r.D2_A,3)}}</td>` +
      `<td>${{r.D2_fosc != null ? fmt(r.D2_fosc,4) : '—'}}</td>` +
      `<td class="separator ${{accClass(r.D2_acc)}}">${{r.D2_acc||'—'}}</td>`;

    tr.innerHTML +=
      `<td>${{fmt(r.dnu_nat,3)}}</td>` +
      `<td>${{fmt(r.sig_nat,3)}}</td>` +
      `<td class="separator">${{fmt(r.Isat_nat,3)}}</td>`;

    tr.innerHTML +=
      `<td>${{fmt(r.dnu_D,3)}}</td>` +
      `<td>${{fmt(r.sig_D,3)}}</td>` +
      `<td class="separator">${{fmt(r.Isat_D,1)}}</td>`;

    tr.innerHTML +=
      `<td style="font-weight:bold">${{fmt(r.R_scatt,3)}}</td>` +
      `<td>${{fmt(r.F_rad,4)}}</td>` +
      `<td>${{fmt(r.v_rec,2)}}</td>` +
      `<td>${{fmt(r.E_rec,2)}}</td>`;

    tbody.appendChild(tr);
  }});

  applyHeatmap();
  updateSortHeaders();
}}

function sortBy(col) {{
  if (sortCol === col) sortAsc = !sortAsc;
  else {{ sortCol = col; sortAsc = true; }}
  renderTable();
}}

function updateSortHeaders() {{
  document.querySelectorAll('thead tr:nth-child(2) th').forEach(th => {{
    th.classList.remove('sorted-asc','sorted-desc');
    const c = th.getAttribute('data-col');
    if (c === sortCol) th.classList.add(sortAsc ? 'sorted-asc' : 'sorted-desc');
  }});
}}

function syncTFromSlider(val) {{
  currentT = parseFloat(val);
  document.getElementById('T-input').value = val;
  document.getElementById('T-head').textContent = Math.round(currentT);
  document.getElementById('T-display').textContent = Math.round(currentT);
  buildRows(currentT);
  renderTable();
}}

function syncTFromInput(val) {{
  const T = parseFloat(val);
  if (isNaN(T) || T <= 0) return;
  currentT = T;
  const sliderVal = Math.min(Math.max(T, 1), 2000);
  document.getElementById('T-slider').value = sliderVal;
  document.getElementById('T-head').textContent = Math.round(currentT);
  document.getElementById('T-display').textContent = Math.round(currentT);
  buildRows(currentT);
  renderTable();
}}

function applyHeatmap() {{
  const col = document.getElementById('heat-col').value;
  const rows = document.querySelectorAll('#table-body tr');
  if (!col) {{
    rows.forEach(tr => {{ tr.style.background = ''; }});
    return;
  }}
  const colMap = {{
    'tau':      r => r.D2_tau,
    'Isat_nat': r => r.Isat_nat,
    'Isat_D':   r => r.Isat_D,
    'soc':      r => r.soc,
    'delta_p':  r => r.delta_p,
    'R_scatt':  r => r.R_scatt,
    'E_rec':    r => r.E_rec,
  }};
  const fn = colMap[col];
  if (!fn) return;
  const vals = displayRows.map(fn).filter(v => v != null && !isNaN(v));
  if (!vals.length) return;
  const mn = Math.min(...vals), mx = Math.max(...vals), rng = mx - mn;
  rows.forEach((tr, i) => {{
    if (i >= displayRows.length) return;
    const v = fn(displayRows[i]);
    if (v == null || isNaN(v)) {{ tr.style.background = ''; return; }}
    const norm = rng > 0 ? (v - mn) / rng : 0.5;
    tr.style.background = norm < 0.33 ? 'rgba(46,204,113,0.12)'
                        : norm < 0.66 ? 'rgba(243,156,18,0.12)'
                        : 'rgba(231,76,60,0.12)';
  }});
}}

function exportCSV() {{
  const headers = [
    'Element','Z','IE(eV)','delta_s','delta_p','delta_d',
    'D1_lambda(nm)','D1_tau(ns)','D1_A(1e7/s)','D1_fosc','SOC(meV)',
    'D2_lambda(nm)','D2_tau(ns)','D2_A(1e7/s)','D2_fosc','D2_Acc',
    'dnu_nat(MHz)','sig_nat(1e-9cm2)','Isat_nat(mW/cm2)',
    'dnu_D(GHz)','sig_D(1e-12cm2)','Isat_D(mW/cm2)',
    'R_scatt(1e6/s)','F_rad(aN)','v_rec(cm/s)','E_rec(nK)'
  ];
  const rows = [headers.join(',')];
  const src = displayRows.length ? displayRows : tableRows;
  src.forEach(r => {{
    rows.push([
      r.element, r.Z, r.IE_eV, r.delta_s, r.delta_p, r.delta_d??'',
      r.D1_wl, r.D1_tau, r.D1_A, r.D1_fosc??'', r.soc??'',
      r.D2_wl, r.D2_tau, r.D2_A, r.D2_fosc??'', r.D2_acc||'',
      fmt(r.dnu_nat,3), fmt(r.sig_nat,3), fmt(r.Isat_nat,3),
      fmt(r.dnu_D,3), fmt(r.sig_D,3), fmt(r.Isat_D,1),
      fmt(r.R_scatt,3), fmt(r.F_rad,3), fmt(r.v_rec,2), fmt(r.E_rec,2)
    ].join(','));
  }});
  const blob = new Blob([rows.join('\\n')], {{type:'text/csv'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'alkali_comparison.csv';
  a.click();
}}

// Init
buildRows(currentT);
renderTable();
</script>
</body>
</html>
"""

html_out = os.path.join(OUT_DIR, 'comparison_table.html')
with open(html_out, 'w', encoding='utf-8') as f:
    f.write(HTML)

print(f"Saved HTML: {html_out}")
print(f"\n{'='*70}")
print(f"  Summary (D2 line, natural broadening):")
print(f"  {'Element':>8}  {'wl_D2(nm)':>10}  {'tau(ns)':>8}  "
      f"{'dnu_nat(MHz)':>13}  {'I_sat(mW/cm2)':>14}")
print(f"  {'-'*65}")
for row in table_data:
    d2 = row.get('D2') or row.get('D1')
    if d2:
        print(f"  {row['element']:>8}  {d2['wl_nm']:>10.3f}  "
              f"{d2['tau_ns']:>8.3f}  {d2['dnu_nat_MHz']:>13.3f}  "
              f"{d2['Isat_nat_mWcm2']:>14.3f}  [{row['source']}]")
print(f"{'='*70}\n")
