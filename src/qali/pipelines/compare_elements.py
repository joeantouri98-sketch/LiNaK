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
  data_json/comparison_table.json    — machine-readable summary (with the C₆ matrix and the
                                       tune-out wavelengths of the elements that have
                                       polarizability data)
  .qali/plots/comparison.html       — the Compare page (``compare_page``): D lines, Laser
                                       cooling, Polarizability, C₆, Trends, Accuracy

Usage:
  qali --pipeline compare_elements
  qali --pipeline compare_elements --T 500      (Doppler at 500 K)
  qali --pipeline compare_elements --elements Na K Rb
  qali --pipeline compare_elements --no-page    (JSON only)
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass

from qali.core.constants import (
    KB_EV as kB,
    AMU_KG as amu,
    C_SI as c,
    EV_TO_J as eV_J,
    H_SI as h,
    HC_EV_NM as HC_nm,
    KB_SI,
)
from qali.core.elements import ELEMENTS
from qali.core.errors import MissingDataError
from qali.core.fileio import write_json
from qali.core.species import require_neutral_species
from qali.core.spectroscopy import LITERATURE
from qali.core.workspace import Workspace

pi = math.pi
_Z_BY_SYMBOL = {sym: Z for Z, (sym, _q, _m) in ELEMENTS.items()}


# ─────────────────────────────────────────────────────────────────────────────
# typed comparison-table row
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ElementData:
    """One cross-element comparison-table row (the record serialized to
    ``comparison_table.json`` and the HTML table payload).

    Field order is the exact historical key order of both row builders
    (``load_element_data`` / ``build_from_literature``); serializing a row
    goes through ``dataclasses.asdict``, which emits the fields in this
    order, so the produced JSON stays byte-identical. "Not available" is a
    ``None`` value, never a magic number. The nested ``D1``/``D2`` D-line
    sub-records remain plain dicts for now (remaining string-keyed
    hotspots).
    """

    element: str
    Z: int
    mass_amu: float
    gs: str
    IE_eV: float
    delta_s: float
    delta_p: float
    delta_d: float
    soc_meV: float | None
    D1: dict | None
    D2: dict | None
    source: str
    T_K: float


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qali --pipeline compare_elements",
        description='Cross-element alkali comparison table')
    parser.add_argument('--T',        type=float, default=300.0,
                        help='Temperature for Doppler broadening in K (default: 300)')
    parser.add_argument('--elements', nargs='+',
                        default=['Li', 'Na', 'K', 'Rb', 'Cs', 'Fr'],
                        help='Elements to include (default: all 6 alkalis)')
    parser.add_argument('--no-page', action='store_true',
                        help='Write the JSON only; skip plots/comparison.html')
    parser.add_argument('--html', action='store_true', help=argparse.SUPPRESS)   # the page is now always written
    from qali.pipelines.compare_page import TAB_FLAGS
    for flag in list(TAB_FLAGS.values())[1:]:
        parser.add_argument(f'--no-tab-{flag}', action='store_true', help=f'Leave out the {flag} tab')
    parser.add_argument('--data-dir', default=str(Workspace.current().data_json_dir),
                        help='Directory containing JSON data files (default: data_json)')
    parser.add_argument('--output-dir', default=str(Workspace.current().plots_dir),
                        help='Output directory for the Compare page (default: plots)')
    return parser

# PHYSICS FUNCTIONS


def doppler_fwhm_Hz(wl_nm, T_K, mass_amu):
    """Doppler FWHM in Hz (Gaussian profile)."""
    nu0   = c / (wl_nm * 1e-9)
    mc2   = mass_amu * amu * c**2 / eV_J   # eV
    return nu0 * math.sqrt(8 * math.log(2) * kB * T_K / mc2)

def sigma_peak_nat(wl_nm, g_upper, g_lower):
    """
    Peak absorption cross-section for natural (Lorentzian) broadening [m²].

    Matches lifetimes.py with Δν_nat = A/(2π):
      σ = (λ²/4π²)·(g_u/g_l)·(A/Δν_nat) = (λ²/2π)·(g_u/g_l)      (P22)
    (Lorentzian peak 2/(πΔν_nat).) Isotropic / m_J-averaged light.
    """
    return (wl_nm * 1e-9)**2 / (2.0 * pi) * (g_upper / g_lower)

def sigma_peak_doppler(wl_nm, A_s, T_K, mass_amu, g_upper, g_lower,
                       sig_nat_m2=None, dnu_nat_Hz=None):
    """
    Peak absorption cross-section for Doppler (Gaussian) broadening [m²].

    Scales the natural peak by linewidths (consistent with the natural σ shown
    in the table):
      σ_D = σ_nat · (Δν_nat / Δν_D) · √(π ln 2)
    with Δν_nat = A/(2π) [Hz] and Δν_D = Doppler FWHM [Hz].
    """
    if sig_nat_m2 is None:
        sig_nat_m2 = sigma_peak_nat(wl_nm, g_upper, g_lower)
    if dnu_nat_Hz is None:
        dnu_nat_Hz = A_s / (2 * pi)
    dnu_D = doppler_fwhm_Hz(wl_nm, T_K, mass_amu)
    if dnu_D <= 0:
        return sig_nat_m2
    return sig_nat_m2 * (dnu_nat_Hz / dnu_D) * math.sqrt(math.log(2) * pi)

def isat_from_sigma(wl_nm, sigma_m2, tau_s):
    """
    Saturation intensity [W/m²] from cross-section and lifetime.
    I_sat = hν / (2 σ τ)
    Returns 0.0 (= "undefined", the file's convention for a missing I_sat) when σ or τ is not
    a positive number, instead of dividing by zero.
    """
    if not (sigma_m2 and tau_s and sigma_m2 > 0 and tau_s > 0):
        return 0.0
    E_J = h * c / (wl_nm * 1e-9)
    return E_J / (2 * sigma_m2 * tau_s)

def recoil_velocity(wl_nm, mass_amu):
    """Single-photon recoil velocity [m/s]."""
    return h / (wl_nm * 1e-9 * mass_amu * amu)

def recoil_energy_nK(wl_nm, mass_amu):
    """Recoil energy in nK (E_rec/k_B).  Uses SI kB to avoid eV/J confusion."""
    v_rec = recoil_velocity(wl_nm, mass_amu)
    E_rec = 0.5 * mass_amu * amu * v_rec**2   # J
    return E_rec / KB_SI * 1e9                  # nK

def max_scattering_rate(A_s):
    """Maximum photon scattering rate [photons/s] at I→∞: R = A/2."""
    return A_s / 2

def rad_pressure_force(wl_nm, A_s):
    """Maximum radiation pressure force [aN] (attonewtons, 10⁻¹⁸ N) at I→∞."""
    p_photon = h / (wl_nm * 1e-9)   # kg·m/s
    return p_photon * max_scattering_rate(A_s) * 1e18   # → aN

# ============================================================================
# D-LINE SUB-RECORD (one assembly, two data sources)
# ============================================================================

def _dline_record(label, wl, tau_ns, tau_s, sigma_tau_ns, fosc, acc, source,
                  A_s, g_upper, g_lower, T_K, mass_amu, *,
                  sig_nat_m2, sig_nat_1e9cm2, dnu_nat_MHz, dnu_doppler_Hz,
                  isat_nat_mWcm2, v_rec_ms, v_rec_cms, E_rec_nK):
    """
    Assemble one D-line sub-record — the shared 19-key dict both row builders
    emit into ``ElementData.D1``/``D2`` (historical key order, byte-identical).

    ``load_element_data`` (lifetime-pipeline JSON) and ``build_from_literature``
    (the ``LITERATURE`` table) resolve the source-specific quantities themselves
    — σ_nat, Δν_nat, I_sat(nat), v_rec, E_rec — and pass them in explicitly;
    this single builder computes what both paths share, once: σ_D (always
    recomputed at CLI ``--T`` so the JSON matches the HTML slider default),
    I_sat(D), Δν_D, R_scatt_max, F_rad, and every record rounding.
    """
    sig_D_m2 = sigma_peak_doppler(
        wl, A_s, T_K, mass_amu, g_upper, g_lower,
        sig_nat_m2=sig_nat_m2, dnu_nat_Hz=dnu_doppler_Hz,
    )
    Isat_D = isat_from_sigma(wl, sig_D_m2, tau_s) * 0.1
    dnu_D = doppler_fwhm_Hz(wl, T_K, mass_amu) / 1e6
    return dict(
        label=label,
        wl_nm=round(wl, 4),
        tau_ns=round(tau_ns, 4),
        sigma_tau_ns=round(sigma_tau_ns, 4),
        A_s=round(A_s, 3),
        fosc=round(fosc, 6) if fosc is not None else None,
        acc=acc,
        dnu_nat_MHz=round(dnu_nat_MHz, 4),
        dnu_D_MHz=round(dnu_D, 4),
        sig_nat_1e9cm2=round(sig_nat_1e9cm2, 4),
        sig_D_1e12cm2=round(sig_D_m2 * 1e4 * 1e12, 4),
        Isat_nat_mWcm2=round(isat_nat_mWcm2, 4),
        Isat_D_mWcm2=round(Isat_D, 2),
        R_scatt_max=round(max_scattering_rate(A_s) / 1e6, 4),
        F_rad_aN=round(rad_pressure_force(wl, A_s), 4),
        v_rec_ms=round(v_rec_ms, 6),
        v_rec_cms=round(v_rec_cms, 4),
        E_rec_nK=round(E_rec_nK, 4),
        source=source,
    )

# ============================================================================
# LOAD PIPELINE DATA


def load_element_data(element, data_dir, T_K):
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

    # Find lowest np state (D-line). For Na: ground=3s, first p = 3p (same n).
    # Cannot assume n_p = n_s+1. Search directly in lifetimes for lowest p state.
    p_states = sorted(
        [x for x in lifetimes if x['l'] == 'p'],
        key=lambda x: (x['n'], x.get('state', ''))
    )
    if not p_states:
        print(f"    No p-state lifetimes found for {element}")
        return None

    # Find D1 and D2 among lowest-n p-states 
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
        sig_nat_m2 = sig_nat_cm2 * 1e-4
        dnu_nat = entry.get('delta_nu_nat_MHz', A_s / (2 * math.pi) / 1e6)

        Isat_nat = (entry.get('I_sat_nat_mWcm2')
                    or entry.get('I_sat_mW_cm2') or 0)
        if not Isat_nat:
            Isat_nat = isat_from_sigma(wl, sig_nat_m2, tau_s) * 0.1

        v_rec_ms = entry.get('v_rec_ms')
        if v_rec_ms is None:
            v_rec_ms = recoil_velocity(wl, mass_amu)
        v_rec_cms = entry.get('v_rec_cms')
        if v_rec_cms is None:
            v_rec_cms = v_rec_ms * 100
        E_rec_nK = (entry.get('E_rec_nK')
                    or recoil_energy_nK(wl, mass_amu))

        return _dline_record(
            label, wl, entry['lifetime_ns'], tau_s,
            entry.get('sigma_tau_ns', 0), entry.get('dominant_fosc', 0),
            entry.get('quality_acc', '?'), entry.get('dominant_source', '?'),
            A_s, g_upper, g_lower, T_K, mass_amu,
            sig_nat_m2=sig_nat_m2, sig_nat_1e9cm2=sig_nat_cm2 * 1e9,
            dnu_nat_MHz=dnu_nat, dnu_doppler_Hz=dnu_nat * 1e6,
            isat_nat_mWcm2=Isat_nat, v_rec_ms=v_rec_ms, v_rec_cms=v_rec_cms,
            E_rec_nK=E_rec_nK,
        )

    d1_data = _entry_data(d1, d1_label, g_upper=2, g_lower=2)
    d2_data = _entry_data(d2, d2_label, g_upper=4, g_lower=2)

    # SOC splitting
    soc_meV = None
    if d1_data and d2_data and d1_data['wl_nm'] != d2_data['wl_nm']:
        E1 = HC_nm / d1_data['wl_nm']
        E2 = HC_nm / d2_data['wl_nm']
        soc_meV = round(abs((E2 - E1) * 1000), 3)

    return ElementData(
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
        if tau <= 0:
            return None
        tau_s = tau * 1e-9
        A_s   = 1.0 / tau_s
        sig_nat_m2 = sigma_peak_nat(wl, g_upper, g_lower)
        dnu_nat_Hz = A_s / (2 * pi)
        Isat_nat = isat_from_sigma(wl, sig_nat_m2, tau_s) * 0.1
        v_ms     = recoil_velocity(wl, mass)

        return _dline_record(
            line, wl, tau, tau_s,
            tau * 0.01, fosc, 'lit', 'literature',
            A_s, g_upper, g_lower, T_K, mass,
            sig_nat_m2=sig_nat_m2,
            sig_nat_1e9cm2=sig_nat_m2 * 1e4 * 1e9,
            dnu_nat_MHz=dnu_nat_Hz / 1e6, dnu_doppler_Hz=dnu_nat_Hz,
            isat_nat_mWcm2=Isat_nat, v_rec_ms=v_ms, v_rec_cms=v_ms * 100,
            E_rec_nK=recoil_energy_nK(wl, mass),
        )

    np_label = lit['np']
    D1 = _lit_entry(f"{np_label}_1/2", 'D1_nm', 'tau_D1_ns', 'fosc_D1',
                    lit['g_upper_D1'], lit['g_lower_D1'])
    D2 = _lit_entry(f"{np_label}_3/2", 'D2_nm', 'tau_D2_ns', 'fosc_D2',
                    lit['g_upper_D2'], lit['g_lower_D2'])

    return ElementData(
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


# BUILD TABLE


def build_table(elements, data_dir, T_K):
    """Assemble one row per element, preferring pipeline JSON over literature."""
    print(f"\n{'='*70}")
    print(f"  Cross-Element Alkali Comparison  (T = {T_K:.0f} K)")
    print(f"{'='*70}")

    table_data = []
    for el in elements:
        print(f"\n  {el}:")
        row = load_element_data(el, data_dir, T_K)
        if row is None:
            print("    No pipeline data — using literature values")
            row = build_from_literature(el, T_K)
        if row is None:
            print("    No data available — skipping")
            continue
        print(f"    Source: {row.source}")
        if row.D1:
            d = row.D1
            print(f"    D1: wl={d['wl_nm']:.3f}nm  tau={d['tau_ns']:.3f}ns  "
                  f"I_sat(nat)={d['Isat_nat_mWcm2']:.2f}mW/cm2")
        if row.D2:
            d = row.D2
            print(f"    D2: wl={d['wl_nm']:.3f}nm  tau={d['tau_ns']:.3f}ns  "
                  f"I_sat(nat)={d['Isat_nat_mWcm2']:.2f}mW/cm2")
        if row.soc_meV:
            print(f"    SOC: {row.soc_meV:.2f} meV")
        table_data.append(row)
    return table_data


# MERGE WITH THE SAVED TABLE


def merge_with_existing(new_rows, json_path):
    """*new_rows* added to (or replacing the same elements in) the rows already saved
    in *json_path*. An unreadable or missing file means no earlier rows."""
    try:
        with open(json_path, encoding='utf-8') as f:
            old = [ElementData(**rec) for rec in json.load(f).get('elements', [])]
    except (OSError, ValueError, TypeError):
        old = []
    replaced = {row.element for row in new_rows}
    return [row for row in old if row.element not in replaced] + list(new_rows)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    T_K = args.T
    elements = [
        require_neutral_species(el, "compare_elements.py")["species_id"]
        for el in args.elements
    ]
    data_dir = args.data_dir
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    table_data = build_table(elements, data_dir, T_K)
    if not table_data:
        # Never replace a good table with an empty one.
        raise MissingDataError(
            f"none of {', '.join(elements)} has the D-line data the table needs in "
            f"{data_dir} (run rydberg, transitions and lifetimes first)")

    # Elements already in the table stay; the requested ones are added or replaced.
    json_out = os.path.join(data_dir, 'comparison_table.json')
    table_data = merge_with_existing(table_data, json_out)
    from qali.pipelines import compare_page as cp
    tabs = [t for t in cp.TABS if t == 'dl' or not getattr(args, f'no_tab_{cp.TAB_FLAGS[t]}', False)]
    payload = cp.build_payload(table_data, data_dir, T_K, tabs=tabs)
    out = {'T_K': T_K, 'elements': [asdict(row) for row in table_data]}
    if payload['c6']:
        out['C6_au'] = [{'pair': [a, b], 'C6_au': c, 'sigma_au': sg, 'reference_au': ref, 'reference_sigma_au': rs,
                         'method': 'Casimir-Polder from each atom\'s alpha(i omega)'} for a, b, c, sg, ref, rs in payload['c6']]
        out['tune_out_nm'] = {el: [{'wl_nm': t[0], 'between': [t[1], t[2]]} for t in p['to']] for el, p in payload['pol'].items()}
    write_json(json_out, out)
    print(f"\nSaved JSON: {json_out}")

    if not args.no_page:
        page = cp.write_page(payload, out_dir)
        print(f"Saved page: {page}")
        if payload['acc']:
            print("  Accuracy (computed vs reference):")
            for r in payload['acc']:
                print(f"    {r['q']:9s} {r['who']:7s} {r['calc']:>14.6g}  ref {r['ref']:>12.6g}  "
                      f"{100 * (r['calc'] - r['ref']) / r['ref']:+7.2f} %")
    print(f"\n{'='*70}")
    print("  Summary (D2 line, natural broadening):")
    print(f"  {'Element':>8}  {'wl_D2(nm)':>10}  {'tau(ns)':>8}  "
          f"{'dnu_nat(MHz)':>13}  {'I_sat(mW/cm2)':>14}")
    print(f"  {'-'*65}")
    for row in table_data:
        d2 = row.D2 or row.D1
        if d2:
            print(f"  {row.element:>8}  {d2['wl_nm']:>10.3f}  "
                  f"{d2['tau_ns']:>8.3f}  {d2['dnu_nat_MHz']:>13.3f}  "
                  f"{d2['Isat_nat_mWcm2']:>14.3f}  [{row.source}]")
    print(f"{'='*70}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
