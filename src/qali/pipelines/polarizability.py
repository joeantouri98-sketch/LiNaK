"""
polarizability.py
-----------------
Computes dynamic electric dipole polarizability α(ω), van der Waals C₆
coefficients, magic wavelengths, and AC Stark shift / optical dipole trap
depth for any element in the pipeline, and writes them as data. The per-species
plots are the Polarizability page (``polarizability_page``), which recomputes
α(ω) from the stored transitions; U(ω) is just α(ω) rescaled by a constant at
fixed intensity, not a second independent calculation.

Physics summary
---------------
Dynamic polarizability (atomic units):
    α_n(ω) = Σ_k  s_k · f_nk / [(E_k - E_n)² - ω²]

where f_nk is the absorption oscillator strength (already the orientation-averaged
value, so there is no extra 2/3 or ΔE prefactor), energies in Hartree, ω in Hartree
(= atomic units of frequency).  s_k = +1 for transitions upward (k above n) and −1
downward (k below n).

Static polarizability: α(0)  [a₀³ = atomic units of volume]

C₆ coefficient via Casimir-Polder integral:
    C₆^AA = (3/π) ∫₀^∞ [α_A(iξ)]² dξ
    C₆^AB = (3/π) ∫₀^∞ α_A(iξ) · α_B(iξ) dξ

where α(iξ) is the polarizability at imaginary frequency, which is smooth
and positive everywhere (no resonances), computed by replacing ω → iξ:
    α_n(iξ) = Σ_k  s_k · f_nk / [(E_k - E_n)² + ξ²]

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
  (α(λ), magic wavelengths and the tweezer are on the Polarizability page; α(0) against
  experiment, the C₆ matrix and the tune-out wavelengths of every element are on the
  Compare page, written by the Compare stage)

Usage:
  qali --pipeline polarizability Na
  qali --pipeline polarizability Na --excited 3p3/2
  qali --pipeline polarizability Na --wl-min 400 --wl-max 1500
  qali --pipeline polarizability Na --wl 1064 --intensity 10   (AC Stark point calc, kW/cm^2)
  qali --pipeline polarizability Na --waist 1.0                (beam waist, micron)
  qali --pipeline polarizability --all
"""

__version__ = '2.1'   # Shared α(ω) core in physics/polarizability_core.py; CLI only under __main__.


import argparse
import itertools
import math
import os

import numpy as np
from qali.core.fileio import write_json
from qali.core.data_inputs import resolve_data_file
from qali.core.errors import MissingDataError
from qali.core.workspace import Workspace
from qali.core.spectroscopy import Transition

from qali.physics.polarizability_core import (
    U_joules,
    U_to_kHz_h,
    U_to_uK,
    alpha0_static,
    alpha_imag_freq,
    alpha_sigma,
    c6_sigma,
    alpha_real,
    build_transitions_for_state,
    get_available_elements,
    kWcm2_to_Wm2,
    load_element,
    nm_from_omega_au,
    omega_au_from_nm,
    trap_frequencies_Hz,
)
from qali.core.constants import (
    ALPHA0_EXPT,
    C6_EXPT,
    ATOMIC_MASS_AMU,
    ISOTOPE_MASS_AMU,
    default_isotope_A,
)

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
    C₆ = (3/π) ∫₀^∞ [α(iξ)]² dξ   (atomic units); see ``compute_C6_pair``.
    """
    return compute_C6_pair(transitions_list, transitions_list, n_xi=n_xi)


def compute_C6_pair(trans_A, trans_B, n_xi=2000):
    """
    Casimir–Polder C₆ of two different atoms (Dalgarno & Davison 1966; Derevianko et al.,
    PRA 63, 052704 (2001), Eq. 3):

        C₆^AB = (3/π) ∫₀^∞ α_A(iξ) α_B(iξ) dξ      (atomic units)

    Each α(iξ) is the atom's own sum over its transitions (core included as a pseudo-line), so the
    heteronuclear value is as good as the two homonuclear ones, unlike a combining rule from α(0).

    Integration on a log-mapped grid x = log(1 + ξ/ξ₀), ξ₀ = 0.1 × the smallest transition energy
    of either atom, up to 10 × the largest; the integrand peaks near ξ ~ ΔE_min and decays as
    ξ⁻⁴, whose tail past the cutoff is added analytically.
    """
    if not trans_A or not trans_B:
        return 0.0
    dE_values = [abs(tr['delta_E_au']) for tr in list(trans_A) + list(trans_B)
                 if tr['delta_E_au'] != 0]
    if not dE_values:
        return 0.0
    xi0 = min(dE_values) * 0.1

    def integrand_mapped(x):
        xi = xi0 * (math.exp(x) - 1.0)
        dxi = xi0 * math.exp(x)
        return alpha_imag_freq(xi, trans_A) * alpha_imag_freq(xi, trans_B) * dxi

    x_max = math.log(1.0 + 10.0 * max(dE_values) / xi0)
    try:
        from scipy.integrate import quad
    except ImportError:
        quad = None
    if quad is not None:
        val, _err = quad(integrand_mapped, 0, x_max, limit=200, epsrel=1e-6)
    else:
        print("  scipy not found - C6 will use trapezoidal integration "
              "(less accurate)")
        x_arr = np.linspace(0, x_max, n_xi)
        y_arr = np.array([integrand_mapped(x) for x in x_arr])
        val = np.trapezoid(y_arr, x_arr)
    # Tail beyond ξ_max: α(iξ) → S/ξ² with S = Σ s f (the TRK sum), so ∫ = S_A S_B/(3 ξ_max³)
    xi_max = xi0 * (math.exp(x_max) - 1.0)
    S_A = sum(tr.get('sign', 1.0) * tr['fosc'] for tr in trans_A)
    S_B = sum(tr.get('sign', 1.0) * tr['fosc'] for tr in trans_B)
    val += S_A * S_B / (3.0 * xi_max ** 3)
    return (3.0 / math.pi) * val


def tune_out_wavelengths(transitions_list, wl_min_nm=200.0, wl_max_nm=2000.0):
    """
    Tune-out wavelengths of a ground state: zeros of the real dynamic polarizability.

    With every term upward (s = +1), α(ω) rises monotonically between two consecutive resonances
    (−∞ just above the lower, +∞ just below the upper), so there is exactly one zero in each gap
    and none below the first resonance. Returns [(λ_nm, lower_label, upper_label)] in the range,
    longest wavelength first; the D1–D2 one is the familiar tune-out (Rb 790.03 nm, K 768.97 nm).
    """
    from scipy.optimize import brentq
    dE = np.array([t['delta_E_au'] for t in transitions_list], dtype=float)
    fs = np.array([t.get('sign', 1) * t['fosc'] for t in transitions_list], dtype=float)

    def alpha(w):          # no resonance guard: the roots are bracketed away from the poles
        return float(np.sum(fs / (dE * dE - w * w)))

    res = sorted({(abs(t['delta_E_au']), t.get('other_label', '')) for t in transitions_list
                  if t['delta_E_au'] != 0 and t.get('other_label') != 'core'})
    out = []
    w_lo, w_hi = omega_au_from_nm(wl_max_nm), omega_au_from_nm(wl_min_nm)
    for (e1, l1), (e2, l2) in itertools.pairwise(res):
        if e2 - e1 < 1e-12 or e2 < w_lo or e1 > w_hi:
            continue
        eps = 1e-9 * (e2 - e1)
        a, b = e1 + eps, e2 - eps
        if alpha(a) * alpha(b) > 0:
            continue
        w0 = brentq(alpha, a, b, xtol=1e-15, rtol=1e-14)
        lam = nm_from_omega_au(w0)
        if wl_min_nm <= lam <= wl_max_nm:
            out.append((lam, l1, l2))
    return sorted(out, reverse=True)


# MAGIC WAVELENGTH FINDER

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


# MAIN COMPUTATION PER ELEMENT


def run_element(element, excited_label=None,
                wl_min=200.0, wl_max=2000.0, n_grid=2000,
                I_ref_kWcm2=10.0, waist_um=1.0, wl_point=None,
                data_dir=None, isotope_A=None, extrapolation=None,
                rydberg_file=None, transitions_file=None, model_f=None, model_file=None):
    """
    Full polarizability pipeline for one element.
    Returns result dict or None on failure.
    """
    if data_dir is None:
        data_dir = str(Workspace.current().data_json_dir)
    print(f"\n{'='*65}")
    print(f"  {element}")
    print(f"{'='*65}")

    data = load_element(element, data_dir=data_dir, extrapolation=extrapolation,
                        rydberg_file=rydberg_file, transitions_file=transitions_file)
    if data is None:
        return None
    if model_f:
        from qali.pipelines.dhf import load_dhf_lines
        data['dhf_lines'] = load_dhf_lines(element, data_dir, model_file, model_f)

    all_transitions = data['transitions']
    gs_label        = data['gs_label']

    # Ground state transitions 
    trans_gs = build_transitions_for_state(gs_label, data, all_transitions)
    print(f"  Ground state {gs_label}: {len(trans_gs)} transitions with f>0")

    n_nist    = sum(1 for t in trans_gs if t['source'] in ('NIST', 'precision'))
    n_prec    = sum(1 for t in trans_gs if t['source'] == 'precision')
    n_coulomb = sum(1 for t in trans_gs if t['source'] == 'Coulomb')
    _prec = f"  (precision D-lines: {n_prec})" if n_prec else ""
    print(f"    NIST: {n_nist}{_prec}   Coulomb: {n_coulomb}")
    n_extrap = sum(1 for t in trans_gs if t['source'] == 'extrapolated')
    if n_extrap:
        print(f"    extrapolated (model + anchors, +-20%): {n_extrap}")

    # Static polarizability 
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
    print("  Computing C₆ (Casimir-Polder integral)...")
    C6 = compute_C6(trans_gs)
    alpha0_sigma = alpha_sigma(0.0, trans_gs)
    C6_sigma = c6_sigma(trans_gs) if trans_gs else 0.0
    print(f"  f-value propagated 1σ (independent errors; core α excluded): "
          f"α(0) ± {alpha0_sigma:.2f} au ({alpha0_sigma/abs(alpha0)*100:.1f}%)   "
          f"C₆ ± {C6_sigma:.1f} au ({C6_sigma/abs(C6)*100:.1f}%)" if alpha0 and C6 else "")
    C6_expt = C6_EXPT.get(element)

    if C6_expt:
        err_pct_c6 = (C6 - C6_expt) / C6_expt * 100
        status_c6  = '✅' if abs(err_pct_c6) < 20 else '⚠'
        print(f"  C₆   = {C6:.1f} au   expt: {C6_expt:.1f} au  "
              f"({err_pct_c6:+.1f}%) {status_c6}")
    else:
        print(f"  C₆   = {C6:.1f} au   (no experimental reference)")

    # Excited state transitions & magic wavelengths
    magic_results = []
    excited_used  = None
    trans_ex      = []

    # Determine which excited state to use
    if excited_label is None:
        # Find the lowest excited state with l = gs_l ± 1 (first dipole-allowed)
        candidates = []
        for t in all_transitions:
            if t['lower_label'] == gs_label:
                candidates.append(t['upper_label'])
        # Only states with a known energy can be ranked; an unknown energy must not
        # masquerade as "lowest" (the old -999 default made it win).
        candidates = [c for c in candidates if data['state_energy_au'].get(c) is not None]
        if candidates:
            # Pick the one with lowest energy
            candidates.sort(key=lambda lbl: data['state_energy_au'][lbl])
            excited_label = candidates[0]

    if excited_label:
        excited_used = excited_label
        trans_ex = build_transitions_for_state(excited_label, data, all_transitions)
        if not trans_ex:
            raise MissingDataError(
                f"excited state {excited_label!r} has no dipole transitions in the "
                f"{element} data (check the label against {element}_rydberg.json)")
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

    # Compute α(ω) spectra 
    print(f"  Computing α(ω) spectrum ({n_grid} points)...")
    wl_gs, alpha_gs = compute_alpha_spectrum(trans_gs, wl_min, wl_max, n_grid)
    if trans_ex:
        wl_ex, alpha_ex = compute_alpha_spectrum(trans_ex, wl_min, wl_max, n_grid)
    else:
        wl_ex, alpha_ex = wl_gs, None

    # AC Stark shift / trap depth 
    # Reuses trans_gs/trans_ex directly — no JSON round-trip, since this used
    # to be a separate script (acstark.py) that re-loaded polarizability.py's
    # saved output. Merged into one process.
    I_ref_Wm2 = kWcm2_to_Wm2(I_ref_kWcm2)
    # Trap frequencies need the mass of the ISOTOPE actually trapped (default: most abundant
    # natural isotope; --isotope overrides), not the IUPAC average atomic weight.
    iso_A = isotope_A if isotope_A is not None else default_isotope_A(element)
    mass_amu = ISOTOPE_MASS_AMU.get((element, iso_A)) if iso_A is not None else None
    if mass_amu is None:
        mass_amu = ATOMIC_MASS_AMU.get(element)
        iso_A = None
    if mass_amu is None:
        print(f"  ⚠  Mass for {element} unknown — trap frequencies will be skipped")

    if trans_ex and magic_results:
        print("  AC Stark cross-check at each magic wavelength "
              "(U_ground should equal U_excited, independent of intensity):")
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

    # Assemble result dict 
    result = dict(
        element          = element,
        extrapolation    = data.get('extrapolation'),
        inputs           = ({k: os.path.basename(v) for k, v in (('rydberg', rydberg_file), ('transitions', transitions_file)) if v}
                            | ({model_f: os.path.basename(model_file or f"{element}_{model_f}.json")} if model_f else {})) or None,
        gs_label         = gs_label,
        excited_label    = excited_used,
        alpha0_au        = round(alpha0, 4),
        alpha0_expt_au   = alpha0_expt,
        alpha0_sigma_au  = round(alpha0_sigma, 4),
        C6_au            = round(C6, 2),
        C6_sigma_au      = round(C6_sigma, 2),
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
        isotope_A        = iso_A,
        waist_um         = waist_um,
        point_result     = point_result,
    )
    return result


# PLOTTING — INDIVIDUAL ELEMENT



def _trim_transitions(trans_list):
    """
    Keep only the fields needed to recompute alpha(omega) / alpha(i*xi) at any
    frequency: delta_E_au, fosc, sign, other_label, source, wl_nm. These are
    exactly the fields alpha_real()/alpha_imag_freq() read, plus sigma_f (the f-value 1σ
    used for error bars).

    Each record is rebuilt through the typed ``Transition`` record
    (core.spectroscopy) at this engine→JSON boundary:
    constructing it asserts the complete field set of the record (a missing
    key fails loudly instead of dropping silently), and the emitted dict
    keeps the legacy JSON keys and order byte-identical
    (``other_label`` ↔ ``upper_label``, ``sigma_f`` ↔ ``sigma``; unknown σ
    stays None).
    """
    trimmed = []
    for t in trans_list:
        tr = Transition(
            delta_E_au=t['delta_E_au'],
            fosc=t['fosc'],
            sign=t['sign'],
            upper_label=t['other_label'],
            source=t['source'],
            wl_nm=t['wl_nm'],
            sigma=t.get('sigma_f'),
        )
        trimmed.append(dict(
            delta_E_au=tr.delta_E_au,
            fosc=tr.fosc,
            sign=tr.sign,
            other_label=tr.upper_label,
            source=tr.source,
            wl_nm=tr.wl_nm,
            sigma_f=tr.sigma,
        ))
    return trimmed

# MAIN

def save_json(result, data_dir=None):
    """Save polarizability results to <data dir>/<element>_polarizability.json."""
    if data_dir is None:
        data_dir = str(Workspace.current().data_json_dir)
    el = result['element']
    out = dict(
        element=el,
        gs_label=result['gs_label'],
        excited_label=result['excited_label'],
        alpha0_au=result['alpha0_au'],
        alpha0_expt_au=result.get('alpha0_expt_au'),
        alpha0_sigma_au=result.get('alpha0_sigma_au'),
        C6_au=result['C6_au'],
        C6_sigma_au=result.get('C6_sigma_au'),
        C6_expt_au=result.get('C6_expt_au'),
        magic_wavelengths=result['magic_wavelengths'],
        n_gs_transitions=len(result['transitions_gs']),
        n_ex_transitions=len(result['transitions_ex']),
        wl_range_nm=[result['wl_min'], result['wl_max']],
        I_ref_kWcm2=result.get('I_ref_kWcm2'),
        point_result=result.get('point_result'),
        extrapolation=result.get('extrapolation'),
        **({'inputs': result['inputs']} if result.get('inputs') else {}),
        transitions_gs=_trim_transitions(result['transitions_gs']),
        transitions_ex=_trim_transitions(result['transitions_ex']),
    )
    path = os.path.join(data_dir, f"{el}_polarizability.json")
    write_json(path, out)
    print(f"  ✓ {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qali --pipeline polarizability",
        description='Dynamic polarizability, C₆, magic wavelengths, and AC Stark / trap depth')
    parser.add_argument('element', nargs='?', default=None)
    parser.add_argument('--excited', default=None)
    parser.add_argument('--wl-min', type=float, default=200.0)
    parser.add_argument('--wl-max', type=float, default=2000.0)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--heteronuclear', action='store_true', help=argparse.SUPPRESS)   # C6 pairs: Compare page
    parser.add_argument('--data-dir', default=str(Workspace.current().data_json_dir))
    parser.add_argument('--rydberg-file', default=None, metavar='FILE',
                        help='Rydberg data to read instead of <species>_rydberg.json '
                             '(a path, or a name under the data folder).')
    parser.add_argument('--transitions-file', default=None, metavar='FILE',
                        help='Transitions data to read instead of <species>_transitions.json '
                             '(a path, or a name under the data folder).')
    from qali.pipelines.dhf import add_model_f_arguments
    add_model_f_arguments(parser, 'Numerov and Coulomb')
    parser.add_argument('--no-plot', action='store_true', help=argparse.SUPPRESS)   # writes no plots any more
    parser.add_argument('--n-grid', type=int, default=2000)
    parser.add_argument('--extrapolation', default=None, metavar='CONFIG.json',
                        help='Extrapolated ns->np f-values (same config as the lifetimes stage; '
                             'mode use fills gaps, tagged extrapolated).')
    parser.add_argument('--isotope', type=int, default=None,
                        help='Mass number A used for the trap-frequency mass (default: most abundant natural isotope)')
    parser.add_argument('--intensity', type=float, default=10.0)
    parser.add_argument('--waist', type=float, default=1.0)
    parser.add_argument('--wl', type=float, default=None)
    parser.add_argument('--state', default=None,
                        help='Print alpha(0) for one label and exit (e.g. 20s)')
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    data_dir = args.data_dir
    print(f"qali --pipeline polarizability (v{__version__})")

    from qali.physics.polarizability_core import require_alpha_species

    if args.all and (args.rydberg_file or args.transitions_file):
        parser.error("--rydberg-file / --transitions-file apply to one species; drop --all")

    if args.state:
        if args.element is None:
            parser.error("--state requires an element, e.g. `polarizability Na --state 20s`")
        el = require_alpha_species(args.element, "polarizability.py")["species_id"]
        data = load_element(el, data_dir, extrapolation=args.extrapolation,
                            rydberg_file=resolve_data_file(args.rydberg_file, el, data_dir, "Rydberg"),
                            transitions_file=resolve_data_file(args.transitions_file, el, data_dir, "Transitions"))
        if data is None:
            raise MissingDataError(f"No polarizability data for {el} in {data_dir}")
        from qali.pipelines.dhf import load_dhf_lines, model_f_choice
        _model, _model_spec = model_f_choice(args)
        if _model:
            data['dhf_lines'] = load_dhf_lines(el, data_dir, _model_spec, _model)
        lab = args.state.strip()
        if lab not in data['state_energy_au']:
            parser.error(f"state {lab!r} not in {el} rydberg set")
        trans = build_transitions_for_state(lab, data, data['transitions'])
        a0 = alpha0_static(trans)
        valence = [t for t in trans if t.get('other_label') != 'core']
        print(f"  {el} {lab}: alpha(0) = {a0:.6g} au  "
              f"({len(valence)} valence + core partners)")
        return 0

    if args.all:
        elements = get_available_elements(data_dir)
        if not elements:
            raise MissingDataError(
                f"No elements found in {data_dir}. Run rydberg + transitions first.")
        print("Running all elements:", ", ".join(elements))
    else:
        if args.element is None:
            available = get_available_elements(data_dir)
            parser.error(
                "missing element argument. Available elements: "
                + (", ".join(available) if available else "none"))
        elements = [require_alpha_species(args.element, "polarizability.py")["species_id"]]

    from qali.pipelines.dhf import model_f_choice
    model_f, model_spec = model_f_choice(args)
    all_results = []
    failed = []
    for el in elements:
        try:
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
                isotope_A=args.isotope,
                extrapolation=args.extrapolation,
                rydberg_file=resolve_data_file(args.rydberg_file, el, data_dir, "Rydberg"),
                transitions_file=resolve_data_file(args.transitions_file, el, data_dir, "Transitions"),
                model_f=model_f,
                model_file=resolve_data_file(model_spec, el, data_dir, "model"),
            )
        except MissingDataError as exc:
            if not args.all:
                raise
            print(f"  ⚠  {el}: {exc}")
            result = None
        if result is None:
            failed.append(el)
            continue
        save_json(result, data_dir)
        all_results.append(result)

    print("\n" + "=" * 65)
    print("  Summary")
    print("=" * 65)
    for r in all_results:
        el = r["element"]
        print(f"    {data_dir}/{el}_polarizability.json")
    if all_results:
        print("    (plots: the Polarizability page, qali --pipeline polarizability_page <species>)")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    print("\n" + "=" * 65 + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
