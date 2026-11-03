"""Line shapes of the interactive absorption spectrum: the pure computation.

Each line is a peak-normalised Doppler (Gaussian), instrumental (Gaussian) or
natural (Lorentzian, ħ/τ of the upper state) profile scaled by f · n_degenerate,
evaluated on a wavelength grid that is fine around every line.

The browser panel's natural width is the same ħ/τ of the upper state
(``line_gamma_nm``), falling back to the line's own A only without a lifetime.
"""

from __future__ import annotations

import numpy as np

from linak.core.constants import C_SI, HBAR_EV_S, HC_EV_NM, KB_EV

def doppler_sigma_nm(wl_nm, mc2_eV, T_K=300.0):
    """Gaussian σ (standard deviation, NOT FWHM) in nm from Doppler broadening;
    *mc2_eV* is the atom's rest energy.
    FWHM = 2√(2ln2)·σ ≈ 2.355·σ. Used directly in exp(-0.5*(x/σ)²)."""
    E_eV     = HC_EV_NM / wl_nm
    # Doppler standard deviation: sigma_nu/nu = sqrt(kT/mc^2)  (FWHM = 2 sqrt(2 ln2) sigma = nu sqrt(8 kT ln2/mc^2))
    sigma_eV = E_eV * np.sqrt(KB_EV * T_K / mc2_eV)
    return wl_nm**2 / HC_EV_NM * sigma_eV   # convert ΔE → Δλ

def natural_gamma_nm(state_label, wl_nm, lifetimes_by_state):
    """Lorentzian FWHM in nm from natural linewidth (ħ/τ)."""
    lt_entry = lifetimes_by_state.get(state_label)
    if lt_entry:
        tau_s = lt_entry.get('lifetime_s',
                             lt_entry.get('lifetime_ns', 0) * 1e-9)
        if tau_s and tau_s > 0:
            gamma_eV = HBAR_EV_S / tau_s
            return wl_nm**2 / HC_EV_NM * gamma_eV
    return None


# Adaptive wavelength grid: the Doppler σ at 589 nm for Na is ~1.5 pm, invisible on a
# 350 nm-wide uniform grid, so a fine sub-grid is built around each line (±6σ_inst,
# pts_per_line points) and merged, sorted and de-duplicated. Every peak is then
# resolved regardless of the overall wavelength span.
SIGMA_INST_DEFAULT = 0.3   # nm — makes all peaks visible on overview plot

def adaptive_grid(lines, sigma_inst=SIGMA_INST_DEFAULT, pts_per_line=300):
    """
    Build a wavelength grid that is fine around every spectral line and
    coarse in the gaps. Each line contributes pts_per_line points spanning
    ±6*sigma_inst. A coarse background grid (200 pts) fills the full range.
    """
    if not lines:
        return np.linspace(100, 1100, 500)
    wls   = [r['wl'] for r in lines]
    lo    = max(10, min(wls) - 10 * sigma_inst)
    hi    = max(wls) + 10 * sigma_inst
    # Coarse background
    grids = [np.linspace(lo, hi, 300)]
    # Fine sub-grid per line
    for wl0 in wls:
        grids.append(np.linspace(wl0 - 6*sigma_inst, wl0 + 6*sigma_inst, pts_per_line))
    combined = np.concatenate(grids)
    combined = np.clip(combined, lo, hi)
    combined = np.unique(combined)   # sort + deduplicate
    return combined

def broadened_spectrum(lines, wl_arr, broadening, mc2_eV, lifetimes_by_state,
                   T_K=300.0, sigma_inst_nm=SIGMA_INST_DEFAULT):
    """
    Compute a normalised spectrum array on wl_arr.
    broadening: 'doppler' | 'instrumental' | 'natural'

    Amplitude = fosc * n_degenerate so degenerate Rydberg levels contribute
    their correct total absorption strength (PHYSICS FIX 6).

    All lineshapes are peak-normalised to 1 before scaling by amplitude so
    that relative line heights correctly reflect oscillator strengths regardless
    of linewidth (PHYSICS FIX 2 & 5).

      Gaussian peak = 1  at x=x0  (already satisfied by exp formula)
      Lorentzian peak = 2/(pi*gamma) at x=x0 — divide out to normalise

    Returns normalised intensity array (max=1 across all lines combined).
    """
    result = np.zeros(len(wl_arr))
    for r in lines:
        wl0  = r['wl']
        # PHYSICS FIX 6: weight by total degeneracy-corrected oscillator strength
        n_deg = r.get('n_deg', 1)
        amp   = r['fosc'] * n_deg
        lbl   = r['ul']
        if broadening == 'doppler':
            sig = max(doppler_sigma_nm(wl0, mc2_eV, T_K), 1e-6)
            # Peak-normalised Gaussian: peak = 1 at x=x0
            result += amp * np.exp(-0.5 * ((wl_arr - wl0) / sig)**2)
        elif broadening == 'instrumental':
            sig = sigma_inst_nm
            result += amp * np.exp(-0.5 * ((wl_arr - wl0) / sig)**2)
        elif broadening == 'natural':
            gamma = natural_gamma_nm(lbl, wl0, lifetimes_by_state)
            if gamma and gamma > 0:
                # PHYSICS FIX 5: peak-normalised Lorentzian
                # Raw Lorentzian peak = (gamma/2pi)/(gamma/2)^2 = 2/(pi*gamma)
                # Divide by peak so max = 1 at x=x0
                lorentz_raw = (gamma / (2*np.pi)) / ((wl_arr - wl0)**2 + (gamma/2)**2)
                lorentz_peak = 2.0 / (np.pi * gamma)
                result += amp * lorentz_raw / lorentz_peak
            else:
                # PHYSICS FIX 3: use Lorentzian with narrow but finite gamma
                # rather than a Gaussian (natural lineshape IS Lorentzian)
                gamma_proxy = 1e-5   # 0.01 pm — essentially a delta function
                lorentz_raw  = (gamma_proxy / (2*np.pi)) / ((wl_arr - wl0)**2 + (gamma_proxy/2)**2)
                lorentz_peak = 2.0 / (np.pi * gamma_proxy)
                result += amp * lorentz_raw / lorentz_peak
    mx = result.max()
    if mx > 0:
        result /= mx
    return result


def line_gamma_nm(r, lifetimes_by_state):
    """Natural linewidth (Lorentzian FWHM) in nm for the JS panel.

    Same quantity as the pre-rendered 'natural' trace: hbar / tau_total of the UPPER state
    (all decay channels), via natural_gamma_nm. Only when the state has no lifetime entry does
    it fall back to the single line's A coefficient (a lower bound on the total rate)."""
    g = natural_gamma_nm(r.get('ul'), r.get('wl', 500.0), lifetimes_by_state)
    if g:
        return g
    _A  = r.get('A', 0)
    _wl = r.get('wl', 500.0)
    if not _A or _A <= 0:
        return 0.0
    return (_wl * 1e-9)**2 / (2 * np.pi * C_SI) * _A * 1e9   # nm
