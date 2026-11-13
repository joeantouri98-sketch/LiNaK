"""Optical-tweezer scattering and heating: the pure computation.

A far-off-resonant Gaussian dipole trap scatters photons through the
imaginary part of the complex polarizability (atomic units; same convention
as ``polarizability_core.alpha_real``)::

    α(ω) = Σ_k  s_k f_k / (ω_k² − ω² − i Γ_k ω)

    Re[α] → trap depth  U = −(2π a₀³/c) · Re[α] · I      (Grimm et al. 2000)
    Im[α] → scattering  R_sc = (4π a₀³ / (ħ c)) · Im[α] · I

Γ_k is the upper state's total decay rate (A_total from the lifetimes
pipeline). The ion-core lines enter Re[α] (the trap depth) with Γ = 0: far below their
resonances they polarise but do not scatter trap light.
Recoil heating is Ė = 2 E_r R_sc with E_r = h²/(2 m λ²), and the
order-of-magnitude trap lifetime is τ_heat ≈ |U| / Ė.

Known issue: the damping uses a frequency-independent Γ (PHYSICS_BUGS.md
P21); it is kept as is so results stay comparable until that is fixed.

No files, prints or plots here: callers pass the transition network, the
upper-state decay rates and the atomic mass in.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from qali.core.constants import A0_SI, AMU_KG, AU_TIME_S, C_SI, H_SI, HBAR_SI, KB_SI
from qali.physics.polarizability_core import (
    U_joules,
    U_to_kHz_h,
    U_to_uK,
    alpha_real,
    kWcm2_to_Wm2,
    nm_from_omega_au,
    omega_au_from_nm,
    trap_frequencies_Hz,
)

# R_sc (s⁻¹) = R_SC_PREFACTOR · Im[α_au] · I[W/m²]
R_SC_PREFACTOR = 4.0 * math.pi * A0_SI**3 / (HBAR_SI * C_SI)
# U (J) = −U_PREFACTOR · Re[α_au] · I[W/m²]  (the report's JS uses it)
U_PREFACTOR = 2.0 * math.pi * A0_SI**3 / C_SI

_POLE_GUARD = 1e-4
_NAN = float("nan")


@dataclass(frozen=True)
class Channel:
    """One transition's share of the scattering rate at the trap wavelength."""

    label: str | None
    wl_nm: float
    fosc: float
    Gamma_s: float
    R_sc_Hz: float
    source: str | None


@dataclass(frozen=True)
class TweezerPoint:
    """Everything at one (wavelength, intensity, waist)."""

    wl_nm: float
    I_kWcm2: float
    I_Wm2: float
    waist_um: float
    mass_amu: float
    alpha_re_au: float
    alpha_im_au: float
    U_J: float
    U_uK: float
    U_kHz_h: float
    R_sc_Hz: float
    R_sc_per_s: float
    E_r_J: float
    E_r_nK: float
    heating_J_s: float
    heating_uK_s: float
    tau_heat_s: float
    tau_scatter_s: float
    nu_r_Hz: float
    nu_z_Hz: float
    zR_um: float
    top_channels: list[Channel]


@dataclass(frozen=True)
class TweezerSpectrum:
    """Wavelength sweep at fixed intensity (lists, one value per wavelength)."""

    wl_nm: list[float]
    alpha_re_au: list[float]
    alpha_im_au: list[float]
    R_sc_Hz: list[float]
    heating_uK_s: list[float]
    tau_heat_s: list[float]
    U_uK: list[float]


@dataclass(frozen=True)
class Resonance:
    label: str
    wl_nm: float
    fosc: float


@dataclass(frozen=True)
class TweezerResult:
    """A full tweezer calculation for one atom."""

    mass_amu: float
    n_transitions: int
    n_with_gamma: int
    n_core_skipped: int
    wl_min: float
    wl_max: float
    I_ref_kWcm2: float
    waist_um: float
    point: TweezerPoint
    spectra: TweezerSpectrum
    resonances: list[Resonance]


def attach_gamma(
    transitions: Sequence[dict], upper_state_rates: Mapping[str, float]
) -> tuple[list[dict], int, int]:
    """Copies of *transitions* with ``Gamma_au``/``Gamma_s`` from the upper state.

    *upper_state_rates* maps a state label to its total decay rate A (s⁻¹).
    Core channels and channels without a rate keep Γ = 0: they count for Re[α] only. (Core
    channels used to be dropped, which left the core polarisability out of the trap depth:
    1.4 % for Cs at 1064 nm.) Returns (channels, number with Γ, number of core channels).
    """
    out = []
    n_with = 0
    n_skip_core = 0
    for tr in transitions:
        t = dict(tr)
        if tr.get('source') == 'core':
            n_skip_core += 1
            t['Gamma_au'] = 0.0
            t['Gamma_s'] = 0.0
            out.append(t)
            continue
        A_s = upper_state_rates.get(t.get('other_label', ''))
        if A_s:
            t['Gamma_au'] = A_s * AU_TIME_S
            t['Gamma_s'] = A_s
            n_with += 1
        else:
            t['Gamma_au'] = 0.0
            t['Gamma_s'] = 0.0
        out.append(t)
    return out, n_with, n_skip_core


def alpha_complex(omega_au: float, transitions: Sequence[dict]) -> tuple[float, float]:
    """(Re[α], Im[α]) in atomic units at real frequency ω.

    Each channel has Lorentzian damping −i Γ_ω ω in the denominator, with the
    frequency-dependent radiative damping of a classical oscillator,
    Γ_ω = Γ (ω/ω_k)² (Grimm, Weidemüller & Ovchinnikov, Adv. At. Mol. Opt. Phys.
    42, 95 (2000), Eqs. 2 and 10), so that the off-resonant scattering rate is
    Γ_sc ∝ (ω/ω_k)³ (Γ/(ω_k−ω) + Γ/(ω_k+ω))². A constant Γ overestimates Im α,
    hence R_sc, by (ω_k/ω)² (≈3.3 for Na at 1064 nm).

    Near an undamped pole the channel is skipped; if every channel was a pole,
    both parts are NaN.
    """
    re = 0.0
    im = 0.0
    near_pole = False
    for tr in transitions:
        dE = tr['delta_E_au']
        f = tr['fosc']
        sgn = tr['sign']
        Gam = tr.get('Gamma_au', 0.0) or 0.0
        x = dE * dE - omega_au * omega_au
        y = Gam * omega_au * (omega_au / dE) ** 2 if dE else 0.0   # Γ_ω ω, Γ_ω = Γ(ω/ω_k)²
        if abs(x) < _POLE_GUARD * abs(dE) and Gam <= 0:
            near_pole = True
            continue
        denom2 = x * x + y * y
        if denom2 <= 0:
            near_pole = True
            continue
        re += sgn * f * x / denom2
        im += sgn * f * y / denom2
    if near_pole and abs(re) == 0 and abs(im) == 0:
        return _NAN, _NAN
    return re, im


def scattering_rate_Hz(im_alpha_au: float, I_Wm2: float) -> float:
    """Photon scattering rate R_sc (s⁻¹) from Im[α]."""
    if not math.isfinite(im_alpha_au):
        return _NAN
    return R_SC_PREFACTOR * im_alpha_au * I_Wm2


def recoil_energy_J(wl_nm: float, mass_amu: float) -> float:
    """E_r = h² / (2 m λ²) for a photon of wavelength *wl_nm*."""
    lam = wl_nm * 1e-9
    m = mass_amu * AMU_KG
    return (H_SI ** 2) / (2.0 * m * lam * lam)


def channel_contributions(
    omega_au: float, transitions: Sequence[dict], I_Wm2: float, top_n: int = 5
) -> list[Channel]:
    """The *top_n* channels by |R_sc| contribution."""
    rows = []
    for tr in transitions:
        Gam = tr.get('Gamma_au', 0.0) or 0.0
        if Gam <= 0:
            continue
        dE = tr['delta_E_au']
        f = tr['fosc']
        sgn = tr['sign']
        x = dE * dE - omega_au * omega_au
        y = Gam * omega_au * (omega_au / dE) ** 2 if dE else 0.0   # Γ_ω ω, Γ_ω = Γ(ω/ω_k)²
        denom2 = x * x + y * y
        if denom2 <= 0:
            continue
        im_k = sgn * f * y / denom2
        wl_res = tr.get('wl_nm') or nm_from_omega_au(abs(dE))
        rows.append(Channel(
            label=tr.get('other_label'),
            wl_nm=round(wl_res, 4),
            fosc=f,
            Gamma_s=tr.get('Gamma_s', 0.0),
            R_sc_Hz=scattering_rate_Hz(im_k, I_Wm2),
            source=tr.get('source'),
        ))
    rows.sort(key=lambda r: abs(r.R_sc_Hz), reverse=True)
    return rows[:top_n]


def evaluate_point(
    transitions: Sequence[dict], wl_nm: float, I_kWcm2: float, waist_um: float,
    mass_amu: float,
) -> TweezerPoint:
    """Depth, scattering, heating and trap frequencies at one operating point."""
    omega = omega_au_from_nm(wl_nm)
    I_Wm2 = kWcm2_to_Wm2(I_kWcm2)
    re_a, im_a = alpha_complex(omega, transitions)
    if not math.isfinite(re_a):
        re_a = alpha_real(omega, transitions)

    U_J = U_joules(re_a, I_Wm2) if math.isfinite(re_a) else _NAN
    U_uK = U_to_uK(U_J) if math.isfinite(U_J) else _NAN
    U_kHz = U_to_kHz_h(U_J) if math.isfinite(U_J) else _NAN

    R_sc = scattering_rate_Hz(im_a, I_Wm2) if math.isfinite(im_a) else _NAN
    E_r = recoil_energy_J(wl_nm, mass_amu)
    E_dot = 2.0 * E_r * R_sc if math.isfinite(R_sc) else _NAN
    heat_uK_s = (E_dot / KB_SI) * 1e6 if math.isfinite(E_dot) else _NAN

    tau_heat_s = abs(U_J) / E_dot if (math.isfinite(U_J) and math.isfinite(E_dot)
                                       and E_dot > 0 and abs(U_J) > 0) else _NAN
    tau_sc_s = 1.0 / R_sc if (math.isfinite(R_sc) and R_sc > 0) else _NAN

    nu_r = nu_z = zR = _NAN
    if math.isfinite(U_J) and abs(U_J) > 0:
        nu_r, nu_z, zR = trap_frequencies_Hz(U_J, waist_um, wl_nm, mass_amu)

    return TweezerPoint(
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
        zR_um=(zR * 1e6) if math.isfinite(zR) else _NAN,
        top_channels=channel_contributions(omega, transitions, I_Wm2),
    )


def compute_spectra(
    transitions: Sequence[dict], wl_min: float, wl_max: float, n_grid: int,
    I_kWcm2: float, mass_amu: float,
) -> TweezerSpectrum:
    """Re[α], Im[α], R_sc, heating and τ_heat versus wavelength at fixed I.

    The uniform grid is refined (80 points, ±2 nm) around every resonance in
    range.
    """
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
        R = scattering_rate_Hz(im, I_Wm2) if math.isfinite(im) else _NAN
        R_sc[i] = R
        Er = recoil_energy_J(float(wl), mass_amu)
        Ed = 2.0 * Er * R if math.isfinite(R) else _NAN
        heat[i] = (Ed / KB_SI) * 1e6 if math.isfinite(Ed) else _NAN
        Uj = U_joules(r, I_Wm2) if math.isfinite(r) else _NAN
        U_uK_arr[i] = U_to_uK(Uj) if math.isfinite(Uj) else _NAN
        if math.isfinite(Uj) and math.isfinite(Ed) and Ed > 0 and abs(Uj) > 0:
            tau[i] = abs(Uj) / Ed
        else:
            tau[i] = _NAN

    return TweezerSpectrum(
        wl_nm=wl_arr.tolist(),
        alpha_re_au=re_a.tolist(),
        alpha_im_au=im_a.tolist(),
        R_sc_Hz=R_sc.tolist(),
        heating_uK_s=heat.tolist(),
        tau_heat_s=tau.tolist(),
        U_uK=U_uK_arr.tolist(),
    )


def strongest_resonances(transitions: Sequence[dict], limit: int = 12) -> list[Resonance]:
    """The damped channels with |f| ≥ 1e-4, strongest first (for plot markers)."""
    found = [
        Resonance(
            label=tr['other_label'],
            wl_nm=tr.get('wl_nm') or nm_from_omega_au(abs(tr['delta_E_au'])),
            fosc=tr['fosc'],
        )
        for tr in transitions
        if (tr.get('Gamma_au') or 0) > 0 and abs(tr.get('fosc', 0)) >= 1e-4
    ]
    found.sort(key=lambda r: -abs(r.fosc))
    return found[:limit]


def compute(
    transitions: Sequence[dict], upper_state_rates: Mapping[str, float], mass_amu: float,
    *, wl_nm: float, I_kWcm2: float, waist_um: float,
    wl_min: float, wl_max: float, n_grid: int,
) -> TweezerResult:
    """The full tweezer calculation: operating point, spectrum and resonances."""
    channels, n_with, n_core = attach_gamma(transitions, upper_state_rates)
    return TweezerResult(
        mass_amu=mass_amu,
        n_transitions=len(channels),
        n_with_gamma=n_with,
        n_core_skipped=n_core,
        wl_min=wl_min,
        wl_max=wl_max,
        I_ref_kWcm2=I_kWcm2,
        waist_um=waist_um,
        point=evaluate_point(channels, wl_nm, I_kWcm2, waist_um, mass_amu),
        spectra=compute_spectra(channels, wl_min, wl_max, n_grid, I_kWcm2, mass_amu),
        resonances=strongest_resonances(channels),
    )
