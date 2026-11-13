"""Two-level laser cooling: scattering rate, molasses force, Doppler and recoil limits.

Pure functions in SI units (Γ is the natural decay rate A in s⁻¹, Δ and kv in rad/s), mirrored in
``report_templates/compare_physics.js`` for the Compare page. References: Metcalf & van der Straten,
*Laser Cooling and Trapping* (1999), ch. 2-3 and 7; Foot, *Atomic Physics* (2005), ch. 9;
Lett et al., JOSA B 6, 2084 (1989) for the saturation-broadened Doppler temperature.
"""

from __future__ import annotations

import math

import numpy as np

from linak.core.constants import AMU_KG, C_SI, H_SI, HBAR_SI, KB_SI


def wavenumber(wl_nm):
    """k = 2π/λ in m⁻¹."""
    return 2.0 * math.pi / (wl_nm * 1e-9)


def scattering_rate(gamma, s, delta):
    """R = (Γ/2) s / (1 + s + (2Δ/Γ)²), photons per second."""
    x = 2.0 * np.asarray(delta, dtype=float) / gamma
    return 0.5 * gamma * s / (1.0 + s + x * x)


def molasses_force(v, gamma, s, delta, wl_nm):
    """1D optical molasses: two counter-propagating beams, each of saturation s, treated
    independently (valid for s ≲ 1): F = ħk [R(Δ − kv) − R(Δ + kv)], in newtons."""
    k = wavenumber(wl_nm)
    v = np.asarray(v, dtype=float)
    return HBAR_SI * k * (scattering_rate(gamma, s, delta - k * v) - scattering_rate(gamma, s, delta + k * v))


def doppler_temperature(gamma, s, delta, n_beams=2):
    """Steady-state 1D temperature of the molasses, k_B T = (ħΓ/4)(1 + N s + (2Δ/Γ)²)/(2|Δ|/Γ),
    for red detuning (Δ < 0; +inf otherwise). Its minimum, ħΓ/(2k_B) at Δ = −Γ/2 and s → 0, is the
    Doppler limit; N s is the saturation broadening by the N beams (Lett et al. 1989)."""
    delta = np.asarray(delta, dtype=float)
    x = 2.0 * delta / gamma
    with np.errstate(divide='ignore'):
        T = HBAR_SI * gamma / (4.0 * KB_SI) * (1.0 + n_beams * s + x * x) / np.abs(x)
    return np.where(delta < 0, T, np.inf)


def doppler_limit_K(gamma):
    """T_D = ħΓ/(2k_B)."""
    return HBAR_SI * gamma / (2.0 * KB_SI)


def recoil_velocity(wl_nm, mass_amu):
    """v_rec = ħk/m, m/s."""
    return HBAR_SI * wavenumber(wl_nm) / (mass_amu * AMU_KG)


def recoil_temperature_K(wl_nm, mass_amu):
    """T_rec = ħ²k²/(m k_B) (Steck's convention: twice the one-photon recoil energy over k_B)."""
    return (HBAR_SI * wavenumber(wl_nm)) ** 2 / (mass_amu * AMU_KG * KB_SI)


def max_acceleration(gamma, wl_nm, mass_amu):
    """a_max = ħkΓ/(2m), m/s²."""
    return HBAR_SI * wavenumber(wl_nm) * gamma / (2.0 * mass_amu * AMU_KG)


def capture_velocity(gamma, wl_nm):
    """v_c ≈ Γ/k, the velocity scale over which the molasses force acts."""
    return gamma / wavenumber(wl_nm)


def isat_cycling_mW_cm2(gamma, wl_nm):
    """Two-level saturation intensity of a closed cycling transition, I_sat = πhcΓ/(3λ³)
    (σ± light on the stretched D2 transition; Steck)."""
    lam = wl_nm * 1e-9
    return math.pi * H_SI * C_SI * gamma / (3.0 * lam ** 3) * 0.1


def most_probable_speed(T_K, mass_amu):
    """v_p = √(2k_BT/m) of the Maxwell-Boltzmann speed distribution, m/s."""
    return math.sqrt(2.0 * KB_SI * T_K / (mass_amu * AMU_KG))


def slower_length(v, gamma, wl_nm, mass_amu, eta=0.5):
    """Shortest Zeeman slower that stops atoms of speed v at a fraction η of a_max: v²/(2η a_max)."""
    return v * v / (2.0 * eta * max_acceleration(gamma, wl_nm, mass_amu))
