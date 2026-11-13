"""Atomic line spectra: the physics behind the Spectrum page.

Every quantity here has a twin in ``report_templates/spectrum_physics.js`` (the page recomputes the
spectra live); the tests run both and require them to agree. Conventions:

* Frequencies in Hz, vacuum wavelengths in nm, cross-sections per atom in m² (the page shows cm²).
* A line's integrated cross-section is ∫σ dν = π r_e c f (absorption oscillator strength f of the
  lower level). Its profile is a Voigt: a Gaussian (Doppler, σ_D = ν₀ √(kT/mc²), combined in
  quadrature with an instrument Gaussian) convolved with a Lorentzian of FWHM (Γ_u + Γ_l)/2π plus
  any extra (pressure, laser) width. The profile is area-normalised, φ = Re w(z) / (σ √(2π)),
  z = (ν − ν₀ + iγ) / (σ √2) with γ the Lorentzian half width and w the Faddeeva function.
* The same w gives the refractive index of a dilute vapour (Kramers–Kronig partner of the
  absorption): n − 1 = −(c / 4πν) N π r_e c f Im w(z) / (σ √(2π)).
* Populations are Boltzmann over the listed levels, N_l ∝ g_l exp(−E_l / kT). Absorption includes
  stimulated emission at the gas temperature through the factor 1 − exp(−hν/kT).
* Emission (optically thin): power per line ∝ N_u A_ul hν with N_u ∝ g_u exp(−E_u / kT_exc); at
  T_exc → ∞ this is g_u A hν (equal population per sublevel).
* Air wavelengths: standard air (the Edlén / Birch & Downs formula NIST's Atomic Spectra Database
  uses; Morton 2000, ApJS 130, 403), valid above 200 nm. Vacuum below 200 nm.
* Vapour pressure: Alcock, Itkin & Horrigan, Can. Metall. Q. 23, 309 (1984), as tabulated in the
  CRC Handbook: log₁₀(P/Pa) = a − b/T for the solid and the liquid.
* The vapour-pressure coefficients, default cell temperatures and isotope shifts are editable in
  Preferences > Tables and variables.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from qali.core import user_tables as _user_tables
from qali.core.constants import AMU_KG, C_SI, H_SI, KB_EV, KB_SI

R_E = 2.8179403262e-15          # classical electron radius (m), CODATA 2018
LINE_K = math.pi * R_E * C_SI   # ∫σ dν per unit f (m² Hz)
SQRT2PI = math.sqrt(2.0 * math.pi)


# ---------------------------------------------------------------- line shape
def faddeeva(z):
    """w(z) = exp(−z²) erfc(−iz) (scipy's reference implementation)."""
    from scipy.special import wofz
    return wofz(z)


def doppler_sigma_hz(nu_hz, mass_amu, T_K):
    """Gaussian standard deviation (Hz) of the Doppler profile: σ_D = ν₀ √(kT / mc²)."""
    return nu_hz * math.sqrt(KB_SI * T_K / (mass_amu * AMU_KG * C_SI ** 2))


def instrument_sigma_hz(nu_hz, fwhm_nm):
    """Gaussian σ (Hz) of an instrument function of FWHM *fwhm_nm* at frequency *nu_hz*."""
    lam_nm = C_SI / nu_hz * 1e9
    return (fwhm_nm / (2.0 * math.sqrt(2.0 * math.log(2.0)))) * C_SI / lam_nm ** 2 * 1e9


def natural_fwhm_hz(tau_upper_s=None, tau_lower_s=None):
    """Lorentzian FWHM (Hz) of a line: (Γ_u + Γ_l)/2π with Γ = 1/τ (an infinite τ adds nothing)."""
    g = 0.0
    for tau in (tau_upper_s, tau_lower_s):
        if tau and tau > 0 and math.isfinite(tau):
            g += 1.0 / tau
    return g / (2.0 * math.pi)


def voigt(nu, nu0, sigma_hz, gamma_hwhm_hz):
    """Area-normalised Voigt profile (1/Hz) at *nu*: Re w(z) / (σ √(2π))."""
    z = (np.asarray(nu, float) - nu0 + 1j * gamma_hwhm_hz) / (sigma_hz * math.sqrt(2.0))
    return faddeeva(z).real / (sigma_hz * SQRT2PI)


def voigt_complex(nu, nu0, sigma_hz, gamma_hwhm_hz):
    """w(z) / (σ √(2π)): real part the absorption profile, imaginary part its dispersion partner."""
    z = (np.asarray(nu, float) - nu0 + 1j * gamma_hwhm_hz) / (sigma_hz * math.sqrt(2.0))
    return faddeeva(z) / (sigma_hz * SQRT2PI)


def cross_section_m2(nu, nu0, f, sigma_hz, gamma_hwhm_hz):
    """Absorption cross-section per atom in the lower level (m²): π r_e c f φ(ν)."""
    return LINE_K * f * voigt(nu, nu0, sigma_hz, gamma_hwhm_hz)


def refractive_index_minus_1(nu, nu0, f, sigma_hz, gamma_hwhm_hz, N_m3):
    """n − 1 of N_m3 absorbers (in the lower level) per m³ near one line."""
    w = voigt_complex(nu, nu0, sigma_hz, gamma_hwhm_hz)
    return -(C_SI / (4.0 * math.pi * np.asarray(nu, float))) * N_m3 * LINE_K * f * w.imag


def stimulated_factor(nu_hz, T_K):
    """1 − exp(−hν/kT): absorption minus stimulated emission in thermal equilibrium."""
    if T_K <= 0:
        return 1.0
    return -math.expm1(-H_SI * nu_hz / (KB_SI * T_K))


# ---------------------------------------------------------------- air wavelengths
def air_index(lam_vac_nm):
    """Refractive index of standard air at the vacuum wavelength (nm); 1 below 200 nm."""
    lam = np.asarray(lam_vac_nm, float)
    s2 = (1e3 / lam) ** 2                        # (1/λ in µm)²
    n = 1.0 + 8.34254e-5 + 2.406147e-2 / (130.0 - s2) + 1.5998e-4 / (38.9 - s2)
    return np.where(lam >= 200.0, n, 1.0)


def vac_to_air_nm(lam_vac_nm):
    return np.asarray(lam_vac_nm, float) / air_index(lam_vac_nm)


# ---------------------------------------------------------------- vapour
# log10(P/Pa) = a − b/T for the solid (below T_melt) and the liquid; Alcock et al. (1984) via CRC.
# Editable in Preferences > Tables and variables (read at call time).
VAPOUR_PRESSURE: dict[str, dict[str, float]] = {
    'Li': dict(solid_a=10.673, solid_b=8310.0, liquid_a=10.061, liquid_b=8023.0, T_melt=453.65),
    'Na': dict(solid_a=10.304, solid_b=5603.0, liquid_a=9.710, liquid_b=5377.0, T_melt=370.944),
    'K': dict(solid_a=9.967, solid_b=4646.0, liquid_a=9.408, liquid_b=4453.0, T_melt=336.53),
    'Rb': dict(solid_a=9.863, solid_b=4215.0, liquid_a=9.318, liquid_b=4040.0, T_melt=312.45),
    'Cs': dict(solid_a=9.717, solid_b=3999.0, liquid_a=9.171, liquid_b=3830.0, T_melt=301.59),
}
_user_tables.apply("spectrum.vapour_pressure", VAPOUR_PRESSURE)
VAPOUR_SOURCE = ("Alcock, Itkin & Horrigan, Can. Metall. Q. 23, 309 (1984), as tabulated in the CRC "
                 "Handbook; ±5% or better")


def vapour_record(element):
    """The page's form of an element's vapour-pressure curve, or None."""
    vp = VAPOUR_PRESSURE.get(element)
    if not vp:
        return None
    return dict(solid=[vp['solid_a'], vp['solid_b']], liquid=[vp['liquid_a'], vp['liquid_b']], T_melt=vp['T_melt'],
                src=VAPOUR_SOURCE)


def vapour_pressure_pa(element, T_K):
    vp = VAPOUR_PRESSURE.get(element)
    if not vp:
        return None
    phase = 'solid' if T_K < vp['T_melt'] else 'liquid'
    return 10.0 ** (vp[phase + '_a'] - vp[phase + '_b'] / T_K)


def vapour_density_cm3(element, T_K):
    """Atoms per cm³ of the saturated vapour above the metal at T (ideal gas, N = P/kT)."""
    p = vapour_pressure_pa(element, T_K)
    return None if p is None else p / (KB_SI * T_K) * 1e-6


# Typical cell temperatures for an optically interesting vapour (a few cm, T ~ 1-100% at the D lines).
DEFAULT_CELL_T = {'Li': 623.0, 'Na': 423.0, 'K': 333.0, 'Rb': 298.15, 'Cs': 298.15}
_user_tables.apply("spectrum.cell_temperature", DEFAULT_CELL_T)


# ---------------------------------------------------------------- populations
def boltzmann_weights(E_eV, g, T_K):
    """Relative populations g e^{−E/kT} normalised over the given levels (T = inf: ∝ g)."""
    E = np.asarray(E_eV, float)
    g = np.asarray(g, float)
    if not math.isfinite(T_K):
        w = g.copy()
    else:
        w = g * np.exp(-(E - E.min()) / (KB_EV * T_K))
    return w / w.sum()


# ---------------------------------------------------------------- curve of growth
def equivalent_width_hz(column_cm2, f, sigma_hz, gamma_hwhm_hz, n_points=20001):
    """W_ν = ∫(1 − e^{−τ(ν)}) dν for one line, τ = Nℓ σ(ν) (Nℓ in cm⁻²)."""
    K = LINE_K * f * column_cm2 * 1e4                         # Nℓ in m⁻² times ∫σ dν
    half = 12.0 * sigma_hz + 40.0 * gamma_hwhm_hz + 12.0 * math.sqrt(max(K * gamma_hwhm_hz / math.pi, 0.0))
    u = np.linspace(-1.0, 1.0, n_points)
    U = 8.0
    d = half * np.sinh(U * u) / math.sinh(U)
    tau = K * voigt(d, 0.0, sigma_hz, gamma_hwhm_hz)
    # beyond ±half the line is thin and Lorentzian: ∫ τ dν over both tails = 2Kγ/(π·half)
    return float(np.trapezoid(-np.expm1(-tau), d)) + 2.0 * K * gamma_hwhm_hz / (math.pi * half)


# ---------------------------------------------------------------- D-line hyperfine components
# Line-centre (hyperfine-free) isotope shifts ν(A) − ν(reference A), MHz.
ISOTOPE_SHIFT_MHZ = {
    ('Li', 6): {'D1': -10534.215, 'D2': -10533.352},   # Das & Natarajan, PRA 75, 052508 (2007)
    ('K', 41): {'D1': 235.5, 'D2': 236.2},             # Falke et al., PRA 74, 032503 (2006)
    ('Rb', 87): {'D1': 77.586, 'D2': 77.992},          # Banerjee et al., arXiv:physics/0209019, /0307094
}
_user_tables.apply("spectrum.isotope_shift", ISOTOPE_SHIFT_MHZ)
ISOTOPE_SHIFT_SOURCE = {
    'Li': "Das & Natarajan, Phys. Rev. A 75, 052508 (2007)",
    'K': "Falke et al., Phys. Rev. A 74, 032503 (2006)",
    'Rb': "Banerjee, Das & Natarajan, Phys. Rev. A 2003 (arXiv:physics/0209019, physics/0307094)",
}


@dataclass
class HfComponent:
    F: float
    Fp: float
    detuning_MHz: float          # from the reference isotope's line centre
    weight: float                # abundance × ground-F population × S_FF' (sums to the abundance)
    S: float                     # S_FF' (sums to 1 over F' for each F)


@dataclass
class IsotopeLine:
    A: int
    I: float
    abundance: float
    mass_amu: float
    shift_MHz: float
    components: list[HfComponent] = field(default_factory=list)


def hyperfine_components(I, J, Jp, A_g, B_g, A_e, B_e, C_e=0.0, shift_MHz=0.0, abundance=1.0):
    """Zero-field hyperfine components of a J → J' line of one isotope."""
    from qali.physics.hyperfine_core import F_levels, hyperfine_energy, hyperfine_line_strengths
    S = hyperfine_line_strengths(I, J, Jp)
    gtot = (2 * I + 1) * (2 * J + 1)
    out = []
    for F in F_levels(I, J):
        Eg = hyperfine_energy(F, I, J, A_g, B_g)
        for Fp in F_levels(I, Jp):
            s = S.get((F, Fp), 0.0)
            if s <= 0:
                continue
            Ee = hyperfine_energy(Fp, I, Jp, A_e, B_e, C_e)
            out.append(HfComponent(F, Fp, shift_MHz + Ee - Eg, abundance * (2 * F + 1) / gtot * s, s))
    return out


# ---------------------------------------------------------------- saturated absorption
def lamb_dip_hwhm_hz(gamma_rate_s, s0, extra_fwhm_hz=0.0):
    """Half width of a Lamb dip in laser frequency: ((Γ/2π)(1 + √(1 + s))/2 + extra FWHM)/2."""
    return 0.5 * (gamma_rate_s / (2 * math.pi) * 0.5 * (1.0 + math.sqrt(1.0 + s0)) + extra_fwhm_hz)


def sas_pairs(components, sigma_d_hz):
    """Lamb dips and crossovers of one isotope's components (pump and probe counter-propagating).

    Weak-saturation rate-equation estimate per pair, no optical pumping between ground levels: a
    pair (i, j) sharing a lower F or an upper F' gives a feature at (ν_i + ν_j)/2 of strength
    (a_i S_j + a_j S_i)/2 (a = component weight, S = its S_FF'), times the Doppler density of the
    velocity class that sees both, exp(−((ν_i − ν_j)/2)² / 2σ_D²). The weak probe sees only these
    narrow features: for one two-level line, α_D(Δ) [1 − (s/2) γ_d² / (Δ² + γ_d²)]. The scan moves the
    pump's hole and the probe's resonance in opposite directions, so the dip's FWHM in laser frequency
    is the mean of the probe width and the saturated hole width, (Γ/2π)(1 + √(1 + s))/2 (Demtröder,
    Laser Spectroscopy, Vol. 2, §2.2). Returns [(centre_MHz, strength)].
    """
    out = []
    n = len(components)
    for i in range(n):
        ci = components[i]
        for j in range(i, n):
            cj = components[j]
            if i != j and ci.F != cj.F and ci.Fp != cj.Fp:
                continue
            k = ci.weight * ci.S if i == j else 0.5 * (ci.weight * cj.S + cj.weight * ci.S)
            half = 0.5 * (ci.detuning_MHz - cj.detuning_MHz) * 1e6
            k *= math.exp(-half * half / (2.0 * sigma_d_hz ** 2))
            out.append((0.5 * (ci.detuning_MHz + cj.detuning_MHz), k))
    return out
