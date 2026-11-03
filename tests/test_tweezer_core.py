# -*- coding: utf-8 -*-
"""``physics.tweezer_core`` against reference values and its own invariants.

These pin the prefactors and conventions, not the known large-detuning issue
(PHYSICS_BUGS.md P21): the two-level check runs close to resonance, where the
frequency dependence of Γ does not matter.
"""
from __future__ import annotations

import math

import pytest

from linak.core.constants import AU_TIME_S, HBAR_SI, KB_SI
from linak.physics.polarizability_core import U_joules, alpha_real, kWcm2_to_Wm2, omega_au_from_nm
from linak.physics.tweezer_core import (
    alpha_complex,
    attach_gamma,
    compute,
    evaluate_point,
    recoil_energy_J,
    scattering_rate_Hz,
)

NA_MASS_AMU = 22.98976928          # Steck, Sodium D Line Data, Table 1
NA_D2_NM = 589.158326              # vacuum wavelength (Steck, Table 3)
NA_D2_A_S = 6.1542e7               # Γ = 2π × 9.7946 MHz (Steck, Table 3)


def _channel(wl_nm, fosc, label, source="NIST"):
    return dict(delta_E_au=omega_au_from_nm(wl_nm), fosc=fosc, sign=1,
                other_label=label, source=source)


def test_recoil_energy_matches_steck_for_the_na_d2_line():
    """T_r = ħ²k²/(m k_B) = 2.3998 µK for Na D2, so E_r/k_B = T_r / 2."""
    E_r = recoil_energy_J(NA_D2_NM, NA_MASS_AMU)
    assert E_r / KB_SI * 1e6 == pytest.approx(2.3998 / 2, rel=1e-4)


def test_near_resonance_reproduces_the_two_level_relation():
    """R_sc = (Γ/|Δ|) · |U|/ħ for one channel with Γ ≪ |Δ| ≪ ω₀ (Grimm et al. 2000)."""
    channels, _, _ = attach_gamma([_channel(NA_D2_NM, 0.64, "3p3/2")], {"3p3/2": NA_D2_A_S})
    wl_trap = 589.5                                   # ~0.3 THz red of D2
    omega = omega_au_from_nm(wl_trap)
    I_Wm2 = kWcm2_to_Wm2(1.0)
    re, im = alpha_complex(omega, channels)
    U = U_joules(re, I_Wm2)
    detuning_s = abs(omega - omega_au_from_nm(NA_D2_NM)) / AU_TIME_S
    expected = NA_D2_A_S / detuning_s * abs(U) / HBAR_SI
    assert scattering_rate_Hz(im, I_Wm2) == pytest.approx(expected, rel=2e-3)


def test_undamped_real_part_equals_alpha_real():
    """With Γ = 0 the complex polarizability reduces to polarizability_core's."""
    transitions = [_channel(589.16, 0.64, "3p3/2"), _channel(589.76, 0.32, "3p1/2"),
                   _channel(330.3, 0.01, "4p3/2")]
    omega = omega_au_from_nm(1064.0)
    re, im = alpha_complex(omega, transitions)
    assert re == pytest.approx(alpha_real(omega, transitions), rel=1e-12)
    assert im == 0.0


def test_attach_gamma_drops_core_and_counts_damped_channels():
    transitions = [_channel(589.16, 0.64, "3p3/2"), _channel(330.3, 0.01, "4p3/2"),
                   dict(_channel(50.0, 1.0, "core"), source="core")]
    channels, n_with, n_core = attach_gamma(transitions, {"3p3/2": NA_D2_A_S})
    assert (len(channels), n_with, n_core) == (2, 1, 1)
    assert channels[0]["Gamma_au"] == pytest.approx(NA_D2_A_S * AU_TIME_S)
    assert channels[1]["Gamma_au"] == 0.0
    assert "Gamma_au" not in transitions[0], "the inputs are not modified"


def test_point_heating_and_lifetime_are_consistent():
    """Ė = 2 E_r R_sc, heating in µK/s = Ė/k_B, τ_heat = |U|/Ė."""
    channels, _, _ = attach_gamma([_channel(NA_D2_NM, 0.64, "3p3/2")], {"3p3/2": NA_D2_A_S})
    pt = evaluate_point(channels, 1064.0, 50.0, 1.0, NA_MASS_AMU)
    assert pt.heating_J_s == pytest.approx(2 * pt.E_r_J * pt.R_sc_Hz)
    assert pt.heating_uK_s == pytest.approx(pt.heating_J_s / KB_SI * 1e6)
    assert pt.tau_heat_s == pytest.approx(abs(pt.U_J) / pt.heating_J_s)
    assert pt.U_J < 0, "red-detuned light attracts"
    assert math.isfinite(pt.nu_r_Hz) and pt.nu_r_Hz > pt.nu_z_Hz


def test_compute_bundles_point_spectrum_and_strongest_resonances():
    transitions = [_channel(589.76, 0.32, "3p1/2"), _channel(589.16, 0.64, "3p3/2")]
    rates = {"3p1/2": 6.1e7, "3p3/2": NA_D2_A_S}
    result = compute(transitions, rates, NA_MASS_AMU, wl_nm=1064.0, I_kWcm2=50.0,
                     waist_um=1.0, wl_min=500.0, wl_max=1200.0, n_grid=200)
    assert result.point == evaluate_point(
        attach_gamma(transitions, rates)[0], 1064.0, 50.0, 1.0, NA_MASS_AMU)
    assert [r.label for r in result.resonances] == ["3p3/2", "3p1/2"]
    spectrum = result.spectra
    assert len(spectrum.wl_nm) > 200, "the grid is refined around resonances"
    assert spectrum.wl_nm == sorted(spectrum.wl_nm)
    assert min(spectrum.wl_nm) >= 500.0 and max(spectrum.wl_nm) <= 1200.0
