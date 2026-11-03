# -*- coding: utf-8 -*-
"""Unit-conversion helper anchors from physics/polarizability_core.py.

These helpers moved into polarizability_core in a recent refactor; the tests
pin the CURRENT conventions (kW/cm^2 -> W/m^2, nm -> atomic-unit angular
frequency, Joule trap depth -> uK / kHz) with expected values derived from the
CODATA constants and prefactors declared in the same modules -- never from
hardcoded magic numbers.
"""
from __future__ import annotations

import math

import pytest

from linak.core.constants import A0_SI, C_SI, EV_TO_HARTREE, H_SI, HC_EV_NM, KB_SI
from linak.physics.polarizability_core import (
    NM_TO_AU_OMEGA,
    NM_TO_HARTREE,
    U_PREFACTOR,
    U_joules,
    U_to_kHz_h,
    U_to_uK,
    alpha_to_uK_factor,
    eV_to_au,
    kWcm2_to_Wm2,
    nm_from_omega_au,
    omega_au_from_nm,
)

from helpers import write_result_summary

# CODATA 2018 (exact, via eV and E_h/e): 1 eV = E_h / 27.211386245988.
_CODATA_EV_TO_AU = 0.036749322


# ---------------------------------------------------------------------------
# Intensity conversion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("i_kWcm2", "expected_Wm2"),
    [(0.001, 1e4), (0.01, 1e5), (1.0, 1e7), (2.5, 2.5e7), (100.0, 1e9)],
)
def test_kWcm2_to_Wm2(i_kWcm2, expected_Wm2):
    """1 kW/cm^2 = 1e3 W over 1e-4 m^2 = 1e7 W/m^2 (spot checks)."""
    # Independent derivation of the factor: kW -> W, cm^2 -> m^2.
    factor = 1e3 / 1e-4
    assert kWcm2_to_Wm2(i_kWcm2) == pytest.approx(i_kWcm2 * factor)
    assert kWcm2_to_Wm2(i_kWcm2) == pytest.approx(expected_Wm2, rel=1e-12)


def test_alpha_to_uK_factor_uses_same_intensity_conversion():
    """No inverse helper exists, so pin alpha_to_uK_factor's internal use of
    the same 1e7 factor: it must be U_to_uK(U_joules(1 au, I))."""
    i_kWcm2 = 1.0
    assert alpha_to_uK_factor(i_kWcm2) == pytest.approx(
        U_to_uK(U_joules(1.0, kWcm2_to_Wm2(i_kWcm2))), rel=1e-12
    )


# ---------------------------------------------------------------------------
# nm <-> atomic-unit angular frequency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("wl_nm", [400.0, 589.158, 589.756, 1064.0])
def test_omega_au_from_nm_is_energy_in_hartree(wl_nm):
    """omega in a.u. equals the photon energy in Hartree (hbar = 1):
    E_eV = hc/lambda, then /E_h(eV)."""
    expected = (HC_EV_NM / wl_nm) * EV_TO_HARTREE
    assert omega_au_from_nm(wl_nm) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("wl_nm", [400.0, 589.158, 589.756, 1064.0])
def test_nm_omega_round_trip(wl_nm):
    """nm -> omega_au -> nm is exact to floating-point precision."""
    omega_au = omega_au_from_nm(wl_nm)
    assert nm_from_omega_au(omega_au) == pytest.approx(wl_nm, rel=1e-12)


def test_nm_from_omega_au_zero_or_negative_is_infinite():
    """A non-positive angular frequency has no wavelength -> +inf."""
    assert math.isinf(nm_from_omega_au(0.0))
    assert math.isinf(nm_from_omega_au(-1.0))


def test_NM_TO_HARTREE_alias_matches_omega_constant():
    """The legacy alias must stay equal to the omega conversion constant."""
    assert NM_TO_HARTREE == NM_TO_AU_OMEGA


# ---------------------------------------------------------------------------
# AC-Stark trap depth helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("i_kWcm2", [0.01, 0.1, 1.0, 10.0])
@pytest.mark.parametrize("alpha_au", [1.0, -25.0, 162.7])
def test_trap_depth_helpers_are_consistent(alpha_au, i_kWcm2):
    """U = -(2*pi*a0^3/c) * alpha * I  (Grimm et al. 2000), and the uK / kHz
    views must be pure unit changes of the same Joule value."""
    i_Wm2 = kWcm2_to_Wm2(i_kWcm2)
    u_J = U_joules(alpha_au, i_Wm2)

    # Joule value derived from the module's own prefactor U_PREFACTOR, which
    # itself is pinned below to 2*pi*a0^3/c from CODATA constants.
    assert u_J == pytest.approx(-U_PREFACTOR * alpha_au * i_Wm2, rel=1e-12)

    # Sign convention: positive alpha (red-detuned) -> attractive U < 0.
    assert math.copysign(1.0, u_J) == -math.copysign(1.0, alpha_au)

    u_uk = U_to_uK(u_J)
    u_khz = U_to_kHz_h(u_J)
    assert u_uk == pytest.approx(u_J / KB_SI * 1e6, rel=1e-12)
    assert u_khz == pytest.approx(u_J / H_SI / 1e3, rel=1e-12)

    # kHz per uK is the fixed conversion k_B/h * 1e-9 (~=
    # 20.837 kHz/uK, i.e. k_B/h ~ 20.837 GHz/K across 1e6 uK/K).
    assert u_khz / u_uk == pytest.approx(KB_SI / H_SI * 1e-9, rel=1e-9)


def test_U_prefactor_equals_two_pi_a0_cubed_over_c():
    """Pin U_PREFACTOR to its documented definition U = -(2*pi*a0^3/c)*alpha*I
    (Grimm et al., Adv. At. Mol. Opt. Phys. 42, 95 (2000))."""
    assert U_PREFACTOR == pytest.approx(2.0 * math.pi * A0_SI**3 / C_SI, rel=1e-12)


# ---------------------------------------------------------------------------
# eV -> atomic units
# ---------------------------------------------------------------------------


def test_eV_to_au_codata():
    """1 eV = 3.6749322e-2 E_h (CODATA 2018) within 1e-4."""
    assert abs(eV_to_au(1.0) - _CODATA_EV_TO_AU) < 1e-4
    # One Hartree of eV must convert back to exactly 1 a.u.
    assert eV_to_au(1.0 / EV_TO_HARTREE) == pytest.approx(1.0, rel=1e-12)


def test_record_trap_depth_sample(results_dir):
    """Record a Na-like sample (alpha = 162.7 au at 1 kW/cm^2) for the log."""
    alpha_au = 162.7  # Na ground-state static polarizability (au), ALPHA0_EXPT
    u_J = U_joules(alpha_au, kWcm2_to_Wm2(1.0))
    write_result_summary(
        results_dir,
        "unit_conversions_trap_depth",
        {
            "alpha_au": alpha_au,
            "I_kWcm2": 1.0,
            "U_joules": u_J,
            "U_uK": U_to_uK(u_J),
            "U_kHz_h": U_to_kHz_h(u_J),
            "omega_au_589nm": omega_au_from_nm(589.158),
        },
    )
