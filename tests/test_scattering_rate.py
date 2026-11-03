# -*- coding: utf-8 -*-
"""Anchors for the closed-form formulas in pipelines/scattering_rate.py.

Pins CURRENT behaviour:
  R_scatt(I, d) = (G/2) * s / (1 + s + (2d/G)^2),  s = I/I_sat
  T_D = hbar*G / (2*k_B)   (Doppler cooling limit)
  v_cap ~ G/k
using the module's own FALLBACK table (its Gamma convention: G = Einstein A
coefficient, treated as an angular decay rate in rad/s).
"""
from __future__ import annotations

import math

import pytest

from linak.core.constants import AMU_KG, HBAR_SI, KB_SI
from linak.pipelines.scattering_rate import (
    FALLBACK,
    R_scatt,
    T_doppler_uK,
    params,
    v_capture,
)

from helpers import write_result_summary

# Literature Doppler limits (T_D = hbar*Gamma/(2*k_B), two-level absorber).
# Na D ~240 uK and Rb D2 ~146 uK are the canonical MOT-textbook values
# (Metcalf & van der Straten, "Laser Cooling and Trapping"; Foot, "Atomic
# Physics", ch. 9).  K and Cs quoted values follow from their D2 lifetimes.
_LITERATURE_T_D_UK = [
    ("Na", "D1", 240.0),
    ("Na", "D2", 240.0),
    ("K", "D2", 145.0),
    ("Rb", "D2", 146.0),
    ("Cs", "D2", 125.0),
]

_ALL_EL_LINES = [(el, line) for el in FALLBACK for line in ("D1", "D2")]


@pytest.mark.parametrize(("element", "line", "lit_uK"), _LITERATURE_T_D_UK)
def test_doppler_temperature_literature_anchor(element, line, lit_uK):
    """T_D = hbar*Gamma/(2 k_B) must land on the textbook value within ~5%.

    T_D = hbar*Gamma/(2 k_B) ~= 240 uK for Na D2 (Gamma/2pi = 9.79 MHz
    -> 234.8 uK from exactly this formula; the 240 uK figure is the usual
    rounded literature anchor).  A few percent is therefore the right band.
    """
    gamma, _k, _isat, _m = params(FALLBACK, element, line)
    td_uk = T_doppler_uK(gamma)

    # Module-consistent value: hbar*Gamma/(2 k_B), expressed in uK.
    expected_uK = HBAR_SI * gamma / (2.0 * KB_SI) * 1e6
    assert td_uk == pytest.approx(expected_uK, rel=1e-12)

    # Absolute literature anchor.
    assert abs(td_uk - lit_uK) / lit_uK < 0.05


@pytest.mark.parametrize("s", [1e2, 1e3, 1e6, 1e9])
def test_scattering_rate_saturates_at_gamma_over_2(s):
    """At zero detuning R_scatt -> Gamma/2 as s = I/I_sat -> infinity."""
    gamma, _k, _isat, _m = params(FALLBACK, "Na", "D2")
    rate = float(R_scatt(gamma, _isat, s, 0.0))

    # Independent closed form at d = 0: (G/2) * s / (1 + s).
    expected = 0.5 * gamma * s / (1.0 + s)
    assert rate == pytest.approx(expected, rel=1e-12)

    # Saturation ceiling: never exceeds Gamma/2, deficit ~ (G/2)/s
    # (small float pad: at s = 1e9 the deficit grazes the float64 ulp).
    gamma_half = 0.5 * gamma
    assert rate <= gamma_half * (1.0 + 1e-12)
    assert abs(rate - gamma_half) <= gamma_half * (1.0 / s + 1e-9)


@pytest.mark.parametrize("x", [-2.0, -0.5, 0.0, 1.0])
@pytest.mark.parametrize("s", [0.01, 0.5, 1.0, 5.0])
def test_r_scatt_matches_docstring_closed_form(s, x):
    """R_scatt = (G/2)*s/(1+s+(2d/G)^2) recomputed independently."""
    gamma, _k, _isat, _m = params(FALLBACK, "Rb", "D2")
    expected = 0.5 * gamma * s / (1.0 + s + 4.0 * x * x)
    assert float(R_scatt(gamma, _isat, s, x)) == pytest.approx(
        expected, rel=1e-12
    )


def test_half_saturation_resonance_rate_is_gamma_over_4():
    """At I = I_sat, d = 0: (G/2)*1/(1+1+0) = G/4 (the classic G/4 anchor)."""
    gamma, _k, isat, _m = params(FALLBACK, "Na", "D2")
    assert float(R_scatt(gamma, isat, 1.0, 0.0)) == pytest.approx(
        0.25 * gamma, rel=1e-12
    )


@pytest.mark.parametrize(("element", "line"), _ALL_EL_LINES)
def test_params_linewidth_matches_lifetime(element, line):
    """The table's Gamma and tau must describe the same linewidth.

    The pipeline treats the Einstein A coefficient as Gamma (an angular decay
    rate), so Gamma = 1/tau, and the displayed Gamma/2pi column equals
    1/(2*pi*tau).
    """
    gamma, k, isat, mass = params(FALLBACK, element, line)
    row = FALLBACK[element][line]

    tau_s = row["tau"] * 1e-9
    assert gamma == pytest.approx(1.0 / tau_s, rel=1e-3)
    assert gamma / (2.0 * math.pi) == pytest.approx(
        1.0 / (2.0 * math.pi * tau_s), rel=1e-3
    )

    # Wavevector k = 2*pi/lambda with lambda in m.
    assert k == pytest.approx(2.0 * math.pi / (row["wl"] * 1e-9), rel=1e-12)

    # Isat positive and in mW/cm^2; mass in kg from the amu table.
    assert isat > 0.0
    assert mass == pytest.approx(FALLBACK[element]["mass"] * AMU_KG, rel=1e-12)


def test_capture_velocity_is_gamma_over_k():
    """v_cap = Gamma/k, the velocity whose Doppler shift is one linewidth."""
    gamma, k, _isat, _m = params(FALLBACK, "Na", "D2")
    assert v_capture(gamma, k) == pytest.approx(gamma / k, rel=1e-12)
    assert v_capture(gamma, k) > 0.0


def test_record_na_d2_anchor(results_dir):
    """Record the Na D2 anchor values in tests/results/ for the regression log."""
    gamma, k, isat, _m = params(FALLBACK, "Na", "D2")
    td_uk = T_doppler_uK(gamma)
    write_result_summary(
        results_dir,
        "scattering_rate_anchor",
        {
            "element": "Na",
            "line": "D2",
            "gamma_rad_s": gamma,
            "gamma_over_2pi_MHz": gamma / (2.0 * math.pi) / 1e6,
            "k_m1": k,
            "Isat_mWcm2": isat,
            "T_doppler_uK": td_uk,
            "R_max_s1": 0.5 * gamma,
            "R_scatt_at_Isat_resonance_s1": 0.25 * gamma,
            "v_capture_m_s": v_capture(gamma, k),
        },
    )
