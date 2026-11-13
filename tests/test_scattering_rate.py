# -*- coding: utf-8 -*-
"""Two-level laser cooling (physics/laser_cooling.py) against Steck's D-line data.

Steck, "Sodium / Rubidium 87 / Cesium D Line Data": Γ/2π = 9.7946, 6.0666, 5.2227 MHz;
λ(D2) = 589.158326, 780.241210, 852.347275 nm. From these: cycling I_sat 6.2600, 1.66933,
1.1023 mW/cm²; Doppler limit 235.03, 145.57 µK; recoil temperature 2.3998 µK, 361.96 nK,
198.34 nK (Cs T_D is left out: Steck's 125.61 µK uses an older Γ); recoil velocity 2.9461 cm/s, 5.8845 mm/s, 3.5225 mm/s.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from linak.core.constants import ISOTOPE_MASS_AMU
from linak.physics import laser_cooling as lc

from helpers import write_result_summary

STECK = {   # element: (A, Γ/2π MHz, λ nm, I_sat cycling, T_D µK, T_rec µK, v_rec m/s)
    'Na': (23, 9.7946, 589.158326, 6.2600, 235.03, 2.3998, 2.9461e-2),
    'Rb': (87, 6.0666, 780.241210, 1.66933, 145.57, 0.36196, 5.8845e-3),
    'Cs': (133, 5.2227, 852.347275, 1.1023, None, 0.19834, 3.5225e-3),   # Steck's T_D uses an older Γ
}


@pytest.mark.parametrize("el", list(STECK))
def test_limits_and_saturation_intensity_match_steck(el):
    A, g, wl, isat, td, trec, vrec = STECK[el]
    G, m = 2 * math.pi * g * 1e6, ISOTOPE_MASS_AMU[(el, A)]
    assert lc.isat_cycling_mW_cm2(G, wl) == pytest.approx(isat, rel=2e-4)
    if td is not None:
        assert lc.doppler_limit_K(G) * 1e6 == pytest.approx(td, rel=2e-4)
    assert lc.recoil_temperature_K(wl, m) * 1e6 == pytest.approx(trec, rel=2e-4)
    assert lc.recoil_velocity(wl, m) == pytest.approx(vrec, rel=2e-4)


@pytest.mark.parametrize("x", [-2.0, -0.5, 0.0, 1.0])
@pytest.mark.parametrize("s", [0.01, 0.5, 1.0, 5.0])
def test_scattering_rate_closed_form(s, x):
    G = 3.7e7
    assert float(lc.scattering_rate(G, s, x * G)) == pytest.approx(0.5 * G * s / (1 + s + 4 * x * x), rel=1e-12)


@pytest.mark.parametrize("s", [1e2, 1e3, 1e6, 1e9])
def test_scattering_rate_saturates_at_gamma_over_2(s):
    G = 6.1e7
    rate = float(lc.scattering_rate(G, s, 0.0))
    assert rate <= 0.5 * G * (1 + 1e-12) and abs(rate - 0.5 * G) <= 0.5 * G * (1 / s + 1e-9)
    assert float(lc.scattering_rate(G, 1.0, 0.0)) == pytest.approx(0.25 * G, rel=1e-12)


def test_doppler_temperature_minimum_is_the_doppler_limit():
    """k_BT = (ħΓ/4)(1 + 2s + (2Δ/Γ)²)/(2|Δ|/Γ): minimum ħΓ/2k_B at Δ = −Γ/2 for s → 0."""
    G = 2 * math.pi * 6.0666e6
    d = np.linspace(-3, -0.05, 5901) * G
    T = lc.doppler_temperature(G, 0.0, d)
    i = int(np.argmin(T))
    assert d[i] / G == pytest.approx(-0.5, abs=1e-3)
    assert T[i] == pytest.approx(lc.doppler_limit_K(G), rel=1e-6)
    assert np.isinf(lc.doppler_temperature(G, 1.0, 0.1 * G))       # blue detuning heats
    assert lc.doppler_temperature(G, 1.0, -0.5 * G) == pytest.approx(2 * lc.doppler_limit_K(G))   # (1 + 2 + 1)/1


def test_molasses_force_is_a_damping_force():
    """Red-detuned molasses: F is odd in v, opposes the motion, and for small v is −βv with
    β = −8ħk² s (Δ/Γ)/(1 + s + (2Δ/Γ)²)² (Metcalf & van der Straten Eq. 7.2, two beams)."""
    G, wl, s, d = 2 * math.pi * 6.0666e6, 780.24, 0.1, -0.5 * 2 * math.pi * 6.0666e6
    k = lc.wavenumber(wl)
    v = np.array([-0.3, -0.01, 0.01, 0.3])
    F = lc.molasses_force(v, G, s, d, wl)
    assert F[0] == pytest.approx(-F[3]) and F[3] < 0 < F[0]
    beta = -8 * lc.HBAR_SI * k * k * s * (d / G) / (1 + s + (2 * d / G) ** 2) ** 2
    assert F[2] / 0.01 == pytest.approx(-beta, rel=1e-3)


def test_slower_length_and_capture_velocity():
    G, wl, m = 2 * math.pi * 9.7946e6, 589.158, ISOTOPE_MASS_AMU[('Na', 23)]
    v = lc.most_probable_speed(600.0, m)
    assert v == pytest.approx(math.sqrt(2 * lc.KB_SI * 600 / (m * lc.AMU_KG)))
    assert lc.slower_length(v, G, wl, m, 0.5) == pytest.approx(v * v / lc.max_acceleration(G, wl, m))
    assert lc.capture_velocity(G, wl) == pytest.approx(G / lc.wavenumber(wl))


def test_record_na_d2_anchor(results_dir):
    G, wl = 2 * math.pi * 9.7946e6, 589.158326
    write_result_summary(results_dir, "scattering_rate_anchor", {
        "element": "Na", "line": "D2", "gamma_rad_s": G, "Isat_cycling_mWcm2": lc.isat_cycling_mW_cm2(G, wl),
        "T_doppler_uK": lc.doppler_limit_K(G) * 1e6, "R_max_s1": 0.5 * G, "v_capture_m_s": lc.capture_velocity(G, wl)})
