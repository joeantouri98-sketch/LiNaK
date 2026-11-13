# -*- coding: utf-8 -*-
"""The dynamic BBR shift: the channel integral F(y), and the limits it must reproduce.

Farley & Wing, Phys. Rev. A 23, 2397 (1981); Itano, Lewis & Wineland, Phys. Rev. A 25,
1233 (1982). F(y) = PV ∫₀^∞ x³/(eˣ-1) dx/(y²-x²).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from qali.physics.blackbody_core import (
    _PV_Y_MAX,
    bbr_static_shift_hz,
    bbr_total_shift_hz,
    bbr_total_shift_pv_hz,
    farley_wing_shift_hz,
    planck_pv_integral,
)

THETA_AU_300K = 300 * 3.1668115634556e-6  # k_B T in hartree


def _channel(delta_E_au, fosc=1.0, sign=1.0):
    return dict(delta_E_au=delta_E_au, fosc=fosc, sign=sign)


def _symmetric_sum(y, cells_per_y=200_000):
    """F(y) by a midpoint sum with cell edges on the pole, where the PV cancels exactly."""
    h = y / cells_per_y if y < 1.0 else y / int(y * cells_per_y)
    x = (np.arange(int(120.0 / h)) + 0.5) * h
    return float(np.sum(x * x * (x / np.expm1(x)) / ((y - x) * (y + x))) * h)


@pytest.mark.parametrize("y", [0.4, 1.0, 2.82, 3.9, 7.3, 25.0])
def test_table_matches_an_independent_integration(y):
    assert float(planck_pv_integral(y)) == pytest.approx(_symmetric_sum(y), rel=1e-6)


def test_near_partner_limit_is_exact():
    assert float(planck_pv_integral(0.0)) == pytest.approx(-math.pi ** 2 / 6, abs=1e-5)


def test_far_partner_limit_is_exact_and_joins_the_table():
    y = 1e5
    assert float(planck_pv_integral(y)) == pytest.approx(math.pi ** 4 / (15 * y ** 2), rel=1e-6)
    below, above = planck_pv_integral(_PV_Y_MAX * 0.999), planck_pv_integral(_PV_Y_MAX * 1.001)
    assert above == pytest.approx(below * 0.999 ** 2 / 1.001 ** 2, rel=1e-5)


def test_trk_closed_network_of_near_partners_gives_farley_wing():
    # Signed f summing to 1 with every partner far below k_B T: the free-electron limit.
    network = [_channel(1e-9, 0.7), _channel(2e-9, 0.2), _channel(-3e-9, 0.1)]
    assert bbr_total_shift_pv_hz(network, 300.0) == pytest.approx(farley_wing_shift_hz(300.0), rel=1e-4)


@pytest.mark.parametrize("T_K", [77.0, 300.0, 1000.0])
def test_farley_wing_scales_as_t_squared(T_K):
    assert bbr_total_shift_pv_hz([_channel(1e-10)], T_K) == pytest.approx(
        farley_wing_shift_hz(T_K), rel=1e-4)


def test_far_partner_gives_the_static_itano_shift():
    delta = 1.0  # far enough above k_B T for the dynamic correction to vanish
    alpha0 = 1.0 / delta ** 2
    assert bbr_total_shift_pv_hz([_channel(delta)], 300.0) == pytest.approx(
        bbr_static_shift_hz(alpha0, 300.0), rel=2e-4)  # Itano's 831.9 V/m is rounded


def test_d_line_partner_has_the_expected_dynamic_correction():
    # The next term of F(y) at large y: 1 + (40 π² / 21) / y² relative to static.
    delta = 0.08
    y = delta / THETA_AU_300K
    assert bbr_total_shift_pv_hz([_channel(delta)], 300.0) == pytest.approx(
        bbr_static_shift_hz(1.0 / delta ** 2, 300.0) * (1 + 40 * math.pi ** 2 / 21 / y ** 2), rel=2e-4)


def test_agrees_with_the_grid_integral_where_the_grid_can_work():
    network = [_channel(0.08, 0.98), _channel(0.15, 0.02)]
    assert bbr_total_shift_pv_hz(network, 300.0) == pytest.approx(
        bbr_total_shift_hz(network, 300.0), rel=1e-5)


def test_partner_inside_the_thermal_band_is_finite_and_resolved():
    # A resonance at k_B T would be a pole for a grid; here it needs nothing special.
    shifts = [bbr_total_shift_pv_hz([_channel(THETA_AU_300K * y)], 300.0) for y in (0.9, 1.0, 1.1)]
    assert np.all(np.isfinite(shifts))
    assert abs(shifts[1] - shifts[0]) < 0.2 * abs(shifts[0])
    assert abs(shifts[2] - shifts[1]) < 0.2 * abs(shifts[1])


def test_empty_network_has_no_shift():
    assert bbr_total_shift_pv_hz([], 300.0) == 0.0
