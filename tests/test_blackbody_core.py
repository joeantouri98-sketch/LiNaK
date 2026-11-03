# -*- coding: utf-8 -*-
"""``physics.blackbody_core`` against closed forms and its own invariants.

The references are independent of the code's atomic-unit route: SI CODATA 2018
values, the Stefan–Boltzmann ⟨E²⟩ = 4σT⁴/(cε₀), the free-electron (Farley–Wing)
shift e²(k_B T)²/(12 m_e ε₀ ħ c³), and the two-level dynamic correction
η = (40π²/21)(k_B T/ħω₀)² (the Planck average of ω²/ω₀²).
"""
from __future__ import annotations

import math

import pytest

from linak.physics.blackbody_core import (
    bbr_depopulation,
    bbr_static_shift_hz,
    bbr_total_shift_hz,
    build_lifetime_graph,
    complete_transitions_for_dyn,
    farley_wing_shift_hz,
    kramers_continuum_partners,
    mix_ionization_rate,
    nbar_photon,
    t_eff_gate,
)
from linak.physics.polarizability_core import alpha_real

H = 6.62607015e-34
HBAR = H / (2 * math.pi)
KB = 1.380649e-23
C = 2.99792458e8
EPS0 = 8.8541878128e-12
E_CHARGE = 1.602176634e-19
M_E = 9.1093837015e-31
ALPHA_AU_SI = 1.64877727436e-41      # C² m² J⁻¹
HARTREE_J = 4.3597447222071e-18


def _line(delta_E_au, f=1.0):
    return dict(delta_E_au=delta_E_au, fosc=f, sign=1, other_label="x", source="test")


def test_static_shift_is_minus_half_alpha_e_squared_with_itano_field():
    """Δν = −½ α(0) (831.9 V/m)² (T/300 K)⁴ / h."""
    expected = -0.5 * ALPHA_AU_SI * 831.9**2 / H
    assert bbr_static_shift_hz(1.0, 300.0) == pytest.approx(expected, rel=1e-8)
    assert bbr_static_shift_hz(1.0, 600.0) == pytest.approx(16 * expected, rel=1e-8)


def test_planck_integral_of_a_static_alpha_gives_stefan_boltzmann():
    """A resonance far above k_B T: the integral sees ⟨E²⟩ = 4σT⁴/(cε₀), Itano's to 0.01 %."""
    T = 300.0
    sigma = 2 * math.pi**5 * KB**4 / (15 * H**3 * C**2)
    planck_over_itano = (4 * sigma * T**4 / (C * EPS0)) / 831.9**2
    far = [_line(50.0)]
    ratio = bbr_total_shift_hz(far, T) / bbr_static_shift_hz(alpha_real(0.0, far), T)
    assert ratio == pytest.approx(planck_over_itano, rel=1e-6)
    assert abs(planck_over_itano - 1) < 2e-4, "the Itano field is the 300 K Planck value"


def test_dynamic_correction_of_a_d_line_matches_the_two_level_closed_form():
    T = 300.0
    kT_au = KB * T / HARTREE_J
    d_line = 0.0772                        # Na 3s-3p, atomic units
    eta = 40 * math.pi**2 / 21 * (kT_au / d_line) ** 2

    def ratio(dE):
        net = [_line(dE)]
        return bbr_total_shift_hz(net, T) / bbr_static_shift_hz(alpha_real(0.0, net), T)

    assert ratio(d_line) / ratio(50.0) - 1 == pytest.approx(eta, rel=2e-2)


def test_farley_wing_is_the_free_electron_shift():
    """+2416.666 Hz at 300 K, ∝ T²."""
    T = 300.0
    expected = E_CHARGE**2 * (KB * T) ** 2 / (12 * M_E * EPS0 * HBAR * C**3) / H
    assert farley_wing_shift_hz(T) == pytest.approx(expected, rel=1e-8)
    assert farley_wing_shift_hz(T) == pytest.approx(2416.666, abs=1e-3)
    assert farley_wing_shift_hz(2 * T) == pytest.approx(4 * expected, rel=1e-8)


def test_photon_number_is_bose_einstein():
    T = 300.0
    assert nbar_photon(KB * T / HBAR, T) == pytest.approx(1 / (math.e - 1))
    assert nbar_photon(1.0, 0.0) == 0.0


def test_depopulation_obeys_detailed_balance():
    """Stimulated emission A·n̄ from the upper level, absorption A·(g_u/g_l)·n̄ from the lower."""
    A, wl_nm, T = 1.0e7, 10_000.0, 300.0
    graph = build_lifetime_graph({"n_start": 4, "lifetimes": [
        {"state": "4p3/2", "n": 4, "l": "p", "A_total_s": A,
         "channels": [{"lower_label": "4s", "A_s": A, "wavelength_nm": wl_nm}]},
    ]})
    nbar = nbar_photon(2 * math.pi * C / (wl_nm * 1e-9), T)
    upper = bbr_depopulation("4p3/2", graph, T)
    lower = bbr_depopulation("4s", graph, T)
    assert upper.Gamma_BBR_s == pytest.approx(A * nbar)
    assert lower.Gamma_BBR_s == pytest.approx(A * (4 / 2) * nbar)
    assert upper.tau_eff_s == pytest.approx(1 / (A + A * nbar))
    assert lower.tau_rad_s is None, "the lower level has no radiative decay here"
    assert [c.kind for c in upper.channels + lower.channels] == ["stim", "abs"]


@pytest.mark.parametrize("valence_f", [0.2, 3.0])
def test_trk_completion_closes_the_oscillator_strength_sum(valence_f):
    """Below 1 the continuum fills the gap; above 1 everything is scaled down to 1."""
    data = {"state_energy_au": {"20s": -0.5 / (20 - 1.35) ** 2},
            "qd": {"s": 1.35, "p": 0.855, "d": 0.015}, "ryd": {"n_max": 30}}
    valence = [_line(1e-3, valence_f / 2), _line(2e-3, valence_f / 2)]
    channels, completion = complete_transitions_for_dyn("20s", data, valence, n_tail_max=60)
    assert sum(t["sign"] * t["fosc"] for t in channels) == pytest.approx(1.0, abs=1e-9)
    assert completion.trk_sum == pytest.approx(1.0, abs=1e-6)
    assert completion.n_tail > 0
    assert valence[0]["fosc"] == valence_f / 2, "the inputs are not modified"


def test_kramers_continuum_carries_exactly_the_residual_strength():
    bins = kramers_continuum_partners(0.01, 0.3, n_bins=40)
    assert len(bins) == 40
    assert sum(b["fosc"] for b in bins) == pytest.approx(0.3)
    assert all(b["delta_E_au"] > 0.01 for b in bins), "every bin is above threshold"


def test_mixing_rate_is_continuous_at_equal_lifetimes():
    """The equal-lifetime branch is the limit of the general rate-equation formula."""
    t1, t2 = 0.3e-6, 2.1e-6
    equal = mix_ionization_rate(1e4, 50.0, 1e-6, 1e-6, t1, t2)
    nearly = mix_ionization_rate(1e4, 50.0, 1e-6, 1e-6 * (1 + 1e-6), t1, t2)
    assert equal > 0
    assert nearly == pytest.approx(equal, rel=1e-5)
    # A long-lived state is collected for the whole gate (τ = 10 s ≫ t₂).
    assert t_eff_gate(10.0, t1, t2) == pytest.approx(t2 - t1, rel=1e-6)
