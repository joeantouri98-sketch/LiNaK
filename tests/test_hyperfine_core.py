# -*- coding: utf-8 -*-
"""``physics.hyperfine_core`` against reference values and its own invariants.

Reference constants and intervals: D. A. Steck, *Sodium D Line Data*
(steck.us/alkalidata), the source of the bundled ``Na_hf_constants.json``. These pin
the conventions, not the known issues of PHYSICS_BUGS.md (P4's diagonal-only quadrupole is avoided by testing the
Breit-Rabi levels on a J = 1/2 state, where there is no quadrupole).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from linak.core.constants import C_SI, MU_B_MHz_G, MU_N_MHz_G
from linak.physics.hyperfine_core import (
    HfConstants,
    Nucleus,
    breit_rabi_energies,
    compute,
    lande_gJ,
)

NA23 = Nucleus(A=23, I=1.5, mu_I=2.2175, Q_barn=0.104, gI=2.2175 / 1.5)
NA_CONSTANTS = {
    "3s1/2": HfConstants(A_MHz=885.81306440),
    "3p1/2": HfConstants(A_MHz=94.44),
    "3p3/2": HfConstants(A_MHz=18.534, B_MHz=2.724),
}
NA_DELTA_S = 1.3505


def _state(result, label):
    return next(st for st in result.states if st.label == label)


def test_na_ground_state_splitting_is_steck_1771_626_mhz():
    """ΔE(3s, F = 2 ↔ 1) = 2A = 1.771 626 128 8 GHz."""
    result = compute(NA23, NA_CONSTANTS, ["3s1/2"], {})
    assert result.clock.state == "3s1/2"
    assert (result.clock.F_upper, result.clock.F_lower) == (2.0, 1.0)
    assert result.clock.frequency_MHz == pytest.approx(1771.6261288, abs=1e-6)
    assert result.clock.wavelength_cm == pytest.approx(C_SI * 1e-4 / 1771.6261288)   # exact c


def test_na_3p3_2_intervals_match_steck():
    """F′ = 3↔2, 2↔1, 1↔0: 58.326, 34.344, 15.810 MHz from A = 18.534, B = 2.724."""
    st = _state(compute(NA23, NA_CONSTANTS, ["3p3/2"], {}), "3p3/2")
    assert [lv.F for lv in st.levels] == [0.0, 1.0, 2.0, 3.0]
    E = [lv.E_MHz for lv in st.levels]
    assert [E[3] - E[2], E[2] - E[1], E[1] - E[0]] == pytest.approx(
        [58.326, 34.344, 15.810], abs=1e-3)
    weights = [2 * lv.F + 1 for lv in st.levels]
    assert sum(w * e for w, e in zip(weights, E)) == pytest.approx(0.0, abs=1e-9), (
        "energies are relative to the centroid")


def test_levels_carry_g_F_and_every_m_F():
    st = _state(compute(NA23, NA_CONSTANTS, ["3s1/2"], {}), "3s1/2")
    lower, upper = st.levels
    assert upper.mF == [2.0, 1.0, 0.0, -1.0, -2.0]
    assert lower.mF == [1.0, 0.0, -1.0]
    # I = 3/2, J = 1/2: g_F(2) = g_J/4 − (3/4) g_I μ_N/μ_B, g_F(1) = −g_J/4 − (5/4) g_I μ_N/μ_B.
    ratio = NA23.gI / 1836.15267
    assert upper.gF == pytest.approx(2.0 / 4 - 0.75 * ratio)
    assert lower.gF == pytest.approx(-2.0 / 4 - 1.25 * ratio)


@pytest.mark.parametrize("B_G", [0.0, 50.0, 315.0, 2000.0])
def test_breit_rabi_matches_the_analytic_j_one_half_formula(B_G):
    """Breit–Rabi: E = −ΔE/(2(2I+1)) + g_I′μ_B m B ± (ΔE/2)√(1 + 4mx/(2I+1) + x²)."""
    I, A, gJ = NA23.I, NA_CONSTANTS["3s1/2"].A_MHz, 2.0
    gI_mu = -NA23.gI * MU_N_MHz_G           # g_I′ μ_B, nuclear term in MHz/G
    dE = A * (I + 0.5)
    x = (gJ * MU_B_MHz_G - gI_mu) * B_G / dE
    expected = []
    for m in np.arange(-I + 0.5, I):         # the two branches of each inner m_F
        root = (dE / 2) * math.sqrt(1 + 4 * m * x / (2 * I + 1) + x * x)
        base = -dE / (2 * (2 * I + 1)) + gI_mu * m * B_G
        expected += [base + root, base - root]
    for s in (+1, -1):                       # the two stretched states
        expected.append(dE * I / (2 * I + 1) + s * (gJ * MU_B_MHz_G / 2 + gI_mu * I) * B_G)

    energies, _ = breit_rabi_energies(I, 0.5, A, 0.0, gJ, NA23.gI, np.array([B_G]))
    assert sorted(energies[0]) == pytest.approx(sorted(expected), abs=1e-6)


def test_lande_g_J_and_the_experimental_override():
    assert lande_gJ(0, 0.5, 0.5) == pytest.approx(2.0)
    assert lande_gJ(1, 0.5, 0.5) == pytest.approx(2 / 3)
    assert lande_gJ(1, 0.5, 1.5) == pytest.approx(4 / 3)
    assert lande_gJ(1, 0.5, 1.5, gJ_exp=1.3342) == 1.3342
    assert lande_gJ(1, 1.0, 0.0, gJ_exp=1.5) == 0.0, "J = 0 has no Zeeman effect"


def test_unmeasured_state_is_scaled_from_the_same_l_j_as_n_star_cubed():
    result = compute(NA23, NA_CONSTANTS, ["3s1/2", "4s1/2", "4d5/2"], {"s": NA_DELTA_S})
    four_s = _state(result, "4s1/2")
    assert four_s.extrapolated_from == "3s1/2"
    expected = 885.81306440 * ((3 - NA_DELTA_S) / (4 - NA_DELTA_S)) ** 3
    assert four_s.A_MHz == pytest.approx(expected)
    no_reference = _state(result, "4d5/2")
    assert (no_reference.extrapolated_from, no_reference.A_MHz) == (None, 0.0)
    assert {lv.E_MHz for lv in no_reference.levels} == {0.0}, "degenerate F levels"


def test_compute_skips_bad_labels_and_keeps_the_order():
    result = compute(NA23, NA_CONSTANTS, ["3p3/2", "bogus", "3s1/2"], {}, B_points=7)
    assert [st.label for st in result.states] == ["3p3/2", "3s1/2"]
    assert result.skipped == ["bogus"]
    assert result.B_G.tolist() == pytest.approx(np.linspace(0, 500, 7).tolist())
    assert result.states[0].BR_energies.shape == (7, 16)


def test_constants_rows_default_b_and_c_to_zero():
    row = HfConstants.from_json({"A_MHz": 10, "B_MHz": None, "gJ_exp": 2.0023})
    assert row == HfConstants(A_MHz=10.0, B_MHz=0.0, C_kHz=0.0, gJ_exp=2.0023, S_exp=None)
