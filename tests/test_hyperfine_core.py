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
    assert upper.gF == pytest.approx(st.gJ / 4 - 0.75 * ratio)
    assert lower.gF == pytest.approx(-st.gJ / 4 - 1.25 * ratio)


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
    # Landé with the free-electron g_S = 2.00231930436, not 2 (0.1% in every Zeeman slope)
    assert lande_gJ(0, 0.5, 0.5) == pytest.approx(2.00231930436, abs=1e-10)
    assert lande_gJ(1, 0.5, 0.5) == pytest.approx(4 / 3 - 2.00231930436 / 3, abs=1e-10)
    assert lande_gJ(1, 0.5, 1.5) == pytest.approx(2 / 3 + 2.00231930436 / 3, abs=1e-10)
    # Na 3p3/2 measured 1.3341(3) (Dodd & Kinnear 1960); the reduced mass changes g_L by m_e/M
    assert lande_gJ(1, 0.5, 1.5, mass_u=23) == pytest.approx(1.3341, abs=3e-4)
    assert lande_gJ(1, 0.5, 1.5, mass_u=23) < lande_gJ(1, 0.5, 1.5)
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


# ---------------------------------------------------------------------------
# Level tracking, field-dependent lines (2026-11 redesign)
# ---------------------------------------------------------------------------
RB87 = Nucleus(A=87, I=1.5, mu_I=2.751818, Q_barn=0.1335, gI=0.0009951414 * 1836.15267343)   # atomic g_I'
RB87_CONSTANTS = {"5s1/2": HfConstants(A_MHz=3417.341305452145, gJ_exp=2.00233113),
                  "5p3/2": HfConstants(A_MHz=84.7185, B_MHz=12.4965)}


def test_breit_rabi_curves_keep_their_m_F_through_crossings():
    """Each column is one level: its eigenvector stays inside one m_F block at every field, and
    levels of different m_F do cross (sorting all eigenvalues together used to swap them)."""
    from linak.physics.hyperfine_core import ZeemanSystem
    zs = ZeemanSystem(1.5, 1.5, 18.534, 2.724, 1.3341, NA23.gI)
    for B in (0.5, 44.0, 120.0):
        _E, V = zs.solve(B, vectors=True)
        for n, (_F, mF) in enumerate(zs.labels):
            inside = [c for c, (mj, mi) in enumerate(zs.basis) if abs(mj + mi - mF) < 1e-9]
            assert np.sum(V[n, inside] ** 2) == pytest.approx(1.0, abs=1e-12)
    E = zs.energies(np.linspace(1, 100, 400))
    d = E[:, :, None] - E[:, None, :]
    crossed = np.any(np.sign(d[0]) * np.sign(d[-1]) < 0)
    assert crossed, "Na 3p3/2 levels of different m_F cross below 100 G"


def test_high_field_labels_follow_steck():
    from linak.physics.hyperfine_core import ZeemanSystem
    zs = ZeemanSystem(1.5, 0.5, 3417.341305452145, 0.0, 2.00233113, RB87.gI)
    lab = dict(zip(zs.labels, zs.high_field))
    assert lab[(2.0, 0.0)] == (0.5, -0.5)          # |2,0> -> |m_J=+1/2, m_I=-1/2>
    assert lab[(1.0, 1.0)] == (-0.5, 1.5)          # |1,+1> -> |m_J=-1/2, m_I=+3/2>
    assert lab[(2.0, -2.0)] == (-0.5, -1.5)


def test_clock_quadratic_zeeman_coefficients_match_steck():
    """87Rb 575.15 Hz/G^2 and 133Cs 427.45 Hz/G^2 (Steck), from the exact Breit-Rabi levels."""
    rb = compute(RB87, RB87_CONSTANTS, ["5s1/2"], {}, B_max=5, B_points=51, lines=True)
    z = next(m for m in rb.microwave if m.mF_lower == 0 and m.mF_upper == 0)
    assert z.linear_Hz_per_G == pytest.approx(0.0, abs=1e-6)
    assert z.quadratic_Hz_per_G2 == pytest.approx(575.15, abs=0.05)
    cs = Nucleus(A=133, I=3.5, mu_I=2.582025, Q_barn=-0.00343, gI=0.00039885395 * 1836.15267343)
    c = compute(cs, {"6s1/2": HfConstants(A_MHz=2298.1579425, gJ_exp=2.002542)}, ["6s1/2"], {},
                B_max=5, B_points=51, lines=True)
    z = next(m for m in c.microwave if m.mF_lower == 0 and m.mF_upper == 0)
    assert z.quadratic_Hz_per_G2 == pytest.approx(427.45, abs=0.05)


def test_rb87_two_photon_magic_field():
    """|1,-1> <-> |2,+1> is first-order field insensitive at 3.228917 G (Harber et al. 2002); this
    needs the atomic (shielded) g_I: the bare nuclear moment gives 3.2418 G."""
    rb = compute(RB87, RB87_CONSTANTS, ["5s1/2"], {}, B_max=10, B_points=201, lines=True)
    line = next(m for m in rb.microwave if (m.mF_lower, m.mF_upper) == (-1.0, 1.0))
    assert line.kind == "two-photon"
    assert [p.B_G for p in line.stationary] == pytest.approx([3.2289], abs=5e-4)


def test_zero_field_line_strengths_and_zeeman_components():
    """S_FF' of the 87Rb D2 line (Steck): 7/10, 1/4, 1/20 from F = 2; 5/12, 5/12, 1/6 from F = 1.
    The Zeeman-resolved components reproduce them at B -> 0 and add to 1 per ground sublevel."""
    from linak.physics.hyperfine_core import hyperfine_line_strengths
    S = hyperfine_line_strengths(1.5, 0.5, 1.5)
    assert S[(2.0, 3.0)] == pytest.approx(0.7) and S[(2.0, 2.0)] == pytest.approx(0.25)
    assert S[(2.0, 1.0)] == pytest.approx(0.05) and S[(1.0, 0.0)] == pytest.approx(1 / 6)
    assert S[(1.0, 1.0)] == pytest.approx(5 / 12) and S[(1.0, 2.0)] == pytest.approx(5 / 12)
    rb = compute(RB87, RB87_CONSTANTS, ["5s1/2", "5p3/2"], {}, B_max=200, B_points=21, lines=True,
                 optical_points=5)
    (ol,) = rb.optical
    g, e = rb.states
    acc, tot = {}, np.zeros((len(ol.B_G), len(g.BR_labels)))
    for c in ol.components:
        key = (g.BR_labels[c.lower][0], e.BR_labels[c.upper][0])
        acc[key] = acc.get(key, 0.0) + c.strength[0] / (2 * key[0] + 1)
        tot[:, c.lower] += c.strength
    for key, v in S.items():
        assert acc[key] == pytest.approx(v, abs=1e-9)
    assert tot == pytest.approx(np.ones_like(tot), abs=1e-9)
