# -*- coding: utf-8 -*-
"""``physics.lifetimes_core`` against reference values and its own invariants.

Na D2 reference values: D. A. Steck, *Sodium D Line Data* (steck.us/alkalidata):
λ = 589.158 nm, f = 0.6408, τ(3p3/2) = 16.2492 ns, v_r = 2.9461 cm/s,
T_D = 235.77 µK. These pin the A ↔ f ↔ g convention and the line-parameter
formulas, not the known issues (P22's peak cross section, P26's Coulomb f).
"""
from __future__ import annotations

import pytest

from qali.core.constants import EINSTEIN_PREFACTOR
from qali.physics.lifetimes_core import (
    Dropped,
    FValueSources,
    TransitionA,
    compute,
    extrapolate_nist_series,
    resolve_multiplet_f,
)

NA_MASS_AMU = 22.98976928
NA_QD = {"s": 1.348, "p": 0.855, "d": 0.015}
EMPTY_NIST = {"fosc": {}, "fosc_j": {}, "acc": {}, "acc_j": {}}
EMPTY_LITERATURE = {"fosc": {}, "fosc_j": {}, "acc_j": {}}


def _line(upper, lower, n_up, n_lo, wl_nm, **extra):
    return dict(upper_label=upper, lower_label=lower, upper_n=n_up, lower_n=n_lo,
                upper_l=upper[len(str(n_up))], lower_l=lower[len(str(n_lo))],
                wavelength_nm=wl_nm, delta_E_eV=1239.84198 / wl_nm, region="visible",
                **extra)


def _na(transitions, sources, **kw):
    return compute(transitions, sources, qd=NA_QD, z_core=1.0, IE=5.139, gs_label="3s",
                   mass_amu=NA_MASS_AMU, **kw)


def test_na_d2_lifetime_recoil_and_doppler_limit_match_steck():
    literature = dict(EMPTY_LITERATURE, fosc_j={("3s", "3p", "3/2"): 0.6408})
    result = _na([_line("3p3/2", "3s", 3, 3, 589.158)],
                 FValueSources(nist=EMPTY_NIST, literature=literature))
    (lt,) = result.lifetimes
    assert lt.dominant.fosc_source == "literature"
    assert (lt.dominant.g_upper, lt.dominant.g_lower) == (4, 2)
    assert lt.lifetime_s * 1e9 == pytest.approx(16.2492, rel=2e-3)
    assert lt.v_rec_ms * 100 == pytest.approx(2.9461, rel=1e-4)
    assert lt.T_dop_uK == pytest.approx(235.77, rel=3e-3)
    assert lt.delta_nu_nat_Hz == pytest.approx(lt.A_total_s / (2 * 3.141592653589793))
    assert lt.B12_m3_J_s2 == pytest.approx(2 * lt.B21_m3_J_s2), "B12 = (g_u/g_l) B21"


def test_s_to_p_multiplet_splits_two_thirds_one_third_with_equal_rates():
    f_mult, wl = 0.9, 589.0
    f32, g_u32, g_l = resolve_multiplet_f(f_mult, 1, 0, "3/2", "1/2", wl)
    f12, g_u12, _ = resolve_multiplet_f(f_mult, 1, 0, "1/2", "1/2", wl)
    assert (f32, f12) == pytest.approx((2 / 3 * f_mult, 1 / 3 * f_mult))
    assert (g_u32, g_u12, g_l) == (4, 2, 2)
    A32 = EINSTEIN_PREFACTOR / wl**2 * g_l / g_u32 * f32
    A12 = EINSTEIN_PREFACTOR / wl**2 * g_l / g_u12 * f12
    assert A32 == pytest.approx(A12), "both fine-structure levels live equally long (LS)"


def test_an_upper_level_decays_to_the_whole_lower_multiplet_at_the_multiplet_rate():
    """d3/2 → p1/2 + d3/2 → p3/2 = A_mult = PREF/λ² (2L_l+1)/(2L_u+1) f_mult."""
    f_mult, wl = 0.4, 818.0
    A_mult = EINSTEIN_PREFACTOR / wl**2 * 3 / 5 * f_mult
    total = 0.0
    for J_lo in ("1/2", "3/2"):
        f, g_u, g_l = resolve_multiplet_f(f_mult, 2, 1, "3/2", J_lo, wl)
        total += EINSTEIN_PREFACTOR / wl**2 * g_l / g_u * f
    assert total == pytest.approx(A_mult)
    assert resolve_multiplet_f(f_mult, 2, 1, "5/2", "1/2", wl) is None, "ΔJ = 2 is forbidden"


def _nist_row(n, f, J="3/2"):
    t = _line(f"{n}p{J}", "3s", n, 3, 300.0)
    return TransitionA(**{k: t[k] for k in ("upper_label", "lower_label", "upper_n",
                                             "lower_n", "upper_l", "lower_l",
                                             "wavelength_nm", "delta_E_eV", "region")},
                       fosc=f, fosc_source="NIST", fosc_acc="B", fosc_delta=0.1,
                       g_upper=4, g_lower=2, A_s=1.0, sigma_A_s=0.1)


def test_below_gate_lines_follow_the_fitted_nist_series():
    """f n*³ = a + b/n*² fitted on n = 4..9 and evaluated at n = 15."""
    a, b, dp = 0.012, -0.004, NA_QD["p"]

    def f_series(n):
        ns = n - dp
        return (a + b / ns**2) / ns**3

    rows = [_nist_row(n, f_series(n)) for n in range(4, 10)]
    gate = Dropped(_line("15p3/2", "3s", 15, 3, 245.0), "below_numerov_gate (n*<=2.5, no NIST f)")
    other = Dropped(_line("4d5/2", "3s", 4, 3, 330.0), "not_E1_or_unlabelled")
    rows_out, new_rows, still_dropped, fits, n_replaced, _ = extrapolate_nist_series(
        rows, [gate, other], NA_QD)
    assert rows_out == rows and n_replaced == 0, "no Numerov/Coulomb rows to replace"
    (row,) = new_rows
    assert row.fosc_source == "NIST-extrap"
    assert row.fosc == pytest.approx(f_series(15), rel=1e-9)
    assert still_dropped == [other]
    assert fits["3s->p3/2"].n_fit_max == 9
    assert fits["3s->p3/2"].rms < 1e-9


def test_metal_lines_take_the_published_gA():
    t = _line("4p3/2", "4s", 4, 4, 400.0, upper_J="3/2", lower_J="1/2", gA=4.0e8)
    result = _na([t], FValueSources(nist=EMPTY_NIST, literature=EMPTY_LITERATURE),
                 is_metal=True)
    assert result.numerov_calibration is None
    (row,) = result.transitions
    assert row.A_s == pytest.approx(1.0e8), "A = gA / g_u"
    assert row.fosc == pytest.approx(1.0e8 * 400.0**2 / EINSTEIN_PREFACTOR * 4 / 2)
    assert result.lifetimes[0].lifetime_s == pytest.approx(1e-8)
