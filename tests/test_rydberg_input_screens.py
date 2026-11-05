# -*- coding: utf-8 -*-
"""Inputs of the high-n network: which NIST levels are used, and where Numerov is scaled."""
from __future__ import annotations

import pytest

from linak.physics.qdt_radial import CALIBRATED_NSTAR_MAX, numerov_scale_for
from linak.physics.rydberg import Ry, fit_ritz_defects, screen_nist_levels

IE = 5.0


def _level(n, delta, unc=1e-7, l_char="p", J="3/2"):
    energy = IE - Ry / (n - delta) ** 2
    return dict(n=n, l=l_char, J=J, energy_eV=energy, uncertainty_eV=unc, theoretical=False)


def test_precise_series_is_kept_whole():
    levels = [_level(n, 1.0) for n in range(4, 30)]
    kept, dropped = screen_nist_levels(IE, levels)
    assert len(kept) == len(levels) and not dropped


def test_level_less_precise_than_the_series_is_replaced():
    # σ_δ = n*³ σ_E / 2Ry = 34³ × 1.2e-4 / 27.2 = 0.17
    levels = [_level(n, 0.05, unc=1e-7) for n in range(4, 12)] + [_level(35, 0.05, unc=1.2e-4)]
    kept, dropped = screen_nist_levels(IE, levels)
    assert [lv["n"] for lv in dropped] == [35]
    assert dropped[0]["reason"].startswith("sigma_delta")
    assert len(kept) == 8


def test_same_uncertainty_is_fine_at_low_n():
    _, dropped = screen_nist_levels(IE, [_level(4, 0.05, unc=1.2e-4)])
    assert not dropped


def test_mislabelled_level_is_an_outlier_even_when_quoted_precisely():
    levels = [_level(n, 3.4, l_char="d", J="5/2") for n in range(8, 22)]
    levels.append(_level(22, 5.78, l_char="d", J="5/2"))
    _, dropped = screen_nist_levels(IE, levels)
    assert [lv["n"] for lv in dropped] == [22]
    assert dropped[0]["reason"].startswith("outlier")


def test_missing_uncertainty_keeps_the_level():
    _, dropped = screen_nist_levels(IE, [_level(40, 0.05, unc=None)])
    assert not dropped


CAL = {"types": {("p", "d"): {"scale": 1.737, "gap_dependent": False},
                 ("s", "p"): {"scale": 2.0, "gap_dependent": True,
                              "scale_near": 1.0, "scale_far": 3.5}},
       "gap_knots": [1.5, 4.0]}


def test_scale_applies_inside_the_calibrated_range():
    assert numerov_scale_for(CAL, "p", "d", gap=2.0, n_star_upper=9.0) == pytest.approx(1.737)


@pytest.mark.parametrize("gap", [0.5, 2.0, 8.0])
def test_scale_stops_above_the_calibrated_range(gap):
    upper = CALIBRATED_NSTAR_MAX + gap + 0.1
    assert numerov_scale_for(CAL, "p", "d", gap=gap, n_star_upper=upper) == 1.0
    assert numerov_scale_for(CAL, "s", "p", gap=gap, n_star_upper=upper) == 1.0


def test_low_lower_level_keeps_its_scale_however_high_the_upper_level():
    assert numerov_scale_for(CAL, "s", "p", gap=20.0, n_star_upper=25.0) == pytest.approx(3.5)


def _ritz_level(n, d0, d2, unc, l_char="p"):
    return _level(n, d0 + d2 / (n - d0) ** 2, unc=unc, l_char=l_char)


def test_precise_series_is_fitted_unweighted():
    levels = [_ritz_level(n, 2.65, 0.3, 1e-7) for n in range(5, 16)]
    fit = fit_ritz_defects(IE, levels)["p"]
    assert fit["delta0"] == pytest.approx(2.65, abs=1e-5)
    assert "weighted" not in fit


def test_noisy_high_members_do_not_lose_the_series():
    levels = [_ritz_level(n, 2.65, 0.3, 1e-7) for n in range(5, 16)]
    for i, n in enumerate(range(16, 37)):
        levels.append(_ritz_level(n, 2.65 + 0.02 * (-1) ** i, 0.3, 2e-5))
    fit = fit_ritz_defects(IE, levels)["p"]
    assert fit["weighted"] is True
    assert fit["delta0"] == pytest.approx(2.65, abs=2e-3)


def test_noisy_series_without_uncertainties_stays_unfitted():
    levels = [_ritz_level(n, 2.65 + 0.02 * (-1) ** n, 0.3, None) for n in range(5, 37)]
    assert "p" not in fit_ritz_defects(IE, levels)


def test_five_precise_points_need_the_weighted_fit():
    levels = [_ritz_level(n, 2.65, 0.3, 1e-7) for n in range(5, 12)]
    assert fit_ritz_defects(IE, levels)["p"]["weighted"] is True
