# -*- coding: utf-8 -*-
"""Inputs of the high-n network: which NIST levels are used, and where Numerov is scaled."""
from __future__ import annotations

import json

import pytest

from qali.physics.qdt_radial import (
    CALIBRATED_NSTAR_MAX,
    level_defect,
    fit_numerov_scale_from_nist,
    level_defects_from_rydberg,
    numerov_fosc,
    numerov_scale_for,
)
from qali.physics.rydberg import (
    Ry,
    fit_ritz_defects,
    generate_rydberg_series,
    screen_nist_levels,
)

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


def _ryd(**extra):
    return dict(kind="neutral", rydberg_constant_eV=Ry, ionization_energy_eV=IE,
                quantum_defects={"s": 3.14, "p": 2.68},
                ground_state={"n": 5, "l": "s"},
                excitations=[
                    dict(n=30, l="p", J="1/2", energy_eV=-Ry / (30 - 2.655) ** 2),
                    dict(n=30, l="p", J="3/2", energy_eV=-Ry / (30 - 2.642) ** 2),
                    dict(n=30, l="s", J=None, energy_eV=-Ry / (30 - 3.131) ** 2),
                    dict(n=31, l="p", J=None, energy_eV=-Ry / (31 - 9.0) ** 2),
                ], **extra)


def test_a_level_has_its_own_defect():
    qd = level_defects_from_rydberg(_ryd())
    assert level_defect(qd, 30, "s") == pytest.approx(3.131, abs=1e-4)
    assert level_defect(qd, 30, "p") == pytest.approx(2.6463, abs=1e-3)   # 2J+1 weighted
    assert qd["p"] == 2.68


def test_the_ground_state_comes_from_the_ionization_energy():
    qd = level_defects_from_rydberg(_ryd())
    assert level_defect(qd, 5, "s") == pytest.approx(5 - (Ry / IE) ** 0.5)


def test_unknown_or_implausible_levels_use_the_average():
    qd = level_defects_from_rydberg(_ryd())
    assert level_defect(qd, 40, "p") == 2.68
    assert level_defect(qd, 31, "p") == 2.68


def test_ions_keep_the_plain_averages():
    averaged = dict(s=1.0)
    ryd = dict(_ryd(), kind="ion", quantum_defects=averaged)
    assert level_defects_from_rydberg(ryd) is averaged
    assert level_defect(averaged, 30, "s") == 1.0


def test_pooled_scale_needs_most_pairs_inside_the_window():
    qd = {"s": 0.0, "p": 0.0}
    fosc = {}
    for i, up in enumerate(range(5, 22)):
        ratio = 1.0 if i < 6 else 0.1
        fosc[("4s", f"{up}p")] = ratio * numerov_fosc(up, 1, 4, 0, 0.0, 0.0)
    info = fit_numerov_scale_from_nist(qd, fosc)
    assert info["reason"] == "median_outside_window"
    assert info["scale"] == 1.0


def test_a_qdt_row_uses_the_series_value_not_its_rounded_energy():
    ryd = _ryd(quantum_defects_ritz={"p": dict(delta0=2.65, delta2=0.3, n_min=7)})
    ryd["excitations"].append(dict(n=40, l="p", J=None, energy_eV=-Ry / (40 - 2.70) ** 2,
                                   source="QDT model"))
    qd = level_defects_from_rydberg(ryd)
    assert level_defect(qd, 40, "p") == pytest.approx(2.65 + 0.3 / (40 - 2.65) ** 2)
    assert level_defect(qd, 60, "p") == pytest.approx(2.65 + 0.3 / (60 - 2.65) ** 2)


def test_an_imprecise_level_takes_the_series_value():
    ryd = _ryd(quantum_defects_ritz={"p": dict(delta0=2.65, delta2=0.3, n_min=7)})
    ryd["excitations"].append(dict(n=35, l="p", J="1/2", energy_eV=-Ry / (35 - 2.70) ** 2,
                                   uncertainty_eV=4e-5))
    qd = level_defects_from_rydberg(ryd)
    assert level_defect(qd, 35, "p") == pytest.approx(2.65 + 0.3 / (35 - 2.65) ** 2)


def test_a_series_starts_at_its_lowest_nist_level():
    lookup = {(5, "d", "5/2", "2D"): dict(energy_eV=-2.0),
              (6, "p", "1/2", "2P*"): dict(energy_eV=-2.5)}
    states = generate_rydberg_series(IE, {"s": 4.0, "p": 3.6, "d": 2.5}, lookup, 6, 7)
    labels = [(st["n"], st["l"]) for st in states]
    assert (5, "d") in labels
    assert (5, "p") not in labels   # never modelled below the measured series


@pytest.mark.parametrize("element,state,low,high", [
    ("K", "4f5/2", 50, 80),      # ARC 62.5 ns; 11 us without 3d
    ("Rb", "4d3/2", 75, 95),     # measured 86(6) ns, Safronova 2004
    ("Rb", "5g9/2", 200, 290),   # ARC 232 ns
    ("Cs", "6g9/2", 330, 470),   # ARC 384 ns; 54 ms without 5f
])
def test_levels_below_the_ground_n_set_these_lifetimes(data_dir, element, state, low, high):
    rows = json.loads((data_dir / f"{element}_lifetimes.json").read_text(encoding="utf-8"))
    tau = {r["state"]: r["lifetime_ns"] for r in rows["lifetimes"]}
    assert low < tau[state] < high


@pytest.mark.parametrize("element,low,high", [("K", 550, 650), ("Rb", 750, 860), ("Cs", 1250, 1500)])
def test_d1_upper_state_polarizability_includes_the_lower_d_level(data_dir, element, low, high):
    bb = json.loads((data_dir / f"{element}_blackbody.json").read_text(encoding="utf-8"))
    assert low < bb["alpha0_ex_au"] < high


def test_published_fr_lines_set_the_8s_and_6d_lifetimes(data_dir):
    rows = json.loads((data_dir / "Fr_lifetimes.json").read_text(encoding="utf-8"))
    tau = {r["state"]: r["lifetime_ns"] for r in rows["lifetimes"]}
    assert tau["8s"] == pytest.approx(53.30, abs=1.0)   # measured 53.30(44) ns
    assert 500 < tau["6d3/2"] < 650
    assert 1600 < tau["6d5/2"] < 2000
    assert rows["dropped_transitions"]["n_total"] == 0


def test_a_literature_line_with_both_j_keeps_them_apart(tmp_path):
    from qali.physics.polarizability_core import load_literature_fvalues
    row = dict(lower="7p", lower_J="3/2", upper="6d", upper_J="5/2", f_abs=0.2, acc="B",
               ref="PUBLISHED: test")
    (tmp_path / "Fr_literature.json").write_text(json.dumps({"transitions": [row]}), "utf-8")
    lit = load_literature_fvalues("Fr", str(tmp_path), verbose=False)
    assert lit["fosc_jl"][("7p", "6d", "5/2", "3/2")] == (0.2, "B")
    assert not lit["fosc_j"] and not lit["fosc"]
