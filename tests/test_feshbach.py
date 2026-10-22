# -*- coding: utf-8 -*-
"""Na magnetic Feshbach a(B) formula and literature-parameter anchors."""
from __future__ import annotations

import json
import math

import numpy as np

from feshbach import multi_resonance_a, scattering_length

from helpers import write_result_summary


def test_feshbach_literature_params(na_feshbach_params, anchors):
    a = anchors["feshbach"]
    assert float(na_feshbach_params["a_bg_default_a0"]) == a["a_bg_a0"]
    by_label = {r["label"]: r for r in na_feshbach_params["resonances"]}
    for exp in a["resonances"]:
        row = by_label[exp["label"]]
        assert float(row["B0_G"]) == exp["B0_G"]
        assert float(row["Delta_G"]) == exp["Delta_G"]
        assert float(row["a_bg_a0"]) == exp["a_bg_a0"]


def test_scattering_length_zero_and_background(anchors, results_dir):
    """a(B0 + Delta) = 0 and a far from resonance -> a_bg."""
    a = anchors["feshbach"]
    res = a["resonances"][0]  # 905 G, Delta=1.04
    a_bg = res["a_bg_a0"]
    B0 = res["B0_G"]
    Delta = res["Delta_G"]

    B_zero = B0 + Delta
    a_zero = float(scattering_length(B_zero, a_bg, B0, Delta))
    assert abs(a_zero - 0.0) <= a["tol_abs_a0"]

    a_far = float(scattering_length(-1.0e4, a_bg, B0, Delta))
    assert abs(a_far - a_bg) / abs(a_bg) < 1e-4

    # Pole is NaN
    assert math.isnan(float(scattering_length(B0, a_bg, B0, Delta)))

    write_result_summary(
        results_dir,
        "feshbach_formula",
        {
            "B_zero": B_zero,
            "a_zero": a_zero,
            "a_far_B0": a_far,
            "a_bg": a_bg,
        },
    )


def test_multi_resonance_product_near_background(na_feshbach_params, anchors):
    """Far below all poles, multi-resonance a(B) recovers a_bg."""
    a = anchors["feshbach"]
    a_bg = float(na_feshbach_params["a_bg_default_a0"])
    B = np.array([0.0, 10.0, 50.0])
    vals = multi_resonance_a(B, na_feshbach_params["resonances"], a_bg)
    for v in vals:
        assert abs(float(v) - a_bg) / abs(a_bg) < 0.02


# ---- PHYSICS_BUGS P1 / P10 / P16 --------------------------------------------------------------
def test_a_bar_is_gribakin_flambaum_for_beta6():
    """abar = 2 pi / Gamma(1/4)^2 * beta6 (= 0.956 * R_vdW with R_vdW = beta6/2)."""
    from feshbach import a_bar_a0, beta6_a0
    import pytest
    assert a_bar_a0(100.0) == pytest.approx(100.0 * 2.0 * math.pi / math.gamma(0.25) ** 2, rel=1e-6)
    b = beta6_a0(4698.0, 86.90918)          # Rb-87, C6 = 4698 au
    assert 164.0 < b < 166.5                 # R_vdW = 82.9 a0
    assert abs(a_bar_a0(b) - 78.99) < 0.2    # literature abar(Rb-87) = 78.99 a0
    assert abs(a_bar_a0(b) - 0.955978 * b / 2.0) < 1e-3


def test_shipped_na_out_json_has_corrected_a_bar(data_dir):
    d = json.load(open(data_dir / "Na_feshbach_out.json"))
    assert abs(d["a_bar_a0"] / d["beta6_a0"] - 0.477989) < 1e-4


def test_nonpositive_c6_is_rejected_with_clear_error():
    from feshbach import beta6_a0
    import pytest
    for bad in (-5.0, 0.0, float("nan")):
        with pytest.raises(ValueError, match="C6"):
            beta6_a0(bad, 23.0)


def test_resonance_sign_convention_and_schema():
    """a_bg > 0, Delta > 0: a -> +inf just below B0 and -inf just above; zero at B0 + Delta."""
    import pytest
    from feshbach import validate_feshbach_params
    assert scattering_length(905.0 - 1e-3, 54.54, 905.0, 1.04) > 1e3
    assert scattering_length(905.0 + 1e-3, 54.54, 905.0, 1.04) < -1e3
    assert abs(float(scattering_length(906.04, 54.54, 905.0, 1.04))) < 1e-9
    # negative Delta is legitimate (e.g. 6Li 834 G) and must pass; Delta = 0 must not
    validate_feshbach_params({"resonances": [{"B0_G": 834.0, "Delta_G": -300.0, "a_bg_a0": -1405.0}]})
    with pytest.raises(ValueError):
        validate_feshbach_params({"resonances": [{"B0_G": 834.0, "Delta_G": 0.0}]})


# ---- PHYSICS_BUGS P24 / P25 (fork review) -----------------------------------------------------
def test_quantum_statistics_from_isotope():
    from feshbach import is_boson
    assert is_boson('Na', 23) and is_boson('K', 39) and is_boson('Rb', 87) and is_boson('Cs', 133)
    assert is_boson('Li', 6) is False and is_boson('K', 40) is False and is_boson('Li', 7)
    assert is_boson('Xx', 3) is None


def test_sigma_el_identical_bosons_is_twice_distinguishable():
    from feshbach import sigma_el_a0
    a = np.array([50.0, 500.0]); k = 0.001
    assert np.allclose(sigma_el_a0(a, k, identical_bosons=True), 2 * sigma_el_a0(a, k))
    assert np.allclose(sigma_el_a0(a, 0.0), 4 * np.pi * a ** 2)
    assert np.allclose(sigma_el_a0(a, 0.0, identical_bosons=True), 8 * np.pi * a ** 2)


def test_channel_parsing_and_grouping(na_feshbach_params):
    from feshbach import identical_particles, group_s_wave_by_channel
    assert identical_particles('|1,+1> + |1,+1>') and not identical_particles('|1> + |2>')
    assert not identical_particles('ee (highest F=2 stretched)')
    g = group_s_wave_by_channel(na_feshbach_params['resonances'])
    assert len(g) == 2 and len(g['|1,+1> + |1,+1>']) == 3 and len(g['|1,-1> + |1,-1>']) == 1
    mixed = [{'channel': 'x', 'partial_wave': 'd'}, {'channel': 'x', 'partial_wave': 's'}]
    assert len(group_s_wave_by_channel(mixed)['x']) == 1       # d-wave poles never enter a_s(B)


def test_combined_curve_does_not_mix_channels():
    import os
    from feshbach import compute_element
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data_json')
    r = compute_element('Na', root, B_points=3000)
    B = r['B_G']
    comb = np.asarray(r['a_combined_a0'])
    # the |1,-1> pole at 1195 G must not appear in the |1,+1> curve ...
    i = int(np.argmin(abs(B - 1195.0)))
    assert abs(comb[i]) < 5000
    # ... but it is in its own channel's curve
    other = next(c for c in r['channels'] if c['channel'] == '|1,-1> + |1,-1>')
    assert np.nanmax(np.abs(other['a_a0'][i - 3:i + 4])) > 1e3
    assert r['sigma_combined_a02'][0] == np.pi * 8 * comb[0] ** 2 / (1 + (r['k_a0inv'] * comb[0]) ** 2)
