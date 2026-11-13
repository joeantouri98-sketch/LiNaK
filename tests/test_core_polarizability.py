# -*- coding: utf-8 -*-
"""The ion core's dynamic polarisability: the bundled DHF + RPA pseudo-spectra
(resources/core/core_polarizability.json) and what they do to α, C₆ and the tweezer depth."""
from __future__ import annotations

import json
import math

import pytest

from linak.core.paths import reference_path
from linak.physics.polarizability_core import ALPHA_CORE_AU, CORE_ELECTRONS, core_oscillators


@pytest.fixture(scope="module")
def table():
    with reference_path("core", "core_polarizability.json").open(encoding="utf-8") as f:
        return json.load(f)["elements"]


def test_rpa_static_core_polarisabilities_match_relativistic_rpa(table):
    # Johnson, Kolb & Huang (relativistic RPA): Li+ 0.1894, Na+ 0.9457, Cs+ 15.81; Fr+ 20.4 (Derevianko)
    for el, ref in {"Li": 0.1894, "Na": 0.9457, "Cs": 15.81, "Fr": 20.4}.items():
        assert table[el]["alpha0_rpa_au"] == pytest.approx(ref, rel=3e-3), el


def test_rpa_restores_the_trk_sum_rule(table):
    """ξ²α(iξ) → N_core at large ξ in RPA (DHF alone overshoots); at ξ = 1000 a.u. the deep 1s
    shell of the heavy cores has not yet responded, so allow 7 %."""
    for el, rec in table.items():
        assert rec["trk_at_xi_1000"] == pytest.approx(rec["core_electrons"], rel=0.07), el
        assert rec["core_electrons"] == CORE_ELECTRONS[el]
        assert rec["fit_max_rel_error"] < 1e-5 and rec["max_imag"] < 1e-2 * rec["alpha0_rpa_au"]
        for w, rpa, fit in rec["real_frequency_check"]:      # the pseudo-spectrum holds at real ω too
            assert fit == pytest.approx(rpa, rel=1e-5), (el, w)


def test_core_lines_are_scaled_to_the_static_table_and_fall_back_without_one():
    for el in ("Na", "Cs"):
        osc = core_oscillators(el)
        assert len(osc) > 10 and sum(f / w ** 2 for w, f in osc) == pytest.approx(ALPHA_CORE_AU[el], rel=1e-12)
        assert sum(f / w ** 2 for w, f in core_oscillators(el, 2 * ALPHA_CORE_AU[el])) == pytest.approx(2 * ALPHA_CORE_AU[el])
    w, f = core_oscillators("Mg", 5.0, 10)[0]                     # not in the table: the old one-line model
    assert f == 10.0 and w == pytest.approx(math.sqrt(10 / 5.0))
    assert core_oscillators("Mg", 0.0, 10) == []


def test_c6_with_the_rpa_core_matches_derevianko(data_dir):
    """With the RPA core, C₆ lands within 0.5 % of Derevianko, Porsev & Babb (2010) for Na–Cs
    (it was 2–4 % high for Rb and Cs with one core line) and within 2 % (1.1σ) for Fr."""
    from linak.pipelines.polarizability import compute_C6
    ref = {"Na": (1556, 0.005), "K": (3897, 0.005), "Rb": (4690, 0.005), "Cs": (6846, 0.005), "Fr": (5256, 0.02)}
    for el, (c6, tol) in ref.items():
        with (data_dir / f"{el}_polarizability.json").open(encoding="utf-8") as f:
            tr = json.load(f)["transitions_gs"]
        assert sum(1 for t in tr if t["other_label"] == "core") > 10, el
        assert compute_C6(tr) == pytest.approx(c6, rel=tol), el


def test_tweezer_depth_includes_the_core():
    from linak.physics.tweezer_core import alpha_complex, attach_gamma
    tr = [dict(delta_E_au=0.05, fosc=1.0, sign=1, other_label="np", source="NIST"),
          dict(delta_E_au=1.5, fosc=10.0, sign=1, other_label="core", source="core")]
    ch, n_with, n_core = attach_gamma(tr, {"np": 3e7})
    assert n_core == 1 and len(ch) == 2 and ch[1]["Gamma_au"] == 0.0
    re, _im = alpha_complex(0.0428, ch)
    assert re == pytest.approx(1.0 / (0.05 ** 2 - 0.0428 ** 2) + 10.0 / (1.5 ** 2 - 0.0428 ** 2), rel=1e-6)
