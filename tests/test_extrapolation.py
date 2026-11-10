"""Extrapolation module: parity with the validated prototype, range guard, refusal of unsupported species, stale-table check."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from linak.physics import extrapolation as E
from linak.physics.extrapolation.tables import OutOfTableRange, load_table

GOLD = json.loads((Path(__file__).parent / "anchors" / "extrapolation_golden.json").read_text())


@pytest.mark.parametrize("key", sorted(GOLD))
def test_matches_prototype(key):
    g = GOLD[key]
    el, J = key.split("_")
    o = E.extrapolate(el, float(J), g["series"], g["targets"])
    for k in ("value", "lo68", "hi95"):
        assert np.allclose(o[k], g[k], rtol=2e-3), (key, k)
    assert math.isclose(o["c0"], g["c0"], rel_tol=2e-3, abs_tol=1e-4)
    assert math.isclose(o["sigma_th_Mb"], g["sigma_th"], rel_tol=2e-3)


def test_unsupported_species_refused():
    with pytest.raises(E.UnsupportedSpeciesError, match="Ra"):
        E.extrapolate("Ra", 0.5, [], [])


@pytest.mark.parametrize("el", E.supported_species())
def test_tables_fresh(el):
    assert E.table_status(el)["fresh"]


def test_out_of_range_targets_are_nan_not_extrapolated():
    g = GOLD["Rb_0.5"]
    o = E.extrapolate("Rb", 0.5, g["series"], [{"n": 1, "nu": 1.5}, {"n": 200, "nu": 300.0}])
    assert o["out_of_range_targets"] == [1.5, 300.0]
    assert all(math.isnan(v) for v in o["value"])
    with pytest.raises(OutOfTableRange):
        load_table("Rb").f(1.5, 0.5)


def test_resonance_anchor_is_excluded_not_fitted():
    g = GOLD["Cs_1.5"]
    low = {"nu": 1.8, "f": 0.8, "unc": 0.01}
    a = E.extrapolate("Cs", 1.5, g["series"], g["targets"])
    b = E.extrapolate("Cs", 1.5, [low, *g["series"]], g["targets"])
    assert b["excluded_series_nu"] == [1.8]
    assert np.allclose(a["value"], b["value"])


def test_stale_hash_is_reported(monkeypatch):
    from linak.physics.extrapolation import params
    mp = params.model_params()
    monkeypatch.setitem(mp["species"]["Rb"], "lam", 0.9)
    assert not E.table_status("Rb")["fresh"]
