# -*- coding: utf-8 -*-
"""Static polarizability of low-lying Na states from the line-resolved sum over states.

Reference: ARC 3.10 (Sodium, Beterov/Marinescu model potential), valence sum over all
n' <= 90, both l' = l +- 1, all allowed j'.  Before the line-level fix alpha(3p1/2) was
223 a.u. (0.62x) and alpha(3p3/2) 441 a.u. (1.22x).  Rydberg states (n >= 10) are NOT
tested here (open item: they are 1.3-2.3x ARC).
"""
from __future__ import annotations

import pytest

import linak.physics.polarizability_core as A

ARC_VALENCE = {          # a.u.
    "3s": 162.09, "3p1/2": 360.55, "3p3/2": 362.27, "4s": 3101.44,
    "3d3/2": 6419.47, "3d5/2": 6395.22, "5s": 21769.68,
}


@pytest.fixture(scope="module")
def na_network(root_dir):
    import os
    cwd = os.getcwd()
    os.chdir(root_dir)
    try:
        data = A.load_element("Na", verbose=False)
    finally:
        os.chdir(cwd)
    return data


@pytest.mark.parametrize("state,ref", sorted(ARC_VALENCE.items()))
def test_low_state_alpha0_matches_model_potential(na_network, state, ref):
    assert na_network.get("ls_mode"), "line-level (LS) mode must be active for Na"
    trans = A.build_transitions_for_state(state, na_network, na_network["transitions"])
    val = sum(t["sign"] * t["fosc"] / t["delta_E_au"] ** 2
              for t in trans if t["other_label"] != "core")
    assert val == pytest.approx(ref, rel=0.02), f"{state}: {val:.1f} vs ARC {ref:.1f}"


def test_fine_structure_partners_have_nearly_equal_alpha0(na_network):
    vals = {}
    for st in ("3p1/2", "3p3/2"):
        tr = A.build_transitions_for_state(st, na_network, na_network["transitions"])
        vals[st] = sum(t["sign"] * t["fosc"] / t["delta_E_au"] ** 2 for t in tr if t["other_label"] != "core")
    assert abs(vals["3p3/2"] / vals["3p1/2"] - 1.0) < 0.02


def test_multiplet_f_rebuilt_from_lines(na_network):
    f = na_network["nist_fosc"]
    assert f[("4p", "8d")] == pytest.approx(0.016573, rel=1e-3)   # legacy value was 0.02403
    assert f[("3s", "4p")] == pytest.approx(0.01346, rel=1e-6)      # s->p unchanged
