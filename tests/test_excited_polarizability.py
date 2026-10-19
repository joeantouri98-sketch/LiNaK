# -*- coding: utf-8 -*-
"""Static polarizability of low-lying Na states from the line-resolved sum over states.

Reference: ARC 3.10 (Sodium, Beterov/Marinescu model potential), valence sum over all
n' <= 90, both l' = l +- 1, all allowed j'.  Before the line-level fix alpha(3p1/2) was
223 a.u. (0.62x) and alpha(3p3/2) 441 a.u. (1.22x).  Rydberg states (n >= 10) are NOT
tested here except 20s (open item: n >= 25 differ by 20-30 % because the QDT-model levels
use constant quantum defects, and 10s is affected by the NIST np->10s column outlier).
"""
from __future__ import annotations

import pytest

import alpha_core as A

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


def test_s_to_p_scale_depends_on_gap(na_network):
    """NIST/Numerov for Na s->p is ~1.0 for near pairs and ~1.37 for far pairs."""
    from qdt_radial import numerov_scale_for
    cal = na_network["numerov_cal_types"]
    info = cal["types"][("s", "p")]
    assert info["gap_dependent"]
    near = numerov_scale_for(cal, "s", "p", gap=0.5)
    mid = numerov_scale_for(cal, "s", "p", gap=2.75)
    far = numerov_scale_for(cal, "s", "p", gap=8.0)
    assert 0.97 < near < 1.05
    assert 1.25 < far < 1.50
    assert near < mid < far
    # flat types keep the single type-wide scale
    assert not cal["types"][("p", "d")]["gap_dependent"]
    assert numerov_scale_for(cal, "p", "d", gap=0.5) == numerov_scale_for(cal, "p", "d", gap=8.0)


def test_rydberg_20s_alpha_matches_model_potential(na_network):
    """Dominated by the near-resonant 20s->20p/19p channels (was 1.88x ARC with one s->p scale)."""
    trans = A.build_transitions_for_state("20s", na_network, na_network["transitions"])
    val = sum(t["sign"] * t["fosc"] / t["delta_E_au"] ** 2
              for t in trans if t["other_label"] != "core")
    assert val == pytest.approx(449.84e6, rel=0.05)


def test_precision_overlay_has_priority_in_line_network():
    """Hierarchy: lifetime-derived precision D-line f (e.g. Fr 7s->7p) beats NIST/Numerov lines."""
    data = dict(nist_fosc_jl={}, qd={"s": 4.0, "p": 3.5}, nist_fosc={}, literature_fosc=None,
                precision_fosc_j={("7s", "7p"): {"1/2": 0.34, "3/2": 0.7357}}, numerov_cal_types=None)
    t = dict(upper_label="7p3/2", lower_label="7s", upper_l="p", lower_l="s",
             lower_n=7, upper_n=7, upper_J="3/2", lower_J="1/2")
    f_up, src_up, sign_up = A._ls_state_term(t, "7s", data)
    f_dn, src_dn, sign_dn = A._ls_state_term(t, "7p3/2", data)
    assert (src_up, src_dn) == ("precision", "precision")
    assert f_up == pytest.approx(0.7357) and sign_up == 1
    assert f_dn == pytest.approx(0.7357 * 2 / 4) and sign_dn == -1
