# -*- coding: utf-8 -*-
"""Fine-structure (6j) resolution, NIST line completion, extrapolation and drop logging
in lifetimes.py / qdt_radial.py.  Requires data_json/Na_lifetimes.json regenerated with
the Tier-1 lifetimes fixes (python lifetimes.py Na)."""
from __future__ import annotations

import json
import math

import pytest

from qdt_radial import (
    allowed_J2,
    build_true_multiplet_f,
    ls_group_branching,
    ls_line_weight,
    wigner_6j,
)


SIXJ_1_1_1_h_h_h = -1.0 / 3.0   # sympy.physics.wigner.wigner_6j(1,1,1,1/2,1/2,1/2)


# ---- pure angular algebra ---------------------------------------------------

def test_wigner_6j_reference_values():
    assert wigner_6j(2, 2, 2, 2, 2, 2) == pytest.approx(1.0 / 6.0, abs=1e-12)   # {1 1 1;1 1 1}
    assert wigner_6j(2, 2, 2, 1, 1, 1) == pytest.approx(SIXJ_1_1_1_h_h_h, abs=1e-12)  # {1 1 1;1/2 1/2 1/2}
    assert wigner_6j(2, 2, 6, 2, 2, 2) == 0.0                                        # violated triangle


@pytest.mark.parametrize("Lu,Ll", [(1, 0), (0, 1), (2, 1), (1, 2), (3, 2), (2, 3), (4, 3)])
def test_line_weights_sum_to_one(Lu, Ll):
    for ju in allowed_J2(Lu):
        assert sum(ls_line_weight(Lu, ju, Ll, jl) for jl in allowed_J2(Ll)) == pytest.approx(1.0, abs=1e-10)


def test_d_to_p_branching_is_5_to_1():
    w12 = ls_line_weight(2, 3, 1, 1)   # d3/2 -> p1/2
    w32 = ls_line_weight(2, 3, 1, 3)   # d3/2 -> p3/2
    assert w12 / w32 == pytest.approx(5.0, rel=1e-9)
    assert ls_line_weight(2, 5, 1, 1) == 0.0        # d5/2 -> p1/2 forbidden
    assert ls_line_weight(2, 5, 1, 3) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("Lu,Ll", [(0, 1), (2, 1), (1, 2), (3, 2)])
def test_unresolved_upper_gives_statistical_lower_split(Lu, Ll):
    """A J-less upper level: A(-> Jl) = A_mult (2Jl+1) / (2 (2Ll+1))  (no double counting)."""
    for jl in allowed_J2(Ll):
        ratio, _gu, _gl = ls_group_branching(Lu, allowed_J2(Lu), Ll, [jl])
        assert ratio == pytest.approx((jl + 1) / (2 * (2 * Ll + 1)), abs=1e-10)
    total = sum(ls_group_branching(Lu, allowed_J2(Lu), Ll, [jl])[0] for jl in allowed_J2(Ll))
    assert total == pytest.approx(1.0, abs=1e-10)


def test_true_multiplet_f_from_lines_is_g_weighted():
    # NIST Na 3p -> 6d lines (J_up, J_lo): 1/2->3/2: 1.36e-2, 3/2->3/2: 1.36e-3, 3/2->5/2: 1.22e-2
    jl = {("3p", "6d", "3/2", "1/2"): (1.36e-2, "B+"),
          ("3p", "6d", "3/2", "3/2"): (1.36e-3, "B+"),
          ("3p", "6d", "5/2", "3/2"): (1.22e-2, "A")}
    f = build_true_multiplet_f(jl)[("3p", "6d")]
    assert f == pytest.approx((2 * 1.36e-2 + 4 * 1.36e-3 + 4 * 1.22e-2) / 6.0, rel=1e-12)
    # a missing weak line is completed with the LS ratio (10:1 in f for d3/2)
    jl2 = {k: v for k, v in jl.items() if k[3] != "1/2"}
    jl2[("3p", "6d", "3/2", "3/2")] = (1.36e-3, "B+")
    assert build_true_multiplet_f(jl2)[("3p", "6d")] == pytest.approx(f, rel=0.01)


# ---- regenerated Na lifetimes ---------------------------------------------------

def _by_state(na_lifetimes):
    return {x["state"]: x for x in na_lifetimes["lifetimes"]}


def test_nd_fine_structure_levels_have_equal_total_rate(na_lifetimes):
    """LS: nd3/2 and nd5/2 decay to the same lower multiplets at the same total rate."""
    st = _by_state(na_lifetimes)
    for n in range(6, 21):
        a, b = st[f"{n}d3/2"]["A_total_s"], st[f"{n}d5/2"]["A_total_s"]
        assert abs(a / b - 1.0) < 0.03, f"{n}d: {a:.4g} vs {b:.4g}"


@pytest.mark.parametrize("state,A_arc", [("20s", 1.141e5), ("25s", 5.596e4)])
def test_jless_upper_total_rate_matches_independent_model_potential(na_lifetimes, state, A_arc):
    """Reference: ARC 3.10 (Sodium, T=0, Beterov model potential). Before the double-count
    fix these were 2.0x and 1.5x too large. 10s is excluded: NIST np->10s f are 1.5-1.9x
    ARC (open item)."""
    A = _by_state(na_lifetimes)[state]["A_total_s"]
    assert abs(A / A_arc - 1.0) < 0.06


def test_extrapolated_channels_continue_nist_smoothly(na_lifetimes):
    ex = na_lifetimes["nist_extrapolation"]
    assert ex["n_channels"] >= 100  # 3s->np n>=19 now come from the curated literature tier
    st = _by_state(na_lifetimes)
    qd = {"s": 1.3505, "p": 0.8650}
    def tau_scaled(n, l):
        label = f"{n}{l}" if (l == "s" or f"{n}{l}1/2" not in st) else f"{n}{l}1/2"
        return st[label]["lifetime_ns"] / (n - qd[l]) ** 3
    for l in ("s", "p"):
        # 3s->np: NIST n=12-18 sit 1.3-1.45x above the smooth physical curve, so the np series steps
        # (by design) where the curated high-n values take over at n=19; check smoothness from n=19 on.
        prev = tau_scaled(18 if l == "s" else 19, l)
        for n in range(19 if l == "s" else 20, 26):
            cur = tau_scaled(n, l)
            assert abs(cur / prev - 1.0) < 0.05, f"{n}{l} jump {cur/prev:.3f}"
            prev = cur


def test_dropped_transitions_are_logged_and_accounted(na_lifetimes, data_dir):
    d = na_lifetimes["dropped_transitions"]
    with (data_dir / "Na_transitions.json").open(encoding="utf-8") as f:
        n_in = len(json.load(f)["transitions"])
    assert d["n_total"] == sum(d["by_reason"].values())
    assert d["n_total"] + na_lifetimes["n_transitions"] == n_in      # every input transition accounted for
    assert not any(r.startswith("below_numerov_gate") for r in d["by_reason"]), \
        "3s/3p channels beyond NIST n_max must be recovered by the extrapolation"


def test_numerov_type_scales_not_inflated_by_multiplet_overcount(na_lifetimes):
    cal = na_lifetimes["numerov_calibration"]
    for key in ("p->d", "d->f"):
        assert 0.9 <= cal["by_type"][key]["scale"] <= 1.15, key
    assert 0.9 <= cal["pooled"]["scale"] <= 1.2
