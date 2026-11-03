# -*- coding: utf-8 -*-
"""``physics.transitions_core``: E1 selection rules and the metal line matching."""
from __future__ import annotations

import pytest

from linak.physics.transitions_core import (
    HC_EV_NM_ROUNDED,
    e1_transitions,
    ground_level,
    j_selection_rule,
    metal_levels,
    metal_transitions,
    qdt_levels,
)

NA_IE = 5.139076
# Na levels, eV above the ground state (NIST ASD)
NA_LEVELS = [
    dict(n=3, l="p", J="1/2", label="3p1/2", energy_eV=2.102297 - NA_IE, source="NIST"),
    dict(n=3, l="p", J="3/2", label="3p3/2", energy_eV=2.104429 - NA_IE, source="NIST"),
    dict(n=4, l="s", J="1/2", label="4s", energy_eV=3.191353 - NA_IE, source="NIST"),
    dict(n=3, l="d", J="5/2", label="3d5/2", energy_eV=3.616977 - NA_IE, source="NIST"),
    dict(n=3, l="d", J="3/2", label="3d3/2", energy_eV=3.616980 - NA_IE, source="NIST"),
]
NA_RYDBERG = {"ground_state": {"n": 3, "l": "s", "J": "1/2", "label": "3s"},
              "excitations": NA_LEVELS}


def test_delta_j_rule():
    assert j_selection_rule(1.5, 0.5)
    assert j_selection_rule(0.5, 0.5)
    assert not j_selection_rule(2.5, 0.5), "ΔJ = 2"
    assert not j_selection_rule(0.0, 0.0), "J = 0 → 0"
    assert j_selection_rule(None, 0.5), "not enforced without a J"


def test_na_low_levels_give_exactly_the_allowed_e1_lines():
    ground, legacy = ground_level(NA_RYDBERG, NA_IE)
    assert (ground.label, ground.energy, legacy) == ("3s", -NA_IE, False)
    lines = e1_transitions(qdt_levels(ground, NA_RYDBERG["excitations"]))
    pairs = {(t.upper_label, t.lower_label) for t in lines}
    assert pairs == {
        ("3p1/2", "3s"), ("3p3/2", "3s"),                      # D1, D2
        ("4s", "3p1/2"), ("4s", "3p3/2"),
        ("3d3/2", "3p1/2"), ("3d3/2", "3p3/2"), ("3d5/2", "3p3/2"),
    }, "no 3d5/2-3p1/2 (ΔJ = 2), 3d-3s (Δl = 2) or 4s-3s (Δl = 0)"
    d2 = next(t for t in lines if t.upper_label == "3p3/2" and t.lower_label == "3s")
    assert d2.wavelength_nm == round(HC_EV_NM_ROUNDED / 2.104429, 4)
    assert d2.wavelength_nm == pytest.approx(589.158, abs=0.01)
    assert (d2.region, d2.delta_n) == ("Visible", 0)


def test_levels_are_deduplicated_and_cut_at_n_max():
    ground, _ = ground_level(NA_RYDBERG, NA_IE)
    excitations = NA_LEVELS + [dict(NA_LEVELS[0]), dict(n=12, l="s", J=None, label="12s",
                                                        energy_eV=-0.11, source="Rydberg")]
    assert [lv.label for lv in qdt_levels(ground, excitations, n_max=4)] == \
        ["3s", "3p1/2", "3p3/2", "4s", "3d5/2", "3d3/2"]
    assert qdt_levels(ground, excitations)[-1].J_float is None, "QDT levels carry no J"


def test_old_json_without_a_ground_block_falls_back_to_the_lowest_s():
    ground, legacy = ground_level({"excitations": NA_LEVELS}, NA_IE)
    assert (ground.label, ground.n, ground.J, legacy) == ("4s", 4, "1/2", True)


def test_metal_lines_match_endpoints_by_label_or_unique_term_and_j():
    gs = {"label": "4s2", "n": 4, "l": "s", "J": "0", "term": "1S"}
    levels = metal_levels(gs, [
        dict(label="4s4p", energy_eV=-4.0, n=4, l="p", J="1", term="1P*"),
    ], IE=6.11)
    lines = [
        # matched by label, ASD wavelength kept (within 5 % of the level-derived one)
        dict(lower_label="4s2", upper_label="4s4p", wavelength_nm=588.0, gA=3.0e8),
        # blank-config line: the upper end matched by its unique (term, J)
        dict(lower_label="4s2", upper_label=None, upper_term="1P*", upper_J="1"),
        # unknown upper level
        dict(lower_label="4s2", upper_label="5p", upper_term="3P", upper_J="2"),
    ]
    found = metal_transitions(lines, levels, IE=6.11)
    (t,) = found.transitions
    assert (t.upper_label, t.lower_label, t.wavelength_nm, t.gA) == ("4s4p", "4s2", 588.0, 3.0e8)
    assert t.delta_E_eV == pytest.approx(2.11)
    assert found.n_resolved == 1, "the second line resolved by term + J (then a duplicate pair)"
    assert found.n_missing_upper == 1
