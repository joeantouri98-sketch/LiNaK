"""Rydberg shifts: series repair (unrounded QDT energies, tied f-g pairs, synthetic h series) and static validity."""
import itertools

import pytest

from qali.core.paths import reference_path
from qali.physics import blackbody_core as bc
from qali.physics.polarizability_core import load_element

FW_HZ = 2416.67      # Farley-Wing limit at 300 K


@pytest.fixture(scope="module")
def data():
    return load_element("Na", str(reference_path("data_json", "Na_rydberg.json").parent), verbose=False)


@pytest.fixture(scope="module")
def refined(data):
    return bc.refine_series_data(data)


def test_model_levels_are_taken_at_the_formula_energy(data, refined):
    # 34f and 34g tie at the stored 1e-6 eV rounding; the formula separates them by (d_f - d_g) / n^3
    assert data["state_energy_au"]["34f"] == data["state_energy_au"]["34g"]
    gap = refined["state_energy_au"]["34g"] - refined["state_energy_au"]["34f"]
    assert gap == pytest.approx((0.0014775678 - 0.0002995855) / 34 ** 3, rel=0.02)
    assert all(abs(refined["state_energy_au"][k] - data["state_energy_au"][k]) < bc.QDT_ROUND_AU
               for k in data["state_energy_au"])


def test_tied_pairs_are_restored_and_the_original_data_is_untouched(data, refined):
    pairs = {frozenset((t["upper_label"], t["lower_label"])) for t in data["transitions"]}
    assert frozenset(("34g", "34f")) not in pairs
    new = {frozenset((t["upper_label"], t["lower_label"])) for t in refined["transitions"]}
    assert frozenset(("34g", "34f")) in new
    assert "34h" not in data["state_energy_au"] and "34h" in refined["state_energy_au"]
    assert bc.refine_series_data(refined) is refined


def test_synthetic_h_states_never_reach_the_table(data):
    rs = bc.compute_rydberg_shifts(data, 300.0, n_min=34, n_max=34, do_dyn=False)
    assert {r.l for r in rs.states} == {"s", "p", "d", "f", "g"}


@pytest.mark.parametrize("label", ["25g", "34g", "41g"])
def test_g_dynamic_shift_approaches_the_farley_wing_limit(refined, label):
    r = bc.rydberg_shift_for_state(label, refined, 300.0)
    assert r.dyn_reliable
    assert 0.9 * FW_HZ < r.shift_dyn_hz < FW_HZ * 1.02      # was -460 .. +1400 Hz without the h series


def test_g_and_f_dynamic_shifts_have_no_steps_at_the_old_ties(refined):
    f = [bc.rydberg_shift_for_state(f"{n}f", refined, 300.0).shift_dyn_hz for n in (33, 34, 35)]
    g = [bc.rydberg_shift_for_state(f"{n}g", refined, 300.0).shift_dyn_hz for n in (33, 34, 35)]
    for series in (f, g):
        assert series[0] < series[1] < series[2] and series[2] - series[0] < 30   # 2374 -> 2345 before


def test_fine_structure_g_states_are_reliable_and_continuous_with_the_collapsed_ones(refined):
    a = bc.rydberg_shift_for_state("20g7/2", refined, 300.0)
    b = bc.rydberg_shift_for_state("21g", refined, 300.0)
    assert a.dyn_reliable and b.dyn_reliable
    assert abs(a.shift_dyn_hz - b.shift_dyn_hz) < 60


def test_static_formula_is_flagged_where_it_does_not_apply(refined):
    g = bc.rydberg_shift_for_state("33g", refined, 300.0)
    assert not g.static_valid and "perturbation" in g.static_note and g.max_coupling > 10
    low = bc.rydberg_shift_for_state("7s", refined, 300.0, do_dyn=False)
    assert low.static_valid and low.static_note == ""
    ryd = bc.rydberg_shift_for_state("25s", refined, 300.0, do_dyn=False)
    assert not ryd.static_valid and "k_BT" in ryd.static_note


def test_validity_rule_on_a_hand_built_network():
    ok = [dict(delta_E_au=0.08, fosc=0.7, sign=1, other_label="3p")]
    assert bc.static_validity(ok, 300.0, 0)[0]
    tiny_gap = [dict(delta_E_au=1e-8, fosc=0.5, sign=1, other_label="4g")]
    valid, note, ratio = bc.static_validity(tiny_gap, 300.0, 3)
    assert not valid and ratio > bc.STATIC_COUPLING_MAX and "perturbation" in note


def test_blackbody_repair_also_covers_a_transitions_file_made_before_the_merged_j_fix(data):
    # files written before transitions_core tested every J of a merged level lack f(5/2,7/2) -> g9/2
    stale = dict(data)
    stale["transitions"] = [t for t in data["transitions"]
                            if {t["upper_label"], t["lower_label"]} != {"17f5/2", "18g9/2"}]
    fixed = bc.refine_series_data(stale)
    assert any({t["upper_label"], t["lower_label"]} == {"17f5/2", "18g9/2"} for t in fixed["transitions"])
    assert fixed["_series_refined"]["n_merged"] == 1
    assert bc.refine_series_data(data)["_series_refined"]["n_merged"] == 0     # current file: nothing to add


def test_f_series_is_smooth_across_the_merged_j_range(refined):
    vals = [bc.rydberg_shift_for_state(f"{n}f5/2", refined, 300.0).shift_dyn_hz for n in (15, 16, 17, 18, 19)]
    assert all(b > a for a, b in itertools.pairwise(vals))      # 17f5/2 was 490 Hz between 1823 and 2130
    assert all(1800 < v < 2300 for v in vals)
