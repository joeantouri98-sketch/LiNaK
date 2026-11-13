"""Francium: model derived for Qali (ARC has no Fr), checked against all-order matrix elements and NIST levels."""
from __future__ import annotations

import json
import math

import pytest

from qali.core.paths import reference_path
from qali.physics.extrapolation import exact, frlevels
from qali.physics.extrapolation import stage as S
from qali.physics.extrapolation.params import anchor_config, species_params
from qali.physics.extrapolation.tables import table_status

# Roberts, Fairhall & Ginges, PRA 107, 052812 (2023), Table X: <7s||D||nP_J> (a.u.)
ROBERTS = {(7, 0.5): 4.2895, (7, 1.5): 5.9065, (8, 0.5): 0.3041, (8, 1.5): 0.9070}


def test_levels_reproduce_nist_and_extend_the_series():
    atom = frlevels.FrAtom()
    assert atom.getEnergy(7, 1, 0.5) == pytest.approx(-2.555496, abs=1e-6)
    assert atom.getEnergy(7, 0, 0.5) == pytest.approx(-4.0727411, abs=1e-6)
    assert atom.getEnergy(30, 1, 1.5) == pytest.approx(-0.020925, abs=1e-6)
    e40 = atom.getEnergy(40, 1, 0.5)                                    # beyond the NIST list: Ritz extension
    assert -0.0110 < e40 < -0.0107 and atom.getEnergy(41, 1, 0.5) > e40


def test_fr_is_supported_and_its_table_matches_the_settings():
    assert species_params("Fr")["nl"] == 7
    assert table_status("Fr")["fresh"]


def test_exact_model_matches_the_all_order_matrix_elements():
    m = exact.ExactModel("Fr")
    for (n, J), d_ref in ROBERTS.items():
        if n == 7:
            continue                                                   # the resonance lines: below the table
        omega = m.energy_ha(n, 1, J) - m.energy_ha(7, 0, 0.5)
        d_model = math.sqrt(3 * m.f(n, J) / omega)
        assert d_model == pytest.approx(d_ref, rel=0.03), (n, J)       # 8p1/2 is a near-cancellation: 2.6% quoted


def test_np_fine_structure_is_reproduced():
    m = exact.ExactModel("Fr")
    fs = (m.energy_ha(7, 1, 1.5) - m.energy_ha(7, 1, 0.5)) * 219474.63
    m._tune()
    e = [m.bound_state(7, 1, j, m.nu(7, j))[0] for j in (0.5, 1.5)]
    assert (e[1] - e[0]) * 219474.63 == pytest.approx(fs, rel=0.01)


def test_the_unreliable_nist_rows_are_left_out_and_the_precision_anchors_used(tmp_path):
    from qali.physics.polarizability_core import load_nist_fvalues

    nist = load_nist_fvalues("Fr", verbose=False)
    ryd = json.loads(reference_path("data_json", "Fr_rydberg.json").read_text(encoding="utf-8"))
    anchors = S.collect_anchors("Fr", nist, ryd)
    assert any(a["unreliable"] for a in anchors)
    used = [a for a in anchors if not a["unreliable"] and a["n"] > 7]
    assert {(a["n"], a["J"]) for a in used} == {(8, 0.5), (8, 1.5)}
    assert all(a["source"] == "precision" for a in used)
    assert not any(a["unreliable"] for a in anchors if a["n"] == 7)    # the resonance line is fine (below the table)
    cfg = S.ExtrapolationConfig(mode="show", n_max=30)
    res = S.run_for_species("Fr", cfg, nist, ryd)
    assert any("left out of the fit" in w for w in res["warnings"])
    for jstr, ch in res["channels"].items():
        assert abs(ch["c0"]) < 0.05                                    # the model already agrees with the anchors
        rows = {r["n"]: r for r in ch["rows"]}
        assert rows[30]["lo68"] < rows[30]["f"] < rows[30]["hi68"]
        assert rows[30]["hi68"] / rows[30]["lo68"] > 1.2              # honest band: no anchor above 8p


def test_ticking_an_unreliable_row_changes_nothing_and_it_stays_excluded():
    from qali.physics.polarizability_core import load_nist_fvalues

    nist = load_nist_fvalues("Fr", verbose=False)
    ryd = json.loads(reference_path("data_json", "Fr_rydberg.json").read_text(encoding="utf-8"))
    res = S.run_for_species("Fr", S.ExtrapolationConfig(mode="show", n_max=30, excluded=[]), nist, ryd)
    flagged = [a for a in res["anchors"] if a["unreliable"]]
    assert flagged and not any(a["included"] for a in flagged)         # the dialog disables these boxes


def test_other_species_keep_their_nist_rows():
    assert "Fr" in anchor_config()["precision_in_stage"] and "Cs" not in anchor_config()["precision_in_stage"]
