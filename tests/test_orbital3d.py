"""Orbital viewer: radial functions, the state list and the page payload."""
from __future__ import annotations

import json
import math
import re
import shutil
import sys
from pathlib import Path

import pytest

from linak.core.paths import reference_path
from linak.physics import orbital_radial as orb
from linak.pipelines.orbital3d import collect_states, radial_payload


def _rydberg(el):
    return json.loads(reference_path("data_json", f"{el}_rydberg.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("n,l,j", [(3, 0, 0.5), (3, 1, 1.5), (4, 2, 2.5), (10, 0, 0.5), (8, 3, 3.5)])
def test_model_state_has_the_right_nodes_and_norm(n, l, j):
    _, r, u = orb.model_state("Na", n, l, j)
    assert len(orb.node_radii(r, u)) == n - l - 1
    assert abs(orb.expectation(r, u, 0) - 1.0) < 1e-6


@pytest.mark.parametrize("el", ["Na", "Cs", "Fr"])
def test_model_eigenvalue_reproduces_the_s_and_p_levels(el):
    ry = _rydberg(el)
    states = [s for s in collect_states(ry, 9) if s["l_int"] <= 1]
    rows = radial_payload({"symbol": el, "Z": {"Na": 11, "Cs": 55, "Fr": 87}[el], "charge": 0}, 1.0, states, "model")
    for st, row in zip(states, rows):
        assert row["method"] == "model"
        assert len(row["nodes"]) == st["n"] - st["l_int"] - 1
        assert abs(row["nu"] - st["nstar"]) < 1e-2, st["label"]


def test_fine_structure_partners_differ_and_follow_the_levels():
    ry = _rydberg("Na")
    st = {s["label"]: s for s in collect_states(ry, 4)}
    a, b = st["3p1/2"], st["3p3/2"]
    assert b["nstar"] - a["nstar"] > 5e-4
    rows = radial_payload({"symbol": "Na", "Z": 11, "charge": 0}, 1.0, [a, b], "model")
    split_model, split_lvl = rows[1]["nu"] - rows[0]["nu"], b["nstar"] - a["nstar"]
    assert abs(split_model - split_lvl) < 0.15 * split_lvl


def test_ground_state_is_listed_first_with_zero_energy():
    states = collect_states(_rydberg("Na"), 5)
    assert states[0]["label"] == "3s" and states[0]["ground"] and states[0]["energy"] == 0.0
    assert abs(states[0]["nstar"] - 1.6271) < 1e-3
    assert abs(states[1]["energy"] - 2.1023) < 1e-3         # 3p1/2 above the ground state


def test_mean_radius_approaches_the_hydrogenic_value_for_high_l():
    d = orb.state_radial("Na", 11, 0, 1.0, 9, 3, 3.5, 8.9983)
    assert d["method"] == "model"
    hyd = (3 * 8.9983**2 - 12) / 2
    assert abs(d["r1"] / hyd - 1) < 5e-3


def test_hydrogenic_values_are_exact():
    d = orb.state_radial("H", 1, 0, 1.0, 3, 1, 1.5, 3.0)
    assert d["method"] == "hydrogenic"
    assert abs(d["r1"] - 12.5) < 1e-4 and abs(d["r2"] - 180.0) < 1e-2
    assert abs(d["rm1"] - 1 / 9) < 1e-6
    assert abs(d["rm3"] - 1 / (27 * 1 * 1.5 * 2)) < 1e-6
    assert len(d["nodes"]) == 1


def test_coulomb_fallback_without_model_parameters():
    # Fr d is not fitted, and --radial coulomb forces the fallback everywhere
    d = orb.state_radial("Fr", 87, 0, 1.0, 7, 2, 1.5, 4.66)
    assert d["method"] == "coulomb" and d["rc"] > 0
    d = orb.state_radial("Na", 11, 0, 1.0, 10, 2, 2.5, 9.9855, prefer="coulomb")
    assert d["method"] == "coulomb"
    assert abs(d["r1"] / ((3 * 9.9855**2 - 6) / 2) - 1) < 5e-3


def test_core_radius_holds_the_core_electrons():
    assert 1.4 < orb.core_radius("Na", 0) < 2.0
    assert 2.2 < orb.core_radius("Cs", 0) < 3.0


def test_core_penetration_falls_as_n_star_cubed_for_s_states():
    ps = [(ns, orb.state_radial("Na", 11, 0, 1.0, n, 0, 0.5, ns)["p_core"]) for n, ns in ((8, 6.6506), (12, 10.6514))]
    (n1, p1), (n2, p2) = ps
    assert abs(math.log(p1 / p2) / math.log(n2 / n1) - 3) < 0.15


@pytest.fixture()
def na_workspace(tmp_path):
    sys.path.insert(0, str(Path(__file__).parent))
    from helpers import make_workspace

    ws = make_workspace(tmp_path / "ws")
    data = ws / ".linak" / "data"
    data.mkdir(parents=True)
    src = reference_path("data_json", "Na_rydberg.json").parent
    for f in src.glob("Na_*.json"):
        shutil.copy(f, data / f.name)
    return ws


@pytest.mark.parametrize("render", ["volume", "isosurface"])
def test_page_payload(na_workspace, render):
    from helpers import run_pipeline

    r = run_pipeline(["orbital3d", "Na", "--n-max", "6", "--render", render], na_workspace)
    assert r.returncode == 0, r.stderr
    page = na_workspace / ".linak" / "plots" / "Na" / "Na_orbital3d.html"
    html = page.read_text(encoding="utf-8")
    assert (page.parent / "plotly.min.js").is_file() and 'src="plotly.min.js"' in html
    assert "$!" not in html and "cdn.plot.ly" not in html
    D = json.loads(re.search(r"const D=(\{.*?\});\n", html).group(1))
    assert D["render"] == render and D["states"][0]["lbl"] == "3s"
    assert all(s["method"] == "model" and len(s["nodes"]) == s["n"] - s["lint"] - 1 for s in D["states"])
    assert len(html) < 300_000
