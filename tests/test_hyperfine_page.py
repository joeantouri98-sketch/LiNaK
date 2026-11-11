# -*- coding: utf-8 -*-
"""The Hyperfine / Zeeman stage's tabbed page and the measured atomic g factors it uses."""
from __future__ import annotations

import json

import pytest

from helpers import load_json, make_workspace, run_pipeline


def _payload(html: str) -> dict:
    start = html.index("const D=") + len("const D=")
    end = html.index(";\nconst $=", start)
    return json.loads(html[start:end])


@pytest.fixture(scope="module")
def rb87(tmp_path_factory):
    root = make_workspace(tmp_path_factory.mktemp("hf") / "ws")
    result = run_pipeline(["hyperfine", "Rb", "--isotope", "87", "--B-max", "10", "--B-points", "201"], root)
    assert result.returncode == 0, result.stdout + result.stderr
    page = root / ".linak" / "plots" / "Rb" / "Rb_hyperfine.html"
    return root, result.stdout, page.read_text(encoding="utf-8")


def test_one_tabbed_page_replaces_the_zeeman_page(rb87):
    root, _out, html = rb87
    for tab in ("pane-lv", "pane-br", "pane-tr", "pane-co"):
        assert f'id="{tab}"' in html
    assert not (root / ".linak" / "plots" / "Rb" / "Rb_zeeman.html").exists()
    assert (root / ".linak" / "plots" / "Rb" / "plotly.min.js").is_file()


def test_page_carries_measured_g_factors_and_the_magic_field(rb87):
    _root, out, html = rb87
    D = _payload(html)
    assert D["gI_src"] == "atomic" and "(atomic, shielded)" in out
    gs = next(s for s in D["states"] if s["label"] == "5s1/2")
    assert gs["gJ_src"] == "experimental" and gs["gJ"] == pytest.approx(2.00233113)
    two = next(m for m in D["microwave"] if m["l"] == [1.0, -1.0] and m["u"] == [2.0, 1.0])
    assert [p[0] for p in two["stat"]] == pytest.approx([3.2289], abs=5e-4)
    assert len(D["B"]) == 201 and all(len(v["E"]) == 201 for v in gs["lv"])
    assert {o["upper"] for o in D["optical"]} == {"5p1/2", "5p3/2"}


def test_hyperfine_json_is_unchanged_in_layout(rb87):
    root, _out, _html = rb87
    data = load_json(root / ".linak" / "data" / "Rb_hyperfine.json")
    assert data["special"]["microwave_clock"]["frequency_MHz"] == pytest.approx(6834.682611, abs=1e-6)
    assert [s["label"] for s in data["states"]] == ["5s1/2", "5p1/2", "5p3/2"]


def test_half_integer_F_for_an_integer_spin(tmp_path):
    # 210Fr (I = 6): F = 11/2, 13/2 must not be rounded on the page
    root = make_workspace(tmp_path / "ws")
    result = run_pipeline(["hyperfine", "Fr", "--isotope", "210", "--states", "7s1/2", "--B-max", "50",
                           "--B-points", "21"], root)
    assert result.returncode == 0, result.stdout + result.stderr
    D = _payload((root / ".linak" / "plots" / "Fr" / "Fr_hyperfine.html").read_text(encoding="utf-8"))
    assert sorted({lv["F"] for lv in D["states"][0]["lv"]}) == [5.5, 6.5]
