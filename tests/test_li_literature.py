"""Li high-n ns/nd lifetimes: the 2p->ns/nd channels beyond NIST (n>8/9) come from the curated ARC tier.

Reference: ARC 3.10 Lithium7, T=0 (Marinescu model potential), tau in seconds. Before the curated 2p rows
and the multiplet split, ns lifetimes were 1.5-1.6x too long because 2p->ns (n*(2p)<=2.5, <6 NIST points)
was dropped by the Coulomb gate.
"""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARC_TAU = {"10s": 7.8728e-07, "20s": 6.4442e-06, "30s": 2.2040e-05, "20d": 3.7878e-06, "30d": 1.2715e-05}


@pytest.fixture(scope="module")
def li():
    p = ROOT / "data_json" / "Li_lifetimes.json"
    if not p.exists():
        pytest.skip("Li_lifetimes.json not generated")
    return json.load(open(p, encoding="utf-8"))


@pytest.mark.parametrize("state", sorted(ARC_TAU))
def test_li_high_n_lifetime_matches_arc(li, state):
    st = {s["state"]: s for s in li["lifetimes"]}
    assert abs(st[state]["lifetime_s"] / ARC_TAU[state] - 1.0) < 0.05


def test_li_literature_rows_only_where_nist_has_none():
    lit = json.load(open(ROOT / "f_values" / "Li_literature.json", encoding="utf-8"))
    nist_last = {("2s", "p"): 13, ("2p", "s"): 8, ("2p", "d"): 9}
    for t in lit["transitions"]:
        if not t["ref"].startswith("ARC "):
            continue
        n = int(re.match(r"\d+", t["upper"]).group())
        assert n > nist_last[(t["lower"], t["upper"][-1])] or t["lower"] == "2s"
