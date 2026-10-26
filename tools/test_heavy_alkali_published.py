"""K, Rb, Cs: np -> ns/nd channels beyond NIST come from published matrix elements (PUBLISHED rows) and the ARC
model potential scaled to them; lifetimes are checked against the published all-order tables, not against ARC.

References: Rb  Safronova, Safronova & Clark, PRA 83, 052508 (2011), Table IV (present column)
            Cs  Safronova, Safronova & Clark, PRA 94, 012505 (2016), Table II (recommended)
            K   Safronova & Safronova, PRA 78, 052504 (2008), Table V (tau(SD))
Before this tier the 5p/6p/4p -> ns/nd channels were dropped (n*<=2.5 gate) and ns lifetimes were 1.7-2.1x too long.
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PUB_NS = {
    "Rb": {"7s": 88.3, "8s": 161.9, "9s": 271.7, "10s": 426.0},
    "Cs": {"7s": 48.4, "8s": 93.0, "9s": 167.0, "10s": 279.0, "11s": 434.0},
    "K": {"7s": 148.8, "8s": 250.7, "9s": 393.6, "10s": 584.7, "11s": 828.9},
}


def _load(el):
    p = ROOT / "data_json" / f"{el}_lifetimes.json"
    if not p.exists():
        pytest.skip(f"{el}_lifetimes.json not generated")
    return json.load(open(p, encoding="utf-8"))


@pytest.mark.parametrize("el,state", [(e, s) for e, d in PUB_NS.items() for s in d])
def test_ns_lifetime_matches_published(el, state):
    st = {s["state"]: s for s in _load(el)["lifetimes"]}
    if state not in st:
        pytest.skip(f"{el} {state} not in table")
    assert abs(st[state]["lifetime_ns"] / PUB_NS[el][state] - 1.0) < 0.06


@pytest.mark.parametrize("el", ["K", "Rb", "Cs"])
def test_np_to_ns_nd_channels_not_dropped(el):
    fam = _load(el)["dropped_transitions"].get("by_family", {})
    bad = [k for k in fam if k.split("->")[0][1:2] == "p" and k.split("->")[1][1:] in ("ns", "nd")]
    assert not bad, bad


@pytest.mark.parametrize("el", ["K", "Rb", "Cs"])
def test_literature_rows_cite_a_source_and_never_override_nist(el):
    lit = json.load(open(ROOT / "f_values" / f"{el}_literature.json", encoding="utf-8"))
    rows = [t for t in lit["transitions"]]
    assert rows and all(str(t["ref"]).startswith(("PUBLISHED", "ARC ")) for t in rows)
    lt = _load(el)
    nist_pairs = {(t["lower_label"][:2], f"{t['upper_n']}{t['upper_l']}") for t in lt["transitions"]
                  if t["fosc_source"] in ("NIST", "precision")}
    used = {(t["lower_label"][:2], f"{t['upper_n']}{t['upper_l']}") for t in lt["transitions"]
            if t["fosc_source"] == "literature"}
    assert used and not (used & nist_pairs)


def test_fr_fit_exclusion_leaves_nist_value_untouched():
    lt = _load("Fr")
    row = [t for t in lt["transitions"] if t["lower_label"] == "7p3/2" and t["upper_label"] == "18d5/2"]
    assert row and all(t["fosc_source"] == "NIST" for t in row)


def test_k_4s_np_tail_comes_from_published_table_and_nothing_is_dropped():
    lt = _load("K")
    assert lt["dropped_transitions"]["n_total"] == 0
    lit = json.load(open(ROOT / "f_values" / "K_literature.json", encoding="utf-8"))
    rows = [t for t in lit["transitions"] if t["lower"] == "4s"]
    assert rows and all("Theodosiou" in t["ref"] and t.get("upper_J") for t in rows)
    # NIST 4s->np at n>=10 is 18-32 % below the published series and carries a >=30 % caveat, values unchanged
    cav = [c for c in lit["nist_series_caveats"] if c["lower"] == "4s"]
    assert cav and cav[0]["rel_unc"] >= 0.3
    nist = [t for t in lt["transitions"] if t["lower_label"] == "4s" and t["fosc_source"] == "NIST" and t["upper_n"] >= 10]
    assert nist and all(t["fosc_delta"] >= 0.3 for t in nist)


def test_cs_only_the_negligible_6s_np_half_series_is_dropped():
    fam = _load("Cs")["dropped_transitions"]["by_family"]
    assert set(fam) <= {"6s->np"}
