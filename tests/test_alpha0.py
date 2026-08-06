# -*- coding: utf-8 -*-
"""Na ground-state static polarizability alpha(0) regression."""
from __future__ import annotations

from alpha_core import alpha0_static, load_cached_transitions

from helpers import write_result_summary


def test_alpha0_json_matches_anchor(na_polarizability, anchors, results_dir):
    a = anchors["alpha0"]
    got = float(na_polarizability["alpha0_au"])
    exp = float(a["alpha0_au"])
    rel = abs(got - exp) / abs(exp)
    assert rel <= a["tol_rel_pipeline"], f"alpha0_au drift: {got} vs {exp}"
    write_result_summary(
        results_dir,
        "alpha0_json",
        {"got": got, "anchor": exp, "rel_err": rel},
    )


def test_alpha0_near_experiment(na_polarizability, anchors):
    a = anchors["alpha0"]
    got = float(na_polarizability["alpha0_au"])
    expt = float(a["alpha0_expt_au"])
    rel = abs(got - expt) / abs(expt)
    assert rel <= a["tol_rel_expt"], f"alpha0 vs expt: {got} vs {expt}"


def test_alpha0_from_cached_network(data_dir, anchors, results_dir):
    """Recompute alpha(0) from stored transition network (code path check)."""
    a = anchors["alpha0"]
    cached = load_cached_transitions("Na", data_dir=str(data_dir))
    assert cached is not None
    got = float(alpha0_static(cached["transitions_gs"]))
    exp = float(a["alpha0_au"])
    rel = abs(got - exp) / abs(exp)
    assert rel <= a["tol_rel_pipeline"], f"SOS alpha0: {got} vs {exp}"
    write_result_summary(
        results_dir,
        "alpha0_sos",
        {"got": got, "anchor": exp, "rel_err": rel, "n_transitions": len(cached["transitions_gs"])},
    )


def test_C6_json_matches_anchor(na_polarizability, anchors):
    a = anchors["alpha0"]
    got = float(na_polarizability["C6_au"])
    exp = float(a["C6_au"])
    rel = abs(got - exp) / abs(exp)
    assert rel <= a["tol_rel_pipeline"]
    expt = float(a["C6_expt_au"])
    assert abs(got - expt) / abs(expt) <= 0.01
