# -*- coding: utf-8 -*-
"""Na D-line (3p) lifetime regression."""
from __future__ import annotations

from helpers import write_result_summary


def test_dline_lifetimes_match_anchor(lifetime_by_state, anchors, results_dir):
    block = anchors["dline_lifetime"]
    summary = {}
    for state, exp in block["states"].items():
        row = lifetime_by_state.get(state)
        assert row is not None, f"missing lifetime row for {state}"
        got_ns = float(row["lifetime_ns"])
        got_A = float(row["A_total_s"])
        exp_ns = float(exp["lifetime_ns"])
        exp_A = float(exp["A_total_s"])
        rel_ns = abs(got_ns - exp_ns) / abs(exp_ns)
        rel_A = abs(got_A - exp_A) / abs(exp_A)
        assert rel_ns <= block["tol_rel"], f"{state} tau: {got_ns} vs {exp_ns}"
        assert rel_A <= block["tol_rel"], f"{state} A: {got_A} vs {exp_A}"
        # Consistency: tau = 1/A
        assert abs(got_ns * 1e-9 * got_A - 1.0) < 1e-6
        summary[state] = {"lifetime_ns": got_ns, "A_total_s": got_A, "rel_ns": rel_ns}
    write_result_summary(results_dir, "dline_lifetime", summary)


def test_dline_near_experimental_scale(lifetime_by_state, anchors):
    """Both fine-structure components should sit near the accepted ~16.2 ns scale."""
    block = anchors["dline_lifetime"]
    expt = float(block["expt_lifetime_ns"])
    for state in block["states"]:
        got = float(lifetime_by_state[state]["lifetime_ns"])
        rel = abs(got - expt) / abs(expt)
        assert rel <= block["tol_rel_expt"], f"{state} vs expt scale: {got} vs {expt}"
