# -*- coding: utf-8 -*-
"""Na BBR continuum PI / SFI / mixing and high-n dyn-shift anchors."""
from __future__ import annotations

from helpers import write_result_summary


def _sanity_map(na_blackbody: dict) -> dict:
    cont = na_blackbody["continuum"]
    return {row["state"]: row for row in cont["sanity"]}


def test_continuum_metadata(na_blackbody, anchors):
    cont = na_blackbody["continuum"]
    a = anchors["continuum_sanity"]
    assert float(cont["T_K"]) == a["T_K"]
    assert float(cont["field_V_cm"]) == a["field_V_cm"]
    assert bool(cont["mix"]) is a["mix"]


def test_pi_mix_sanity_anchors(na_blackbody, anchors, results_dir):
    a = anchors["continuum_sanity"]
    got_map = _sanity_map(na_blackbody)
    summary = {}
    keys = (
        "Gamma_PI_s",
        "Gamma_SFI_s",
        "Gamma_mix_PI_s",
        "Gamma_mix_SFI_s",
        "Gamma_ion_s",
    )
    for state, exp in a["states"].items():
        row = got_map[state]
        summary[state] = {}
        for key in keys:
            got = float(row[key])
            ref = float(exp[key])
            rel = abs(got - ref) / max(abs(ref), 1e-30)
            assert rel <= a["tol_rel"], f"{state}.{key}: {got} vs {ref}"
            summary[state][key] = {"got": got, "anchor": ref, "rel_err": rel}
        # Ion total should be the sum of the four channels
        parts = sum(float(row[k]) for k in keys if k != "Gamma_ion_s")
        assert abs(parts - float(row["Gamma_ion_s"])) / max(parts, 1e-30) < 1e-5
    write_result_summary(results_dir, "pi_mix_sanity", summary)


def test_mix_dominates_direct_pi_at_10s(na_blackbody):
    """Sanity physics: at 10s with field, mix_PI > direct PI (Beterov / Log)."""
    row = _sanity_map(na_blackbody)["10s"]
    assert float(row["Gamma_mix_PI_s"]) > float(row["Gamma_PI_s"])


def test_direct_pi_dominates_mix_at_20s(na_blackbody):
    """At 20s, direct PI should still exceed mix_PI after Numerov A updates."""
    row = _sanity_map(na_blackbody)["20s"]
    assert float(row["Gamma_PI_s"]) > float(row["Gamma_mix_PI_s"])


def test_farley_wing_and_dyn_shifts(na_blackbody, anchors, results_dir):
    rs = na_blackbody["rydberg_shifts"]
    a = anchors["rydberg_dyn"]
    fw = float(rs["farley_wing_hz"])
    rel_fw = abs(fw - a["farley_wing_hz"]) / a["farley_wing_hz"]
    assert rel_fw <= a["tol_fw_rel"]

    by_state = {row["state"]: row for row in rs["states"]}
    summary = {"farley_wing_hz": fw}
    for state, exp in a["states"].items():
        got = float(by_state[state]["shift_dyn_hz"])
        ref = float(exp["shift_dyn_hz"])
        rel = abs(got - ref) / abs(ref)
        assert rel <= exp["tol_rel"], f"{state} dyn: {got} vs {ref}"
        summary[state] = {"shift_dyn_hz": got, "anchor": ref, "rel_err": rel}
    write_result_summary(results_dir, "rydberg_dyn", summary)
