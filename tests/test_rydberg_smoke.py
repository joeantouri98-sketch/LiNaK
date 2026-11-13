# -*- coding: utf-8 -*-
"""End-to-end smoke test for the ``rydberg`` pipeline and the Rydberg formula.

Runs the full ``qali --pipeline rydberg Na`` CLI against an isolated test
workspace (subprocess, like test_pipeline_smoke.py), validates its JSON output
and anchors the series against

    E = E_ion - Ry_eff / (n - delta_l)^2

Energy convention: the pipeline stores ionization-relative energies, i.e.
the series limit is E_ion = 0 and every bound member has E < 0; the absolute
energy above the ground state is therefore IE + E(n).
"""
from __future__ import annotations

from itertools import pairwise

import pytest

from helpers import load_json, run_pipeline, write_result_summary


def _na_rydberg_output(workspace):
    """The pipeline writes <workspace>/.qali/data/Na_rydberg.json."""
    return workspace / ".qali" / "data" / "Na_rydberg.json"


def test_rydberg_na_end_to_end_writes_finite_json(workspace):
    """The CLI run must exit 0 and emit finite, well-formed JSON."""
    result = run_pipeline(["rydberg", "Na"], workspace)
    assert result.returncode == 0, f"rydberg Na failed:\n{result.stderr}"

    out = _na_rydberg_output(workspace)
    assert out.is_file(), "rydberg did not write Na_rydberg.json"

    payload = load_json(out)
    assert payload, "rydberg output JSON is empty"

    # Series metadata sanity: Na ground state is 3s, neutral species.
    assert payload["n_start"] == 3
    assert payload["n_max"] == 50
    assert payload["z_core"] == 1
    # Literature Na ionization energy: 5.139076 eV (NIST ASD).
    assert 5.13 < payload["ionization_energy_eV"] < 5.15
    # Effective Rydberg for a neutral: the ordinary Ry constant (eV).
    assert payload["rydberg_constant_eV"] == pytest.approx(13.6057, rel=1e-4)


def test_rydberg_members_lie_below_the_ionisation_limit(workspace):
    """Every emitted state must be bound: E < 0 in the ionization-relative
    convention, i.e. strictly below the ionisation energy (series limit 0)."""
    result = run_pipeline(["rydberg", "Na"], workspace)
    assert result.returncode == 0, f"rydberg Na failed:\n{result.stderr}"

    payload = load_json(_na_rydberg_output(workspace))
    excitations = payload["excitations"]
    assert len(excitations) == payload["num_states"] > 0

    for state in excitations:
        assert state["energy_eV"] < 0.0, f"{state['label']} is not bound"


def test_rydberg_qdt_states_follow_rydberg_formula(workspace):
    """QDT-model members satisfy E = -Ry_eff/(n - delta_l(n))^2 exactly, with the Ritz
    delta_l(n) where it was fitted and the constant delta_l otherwise (the series limit
    E_ion = 0 in this ionization-relative convention)."""
    from qali.physics.rydberg import ritz_delta
    result = run_pipeline(["rydberg", "Na"], workspace)
    assert result.returncode == 0, f"rydberg Na failed:\n{result.stderr}"

    payload = load_json(_na_rydberg_output(workspace))
    ry_eff = payload["rydberg_constant_eV"]
    quantum_defects = payload["quantum_defects"]
    ritz = payload["quantum_defects_ritz"]

    qdt_states = [s for s in payload["excitations"] if s["source"] == "QDT model"]
    # Emitted energies are rounded to 6 dp; allow that rounding.
    n_checked = 0
    for state in qdt_states:
        delta = ritz_delta(ritz, state["l"], state["n"], quantum_defects[state["l"]])
        expected = -ry_eff / (state["n"] - delta) ** 2
        assert state["energy_eV"] == pytest.approx(expected, abs=2e-6)
        n_checked += 1
    assert n_checked >= 10, "unexpectedly few QDT-model states for Na"


@pytest.mark.parametrize("l_char", ["s", "p", "d", "f", "g"])
def test_rydberg_binding_energy_decreases_with_n(workspace, l_char):
    """For each orbital the mean energy rises monotonically towards the
    series limit with n (equivalently: the binding energy |E| decreases)."""
    result = run_pipeline(["rydberg", "Na"], workspace)
    assert result.returncode == 0, f"rydberg Na failed:\n{result.stderr}"

    payload = load_json(_na_rydberg_output(workspace))
    by_n = {}
    for state in payload["excitations"]:
        if state["l"] == l_char:
            by_n.setdefault(state["n"], []).append(state["energy_eV"])

    assert by_n, f"no {l_char}-series members in the Na output"
    mean_energy_per_n = [sum(energy) / len(energy) for _, energy in sorted(by_n.items())]
    assert all(
        later > earlier
        for earlier, later in pairwise(mean_energy_per_n)
    ), f"{l_char}-series energies are not monotonically increasing with n"


def test_rydberg_d_line_wavelength_anchor(workspace, results_dir):
    """Ground->3p members must land on the Na D lines (~589 nm vacuum)."""
    result = run_pipeline(["rydberg", "Na"], workspace)
    assert result.returncode == 0, f"rydberg Na failed:\n{result.stderr}"

    payload = load_json(_na_rydberg_output(workspace))
    excitations = payload["excitations"]

    # Both fine-structure components are NIST-resolved for Na.
    nist = [s for s in excitations if s["source"] == "NIST"]
    assert nist, "no NIST-resolved states in the Na output"

    d_lines = [
        s["wavelength_nm"]
        for s in excitations
        if s["n"] == 3 and s["l"] == "p"
    ]
    assert d_lines, "no 3p members in the Na output"
    for wl in d_lines:
        assert 588.5 < wl < 590.5, f"3p wavelength {wl} nm off the D lines"

    write_result_summary(
        results_dir,
        "rydberg_smoke",
        {
            "ionization_energy_eV": payload["ionization_energy_eV"],
            "n_start": payload["n_start"],
            "num_states": payload["num_states"],
            "num_nist": sum(1 for s in excitations if s["source"] == "NIST"),
            "num_qdt_model": sum(1 for s in excitations if s["source"] == "QDT model"),
            "quantum_defects": payload["quantum_defects"],
            "d_line_wavelengths_nm": d_lines,
        },
    )
