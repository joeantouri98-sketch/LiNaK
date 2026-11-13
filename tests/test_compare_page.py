# -*- coding: utf-8 -*-
"""The Compare page: heteronuclear C₆ by Casimir–Polder, tune-out wavelengths, the browser twin
of the laser-cooling and α(λ) physics (run in node), and the stage that writes the page."""
from __future__ import annotations

import json
import math
import shutil
import subprocess

import numpy as np
import pytest

from helpers import make_workspace, run_pipeline
from linak.physics import laser_cooling as lc
from linak.physics.polarizability_core import alpha_real, omega_au_from_nm
from linak.pipelines.polarizability import compute_C6, compute_C6_pair, tune_out_wavelengths
from linak.report_templates import load as load_template

NODE = shutil.which("node")
PHYS_JS = load_template("compare_physics.js")


def js(expr: str, data=None):
    if NODE is None:
        pytest.skip("node is not installed")
    code = PHYS_JS.replace("if (typeof module !== 'undefined') module.exports = CP;", "") + \
        f"\nconst DATA={json.dumps(data)};\nprocess.stdout.write(JSON.stringify({expr}));"
    out = subprocess.run([NODE, "-e", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def _payload(html: str) -> dict:
    start = html.index("const D=") + len("const D=")
    end = html.index(";\nconst $=", start)
    return json.loads(html[start:end])


@pytest.fixture(scope="module")
def trans(data_dir):
    out = {}
    for el in ("Li", "Na", "K", "Rb", "Cs"):
        with (data_dir / f"{el}_polarizability.json").open(encoding="utf-8") as f:
            out[el] = json.load(f)["transitions_gs"]
    return out


# ------------------------------------------------------------------ C6 and tune-outs
def test_pair_c6_reduces_to_the_homonuclear_one(trans):
    assert compute_C6_pair(trans["Rb"], trans["Rb"]) == pytest.approx(compute_C6(trans["Rb"]), rel=1e-9)
    assert compute_C6_pair(trans["K"], trans["Rb"]) == pytest.approx(compute_C6_pair(trans["Rb"], trans["K"]), rel=1e-9)


def test_pair_c6_against_two_oscillators_in_closed_form():
    """One oscillator per atom: C₆ = (3/2) α_A α_B ω_A ω_B/(ω_A + ω_B) (London)."""
    a = [dict(delta_E_au=0.07, fosc=0.9, sign=1)]
    b = [dict(delta_E_au=0.05, fosc=1.1, sign=1)]
    aA, aB = 0.9 / 0.07 ** 2, 1.1 / 0.05 ** 2
    assert compute_C6_pair(a, b) == pytest.approx(1.5 * aA * aB * 0.07 * 0.05 / 0.12, rel=1e-5)


def test_heteronuclear_c6_tracks_derevianko(trans):
    """Casimir–Polder from LiNaK's own α(iω), RPA core included, is within 0.5 % of Derevianko et
    al. (2010) for the pairs without Li (Li's own C₆ is 2 % low: its valence sum lacks the continuum)."""
    for (a, b), ref in {("Na", "K"): 2447, ("K", "Rb"): 4274, ("Rb", "Cs"): 5663, ("Na", "Cs"): 3227}.items():
        assert compute_C6_pair(trans[a], trans[b]) == pytest.approx(ref, rel=0.005), (a, b)
    assert compute_C6_pair(trans["Li"], trans["Na"]) == pytest.approx(1467, rel=0.02)


def test_tune_outs_are_zeros_between_neighbouring_resonances(trans):
    to = tune_out_wavelengths(trans["Rb"], 300, 2000)
    lam = [t[0] for t in to]
    assert lam == sorted(lam, reverse=True)
    d1d2 = next(t for t in to if t[1:] == ("5p1/2", "5p3/2"))
    # Leonard et al. 2015: 790.032388(32) nm (87Rb); the bundled f-values land within 5 pm
    assert d1d2[0] == pytest.approx(790.032388, abs=0.005)
    dE = np.array([t["delta_E_au"] for t in trans["Rb"]])
    sf = np.array([t.get("sign", 1) * t["fosc"] for t in trans["Rb"]])
    for l, *_ in to[:4]:
        w = omega_au_from_nm(l)
        assert abs(np.sum(sf / (dE ** 2 - w ** 2))) < 1e-6
    k = next(t for t in tune_out_wavelengths(trans["K"]) if t[1:] == ("4p1/2", "4p3/2"))
    assert k[0] == pytest.approx(768.9712, abs=0.005)            # Holmgren et al. 2012


def test_fr_c6_reference_is_derevianko():
    from linak.core.constants import C6_EXPT
    assert C6_EXPT["Fr"] == 5256.0                                 # was an unsourced 5500


# ------------------------------------------------------------------ browser twin
def test_js_twin_matches_python(trans):
    G, wl, m = 2 * math.pi * 6.0666e6, 780.24, 86.909
    got = js(f"[CP.rate({G},1.3,{-0.7 * G}),CP.force(0.2,{G},0.5,{-0.5 * G},{wl}),CP.temp({G},0.4,{-0.8 * G}),CP.Trec({wl},{m}),"
             f"CP.isatCyc({G},{wl}),CP.amax({G},{wl},{m}),CP.slower(300,{G},{wl},{m},0.5),CP.alpha(1064,DATA),CP.alpha(532,DATA)]",
             [[t["delta_E_au"], t.get("sign", 1) * t["fosc"]] for t in trans["Rb"]])
    want = [float(lc.scattering_rate(G, 1.3, -0.7 * G)), float(lc.molasses_force(0.2, G, 0.5, -0.5 * G, wl)),
            float(lc.doppler_temperature(G, 0.4, -0.8 * G)), lc.recoil_temperature_K(wl, m), lc.isat_cycling_mW_cm2(G, wl),
            lc.max_acceleration(G, wl, m), lc.slower_length(300, G, wl, m, 0.5),
            alpha_real(omega_au_from_nm(1064), trans["Rb"]), alpha_real(omega_au_from_nm(532), trans["Rb"])]
    for g, w in zip(got, want):
        assert g == pytest.approx(w, rel=1e-9)


# ------------------------------------------------------------------ the stage
@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    return make_workspace(tmp_path_factory.mktemp("cmp"))


def test_compare_stage_writes_the_page_and_the_c6_matrix(ws):
    res = run_pipeline(["compare_elements", "--elements", "Na", "K", "Rb"], ws)
    assert res.returncode == 0, res.stdout + res.stderr
    html = (ws / ".linak" / "plots" / "comparison.html").read_text(encoding="utf-8")
    d = _payload(html)
    assert d["opts"]["tabs"] == ["dl", "lc", "pol", "c6", "tr", "acc"]
    els = {r["element"] for r in d["rows"]}            # added to the saved table's elements
    assert {"Na", "K", "Rb"} <= els and set(d["pol"]) <= els
    n = len(d["pol"])
    assert len(d["c6"]) == n * (n + 1) // 2 and all(p[4] for p in d["c6"])   # every pair has a reference
    qs = {r["q"] for r in d["acc"]}
    assert {"τ D1", "τ D2", "α(0)", "C₆", "C₆ pair", "tune-out"} <= qs
    out = json.loads((ws / ".linak" / "data" / "comparison_table.json").read_text(encoding="utf-8"))
    assert len(out["C6_au"]) == n * (n + 1) // 2 and "Rb" in out["tune_out_nm"]


def test_scattering_rate_opens_the_laser_cooling_tab(ws):
    res = run_pipeline(["scattering_rate", "--element", "Rb", "--line", "D1"], ws)
    assert res.returncode == 0, res.stdout + res.stderr
    d = _payload((ws / ".linak" / "plots" / "comparison.html").read_text(encoding="utf-8"))
    assert d["opts"]["tab"] == "lc" and d["opts"]["el"] == "Rb" and d["opts"]["line"] == "D1"


def test_tabs_can_be_left_out_and_the_page_skipped(ws):
    res = run_pipeline(["compare_elements", "--elements", "Na", "--no-tab-c6", "--no-tab-trends"], ws)
    assert res.returncode == 0, res.stdout + res.stderr
    d = _payload((ws / ".linak" / "plots" / "comparison.html").read_text(encoding="utf-8"))
    assert d["opts"]["tabs"] == ["dl", "lc", "pol", "acc"]
    page = ws / ".linak" / "plots" / "comparison.html"
    page.unlink()
    assert run_pipeline(["compare_elements", "--elements", "Na", "--no-page"], ws).returncode == 0
    assert not page.exists()
