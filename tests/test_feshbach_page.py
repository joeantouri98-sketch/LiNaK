# -*- coding: utf-8 -*-
"""The Feshbach page: the published a(B) curves (6Li table, Cs fits) against the literature, the
derived physics (thermal σ, dimer energies, s_res, L₃), its browser twin run in node, and the stage."""
from __future__ import annotations

import json
import math
import shutil
import subprocess

import numpy as np
import pytest

from helpers import make_workspace, run_pipeline
from qali.physics import feshbach as fb
from qali.report_templates import load as load_template

NODE = shutil.which("node")
PHYS_JS = load_template("feshbach_physics.js")


def js(expr: str):
    if NODE is None:
        pytest.skip("node is not installed")
    code = PHYS_JS.replace("if (typeof module !== 'undefined') module.exports = FP;", "") + \
        f"\nprocess.stdout.write(JSON.stringify({expr}));"
    out = subprocess.run([NODE, "-e", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def _payload(html: str) -> dict:
    start = html.index("const D=") + len("const D=")
    end = html.index(";\nconst $=", start)
    return json.loads(html[start:end])


@pytest.fixture(scope="module")
def li_inputs(data_dir):
    from qali.pipelines.feshbach import load_inputs
    return load_inputs("Li", str(data_dir))


@pytest.fixture(scope="module")
def cs_inputs(data_dir):
    from qali.pipelines.feshbach import load_inputs
    return load_inputs("Cs", str(data_dir))


# ------------------------------------------------------------------ published a(B)
def test_li6_table_reproduces_the_published_poles_and_zeros(li_inputs):
    """Zürn et al. PRL 110, 135301: poles 832.18, 689.68, 809.76 G; |12> zero crossing 527.5 G
    (O'Hara/Jochim measurements 527-528 G), which the local fits miss (they give 569.9 G)."""
    t12 = li_inputs.published["|1> + |2>"]
    poles, zeros = t12.poles_and_zeros()
    assert min(abs(p - 832.18) for p in poles) < 0.05
    assert min(abs(z - 527.5) for z in zeros) < 0.5
    assert 543.286 in poles and any(abs(z - 543.386) < 1e-9 for z in zeros)   # the narrow line, analytic
    assert min(abs(p - 689.68) for p in li_inputs.published["|1> + |3>"].poles_and_zeros()[0]) < 0.05
    assert min(abs(p - 809.76) for p in li_inputs.published["|2> + |3>"].poles_and_zeros()[0]) < 0.05
    # interpolation in arctan(a/L) keeps the tabulated values and the 1/(B - B0) divergence
    assert float(t12.a(1000.0)) == pytest.approx(-4134.6, abs=0.05)
    assert t12.a(832.0) > 1e5 and t12.a(832.4) < -1e5


def test_cs_fits_match_kraemer_and_jachymski(cs_inputs):
    fc = cs_inputs.published["|3,+3> + |3,+3>"]
    a, br = fc.a_and_flag(np.array([0.0, 20.0, 556.9, 878.43, 300.0]))
    # Kraemer et al. 2006: a(0) = 1722 (1 - 28.72/11.74) = -2490 a0 (the 47.97 G d-wave changes it by < 1 %)
    assert a[0] == pytest.approx(1722 * (1 - 28.72 / 11.74), rel=0.01)
    assert a[1] == pytest.approx(1752.4 * (1 - 28.72 / 31.74), rel=0.02)
    assert abs(a[3]) < 5                            # Jachymski & Julienne: zero at 786.17 + 92.26 G
    assert list(br) == [False, False, False, False, True]   # 300 G lies between the fits
    zeros = fb._product_poles_and_zeros(np.linspace(0, 30, 3001), fc.a(np.linspace(0, 30, 3001)), [-11.74])[1]
    assert zeros[0] == pytest.approx(16.98, abs=0.05)    # Kraemer: 17.0 G (Berninger: 17.12 G)


def test_cs_bridge_is_continuous_at_every_fit_edge(cs_inputs):
    fc = cs_inputs.published["|3,+3> + |3,+3>"]
    for s in fc.segments:
        for b in (s.B_min, s.B_max):
            lo, hi = fc.a(np.array([b - 1e-6, b + 1e-6]))
            assert lo == pytest.approx(hi, rel=1e-4), b


def test_even_l_lines_enter_the_channel_and_are_not_doubled(cs_inputs):
    fc = cs_inputs.published["|3,+3> + |3,+3>"]
    # only the 47.97 G d-wave line is missing from the fits; the others are already in them
    assert fc.extra_poles == ((47.97, 0.16),)


def test_compute_uses_the_published_curve_and_refines_the_grid(cs_inputs):
    r = fb.compute(cs_inputs.resonances, cs_inputs.a_bg_default_a0, cs_inputs.C6_au, cs_inputs.mass_amu,
                   boson=True, published=cs_inputs.published, refine=True, B_points=500)
    ch = r.channels[0]
    assert ch.source == "fit" and ch.bridged is not None and ch.bridged.any()
    near = r.B_G[np.abs(r.B_G - 47.97) < 0.16]
    assert len(near) > 20                           # the 0.16 G line is resolved on a 0-940 G axis


def test_stale_workspace_copy_is_reported(tmp_path, data_dir):
    from qali.pipelines.feshbach import _stale_copy_hint
    assert _stale_copy_hint("Cs", str(tmp_path), {"resonances": []}) is not None
    assert _stale_copy_hint("Cs", str(tmp_path), {"curves": {"x": {}}}) is None


# ------------------------------------------------------------------ derived physics
def test_thermal_factor_matches_the_closed_form():
    from scipy.special import exp1
    for b in (1e-3, 0.3, 1.0, 10.0, 1e3):
        z = 1.0 / b
        exact = (1.0 / b) * (1.0 - z * math.exp(z) * exp1(z)) if z < 600 else 1.0 - 2.0 * b
        assert float(fb.thermal_factor(b)) == pytest.approx(exact, rel=2e-4)
    assert float(fb.thermal_factor(0.0)) == pytest.approx(1.0, rel=1e-9)


def test_thermal_sigma_limits():
    m = 132.905452
    assert float(fb.thermal_sigma_a0(1.0, 1.0, m, True)) == pytest.approx(8 * math.pi, rel=1e-4)
    k = fb.thermal_k_a0(1.0, m)
    assert float(fb.thermal_sigma_a0(1e9, 1.0, m)) == pytest.approx(4 * math.pi / k**2, rel=1e-4)


def test_resonance_strength_matches_chin_table_iv():
    """Chin et al., RMP 82, 1225 (2010) Table IV: s_res 2.1 (39K 402 G), 0.09 (Na 907 G), 59 (6Li 834 G)."""
    assert fb.resonance_strength(-29, -52, 1.5, 61.1, 38.963707) == pytest.approx(2.1, abs=0.05)
    assert fb.resonance_strength(63, 1.0, 3.8, 44.93, 22.98977) == pytest.approx(0.09, abs=0.01)
    assert fb.resonance_strength(-1405, -300, 2.0, 29.9, 6.015122) == pytest.approx(59, abs=1)


def test_binding_energy_universal_and_corrected():
    from qali.core.constants import A0_M, AMU_KG, HBAR_SI, H_SI
    m = 6.015122
    univ = HBAR_SI**2 / (m * AMU_KG * (1000 * A0_M) ** 2) / H_SI
    assert float(fb.binding_energy_hz(1000.0, m)) == pytest.approx(univ, rel=1e-12)
    assert float(fb.binding_energy_hz(1029.69, m, 29.69)) == pytest.approx(univ, rel=1e-12)
    assert math.isnan(float(fb.binding_energy_hz(-10.0, m)))


def test_three_body_coefficient_scaling_and_cap():
    m = 132.905452
    assert float(fb.three_body_L3_cm6_s(200.0, m) / fb.three_body_L3_cm6_s(100.0, m)) == pytest.approx(16.0)
    cap = float(fb.three_body_L3_cm6_s(1e7, m, T_uK=1.0))
    assert float(fb.three_body_L3_cm6_s(2e7, m, T_uK=1.0)) == cap
    # Kraemer 2006 order of magnitude: L3 ~ 1e-25 cm^6/s at a ~ 1000 a0 with C ~ 1 ... 70
    assert 1e-26 < float(fb.three_body_L3_cm6_s(1000.0, m, C=10.0)) < 1e-23


def test_js_twin_matches_python():
    m = 6.015122
    got = js(f"[FP.thermalFactor(0.37),FP.thermalSigma(3000,0.5,{m!r},false),FP.bindingHz(800,{m!r},29.69),"
             f"FP.kF(1e12),FP.TF(1e12,{m!r}),FP.L3(500,132.905452,67.1,1),FP.sRes(-29,-52,1.5,61.1,38.963707)]")
    want = [float(fb.thermal_factor(0.37)), float(fb.thermal_sigma_a0(3000, 0.5, m)),
            float(fb.binding_energy_hz(800, m, 29.69)), fb.fermi_wave_number_a0(1e12), fb.fermi_temperature_K(1e12, m),
            float(fb.three_body_L3_cm6_s(500, 132.905452, 67.1, 1.0)), fb.resonance_strength(-29, -52, 1.5, 61.1, 38.963707)]
    for g, w in zip(got, want):
        assert g == pytest.approx(w, rel=1e-9)


# ------------------------------------------------------------------ the stage
@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    return make_workspace(tmp_path_factory.mktemp("fesh"))


@pytest.mark.parametrize("el", ["Li", "Cs"])
def test_page_carries_every_tab_and_the_thresholds(ws, el):
    res = run_pipeline(["feshbach", el, "--field-unit", "mT"], ws)
    assert res.returncode == 0, res.stdout + res.stderr
    html = (ws / ".qali" / "plots" / el / f"{el}_feshbach.html").read_text(encoding="utf-8")
    d = _payload(html)
    assert d["opts"]["tabs"] == ["a", "cs", "mo", "th", "rs"] and d["opts"]["unit"] == "mT"
    assert d["thr"] and d["thr"]["chs"]
    src = {c["name"]: c["src"] for c in d["ch"]}
    if el == "Li":
        assert d["boson"] is False and set(src.values()) == {"table"}
        assert d["thr"]["lab"][d["thr"]["chs"]["|1> + |2>"][0]] == [0.5, 0.5]   # |1> = |F=1/2, mF=+1/2>
        assert d["thr"]["lab"][d["thr"]["chs"]["|1> + |2>"][1]] == [0.5, -0.5]
    else:
        assert src == {"|3,+3> + |3,+3>": "fit"} and d["ch"][0]["br"]
    out = json.loads((ws / ".qali" / "data" / f"{el}_feshbach_out.json").read_text(encoding="utf-8"))
    assert out["channels"][0]["source"] == src[out["channels"][0]["channel"]]


def test_tabs_can_be_left_out(ws):
    res = run_pipeline(["feshbach", "Na", "--no-tab-thresholds", "--no-tab-molecules"], ws)
    assert res.returncode == 0, res.stdout + res.stderr
    d = _payload((ws / ".qali" / "plots" / "Na" / "Na_feshbach.html").read_text(encoding="utf-8"))
    assert d["opts"]["tabs"] == ["a", "cs", "rs"] and d["thr"] is None
