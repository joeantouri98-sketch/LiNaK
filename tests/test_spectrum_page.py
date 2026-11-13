# -*- coding: utf-8 -*-
"""The Spectrum page: its physics (against published values), the browser twin of that physics
(run in node and compared with the Python), and the stage that writes the page."""
from __future__ import annotations

import json
import math
import shutil
import subprocess

import numpy as np
import pytest

from helpers import make_workspace, run_pipeline
from qali.core.constants import C_SI
from qali.physics import spectrum_core as sc
from qali.report_templates import load as load_template

NODE = shutil.which("node")
PHYS_JS = load_template("spectrum_physics.js")


def js(expr: str, data=None):
    """Evaluate *expr* (JS, may use SP and DATA) in node and return its JSON value."""
    if NODE is None:
        pytest.skip("node is not installed")
    code = PHYS_JS.replace("if (typeof module !== 'undefined') module.exports = SP;", "") + \
        f"\nconst DATA={json.dumps(data)};\nprocess.stdout.write(JSON.stringify({expr}));"
    out = subprocess.run([NODE, "-e", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def _payload(html: str) -> dict:
    start = html.index("const D=") + len("const D=")
    end = html.index(";\nconst $=", start)
    return json.loads(html[start:end])


# ------------------------------------------------------------------ physics
def test_voigt_is_area_normalised_and_its_integral_is_pi_re_c_f():
    s, g = 3e8, 3e6
    d = np.linspace(-200 * s, 200 * s, 400001)
    assert np.trapezoid(sc.voigt(d, 0.0, s, g), d) == pytest.approx(1.0, abs=2e-3)   # Lorentz tails beyond ±200σ
    sig = sc.cross_section_m2(d, 0.0, 0.64, s, g)
    assert np.trapezoid(sig, d) == pytest.approx(math.pi * sc.R_E * C_SI * 0.64, rel=2e-3)


def test_air_wavelengths_match_nist():
    # NIST ASD Ritz wavelengths of Na D2: vacuum 589.1583264 nm, air 588.9950954 nm
    assert float(sc.vac_to_air_nm(589.1583264)) == pytest.approx(588.9950954, abs=2e-5)
    assert float(sc.vac_to_air_nm(150.0)) == 150.0                       # vacuum below 200 nm


def test_vapour_pressure_matches_steck():
    torr = 133.322368
    # Steck, Rb and Cs D Line Data: 3.92e-7 torr (Rb) and 1.488e-6 torr (Cs) at 25 °C
    assert sc.vapour_pressure_pa("Rb", 298.15) / torr == pytest.approx(3.92e-7, rel=0.03)
    assert sc.vapour_pressure_pa("Cs", 298.15) / torr == pytest.approx(1.488e-6, rel=0.03)
    # solid below the melting point, liquid above, continuous at it
    for el, vp in sc.VAPOUR_PRESSURE.items():
        Tm = vp["T_melt"]
        assert sc.vapour_pressure_pa(el, Tm - 1e-6) == pytest.approx(sc.vapour_pressure_pa(el, Tm + 1e-6), rel=0.06)
    assert sc.vapour_density_cm3("Fr", 300.0) is None


def test_refractive_index_is_normal_below_and_anomalous_above_resonance():
    nu0, s, g, N = 3.84e14, 2.2e8, 3e6, 1e16
    below = sc.refractive_index_minus_1(nu0 - 5 * s, nu0, 0.7, s, g, N)
    above = sc.refractive_index_minus_1(nu0 + 5 * s, nu0, 0.7, s, g, N)
    assert below > 0 > above


def test_curve_of_growth_limits():
    f, s, g = 0.7, 2.2e8, 3e6
    K = sc.LINE_K * f
    thin = 1e6                                            # cm⁻²: τ₀ ~ 1e-5
    assert sc.equivalent_width_hz(thin, f, s, g) == pytest.approx(K * thin * 1e4, rel=1e-3)
    # pure Lorentzian line deep in the damping regime: W = 2 √(γ K Nℓ)
    thick = 1e18
    assert sc.equivalent_width_hz(thick, f, 1.0, g) == pytest.approx(2 * math.sqrt(g * K * thick * 1e4), rel=2e-3)


def test_hyperfine_components_match_steck():
    from qali.physics.hyperfine_core import hyperfine_energy
    # Cs D2 F=4 -> F'=5 lies 4021.776 - 263.8906 MHz below the line centroid (Steck, Cs D Line Data)
    comps = sc.hyperfine_components(3.5, 0.5, 1.5, 2298.1579425, 0.0, 50.28827, -0.4934, 0.56)
    c = next(c for c in comps if (c.F, c.Fp) == (4, 5))
    assert c.detuning_MHz == pytest.approx(263.8906 - 4021.776, abs=0.01)
    assert sum(c.weight for c in comps) == pytest.approx(1.0)
    assert hyperfine_energy(4, 3.5, 0.5, 2298.1579425) == pytest.approx(4021.776, abs=1e-3)
    # a forbidden ΔF = 2 component never appears
    assert all(abs(c.F - c.Fp) <= 1 for c in comps)


def test_sas_two_level_limit_and_crossovers():
    one = [sc.HfComponent(1.0, 2.0, 0.0, 1.0, 1.0)]
    assert sc.sas_pairs(one, 2e8) == [(0.0, 1.0)]
    # Rb-87 D1 from F=2: F'=1 and F'=2 share the lower level, so a crossover sits halfway
    comps = sc.hyperfine_components(1.5, 0.5, 0.5, 3417.341305, 0.0, 407.25, 0.0)
    feats = sc.sas_pairs(comps, 2.2e8)
    f21 = next(c for c in comps if (c.F, c.Fp) == (2, 1))
    f22 = next(c for c in comps if (c.F, c.Fp) == (2, 2))
    assert any(abs(d - 0.5 * (f21.detuning_MHz + f22.detuning_MHz)) < 1e-9 for d, _k in feats)
    assert len(feats) == 4 + 4            # 4 Lamb dips, 2 ground-level and 2 upper-level crossovers
    # Lamb-dip width: natural FWHM in the weak-pump limit
    assert 2 * sc.lamb_dip_hwhm_hz(1 / 26.2e-9, 0.0) == pytest.approx(1 / (2 * math.pi * 26.2e-9))


# ------------------------------------------------------------------ browser twin
def test_js_faddeeva_matches_scipy():
    from scipy.special import wofz
    pts = [[x, y] for x in (-30, -12, -6, -3, -1.5, -0.5, 0, 0.2, 1, 2.5, 4, 5, 7, 10, 20)
           for y in (1e-6, 1e-3, 0.01, 0.1, 0.5, 1, 3, 8, 20)]
    got = js("DATA.map(([x,y])=>SP.faddeeva(x,y))", pts)
    for (x, y), (re, im) in zip(pts, got):
        w = wofz(x + 1j * y)
        assert abs(complex(re, im) - w) / abs(w) < 2e-4, (x, y)
        assert abs(re - w.real) / abs(w.real) < 1e-3, (x, y)


def test_js_physics_matches_python():
    lam = [150.0, 250.0, 589.1575, 780.2412, 1500.0]
    got = js("""({air: DATA.lam.map(SP.vacToAir), back: DATA.lam.map(l => SP.airToVac(SP.vacToAir(l))),
                  dop: SP.dopplerSigma(3.84e14, 86.9, 300), vap: SP.vapourDensityCm3(DATA.vp, 350),
                  bz: SP.boltzmann([0, 2.1, 3.0], [2, 6, 10], 3000), ew: SP.equivalentWidth(1e11, 0.7, 2.2e8, 3e6),
                  inst: SP.instrumentSigma(5e14, 0.01)})""",
             dict(lam=lam, vp=sc.vapour_record("Rb")))
    assert got["air"] == pytest.approx(list(sc.vac_to_air_nm(lam)), rel=1e-12)
    assert got["back"] == pytest.approx(lam, rel=1e-9)
    assert got["dop"] == pytest.approx(sc.doppler_sigma_hz(3.84e14, 86.9, 300), rel=1e-12)
    assert got["vap"] == pytest.approx(sc.vapour_density_cm3("Rb", 350), rel=1e-9)
    assert got["bz"] == pytest.approx(list(sc.boltzmann_weights([0, 2.1, 3.0], [2, 6, 10], 3000)), rel=1e-9)
    assert got["ew"] == pytest.approx(sc.equivalent_width_hz(1e11, 0.7, 2.2e8, 3e6), rel=2e-3)
    assert got["inst"] == pytest.approx(sc.instrument_sigma_hz(5e14, 0.01), rel=1e-12)


def test_js_spectrum_keeps_line_areas_at_any_resolution():
    """Unresolved lines put their exact area into the plotted bins; resolved ones are sampled."""
    lines = [dict(nu=5.0e14, s=5e8, g=5e6, a=2.0), dict(nu=5.0e14 + 3e11, s=5e8, g=5e6, a=1.0)]
    for n, lo, hi in ((300, 4.99e14, 5.01e14), (4000, 4.99e14, 5.01e14)):
        res = js("""(()=>{const nu=Float64Array.from({length:DATA.n},(_,k)=>DATA.lo+(DATA.hi-DATA.lo)*k/(DATA.n-1));
                  const r=SP.sumProfiles(nu,DATA.lines,false);let A=0;for(let k=1;k<nu.length;k++)A+=0.5*(r.re[k]+r.re[k-1])*(nu[k]-nu[k-1]);
                  return A})()""", dict(n=n, lo=lo, hi=hi, lines=lines))
        assert res == pytest.approx(3.0, rel=0.02), n
    # point values of a resolved line equal the Python Voigt
    nu = [5.0e14 - 1e9, 5.0e14, 5.0e14 + 2e8]
    got = js("""SP.sumProfiles(Float64Array.from(DATA.nu),[DATA.L],true)""",
             dict(nu=nu, L=dict(nu=5.0e14, s=5e8, g=5e6, a=1.0)))
    want = sc.voigt_complex(np.array(nu), 5.0e14, 5e8, 5e6)
    assert [got["re"][str(k)] for k in range(3)] == pytest.approx(list(want.real), rel=2e-4)
    assert [got["im"][str(k)] for k in range(3)] == pytest.approx(list(want.imag), rel=2e-3, abs=1e-15)


def test_js_sas_pairs_match_python():
    comps = sc.hyperfine_components(1.5, 0.5, 1.5, 3417.341305, 0.0, 84.7185, 12.4965, 0.0, 77.992, 0.2783)
    got = js("SP.sasPairs(DATA.c, 2.2e8).map(f=>[f.d,f.k])",
             dict(c=[dict(F=c.F, Fp=c.Fp, d=c.detuning_MHz, w=c.weight, S=c.S) for c in comps]))
    want = sc.sas_pairs(comps, 2.2e8)
    assert len(got) == len(want)
    for (d, k), (d2, k2) in zip(got, want):
        assert d == pytest.approx(d2, abs=1e-9) and k == pytest.approx(k2, rel=1e-12)


# ------------------------------------------------------------------ the stage
@pytest.fixture(scope="module")
def pages(tmp_path_factory):
    root = make_workspace(tmp_path_factory.mktemp("spec") / "ws")
    out = {}
    for sp in ("Rb", "Na_c1"):
        r = run_pipeline(["spectra", sp], root)
        assert r.returncode == 0, r.stdout + r.stderr
        page = root / ".qali" / "plots" / sp / f"{sp}_spectrum.html"
        out[sp] = (r.stdout, page.read_text(encoding="utf-8"), page.parent)
    return out


def test_rb_page_has_every_tab_and_the_isotope_resolved_d_lines(pages):
    out, html, folder = pages["Rb"]
    for tab in ("ab", "em", "dl", "cg", "st", "li"):
        assert f'id="pane-{tab}"' in html
    assert (folder / "plotly.min.js").is_file()
    assert not (folder / "Rb_spectra.html").exists() and not (folder / "Rb_spectrum_levels.html").exists()
    D = _payload(html)
    assert D["gs"] == "5s" and D["opts"]["tabs"] == ["ab", "em", "dl", "cg", "st", "li"]
    d2 = next(d for d in D["dlines"] if d["name"] == "D2")
    assert [i["A"] for i in d2["iso"]] == [85, 87]
    assert sum(i["ab"] for i in d2["iso"]) == pytest.approx(1.0, abs=1e-6)
    assert sum(c[3] for i in d2["iso"] for c in i["comps"]) == pytest.approx(1.0, abs=1e-6)
    assert next(i for i in d2["iso"] if i["A"] == 87)["shift"] == pytest.approx(77.992)
    assert "Rb-85 (72.17%, 6 components), Rb-87 (27.83%, 6 components)" in out
    assert "Vapour at 298.15 K: 1.29e+10 atoms/cm³" in out


def test_ion_page_finds_the_true_ground_state_and_has_no_d_lines(pages):
    _out, html, _folder = pages["Na_c1"]
    D = _payload(html)
    assert D["gs"] == "2p" and D["is_ion"] and D["vapour"] is None and D["dlines"] == []
    assert "dl" not in D["opts"]["tabs"]
    assert 28 < D["opts"]["wl"][0] < D["opts"]["wl"][1] < 40      # the 2p⁶ → 2p⁵3s/3d resonance lines


def test_rounding_artefacts_are_left_out(pages):
    D = _payload(pages["Rb"][1])
    E = {lv[0]: lv[2] for lv in D["levels"]}
    assert all(abs(E[D["levels"][r[1]][0]] - E[D["levels"][r[0]][0]]) > 2e-6 for r in D["lines"])
    assert D["dropped"] >= 1
