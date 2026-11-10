"""The Grotrian page: payload content, extrapolation bands, template tokens, determinism."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from linak.pipelines.plotinteractive import build_parser, grotrian_page
from linak.pipelines.plotinteractive.inputs import load_inputs
from linak.pipelines.plotinteractive.view import prepare_view
from linak.core.species import resolve_species
from linak.report_templates import load


def _inputs(data_dir, tmp_path):
    cli = build_parser().parse_args(["Na", "--no-soc-overlay"])
    inputs = load_inputs(resolve_species("Na"), cli, str(data_dir))
    return inputs, prepare_view(inputs, cli)


@pytest.fixture()
def na(data_dir, tmp_path):
    return _inputs(data_dir, tmp_path)


def test_payload_carries_every_level_and_transition(na):
    inputs, view = na
    p = grotrian_page.build_payload(inputs, view)
    assert len(p["levels"]) == len(inputs.rydberg_excitations) + 1
    assert len(p["trans"]) == len(view.visible_transitions) > 10000
    assert p["gs"] == "3s" and p["levels"][0]["E"] == 0.0
    assert p["ieText"].startswith("Ionization (5.139")
    row = next(r for r in p["trans"] if r[:2] == ["3p1/2", "3s"])
    assert row[4] == pytest.approx(0.32) and row[5] == "NIST"


def test_theoretical_levels_keep_their_flag(na):
    p = grotrian_page.build_payload(*na)
    assert sum(1 for lv in p["levels"] if lv["th"]) == na[0].rydberg["num_theoretical"]


def test_bands_attach_to_ground_to_np_lines(na):
    inputs, view = na
    ext = {"mode": "use", "channels": {"3/2": {"rows": [
        {"n": 12, "f": 0.01, "f_model": 0.011, "lo68": 0.009, "hi68": 0.012, "lo95": 0.008, "hi95": 0.014}]}}}
    inputs = SimpleNamespace(**{**inputs.__dict__, "extrapolation": ext}) if False else _with(inputs, extrapolation=ext)
    p = grotrian_page.build_payload(inputs, view)
    hit = [e["xb"] for e in p["extra"].values() if "xb" in e]
    assert len(hit) == 1 and hit[0]["lo95"] == 0.008 and hit[0]["hi68"] == 0.012 and hit[0]["fm"] == 0.011
    i = next(int(k) for k, e in p["extra"].items() if "xb" in e)
    assert p["trans"][i][:2] == ["12p3/2", "3s"]


def _with(inputs, **kw):
    import dataclasses
    return dataclasses.replace(inputs, **kw)


def test_synthetic_extrapolated_lines_carry_bands(na):
    inputs, view = na
    syn = {"upper_label": "3p1/2", "lower_label": "3s", "wavelength_nm": 600.0, "extrapolated_f": 0.3,
           "extrapolated_lo68": 0.2, "extrapolated_hi68": 0.4, "extrapolated_lo95": 0.1, "extrapolated_hi95": 0.5}
    p = grotrian_page.build_payload(_with(inputs, extrapolated_overlay=(syn,)), view)
    last = p["trans"][-1]
    assert last[5] == "extrapolated" and last[4] == 0.3
    assert p["extra"][str(len(p["trans"]) - 1)]["xb"]["hi95"] == 0.5


def test_template_tokens_and_output_are_clean(na, tmp_path):
    assert set(re.findall(r"\$!(\w+)", load("plotinteractive_grotrian.html"))) == {"TITLE", "DATA_JS"}
    path = grotrian_page.plot_grotrian(*na, str(tmp_path))
    text = Path(path).read_text(encoding="utf-8")
    assert "$!" not in text and 'src="plotly.min.js"' in text
    blob = re.search(r"const D=(\{.*?\});\nconst F=", text, re.S).group(1)
    assert json.loads(blob)["element"] == "Na"
    (tmp_path / "b").mkdir()
    again = grotrian_page.plot_grotrian(*na, str(tmp_path / "b"))
    with open(again, encoding="utf-8") as fh:
        assert fh.read() == text


def test_features_requested_are_in_the_page():
    t = load("plotinteractive_grotrian.html")
    assert 'data-g="kind"' in t and 'id="fmin"' in t and "68% band" in t and "95% band" in t
