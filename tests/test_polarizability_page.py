"""The polarizability page stage: payload, page, and its place in the GUI stage table."""
import json
import re
import shutil

import pytest

from linak.core.errors import MissingDataError
from linak.core.paths import reference_path
from linak.gui.stages import STAGE_BY_ID, STAGE_GROUPS, flag_section
from linak.pipelines import polarizability_page as pp


@pytest.fixture()
def data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    for name in ("Na_polarizability.json", "Na_lifetimes.json"):
        shutil.copy(reference_path("data_json", name), d / name)
    return d


def _run(data_dir, tmp_path, *extra):
    out = tmp_path / "plots"
    code = pp.main(["Na", "--data-dir", str(data_dir), "--output-dir", str(out), *extra])
    assert code == 0
    return out / "Na" / "Na_response.html"


def _payload(html):
    m = re.search(r"const D=(\{.*?\});\n", html, re.S)
    return json.loads(m.group(1))


def test_page_has_all_three_tabs_and_the_data(data_dir, tmp_path):
    page = _run(data_dir, tmp_path)
    assert (page.parent / "plotly.min.js").exists()
    html = page.read_text(encoding="utf-8")
    assert "$!" not in html
    d = _payload(html)
    assert d["tabs"] == ["pol", "mag", "tw"]
    assert d["gs"] == "3s" and d["gt"] and d["et"] and d["magic"]
    assert d["tw"]["wl"] and len(d["tw"]["wl"]) == len(d["tw"]["rsc"]) == len(d["tw"]["tau"])
    assert d["tw"]["point"]["top_channels"]
    assert d["opts"]["unit"] == "au" and d["opts"]["lam"] == 1064.0


def test_static_polarizability_from_the_rows_matches_the_stored_value(data_dir, tmp_path):
    d = _payload(_run(data_dir, tmp_path).read_text(encoding="utf-8"))
    static = sum(t[2] * t[1] / t[0] ** 2 for t in d["gt"])
    assert static == pytest.approx(d["alpha0"], rel=1e-4)


def test_output_is_deterministic(data_dir, tmp_path):
    first = _run(data_dir, tmp_path).read_text(encoding="utf-8")
    second = _run(data_dir, tmp_path).read_text(encoding="utf-8")
    assert first == second


def test_options_reach_the_page(data_dir, tmp_path):
    d = _payload(_run(data_dir, tmp_path, "--no-tweezer", "--units", "uk", "--scale", "lin",
                      "--wl-min", "500", "--wl-max", "900", "--cursor-wl", "820",
                      "--intensity", "25", "--waist", "2").read_text(encoding="utf-8"))
    assert d["tabs"] == ["pol", "mag"] and d["tw"] is None
    assert d["opts"] == dict(unit="uk", scale="lin", wmin=500.0, wmax=900.0, lam=820.0, I=25.0, waist=2.0)


def test_each_tab_can_be_ticked_off(data_dir, tmp_path):
    d = _payload(_run(data_dir, tmp_path, "--no-polarizability", "--no-magic").read_text(encoding="utf-8"))
    assert d["tabs"] == ["tw"]
    with pytest.raises(SystemExit):
        pp.main(["Na", "--data-dir", str(data_dir), "--output-dir", str(tmp_path / "p"),
                 "--no-polarizability", "--no-magic", "--no-tweezer"])


def test_tweezer_tab_is_left_out_without_lifetimes_data(data_dir, tmp_path):
    (data_dir / "Na_lifetimes.json").unlink()
    d = _payload(_run(data_dir, tmp_path).read_text(encoding="utf-8"))
    assert d["tabs"] == ["pol", "mag"] and d["tw"] is None


def test_missing_polarizability_data_is_a_clear_error(data_dir, tmp_path):
    (data_dir / "Na_polarizability.json").unlink()
    with pytest.raises(MissingDataError, match="Polarizability stage"):
        pp.main(["Na", "--data-dir", str(data_dir), "--output-dir", str(tmp_path / "plots")])


def test_chosen_input_files_are_used(data_dir, tmp_path):
    shutil.copy(data_dir / "Na_polarizability.json", data_dir / "Na_polarizability__hot.json")
    (data_dir / "Na_polarizability.json").unlink()
    d = _payload(_run(data_dir, tmp_path, "--polarizability-file", "polarizability__hot").read_text(encoding="utf-8"))
    assert d["gt"]


def test_stage_is_in_visualise_with_input_and_output_options():
    stage = STAGE_BY_ID["response"]
    assert stage["pipeline"] == "polarizability_page"
    group = dict(STAGE_GROUPS)["Visualise"]
    assert "response" in group
    ticks = {f["name"] for f in stage["flags"] if f["kind"] == "invflag"}
    assert ticks == {"--no-polarizability", "--no-magic", "--no-tweezer"}
    sections = {f["name"]: flag_section(f) for f in stage["flags"]}
    assert sections["--polarizability-file"] == "input" and sections["--lifetimes-file"] == "input"
    assert sections["--output-dir"] == "output"
    parser_flags = {o.option_strings[0] for o in pp.build_parser()._actions if o.option_strings}
    assert {f["name"] for f in stage["flags"]} <= parser_flags
    assert stage["outputs"] == ["plots/{el}/{el}_response.html"]


def test_stage_flags_parse_to_the_stage_defaults():
    args = pp.build_parser().parse_args(["Na"])
    for f in STAGE_BY_ID["response"]["flags"]:
        if f["kind"] in ("float", "choice") and f["default"] != "":
            dest = f["name"].lstrip("-").replace("-", "_")
            assert str(getattr(args, dest)).rstrip("0").rstrip(".") == str(f["default"]), f["name"]


def test_polarizability_and_tweezer_stages_write_data_only():
    from linak.gui.stages import stage_argv

    for stage_id in ("polarizability", "tweezer"):
        stage = STAGE_BY_ID[stage_id]
        assert all(f["name"] != "--no-plot" for f in stage["flags"])
        assert all(o.startswith("data/") for o in stage["outputs"])
        assert stage_argv(stage, "Na", [])[-1] == "--no-plot"
