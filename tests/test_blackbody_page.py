"""The blackbody page stage: payload, page, missing data and its place in the GUI stage table."""
import json
import re
import shutil

import pytest

from linak.core.errors import MissingDataError
from linak.core.paths import reference_path
from linak.gui.stages import STAGE_BY_ID, STAGE_GROUPS, flag_section, stage_argv
from linak.pipelines import blackbody_page as bp

FILES = ("Na_blackbody.json", "Na_polarizability.json", "Na_lifetimes.json")


@pytest.fixture()
def data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    for name in FILES:
        shutil.copy(reference_path("data_json", name), d / name)
    return d


def _run(data_dir, tmp_path, *extra):
    out = tmp_path / "plots"
    assert bp.main(["Na", "--data-dir", str(data_dir), "--output-dir", str(out), *extra]) == 0
    return out / "Na" / "Na_blackbody_page.html"


def _payload(html):
    return json.loads(re.search(r"const D=(\{.*?\});\n", html, re.S).group(1))


def test_page_has_all_tabs_and_the_data(data_dir, tmp_path):
    page = _run(data_dir, tmp_path)
    assert (page.parent / "plotly.min.js").exists()
    html = page.read_text(encoding="utf-8")
    assert "$!" not in html
    d = _payload(html)
    assert d["tabs"] == ["sh", "rt", "pi", "ry"]
    assert d["gs"] == "3s" and d["Tdyn"] and len(d["Tdyn"]) == len(d["dyn_gs"])
    assert d["rates"]["states"] and d["labels"] and d["pi"]["states"] and d["ry"]["states"]
    edge = d["rates"]["states"][0]["e"][0]
    assert 0 <= edge[2] < len(d["labels"]) and edge[3] in (0, 1)
    assert d["opts"]["unit"] == "hz" and d["opts"]["T"] == 300.0


def test_static_shift_matches_the_stored_value(data_dir, tmp_path):
    d = _payload(_run(data_dir, tmp_path).read_text(encoding="utf-8"))
    stored = json.loads((data_dir / "Na_blackbody.json").read_text())
    assert d["a0g"] == pytest.approx(stored["alpha0_gs_au"], rel=1e-6)


def test_output_is_deterministic(data_dir, tmp_path):
    assert _run(data_dir, tmp_path).read_text(encoding="utf-8") == _run(data_dir, tmp_path).read_text(encoding="utf-8")


def test_options_reach_the_page(data_dir, tmp_path):
    d = _payload(_run(data_dir, tmp_path, "--units", "frac", "--temperature", "77", "--t-min", "20",
                      "--t-max", "600", "--field", "50", "--no-rydberg").read_text(encoding="utf-8"))
    assert d["tabs"] == ["sh", "rt", "pi"] and d["ry"] is None
    assert d["opts"] == dict(unit="frac", T=77.0, tmin=20.0, tmax=600.0, field=50.0)


def test_each_tab_can_be_ticked_off(data_dir, tmp_path):
    d = _payload(_run(data_dir, tmp_path, "--no-shifts", "--no-depopulation", "--no-rydberg").read_text(encoding="utf-8"))
    assert d["tabs"] == ["pi"]
    with pytest.raises(MissingDataError):
        bp.main(["Na", "--data-dir", str(data_dir), "--output-dir", str(tmp_path / "p"), "--no-shifts",
                 "--no-depopulation", "--no-photoionization", "--no-rydberg"])


def test_missing_lifetimes_or_polarizability_leave_tabs_out(data_dir, tmp_path):
    (data_dir / "Na_lifetimes.json").unlink()
    assert _payload(_run(data_dir, tmp_path).read_text(encoding="utf-8"))["tabs"] == ["sh", "pi", "ry"]
    (data_dir / "Na_polarizability.json").unlink()
    assert _payload(_run(data_dir, tmp_path).read_text(encoding="utf-8"))["tabs"] == ["pi", "ry"]


def test_missing_blackbody_data_is_a_clear_error(data_dir, tmp_path):
    (data_dir / "Na_blackbody.json").unlink()
    with pytest.raises(MissingDataError, match="Blackbody stage"):
        bp.main(["Na", "--data-dir", str(data_dir), "--output-dir", str(tmp_path / "plots")])


def test_other_excited_state_drops_the_excited_curves(data_dir, tmp_path):
    path = data_dir / "Na_polarizability.json"
    pol = json.loads(path.read_text())
    pol["excited_label"] = "9s"
    path.write_text(json.dumps(pol))
    d = _payload(_run(data_dir, tmp_path).read_text(encoding="utf-8"))
    assert d["dyn_ex"] is None and d["a0e"] is None and "sh" in d["tabs"]


def test_stage_is_in_visualise_with_input_and_output_options():
    stage = STAGE_BY_ID["bbr"]
    assert stage["pipeline"] == "blackbody_page" and "bbr" in dict(STAGE_GROUPS)["Visualise"]
    ticks = {f["name"] for f in stage["flags"] if f["kind"] == "invflag"}
    assert ticks == {"--no-shifts", "--no-depopulation", "--no-photoionization", "--no-rydberg"}
    sections = {f["name"]: flag_section(f) for f in stage["flags"]}
    for name in ("--blackbody-file", "--polarizability-file", "--lifetimes-file"):
        assert sections[name] == "input"
    assert sections["--output-dir"] == "output"
    parser_flags = {o.option_strings[0] for o in bp.build_parser()._actions if o.option_strings}
    assert {f["name"] for f in stage["flags"]} <= parser_flags
    assert stage["outputs"] == ["plots/{el}/{el}_blackbody_page.html"]


def test_stage_flags_parse_to_the_stage_defaults():
    args = bp.build_parser().parse_args(["Na"])
    for f in STAGE_BY_ID["bbr"]["flags"]:
        if f["kind"] in ("float", "choice") and f["default"] != "":
            dest = f["name"].lstrip("-").replace("-", "_")
            assert str(getattr(args, dest)).rstrip("0").rstrip(".") == str(f["default"]), f["name"]


def test_blackbody_stage_keeps_its_generate_plot_tick():
    flags = {f["name"]: f for f in STAGE_BY_ID["blackbody"]["flags"]}
    assert flags["--no-plot"]["kind"] == "invflag"
    assert "--no-plot" not in stage_argv(STAGE_BY_ID["blackbody"], "Na", [])
