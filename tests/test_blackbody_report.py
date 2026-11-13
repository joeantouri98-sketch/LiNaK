"""The blackbody report: written by the Blackbody stage, tabs chosen by ticks, old plots gone."""
import json
import re
import shutil
import sys
from pathlib import Path

import pytest

from qali.core.errors import MissingDataError
from qali.core.paths import reference_path
from qali.gui.stages import STAGE_BY_ID, STAGE_GROUPS, flag_section, stage_argv
from qali.pipelines import blackbody
from qali.pipelines import blackbody_report as br
from qali.pipelines.runner import IMPLEMENTATIONS


@pytest.fixture(scope="module")
def reference():
    bb = json.loads(Path(reference_path("data_json", "Na_blackbody.json")).read_text(encoding="utf-8"))
    life = json.loads(Path(reference_path("data_json", "Na_lifetimes.json")).read_text(encoding="utf-8"))
    pol = json.loads(Path(reference_path("data_json", "Na_polarizability.json")).read_text(encoding="utf-8"))
    return bb, life, pol


def _networks(pol):
    return pol.get("transitions_gs"), (pol.get("transitions_ex") if pol.get("excited_label") else None)


def _opts(*extra):
    return br.options_from_args(blackbody.build_parser().parse_args(["Na", *extra]))


def _payload(path):
    html = Path(path).read_text(encoding="utf-8")
    return json.loads(re.search(r"const D=(\{.*?\});\n", html, re.S).group(1))


def _write(reference, tmp_path, *extra, life=True, pol=True):
    bb, lf, pl = reference
    gs, ex = _networks(pl) if pol else (None, None)
    return br.write_report("Na", "Na", bb, gs, ex, lf if life else None, _opts(*extra), str(tmp_path))


def test_all_tabs_and_the_data(reference, tmp_path):
    path = _write(reference, tmp_path)
    assert (tmp_path / "plotly.min.js").exists()
    assert "$!" not in Path(path).read_text(encoding="utf-8")
    d = _payload(path)
    assert d["tabs"] == ["sh", "rt", "pi", "ry"]
    assert d["rates"]["states"] and d["labels"] and d["pi"]["states"] and d["ry"]["states"]
    assert d["opts"]["unit"] == "hz"


def test_output_is_deterministic(reference, tmp_path):
    a = Path(_write(reference, tmp_path / "a")).read_text(encoding="utf-8")
    b = Path(_write(reference, tmp_path / "b")).read_text(encoding="utf-8")
    assert a == b


def test_options_reach_the_page(reference, tmp_path):
    d = _payload(_write(reference, tmp_path, "--plot-units", "frac", "--T", "77", "--plot-t-min", "20",
                        "--plot-t-max", "600", "--field", "50", "--no-tab-rydberg"))
    assert d["tabs"] == ["sh", "rt", "pi"] and d["ry"] is None
    assert d["opts"] == dict(unit="frac", T=77.0, tmin=20.0, tmax=600.0, field=50.0)


def test_each_tab_can_be_ticked_off(reference, tmp_path):
    d = _payload(_write(reference, tmp_path, "--no-tab-shifts", "--no-tab-depopulation", "--no-tab-rydberg"))
    assert d["tabs"] == ["pi"]
    assert _write(reference, tmp_path / "none", "--no-tab-shifts", "--no-tab-depopulation",
                  "--no-tab-photoionization", "--no-tab-rydberg") is None


def test_missing_lifetimes_or_polarizability_leave_tabs_out(reference, tmp_path):
    assert _payload(_write(reference, tmp_path / "a", life=False))["tabs"] == ["sh", "pi", "ry"]
    assert _payload(_write(reference, tmp_path / "b", life=False, pol=False))["tabs"] == ["pi", "ry"]


def test_a_payload_with_nothing_to_draw_is_an_error(reference, tmp_path):
    bb = {k: v for k, v in reference[0].items() if k not in ("rates", "continuum", "rydberg_shifts")}
    with pytest.raises(MissingDataError):
        br.write_report("Na", "Na", bb, None, None, None, _opts(), str(tmp_path))


def test_stage_flags_and_removed_pieces():
    stage = STAGE_BY_ID["blackbody"]
    flags = {f["name"]: f for f in stage["flags"]}
    assert flags["--no-plot"]["kind"] == "invflag"
    assert "--no-plot" not in stage_argv(stage, "Na", [])
    ticks = {n for n, f in flags.items() if f["kind"] == "invflag" and n.startswith("--no-tab-")}
    assert ticks == {"--no-tab-shifts", "--no-tab-depopulation", "--no-tab-photoionization",
                     "--no-tab-rydberg"}
    assert flag_section(flags["--output-dir"]) == "output"
    parser_flags = {o.option_strings[0] for o in blackbody.build_parser()._actions if o.option_strings}
    assert set(flags) <= parser_flags
    assert stage["outputs"] == ["data/{el}_blackbody.json", "plots/{el}/{el}_blackbody.html"]
    assert "bbr" not in STAGE_BY_ID and "bbr" not in dict(STAGE_GROUPS)["Visualise"]
    assert "blackbody_page" not in IMPLEMENTATIONS
    for gone in ("plot_blackbody", "_write_blackbody_hub", "_fig_shifts", "_inject_point_calculator"):
        assert not hasattr(blackbody, gone)


def test_stage_flags_parse_to_the_stage_defaults():
    args = blackbody.build_parser().parse_args(["Na"])
    for f in STAGE_BY_ID["blackbody"]["flags"]:
        if f["name"] in ("--plot-units", "--plot-t-min", "--plot-t-max"):
            dest = f["name"].lstrip("-").replace("-", "_")
            assert str(getattr(args, dest)).rstrip("0").rstrip(".") == str(f["default"]), f["name"]


def test_blackbody_stage_writes_the_page_with_its_data(tmp_path):
    sys.path.insert(0, str(Path(__file__).parent))
    from helpers import make_workspace, run_pipeline

    ws = make_workspace(tmp_path / "ws")
    data = ws / ".qali" / "data"
    data.mkdir(parents=True)
    for f in Path(reference_path("data_json", "Na_rydberg.json")).parent.glob("Na*.json"):
        shutil.copy(f, data / f.name)
    res = run_pipeline(["blackbody", "Na", "--no-rydberg-shifts", "--no-pi"], ws)
    assert res.returncode == 0, res.stderr
    plots = list((ws / ".qali").rglob("Na_blackbody.html"))
    assert len(plots) == 1
    assert _payload(plots[0])["tabs"] == ["sh", "rt"]
    for old in ("_shifts", "_rates", "_pi", "_alpha"):
        assert not list((ws / ".qali").rglob(f"Na_blackbody{old}.html"))
