from pathlib import Path

from qali.gui.save_as import OutputSnapshot, named_path, sanitize


def test_sanitize_keeps_safe_names_only():
    assert sanitize("hot 350K") == "hot_350K"
    assert sanitize("  ../x/y  ") == "x_y"
    assert sanitize("***") == ""
    assert len(sanitize("a" * 100)) == 40


def test_named_path_goes_before_the_extension():
    assert named_path(Path("data/Na_blackbody.json"), "hot") == Path("data/Na_blackbody__hot.json")


def _write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_named_run_keeps_the_new_files_and_restores_the_usual_ones(tmp_path):
    data, html = tmp_path / "Na_x.json", tmp_path / "plots" / "Na_x.html"
    _write(data, "old-data")                               # existed before; html did not
    snap = OutputSnapshot([data, html], backup_root=tmp_path)
    _write(data, "new-data!")                              # the stage overwrites / creates them
    _write(html, "new-html")
    kept = snap.finish("hot")
    assert {p.name for p in kept} == {"Na_x__hot.json", "Na_x__hot.html"}
    assert data.read_text() == "old-data" and not html.exists()
    assert (tmp_path / "Na_x__hot.json").read_text() == "new-data!"
    assert (tmp_path / "plots" / "Na_x__hot.html").read_text() == "new-html"
    assert not any(p.name.startswith("qali_saveas_") for p in tmp_path.iterdir())


def test_failed_run_keeps_nothing_and_leaves_the_usual_files(tmp_path):
    data = tmp_path / "Na_x.json"
    _write(data, "old")
    snap = OutputSnapshot([data], backup_root=tmp_path)
    _write(data, "half-written")
    assert snap.finish("hot", ok=False) == []
    assert data.read_text() == "old" and not (tmp_path / "Na_x__hot.json").exists()


def test_stage_that_wrote_nothing_changes_nothing(tmp_path):
    data = tmp_path / "Na_x.json"
    _write(data, "same")
    snap = OutputSnapshot([data], backup_root=tmp_path)
    assert snap.finish("hot") == [] and data.read_text() == "same"


def test_second_run_under_the_same_name_replaces_the_first(tmp_path):
    data = tmp_path / "Na_x.json"
    for text in ("run1", "run2"):
        snap = OutputSnapshot([data], backup_root=tmp_path)
        _write(data, text)
        snap.finish("hot")
    assert (tmp_path / "Na_x__hot.json").read_text() == "run2" and not data.exists()


def test_grotrian_soc_controls_are_in_the_parser_and_the_gui_stage():
    from qali.gui.stages import STAGE_BY_ID
    from qali.pipelines.plotinteractive import build_parser

    args = build_parser().parse_args(["Na", "--soc-file", "Na_soc_norb8", "--no-soc-overlay"])
    assert args.soc_file == "Na_soc_norb8" and args.no_soc_overlay is True
    assert build_parser().parse_args(["Na"]).soc_file is None
    names = {f["name"] for f in STAGE_BY_ID["grotrian"]["flags"]}
    assert {"--soc-file", "--no-soc-overlay"} <= names


def test_filter_rail_template_has_no_emoji_icon_and_is_self_contained():
    html = Path(__file__).resolve().parents[1] / "src" / "qali" / "report_templates" / "plotinteractive_grotrian.html"
    text = html.read_text(encoding="utf-8")
    assert "\U0001F52C" not in text and "googleapis" not in text and "cdn." not in text
