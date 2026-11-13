import json
import shutil
import sys
from pathlib import Path

import pytest

from qali.core.data_inputs import resolve_data_file
from qali.core.errors import MissingDataError
from qali.core.paths import reference_path
from qali.gui.stages import STAGE_BY_ID


def test_blank_means_the_default_file(tmp_path):
    assert resolve_data_file("", "Na", tmp_path, "lifetimes") is None
    assert resolve_data_file(None, "Na", tmp_path, "lifetimes") is None


def test_names_resolve_under_the_data_folder_and_must_belong_to_the_species(tmp_path):
    for name in ("Na_lifetimes__hot.json", "K_lifetimes.json"):
        (tmp_path / name).write_text("{}")
    expected = str(tmp_path / "Na_lifetimes__hot.json")
    for spec in ("Na_lifetimes__hot.json", "Na_lifetimes__hot", "lifetimes__hot", expected):
        assert resolve_data_file(spec, "Na", tmp_path, "lifetimes") == expected
    with pytest.raises(MissingDataError, match="not a Na file"):
        resolve_data_file(str(tmp_path / "K_lifetimes.json"), "Na", tmp_path, "lifetimes")
    with pytest.raises(MissingDataError, match="not found"):
        resolve_data_file("nope", "Na", tmp_path, "lifetimes")


def test_blackbody_and_tweezer_offer_the_input_files():
    from qali.pipelines.blackbody import build_parser as bb
    from qali.pipelines.tweezer import build_parser as tw

    for stage_id, parser in (("blackbody", bb()), ("tweezer", tw())):
        names = {f["name"]: f for f in STAGE_BY_ID[stage_id]["flags"]}
        for flag in ("--polarizability-file", "--lifetimes-file"):
            assert names[flag]["kind"] == "path" and names[flag]["browse"] == "data"
        args = parser.parse_args(["Na", "--lifetimes-file", "x", "--polarizability-file", "y"])
        assert args.lifetimes_file == "x" and args.polarizability_file == "y"


def test_rydberg_and_reparse_stages_have_browse_buttons():
    rydberg = {f["name"]: f for f in STAGE_BY_ID["rydberg"]["flags"]}
    reparse = {f["name"]: f for f in STAGE_BY_ID["orca_to_json"]["flags"]}
    assert rydberg["--nist-dir"]["kind"] == "path" and rydberg["--nist-dir"]["browse"] == "dir"
    assert reparse["--run"]["kind"] == "path" and reparse["--run"]["browse"] == "orca_run"


@pytest.fixture()
def na_workspace(tmp_path):
    sys.path.insert(0, str(Path(__file__).parent))
    from helpers import make_workspace

    ws = make_workspace(tmp_path / "ws")
    data = ws / ".qali" / "data"
    data.mkdir(parents=True)
    src = reference_path("data_json", "Na_rydberg.json").parent
    for f in src.glob("Na*.json"):
        shutil.copy(f, data / f.name)
    return ws, data


def test_blackbody_reads_a_chosen_lifetimes_file_and_records_it(na_workspace):
    from helpers import run_pipeline

    ws, data = na_workspace
    shutil.copy(data / "Na_lifetimes.json", data / "Na_lifetimes__copy.json")
    quick = ["blackbody", "Na", "--no-plot", "--no-rydberg-shifts", "--no-pi"]
    plain = run_pipeline(quick, ws)
    assert plain.returncode == 0, plain.stderr
    assert "inputs" not in json.loads((data / "Na_blackbody.json").read_text())
    named = run_pipeline([*quick, "--lifetimes-file", "Na_lifetimes__copy"], ws)
    assert named.returncode == 0, named.stderr
    assert json.loads((data / "Na_blackbody.json").read_text())["inputs"] == {"lifetimes": "Na_lifetimes__copy.json"}
    missing = run_pipeline([*quick, "--lifetimes-file", "nope"], ws)
    assert missing.returncode != 0


def test_tweezer_reads_chosen_files_and_records_them(na_workspace):
    from helpers import run_pipeline

    ws, data = na_workspace
    for kind in ("lifetimes", "polarizability"):
        shutil.copy(data / f"Na_{kind}.json", data / f"Na_{kind}__alt.json")
    res = run_pipeline(["tweezer", "Na", "--no-plot", "--lifetimes-file", "lifetimes__alt",
                        "--polarizability-file", "polarizability__alt"], ws)
    assert res.returncode == 0, res.stderr
    out = json.loads((data / "Na_tweezer.json").read_text())
    assert out["inputs"] == {"polarizability": "Na_polarizability__alt.json", "lifetimes": "Na_lifetimes__alt.json"}


def test_browse_modes_fill_the_field(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QFileDialog, QLineEdit

    from qali.core.workspace import Workspace
    from qali.gui import main_window as mw

    QApplication.instance() or QApplication([])
    runs = tmp_path / ".qali" / "orca_outputs" / "Na" / "casscf_a"
    runs.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("#")
    (tmp_path / "src" / "qali").mkdir(parents=True)

    class Fake:
        root = tmp_path
        current_species = "Na"

    asked = {}
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda _p, _t, start: asked.update(start=start) or str(runs)))
    edit = QLineEdit()
    mw.MainWindow._browse_for_path(Fake(), edit, mode="orca_run")
    assert edit.text() == "casscf_a" and asked["start"] == str(Workspace(root=tmp_path).orca_outputs_dir / "Na")
    mw.MainWindow._browse_for_path(Fake(), edit, mode="dir")
    assert edit.text() == str(runs)


@pytest.mark.parametrize("stage_id,flags", [
    ("grotrian", ("rydberg", "transitions", "lifetimes")),
    ("spectra", ("lifetimes", "hf")),
    ("orbital3d", ("rydberg", "lifetimes")),
    ("response", ("polarizability", "lifetimes")),
])
def test_plot_stages_offer_their_input_files(stage_id, flags):
    import importlib

    from qali.pipelines.runner import IMPLEMENTATIONS

    parser = importlib.import_module(IMPLEMENTATIONS[STAGE_BY_ID[stage_id]["pipeline"]]).build_parser()
    names = {f["name"]: f for f in STAGE_BY_ID[stage_id]["flags"]}
    for kind in flags:
        assert names[f"--{kind}-file"]["kind"] == "path" and names[f"--{kind}-file"]["browse"] == "data"
        assert getattr(parser.parse_args(["Na", f"--{kind}-file", "x"]), f"{kind}_file") == "x"
        assert getattr(parser.parse_args(["Na"]), f"{kind}_file") is None


def test_spectra_reads_a_chosen_lifetimes_file(na_workspace):
    from helpers import run_pipeline

    ws, data = na_workspace
    alt = json.loads((data / "Na_lifetimes.json").read_text())
    alt["transitions"] = alt["transitions"][:7]
    (data / "Na_lifetimes__few.json").write_text(json.dumps(alt))
    res = run_pipeline(["spectra", "Na", "--lifetimes-file", "lifetimes__few"], ws)
    assert res.returncode == 0, res.stderr
    assert "Na_lifetimes__few.json" in res.stdout
    missing = run_pipeline(["spectra", "Na", "--lifetimes-file", "nope"], ws)
    assert missing.returncode != 0


def test_orca_stages_have_no_save_as():
    from qali.gui.main_window import NO_SAVE_AS

    assert set(NO_SAVE_AS) == {"runorca", "orca_to_json"}
    assert all("--outname" in {f["name"] for f in STAGE_BY_ID[s]["flags"]} for s in NO_SAVE_AS)


def test_polarizability_reads_chosen_rydberg_and_transitions_files(na_workspace):
    from helpers import run_pipeline

    ws, data = na_workspace
    for kind in ("rydberg", "transitions"):
        shutil.copy(data / f"Na_{kind}.json", data / f"Na_{kind}__alt.json")
    res = run_pipeline(["polarizability", "Na", "--no-plot", "--rydberg-file", "rydberg__alt",
                        "--transitions-file", "transitions__alt"], ws)
    assert res.returncode == 0, res.stderr
    out = json.loads((data / "Na_polarizability.json").read_text())
    assert out["inputs"] == {"rydberg": "Na_rydberg__alt.json", "transitions": "Na_transitions__alt.json"}
    plain = run_pipeline(["polarizability", "Na", "--no-plot"], ws)
    assert plain.returncode == 0 and "inputs" not in json.loads((data / "Na_polarizability.json").read_text())
    assert run_pipeline(["polarizability", "Na", "--no-plot", "--rydberg-file", "nope"], ws).returncode != 0


def test_transitions_and_lifetimes_read_chosen_files(na_workspace):
    from helpers import run_pipeline

    ws, data = na_workspace
    for kind in ("rydberg", "transitions"):
        shutil.copy(data / f"Na_{kind}.json", data / f"Na_{kind}__alt.json")
    res = run_pipeline(["transitions", "Na", "--rydberg-file", "rydberg__alt"], ws)
    assert res.returncode == 0, res.stderr
    assert json.loads((data / "Na_transitions.json").read_text())["inputs"] == {"rydberg": "Na_rydberg__alt.json"}
    res = run_pipeline(["lifetimes", "Na", "--rydberg-file", "rydberg__alt", "--transitions-file", "transitions__alt"], ws)
    assert res.returncode == 0, res.stderr
    assert json.loads((data / "Na_lifetimes.json").read_text())["inputs"] == {
        "rydberg": "Na_rydberg__alt.json", "transitions": "Na_transitions__alt.json"}
    assert run_pipeline(["lifetimes", "Na", "--orca-file", "nope"], ws).returncode != 0
    plain = run_pipeline(["lifetimes", "Na"], ws)
    assert plain.returncode == 0 and "inputs" not in json.loads((data / "Na_lifetimes.json").read_text())
    from qali.pipelines.lifetimes import build_parser
    assert build_parser().parse_args(["Na", "--orca-file", "x"]).orca_file == "x"
