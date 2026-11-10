"""What a new species needs, importing its NIST files, and the stages that then run."""
from __future__ import annotations

import json
import shutil

import pytest

from linak.core import paths, species_data as D, user_tables as U
from linak.core.paths import resource_root

CR_LEVELS = resource_root() / "NIST Levels" / "Cr.csv"
CR_LINES = resource_root() / "f_values" / "Cr_lines.csv"


@pytest.fixture()
def ws(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.PROJECT_ROOT_ENV, str(tmp_path))
    yield tmp_path
    shutil.rmtree(tmp_path / "tables", ignore_errors=True)
    for t in U.TABLES:
        try:
            U.reload(t.table_id)
        except Exception:
            pass


def test_only_species_without_levels_or_lines_need_setup(ws):
    assert D.needs_setup("Mg") and D.needs_setup("mg") and D.needs_setup("Mg_c1")
    assert not any(D.needs_setup(s) for s in ("Na", "Cs", "Fr", "Cr", "Fe", "Na_c1", "Xx", ""))


def test_stage_ids_exist_in_the_gui_stage_table():
    from linak.gui.stages import PIPELINE

    ids = {s["id"] for s in PIPELINE if s.get("id")}
    assert {sid for sid, _l, _n in D.STAGES} <= ids


def test_importing_the_two_files_enables_the_nist_stages(ws):
    states = {sid: ok for sid, _l, ok, _w in D.stage_states("Mg")}
    assert not states["rydberg"] and not states["lifetimes"] and states["runorca"]
    assert "levels" in D.import_file("levels", "Mg", CR_LEVELS)
    states = {sid: ok for sid, _l, ok, _w in D.stage_states("Mg")}
    assert states["rydberg"] and states["transitions"] and not states["lifetimes"]
    assert "lines" in D.import_file("lines", "Mg", CR_LINES)
    states = {sid: (ok, why) for sid, _l, ok, why in D.stage_states("Mg")}
    assert all(states[s][0] for s in ("lifetimes", "polarizability", "blackbody", "tweezer", "spectra"))
    assert not states["grotrian"][0] and "ORCA" in states["grotrian"][1]
    assert not states["hyperfine"][0] and not states["compare"][0]
    assert not D.needs_setup("Mg")
    assert paths.reference_path("NIST Levels", "Mg.csv").is_file() and paths.reference_path("f_values", "Mg_lines.csv").is_file()


def test_a_file_that_is_not_a_nist_export_is_refused(ws, tmp_path):
    bad = tmp_path / "junk.csv"
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    for key in ("levels", "lines"):
        with pytest.raises(ValueError):
            D.import_file(key, "Mg", bad)
    with pytest.raises(ValueError):
        D.import_file("levels", "Mg", tmp_path / "missing.csv")
    assert D.needs_setup("Mg")


def test_ground_numbers_count_once_added(ws):
    assert {i.key for i in D.missing_items("Mg")} >= {"mass", "alpha_core"}
    U.save_additions({"constants.alkali_mass": {"Mg": {"value": 24.305, "source": ""}}})
    U.reload("constants.alkali_mass")
    assert "mass" not in {i.key for i in D.missing_items("Mg")}


def test_rydberg_uses_nist_levels_only_for_a_neutral_that_is_not_an_alkali(ws):
    from linak.pipelines import rydberg

    D.import_file("levels", "Mg", CR_LEVELS)
    assert rydberg.main(["Mg", "--n-max", "10"]) == 0
    out = json.loads((ws / "data" / "Mg_rydberg.json").read_text(encoding="utf-8"))
    assert "no QDT" in str(out.get("method"))


def test_hyperfine_constants_are_checked_written_and_read_back(ws):
    with pytest.raises(ValueError, match="not in the nuclear data"):
        D.check_hf("Mg", "24", [("3p3/2", "1", "0", "0")])
    for rows, needle in (([("3p", "x", "", "")], "must be a number"), ([("p3/2", "1", "", "")], "looks like"),
                         ([("3p3/2", "1", "", ""), ("3p3/2", "2", "", "")], "twice"), ([("", "", "", "")], "at least one"),
                         ([("3p3/2", "", "", "")], "Fill in A")):
        with pytest.raises(ValueError, match=needle):
            D.check_hf("Mg", "25", rows)
    with pytest.raises(ValueError):
        D.check_hf("Mg", "abc", [("3p3/2", "1", "", "")])
    mass, rows = D.check_hf("Mg", "25", [("3p3/2", "-10,5", "1", ""), ("", "", "", "")])
    assert mass == 25 and rows == [("3p3/2", -10.5, 1.0, 0.0)]
    data = ws / "data"
    assert not D.present(D.BY_KEY["hf"], "Mg", "Mg", data_dir=data)
    D.write_hf_constants(data, "Mg", mass, rows)
    assert D.present(D.BY_KEY["hf"], "Mg", "Mg", data_dir=data) and D.read_hf_constants(data, "Mg") == (25, rows)
    states = {sid: ok for sid, _l, ok, _w in D.stage_states("Mg", data)}
    assert states["hyperfine"] is True
    assert not D.read_hf_constants(data, "Na")[1]
