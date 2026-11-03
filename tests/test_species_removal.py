"""Removing a species: what is touched, and that bundled data stays gone."""

from __future__ import annotations

import pytest

from linak.core import workspace as workspace_module
from linak.core.workspace import (
    Workspace,
    hidden_species,
    mark_removed,
    restore_species,
    seed_bundled_data,
    species_files,
)
from linak.gui.project_io import workspace_species


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")


@pytest.fixture
def ws(tmp_path):
    return Workspace(root=tmp_path / "out")


@pytest.fixture
def bundled(tmp_path, monkeypatch):
    """A small stand-in for the bundled resources."""
    root = tmp_path / "resources"
    for name in ("data_json/Na.json", "data_json/Na_c1.json", "data_json/K.json",
                 "plots/Na/a.html", "plots/shared.html", "orca_outputs/Na/run/x.out"):
        _touch(root / name)
    monkeypatch.setattr(workspace_module, "reference_path", lambda *parts: root.joinpath(*parts))


def test_files_are_the_species_own(ws):
    for name in ("Na.json", "Na_lifetimes.json", "Na_c1.json", "Na_c1_soc.json",
                 "nuclear_data.json", "comparison_table.json", "arc/Na_rydberg.json"):
        _touch(ws.data_json_dir / name)
    (ws.plots_dir / "Na").mkdir(parents=True)
    _touch(ws.plots_dir / "C6_matrix.html")
    (ws.orca_outputs_dir / "Na").mkdir(parents=True)

    files = species_files(ws, "Na")
    assert [p.name for p in files["data_json"]] == ["Na.json", "Na_lifetimes.json", "Na_rydberg.json"]
    assert files["plots"] == [ws.plots_dir / "Na"]
    assert files["orca_outputs"] == [ws.orca_outputs_dir / "Na"]

    ion = species_files(ws, "Na_c1")
    assert [p.name for p in ion["data_json"]] == ["Na_c1.json", "Na_c1_soc.json"]


@pytest.mark.parametrize("name", ["..", "../Na", "a/b", ""])
def test_unsafe_names_match_nothing(ws, name):
    _touch(ws.data_json_dir / "Na.json")
    assert not any(species_files(ws, name).values())


def test_deleted_bundled_species_is_not_reseeded(ws, bundled):
    seed_bundled_data(ws, include_examples=True)
    (ws.data_json_dir / "Na.json").unlink()
    mark_removed(ws, "Na", no_seed=("data_json",))

    seed_bundled_data(ws, include_examples=True)

    assert not (ws.data_json_dir / "Na.json").exists()
    assert (ws.data_json_dir / "Na_c1.json").exists()  # the ion is another species
    assert (ws.data_json_dir / "K.json").exists()


def test_removed_trees_are_skipped_one_by_one(ws, bundled):
    mark_removed(ws, "Na", no_seed=("plots", "orca_outputs"))
    seed_bundled_data(ws, include_examples=True)
    assert (ws.data_json_dir / "Na.json").exists()
    assert not (ws.plots_dir / "Na").exists()
    assert not (ws.orca_outputs_dir / "Na").exists()
    assert (ws.plots_dir / "shared.html").exists()


def test_restore_brings_bundled_data_back(ws, bundled):
    mark_removed(ws, "Na", no_seed=("data_json",))
    seed_bundled_data(ws)
    assert not (ws.data_json_dir / "Na.json").exists()

    assert restore_species(ws, "Na") is True
    seed_bundled_data(ws)
    assert (ws.data_json_dir / "Na.json").exists()
    assert restore_species(ws, "Na") is False


def test_list_only_keeps_files_and_hides_species(ws):
    _touch(ws.data_json_dir / "Fe.json")
    mark_removed(ws, "Fe", hide=True)

    assert (ws.data_json_dir / "Fe.json").exists()
    assert hidden_species(ws) == {"Fe"}
    assert workspace_species(ws.root) == []

    restore_species(ws, "Fe")
    assert workspace_species(ws.root) == ["Fe"]


@pytest.mark.parametrize("text", ["not json", "[1, 2]", '{"hidden": 3, "no_seed": 4}'])
def test_damaged_record_is_ignored(ws, bundled, text):
    ws.runtime_root.mkdir(parents=True)
    (ws.runtime_root / workspace_module.REMOVED_SPECIES_FILE).write_text(text, encoding="utf-8")
    seed_bundled_data(ws)
    assert (ws.data_json_dir / "Na.json").exists()
    assert hidden_species(ws) == set()
