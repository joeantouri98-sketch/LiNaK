"""The species list finds a species at any point of the pipeline chain."""

from qali.core.workspace import Workspace
from qali.gui.project_io import workspace_species


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")


def test_orca_only_species_is_listed(tmp_path):
    ws = Workspace(root=tmp_path)
    _touch(ws.data_json_dir / "Fe.json")
    (ws.orca_outputs_dir / "Fe").mkdir(parents=True)
    assert workspace_species(tmp_path) == ["Fe"]


def test_each_kind_of_result_counts(tmp_path):
    ws = Workspace(root=tmp_path)
    _touch(ws.data_json_dir / "Na_rydberg.json")
    _touch(ws.data_json_dir / "Mg_c1.json")
    (ws.orca_outputs_dir / "Ca").mkdir(parents=True)
    (ws.plots_dir / "K").mkdir(parents=True)
    assert workspace_species(tmp_path) == ["Ca", "K", "Mg_c1", "Na"]


def test_shared_files_and_folders_are_not_species(tmp_path):
    ws = Workspace(root=tmp_path)
    for name in ("nuclear_data.json", "comparison_table.json", "Na_lifetimes.json"):
        _touch(ws.data_json_dir / name)
    (ws.data_json_dir / "arc").mkdir()
    _touch(ws.plots_dir / "C6_matrix.html")
    assert workspace_species(tmp_path) == []


def test_empty_folder_has_no_species(tmp_path):
    assert workspace_species(tmp_path) == []
