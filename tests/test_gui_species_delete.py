"""The Delete species dialog, offscreen, and the window's handling of it."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtWidgets import QApplication, QDialog, QDialogButtonBox  # noqa: E402

from qali.core.workspace import Workspace, hidden_species, species_files  # noqa: E402
from qali.gui.species_delete import DeleteSpeciesDialog  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def app():
    yield QApplication.instance() or QApplication([])


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")


@pytest.fixture
def populated(tmp_path):
    ws = Workspace(root=tmp_path)
    _touch(ws.data_json_dir / "Fe.json")
    _touch(ws.orca_outputs_dir / "Fe" / "run" / "Fe.out")
    return ws


def test_default_keeps_the_files(populated):
    dialog = DeleteSpeciesDialog("Fe", species_files(populated, "Fe"))
    assert dialog.selected_trees() == ()


def test_delete_selects_only_trees_with_files(populated):
    dialog = DeleteSpeciesDialog("Fe", species_files(populated, "Fe"))
    dialog.delete_data.setChecked(True)
    assert dialog.selected_trees() == ("data_json", "orca_outputs")
    assert not dialog.tree_boxes["plots"].isEnabled()


def test_cancel_stays_the_default_button_once_shown(populated):
    dialog = DeleteSpeciesDialog("Fe", species_files(populated, "Fe"))
    dialog.show()
    QApplication.processEvents()
    assert dialog.buttons.button(QDialogButtonBox.Cancel).isDefault()
    assert not dialog.buttons.button(QDialogButtonBox.Ok).isDefault()
    dialog.close()


def test_trees_can_be_unticked(populated):
    dialog = DeleteSpeciesDialog("Fe", species_files(populated, "Fe"))
    dialog.delete_data.setChecked(True)
    dialog.tree_boxes["orca_outputs"].setChecked(False)
    assert dialog.selected_trees() == ("data_json",)
    dialog.tree_boxes["data_json"].setChecked(False)
    assert not dialog.buttons.button(QDialogButtonBox.Ok).isEnabled()


def test_species_without_files_can_only_be_unlisted(tmp_path):
    dialog = DeleteSpeciesDialog("Ca", species_files(Workspace(root=tmp_path), "Ca"))
    assert not dialog.delete_data.isEnabled()


@pytest.fixture
def window(populated, monkeypatch):
    pytest.importorskip("PySide6.QtWebEngineWidgets")
    from qali.gui import main_window as module

    monkeypatch.setenv("QALI_PROJECT_ROOT", str(populated.root))
    monkeypatch.setattr(module, "seed_bundled_data", lambda *a, **k: None)
    # Whether a Recycle Bin exists depends on the machine, not on this code.
    monkeypatch.setattr(module, "move_to_trash", module.delete_permanently)
    win = module.MainWindow()
    for worker in list(win._seed_workers):
        worker.wait()
    QApplication.processEvents()
    yield win
    win.close()


def _answer(monkeypatch, delete_trees):
    from qali.gui import main_window as module

    class Answer(module.DeleteSpeciesDialog):
        def exec(self):
            return QDialog.Accepted

        def selected_trees(self):
            return delete_trees

    monkeypatch.setattr(module, "DeleteSpeciesDialog", Answer)


def _listed(win):
    return [win.species_list.item(i).text() for i in range(win.species_list.count())]


def test_window_list_only(window, populated, monkeypatch):
    window._set_species("Fe")
    _answer(monkeypatch, ())
    window._delete_species("Fe")
    assert _listed(window) == []
    assert window.current_species == ""
    assert (populated.data_json_dir / "Fe.json").exists()
    assert hidden_species(populated) == {"Fe"}


def test_window_delete_everything_then_type_it_back(window, populated, monkeypatch):
    window._set_species("Fe")
    _answer(monkeypatch, ("data_json", "orca_outputs"))
    window._delete_species("Fe")
    assert _listed(window) == []
    assert not (populated.data_json_dir / "Fe.json").exists()
    assert not (populated.orca_outputs_dir / "Fe").exists()

    window.species_input.setText("Fe")
    window._set_species_from_input()
    assert window.current_species == "Fe"
    assert "Fe" in _listed(window)


def test_no_recycle_bin_asks_before_deleting_for_good(window, populated, monkeypatch):
    from qali.gui import main_window as module

    monkeypatch.setattr(module, "move_to_trash", list)
    asked = []
    monkeypatch.setattr(module.QMessageBox, "question", lambda *a: asked.append(a) or module.QMessageBox.Yes)
    _answer(monkeypatch, ("data_json", "orca_outputs"))
    window._delete_species("Fe")
    assert len(asked) == 1
    assert not (populated.data_json_dir / "Fe.json").exists()
    assert not (populated.orca_outputs_dir / "Fe").exists()


def test_declining_permanent_delete_keeps_everything(window, populated, monkeypatch):
    from qali.gui import main_window as module

    window._set_species("Fe")
    monkeypatch.setattr(module, "move_to_trash", list)
    monkeypatch.setattr(module.QMessageBox, "question", lambda *a: module.QMessageBox.No)
    monkeypatch.setattr(module.QMessageBox, "warning", lambda *a: None)
    _answer(monkeypatch, ("data_json",))
    window._delete_species("Fe")
    assert (populated.data_json_dir / "Fe.json").exists()
    assert not (populated.runtime_root / "removed_species.json").exists()
    assert _listed(window) == ["Fe"]
    assert window.current_species == "Fe"


def test_window_partial_delete_keeps_species_listed(window, populated, monkeypatch):
    window._set_species("Fe")
    _answer(monkeypatch, ("orca_outputs",))
    window._delete_species("Fe")
    assert _listed(window) == ["Fe"]
    assert window.current_species == "Fe"
    assert (populated.data_json_dir / "Fe.json").exists()
