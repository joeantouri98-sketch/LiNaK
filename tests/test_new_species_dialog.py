"""The set-up dialog for a species with no bundled data."""
from __future__ import annotations

import os
import shutil

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")
from PySide6.QtWidgets import QApplication, QFileDialog  # noqa: E402

from qali.core import paths, user_tables as U  # noqa: E402
from qali.core.paths import resource_root  # noqa: E402
from qali.gui import new_species_dialog as N  # noqa: E402


@pytest.fixture()
def dlg(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setenv(paths.PROJECT_ROOT_ENV, str(tmp_path))
    d = N.NewSpeciesDialog(tmp_path, "Mg")
    yield d
    shutil.rmtree(tmp_path / "tables", ignore_errors=True)
    for t in U.TABLES:
        try:
            U.reload(t.table_id)
        except Exception:
            pass
    assert app is not None


def _stage(d, name):
    return next(d.stages.item(i).text() for i in range(d.stages.count()) if name in d.stages.item(i).text())


def test_opens_with_everything_missing_and_updates_as_data_arrives(dlg, monkeypatch):
    assert _stage(dlg, "Lifetimes").startswith("✗") and dlg._status["levels"].text() == "missing"
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(resource_root() / "NIST Levels" / "Cr.csv"), "")))
    dlg._pick("levels")
    assert dlg._status["levels"].text() == "✓ present" and _stage(dlg, "Rydberg").startswith("✓")
    assert _stage(dlg, "Lifetimes").startswith("✗")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(resource_root() / "f_values" / "Cr_lines.csv"), "")))
    dlg._pick("lines")
    assert _stage(dlg, "Lifetimes").startswith("✓") and _stage(dlg, "Grotrian").startswith("✗")


def test_a_bad_file_leaves_a_message_and_changes_nothing(dlg, monkeypatch, tmp_path):
    bad = tmp_path / "x.csv"
    bad.write_text("nothing useful\n", encoding="utf-8")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(bad), "")))
    dlg._pick("levels")
    assert dlg._status["levels"].text() == "missing"
    assert "Could not read" in dlg.message.text() or "No levels" in dlg.message.text()


def test_typed_numbers_are_saved_into_the_tables(dlg):
    from qali.core import constants as C

    dlg._boxes["mass"].setText("24.305")
    assert dlg._status["mass"].text() == "✓ ready to save"
    dlg._boxes["alpha_core"].setText("-3")                      # not a valid value: stays missing
    assert dlg._status["alpha_core"].text() == "missing"
    dlg._save()
    assert C.ALKALI_MASS_AMU["Mg"] == 24.305
    assert "Mg" in U.load_additions()["constants.alkali_mass"]


def test_hyperfine_constants_are_entered_and_saved(dlg, monkeypatch):
    from qali.core import species_data as sd
    from qali.gui import hyperfine_dialog as H

    def fill(self):
        self.mass.setText("25")
        self.grid.setItem(0, 0, H.QTableWidgetItem("3p3/2"))
        self.grid.setItem(0, 1, H.QTableWidgetItem("-10.5"))
        self.accept()
        return 1

    monkeypatch.setattr(H.HyperfineDialog, "exec", fill)
    dlg._pick("hf")
    assert dlg._status["hf"].text() == "\u2713 ready to save" and _stage(dlg, "Hyperfine").startswith("\u2713")
    dlg._save()
    assert sd.read_hf_constants(dlg.data_dir, "Mg") == (25, [("3p3/2", -10.5, 0.0, 0.0)])


def test_the_hyperfine_dialog_refuses_a_spin_zero_isotope(dlg):
    from qali.gui.hyperfine_dialog import HyperfineDialog, QTableWidgetItem

    h = HyperfineDialog("Mg", dlg.data_dir)
    h.mass.setText("24")
    h.grid.setItem(0, 0, QTableWidgetItem("3p3/2"))
    h.grid.setItem(0, 1, QTableWidgetItem("1"))
    h.accept()
    assert "not in the nuclear data" in h.error.text() and h.result_rows is None
