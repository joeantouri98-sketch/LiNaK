# -*- coding: utf-8 -*-
"""The folder in use is seeded from the bundle, and the GUI starts in it.

Regression context: the GUI read its folder from the launch directory while
the bundled data was seeded into the per-user home (and, in the Nuitka
bundle, not found at all), so a fresh launch showed species without any
data. One rule now: the selected folder, else the per-user home; the GUI
copies the bundled data *and* the example plots/ORCA runs into it.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import linak.core.workspace as workspace_mod
from linak.core.workspace import Workspace, seed_bundled_data


@pytest.fixture
def fake_bundle(tmp_path, monkeypatch) -> Path:
    """A tiny stand-in for the bundled resources/ tree."""
    bundle = tmp_path / "bundle"
    for relative, text in {
        "data_json/Na_rydberg.json": '{"n": 1}',
        "data_json/arc/Na_rydberg.json": '{"arc": 1}',
        "plots/Na/Na_feshbach.html": "<html></html>",
        "orca_outputs/Na/Na_gs.out": "ORCA TERMINATED NORMALLY",
    }.items():
        path = bundle / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(workspace_mod, "reference_path", lambda *parts: bundle.joinpath(*parts))
    return bundle


def test_seeding_copies_only_pipeline_inputs_by_default(tmp_path, fake_bundle):
    ws = Workspace(root=tmp_path / "out")
    seed_bundled_data(ws)
    assert (ws.data_json_dir / "Na_rydberg.json").is_file()
    assert (ws.data_json_dir / "arc" / "Na_rydberg.json").is_file()
    assert not ws.plots_dir.exists()
    assert not ws.orca_outputs_dir.exists()


def test_seeding_with_examples_copies_plots_and_orca_outputs(tmp_path, fake_bundle):
    ws = Workspace(root=tmp_path / "out")
    seed_bundled_data(ws, include_examples=True)
    assert (ws.plots_dir / "Na" / "Na_feshbach.html").is_file()
    assert (ws.orca_outputs_dir / "Na" / "Na_gs.out").is_file()


def test_seeding_never_overwrites_and_leaves_no_temp_files(tmp_path, fake_bundle):
    ws = Workspace(root=tmp_path / "out")
    mine = ws.data_json_dir / "Na_rydberg.json"
    mine.parent.mkdir(parents=True)
    mine.write_text("my own run", encoding="utf-8")
    seed_bundled_data(ws, include_examples=True)
    assert mine.read_text(encoding="utf-8") == "my own run"
    leftovers = [p for p in ws.runtime_root.rglob("*") if p.name.endswith(".part")]
    assert leftovers == []


def test_seeding_into_a_checkout_stays_under_dot_linak(tmp_path, fake_bundle):
    checkout = tmp_path / "repo"
    (checkout / "src" / "linak").mkdir(parents=True)
    (checkout / "pyproject.toml").write_text("", encoding="utf-8")
    seed_bundled_data(Workspace(root=checkout), include_examples=True)
    assert (checkout / ".linak" / "data" / "Na_rydberg.json").is_file()
    assert not (checkout / "data").exists()


# ---------------------------------------------------------------------------
# GUI: start folder + seeding (real bundle)
# ---------------------------------------------------------------------------
@pytest.fixture
def window(tmp_path, monkeypatch, isolated_user_data):
    """A MainWindow with QSettings redirected into *tmp_path*."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path / "settings"))
    from linak.gui.main_window import MainWindow

    created = []

    def make():
        win = MainWindow()
        created.append(win)
        return win

    yield make
    for win in created:
        win.close()
    app.processEvents()


def _wait_for_seeding(win) -> None:
    for worker in list(win._seed_workers):
        worker.wait()
    from PySide6.QtWidgets import QApplication

    QApplication.processEvents()  # deliver the queued finished() -> refresh


def _species(win) -> list[str]:
    return [win.species_list.item(i).text() for i in range(win.species_list.count())]


def test_gui_starts_in_the_user_folder_and_populates_species(window, isolated_user_data):
    win = window()
    assert win.root == isolated_user_data / "LiNaK"
    _wait_for_seeding(win)
    assert {"Li", "Na", "Na_c1"} <= set(_species(win))

    win._set_species("Na")
    names = [win.species_json_list.item(i).text() for i in range(win.species_json_list.count())]
    plots = [win.species_html_list.item(i).text() for i in range(win.species_html_list.count())]
    assert "Na_rydberg.json" in names
    assert "Na_feshbach.html" in plots
    assert (win.root / "orca_outputs" / "Na").is_dir()


def test_gui_switches_folder_and_remembers_it(window, tmp_path):
    win = window()
    _wait_for_seeding(win)
    chosen = tmp_path / "project"
    chosen.mkdir()
    win.settings.setValue("root_path", str(chosen))
    win._switch_root(chosen)
    assert _species(win) == [], "the previous folder's species must not linger"
    _wait_for_seeding(win)
    assert "Na" in _species(win)
    assert (chosen / "data" / "Na_rydberg.json").is_file()

    assert window().root == chosen, "the chosen folder is remembered"

    win._use_default_root()
    _wait_for_seeding(win)
    assert win.root != chosen
    assert window().root == win.root, "going back to the default forgets the choice"
