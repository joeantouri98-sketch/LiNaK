"""Themes and window layout: readable colours, and the panels doing their jobs."""

from __future__ import annotations

import os
import re
from dataclasses import fields

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QEvent, QPoint, QPointF, QSettings, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent, QPalette  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from linak.gui import theme  # noqa: E402
from linak.gui.stage_list import DONE_ROLE  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def app():
    yield QApplication.instance() or QApplication([])


def _luminance(colour: str) -> float:
    channels = [int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    r, g, b = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a: str, b: str) -> float:
    high, low = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


@pytest.mark.parametrize("name", [theme.DARK, theme.LIGHT])
@pytest.mark.parametrize("foreground, background", [
    ("text", "bg"), ("text", "surface"), ("text", "raised"), ("text", "selection"),
    ("text_muted", "bg"), ("text_muted", "surface"), ("text_muted", "raised"),
    ("accent_text", "accent"), ("accent_ink", "bg"), ("accent_ink", "surface"),
    ("accent_ink", "selection"), ("danger", "bg"), ("danger", "raised"),
])
def test_text_meets_wcag_aa(name, foreground, background):
    tokens = theme.THEMES[name]
    ratio = _contrast(getattr(tokens, foreground), getattr(tokens, background))
    assert ratio >= 4.5, f"{name}: {foreground} on {background} is {ratio:.2f}:1"


@pytest.mark.parametrize("name", [theme.DARK, theme.LIGHT])
def test_input_outlines_are_visible(name):
    tokens = theme.THEMES[name]
    assert _contrast(tokens.border_strong, tokens.surface) >= 2.0


@pytest.mark.parametrize("name", [theme.DARK, theme.LIGHT])
def test_stylesheet_is_fully_filled_in(name):
    sheet = theme.build_stylesheet(theme.THEMES[name])
    assert "{bg}" not in sheet and "{{" not in sheet
    used = set(re.findall(r"#[0-9a-f]{6}", sheet))
    assert used <= {getattr(theme.THEMES[name], f.name) for f in fields(theme.Tokens)}


def test_apply_theme_sets_the_palette(app):
    theme.apply_theme(app, theme.LIGHT)
    assert app.palette().color(QPalette.Window).name() == theme.THEMES[theme.LIGHT].bg
    theme.apply_theme(app, theme.DARK)
    assert app.palette().color(QPalette.Window).name() == theme.THEMES[theme.DARK].bg


@pytest.fixture
def window(tmp_path, monkeypatch):
    pytest.importorskip("PySide6.QtWebEngineWidgets")
    from linak.core.workspace import Workspace
    from linak.gui import main_window as module

    workspace = Workspace(root=tmp_path / "out")
    workspace.data_json_dir.mkdir(parents=True)
    (workspace.data_json_dir / "Fe.json").write_text('{"a": 1}', encoding="utf-8")
    monkeypatch.setenv("LINAK_PROJECT_ROOT", str(workspace.root))
    monkeypatch.setattr(module, "seed_bundled_data", lambda *a, **k: None)
    win = module.MainWindow()
    for worker in list(win._seed_workers):
        worker.wait()
    QApplication.processEvents()
    yield win
    win.close()


def test_group_headings_are_not_stages(window):
    headings = [window.stage_list.item(i) for i in range(window.stage_list.count())
                if window.stage_list.item(i).data(Qt.UserRole) is None]
    assert [item.text() for item in headings][0] == "Structure models"
    assert all(not (item.flags() & Qt.ItemIsSelectable) for item in headings)


def test_status_dots_follow_the_files(window):
    window._set_species("Fe")
    done = {window.stage_list.item(i).data(Qt.UserRole): window.stage_list.item(i).data(DONE_ROLE)
            for i in range(window.stage_list.count())}
    done.pop(None)
    assert done["rydberg"] is False
    assert all(isinstance(value, bool) for value in done.values())


def test_selecting_a_species_highlights_its_chip_and_clears_the_box(window):
    window.species_input.setText("Fe")
    window._set_species_from_input()
    assert window.species_list.currentItem().text() == "Fe"
    assert window.species_input.text() == ""


def test_opening_an_output_shows_the_viewer(window, tmp_path):
    window.center_tabs.setCurrentWidget(window.overview_view)
    window._load_output(tmp_path / "out" / "data" / "Fe.json", "JSON")
    assert window.center_tabs.currentWidget() is window.viewer_stack
    assert window.viewer_stack.currentWidget() is window.json_view_box


def _drag_shut(splitter, index):
    """Collapse a panel the way a drag does: its handle moves to the edge."""
    handle = index if index else 1
    splitter.moveSplitter(0 if index == 0 else 10 ** 5, handle)
    QApplication.processEvents()


def test_console_can_fold_and_comes_back_for_a_run(window):
    window.show()
    QApplication.processEvents()
    window.bottom_tabs.setCurrentWidget(window.bottom_overview)
    window.bottom_action.setChecked(False)
    assert window.center_splitter.sizes()[1] == 0
    window._show_console()
    assert window.center_splitter.sizes()[1] > 0
    assert window.bottom_tabs.currentWidget() is window.console


def _record_refreshes(window, monkeypatch):
    calls = []
    for name, view in (("center", window.overview_view), ("bottom", window.bottom_overview)):
        monkeypatch.setattr(view, "refresh", lambda _root, species, name=name: calls.append((name, species)))
    return calls


def test_overview_is_offered_in_both_places(window):
    assert window.center_tabs.indexOf(window.overview_view) >= 0
    assert window.bottom_tabs.indexOf(window.bottom_overview) >= 0
    assert window.center_tabs.tabText(0) == "Viewer"
    assert window.bottom_tabs.tabText(0) == "Console"


def test_a_hidden_overview_is_not_refreshed(window, monkeypatch):
    calls = _record_refreshes(window, monkeypatch)
    window._set_species("Fe")
    assert calls == []


def test_each_overview_catches_up_when_shown(window, monkeypatch):
    calls = _record_refreshes(window, monkeypatch)
    window._set_species("Fe")
    window.bottom_tabs.setCurrentWidget(window.bottom_overview)
    assert calls == [("bottom", "Fe")]
    window.center_tabs.setCurrentWidget(window.overview_view)
    assert calls == [("bottom", "Fe"), ("center", "Fe")]


def test_a_shown_overview_follows_the_species(window, monkeypatch):
    calls = _record_refreshes(window, monkeypatch)
    window.bottom_tabs.setCurrentWidget(window.bottom_overview)
    calls.clear()
    window._set_species("Fe")
    assert ("bottom", "Fe") in calls
    assert not any(place == "center" for place, _species in calls)


@pytest.mark.parametrize("index", [0, 2])
def test_dragging_a_side_panel_shut_gives_the_viewer_its_room(window, index):
    window.show()
    QApplication.processEvents()
    before = window.main_splitter.sizes()
    _drag_shut(window.main_splitter, index)
    after = window.main_splitter.sizes()
    assert after[index] == 0
    assert after[1] > before[1]


def test_view_menu_follows_a_drag_and_reopens_at_the_old_size(window):
    window.show()
    QApplication.processEvents()
    action = next(a for a, _s, _p in window._panel_actions if a.text() == "Options panel")
    width = window.main_splitter.sizes()[2]
    action.setChecked(False)
    assert window.main_splitter.sizes()[2] == 0
    action.setChecked(True)
    assert window.main_splitter.sizes()[2] == pytest.approx(width, abs=2)

    _drag_shut(window.main_splitter, 2)
    assert not action.isChecked()
    action.setChecked(True)
    assert window.main_splitter.sizes()[2] > 0


def test_the_viewer_never_collapses(window):
    assert not window.main_splitter.isCollapsible(1)
    assert not window.center_splitter.isCollapsible(0)
    window._set_console_panel_on_top(True)
    assert not window.center_splitter.isCollapsible(1)
    assert window.center_splitter.isCollapsible(0)


def _shown(window):
    window.show()
    QApplication.processEvents()
    return window


def test_panel_moves_above_the_viewer_and_keeps_its_height(window):
    _shown(window)
    viewer_height, panel_height = window.center_splitter.sizes()
    window.move_panel_btn.click()
    assert window.center_splitter.widget(0) is window.bottom_panel
    assert window.center_splitter.sizes()[0] == pytest.approx(panel_height, abs=3)
    assert window.panel_on_top_action.isChecked()
    window.move_panel_btn.click()
    assert window.center_splitter.widget(1) is window.bottom_panel
    assert window.center_splitter.sizes()[1] == pytest.approx(panel_height, abs=3)


def test_view_menu_moves_the_panel(window):
    _shown(window)
    window.panel_on_top_action.setChecked(True)
    assert window.center_splitter.widget(0) is window.bottom_panel


def test_panel_position_is_saved(window):
    _shown(window)
    window._set_console_panel_on_top(True)
    assert window.settings.value("layout/console_panel_on_top", type=bool) is True


def _drag_tab_bar_to(window, y_fraction):
    bar = window.bottom_tabs.tabBar()
    splitter = window.center_splitter
    target = splitter.mapToGlobal(QPoint(50, int(splitter.height() * y_fraction)))
    event = QMouseEvent(QEvent.MouseMove, QPointF(5, 5), QPointF(target), Qt.NoButton,
                        Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(bar, event)


def test_dragging_the_tab_bar_moves_the_panel_both_ways(window):
    _shown(window)
    _drag_tab_bar_to(window, 0.5)  # the middle does nothing, so it can't flicker
    assert window.center_splitter.widget(1) is window.bottom_panel
    _drag_tab_bar_to(window, 0.2)
    assert window.center_splitter.widget(0) is window.bottom_panel
    _drag_tab_bar_to(window, 0.5)
    assert window.center_splitter.widget(0) is window.bottom_panel
    _drag_tab_bar_to(window, 0.8)
    assert window.center_splitter.widget(1) is window.bottom_panel


def test_collapse_still_works_with_the_panel_on_top(window):
    _shown(window)
    window._set_console_panel_on_top(True)
    window.bottom_action.setChecked(False)
    assert window.center_splitter.sizes()[0] == 0
    window._show_console()
    assert window.center_splitter.sizes()[0] > 0


@pytest.fixture
def overview_data(tmp_path):
    import json

    from linak.core.workspace import Workspace

    ws = Workspace(root=tmp_path / "overview")
    ws.data_json_dir.mkdir(parents=True)
    (ws.data_json_dir / "Na_hyperfine.json").write_text(json.dumps({"states": [{
        "label": "3s1/2", "J": 0.5, "A_MHz": 885.813,
        "F_levels": [{"F": 1, "E_MHz": -1107.2, "gF": -0.5}]}]}), encoding="utf-8")
    (ws.data_json_dir / "Na_feshbach_out.json").write_text(json.dumps({"resonances": [{
        "label": "broad s-wave resonance", "channel": "|1,1> + |1,1> entrance", "B0_G": 907.0, "ref": "Long reference " * 12}]}),
        encoding="utf-8")
    return ws.root


@pytest.mark.parametrize("width, columns", [(420, 1), (760, 2), (1300, 3)])
def test_overview_fits_its_width(overview_data, width, columns):
    from linak.gui.overview import SpeciesOverviewWidget

    view = SpeciesOverviewWidget()
    view.resize(width, 600)
    view.show()
    QApplication.processEvents()
    view.refresh(overview_data, "Na")
    QTest.qWait(150)  # the vertical scrollbar's arrival narrows the viewport a moment later
    assert view._column_count == columns
    assert view._root_widget.width() <= view._scroll.viewport().width()
    assert view._scroll.horizontalScrollBar().maximum() == 0
    view.close()


def test_overview_reflows_when_resized(overview_data):
    from linak.gui.overview import SpeciesOverviewWidget

    view = SpeciesOverviewWidget()
    view.resize(1300, 600)
    view.show()
    view.refresh(overview_data, "Na")
    QApplication.processEvents()
    assert view._column_count == 3
    view.resize(420, 600)
    QApplication.processEvents()
    assert view._column_count == 1
    assert all(not card.isHidden() for card in view._cards)
    view.close()


def test_tests_never_touch_the_real_settings(window, tmp_path):
    assert window.settings.format() == QSettings.IniFormat
    assert window.settings.fileName().endswith("gui_settings.ini")
    assert str(tmp_path.name) in window.settings.fileName()


def test_theme_choice_is_saved(window):
    window._set_theme(theme.LIGHT)
    assert window.settings.value("appearance/theme") == theme.LIGHT
    window._set_theme(theme.DARK)
