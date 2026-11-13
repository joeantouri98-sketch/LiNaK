"""
Qali Desktop Control Panel
----------------------------
A plain, native desktop GUI (PySide6/Qt) for the Qali alkali-atom AMO
pipeline: run pipeline stages (including ORCA) for a species and browse
both freshly produced and already-on-disk plots/JSON, with the Plotly
HTML plots rendered directly inside the window.

This does not reimplement any physics: each stage runs the pipeline as a
``qali --pipeline <name> ...`` subprocess with the same flags you'd use
from the command line, then loads the JSON/HTML files it produces.

Run with ``qali-gui`` (or ``uv run qali-gui`` in a checkout).

Outputs go to Qali's per-user data folder unless another folder is picked
with File > Open output folder...; the bundled data and example results are
copied into the folder in use (never overwriting). See Help in the menu bar
for a flags reference and short physics notes per stage.

The stage table lives in :mod:`.stages`, the Overview tab in
:mod:`.overview`, subprocess helpers in :mod:`.stage_process` and the
offline report cache in :mod:`.plot_cache`.
"""
from __future__ import annotations

import codecs
import csv
import json
import os
import shutil
import sys
import tempfile
import time
from html import escape
from pathlib import Path

from PySide6.QtCore import (
    QEvent,
    QProcess,
    QProcessEnvironment,
    QSettings,
    QSize,
    Qt,
    QThread,
    QTimer,
    QUrl,
    Signal,
)
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QDesktopServices,
    QKeySequence,
    QPainter,
    QShortcut,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplashScreen,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from qali.core import species_data
from qali.core.elements import expand_element_families
from qali.core.paths import is_frozen_app, project_root_from_env
from qali.core.species import resolve_species
from qali.core.workspace import (
    Workspace,
    default_workspace,
    mark_removed,
    restore_species,
    seed_bundled_data,
    species_files,
)

from .compare_table_view import CompareTableView
from .dataset_editor import DatasetEditorDialog
from .find_bar import with_find_bar
from .extrapolation_dialog import active_config_path
from . import wheel_guard
from .save_as import OutputSnapshot, sanitize
from .web_downloads import attach_web_downloads
from .preferences_dialog import PreferencesDialog, load_overrides as load_colour_overrides
from .extrapolation_dialog import open_for_species as open_extrapolation_dialog
from .dataset_templates import CSV_DATASET_TEMPLATES, JSON_DATASET_TEMPLATES
from .help_dialog import HelpDialog
from .overview import SpeciesOverviewWidget
from .plot_cache import local_plot_copy
from .new_species_dialog import NewSpeciesDialog
from .properties_dialog import PropertiesDialog
from .species_delete import DeleteSpeciesDialog, delete_permanently, move_to_trash
from .project_io import (
    open_pipeline_overview_html,
    output_actions,
    shared_outputs,
    species_outputs,
    workspace_species,
)
from .stage_process import (
    cli_log_paths,
    kill_process_tree,
    stage_env_values,
    stage_program,
    start_in_own_session,
)
from .stage_list import DONE_ROLE, SYMBOL_ROLE, StageDelegate
from .stages import (
    PIPELINE,
    STAGE_BY_ID,
    STAGE_GROUPS,
    STAGE_PHYSICS,
    FLAG_SECTIONS,
    flag_section,
    input_hint,
    output_hint,
    isotope_prerequisite_issues,
    stage_argv,
    stage_has_output,
)
from . import brand
from .theme import DARK, SYSTEM, THEME_LABELS, THEME_NAMES, apply_theme, is_dark, mono_font

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    HAS_WEBENGINE = True
except ImportError:
    HAS_WEBENGINE = False


ORG_NAME = "Qali"
APP_NAME = "ControlPanel"
SPLASH_TAGLINE = "An integrated computational toolkit for alkali-atom AMO physics"
LEGACY_ORG_NAME = "LiNaK"   # the app's name until 2026-11; its saved settings are carried over once
# QSettings key remembering the folder picked with File > Open output folder...
ROOT_SETTING = "root_path"
# Set to an .ini path to keep the GUI's settings there instead of the user's
# own store (the registry on Windows). The test suite always sets it.
SETTINGS_FILE_ENV = "QALI_SETTINGS_FILE"
LEGACY_SETTINGS_FILE_ENV = "LINAK_SETTINGS_FILE"
THEME_SETTING = "appearance/theme"
CENTER_PANEL = 1  # the viewer: the one panel that never collapses
SPLITTER_HANDLE_WIDTH = 7
PANEL_REOPEN_SIZE = 280
CONSOLE_ON_TOP_SETTING = "layout/console_panel_on_top"
# Dragging the panel's tab bar past these fractions of the height moves it.
MOVE_PANEL_UP_BELOW = 0.35
MOVE_PANEL_DOWN_ABOVE = 0.65


def open_settings() -> QSettings:
    """The GUI's settings store."""
    path = os.environ.get(SETTINGS_FILE_ENV) or os.environ.get(LEGACY_SETTINGS_FILE_ENV)
    if path:
        return QSettings(path, QSettings.IniFormat)
    settings = QSettings(ORG_NAME, APP_NAME)
    if not settings.allKeys():
        adopt_settings(QSettings(LEGACY_ORG_NAME, APP_NAME), settings)
    return settings


def adopt_settings(old: QSettings, new: QSettings) -> int:
    """Copy every key of *old* into an empty *new*; returns how many were copied."""
    keys = old.allKeys()
    for key in keys:
        new.setValue(key, old.value(key))
    return len(keys)


class _SeedWorker(QThread):
    """Copies the bundled data and example results into a folder.

    A first copy is ~170 MB, so it runs off the GUI thread; existing files
    are never overwritten.
    """

    failed = Signal(str)

    def __init__(self, root: Path, parent=None):
        super().__init__(parent)
        self.root = root

    def run(self):
        try:
            seed_bundled_data(Workspace(root=self.root), include_examples=True)
        except OSError as exc:
            self.failed.emit(f"Could not copy the bundled data into {self.root}: {exc}")


NO_SAVE_AS = ("runorca", "orca_to_json")


class _OptionLabel(QLabel):
    """A one-line label for the options form: it never asks for more width than the panel has (no sideways scrolling)
    and shortens its text with an ellipsis only when the panel really is too narrow."""

    def minimumSizeHint(self):
        return QSize(0, super().minimumSizeHint().height())

    def paintEvent(self, event):
        text = self.text()
        shown = self.fontMetrics().elidedText(text, Qt.ElideRight, self.width())
        if shown == text:
            super().paintEvent(event)
            return
        painter = QPainter(self)
        painter.setPen(self.palette().color(self.foregroundRole()))
        painter.drawText(self.rect(), int(self.alignment()) | Qt.AlignVCenter, shown)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Qali")
        self.resize(1440, 900)

        self.settings = open_settings()
        self.root = self._initial_root()
        self.current_species = ""
        self.process: QProcess | None = None
        self.current_stage = None
        self._snapshot = None            # outputs before a run filed under a name (Save as)
        self._snapshot_name = None
        self._save_as_edit = None
        self.fallback_path: Path | None = None
        self._flag_widgets: dict = {}
        self._basis_picker_checkbox = None
        self._basis_picker_combo = None
        self._basis_cache_dir = Path(tempfile.mkdtemp(prefix="qali_gui_basis_"))
        self._plot_cache_dir = Path(tempfile.mkdtemp(prefix="qali_gui_"))
        self._seed_workers: set[_SeedWorker] = set()

        brand.load_colours(self.settings)
        self._build_ui()
        self._build_menu()
        load_colour_overrides(self.settings)
        saved_theme = self.settings.value(THEME_SETTING, DARK)
        wheel_guard.install(QApplication.instance())
        self._set_theme(saved_theme if saved_theme in THEME_NAMES else DARK, save=False)
        self._theme_actions[self.theme_name].setChecked(True)
        QApplication.instance().styleHints().colorSchemeChanged.connect(
            self._on_system_scheme_changed)
        self._set_console_panel_on_top(
            self.settings.value(CONSOLE_ON_TOP_SETTING, False, type=bool), save=False)
        self._seed_root()
        self._refresh_all()

    def closeEvent(self, event):
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            kill_process_tree(self.process)
            self.process.waitForFinished(3000)
        self.process = None
        if getattr(self, "_tests_dialog", None) is not None:
            self._tests_dialog.shutdown()
        # A QThread destroyed while running aborts the process; copies are
        # atomic, so waiting for them is the only safe exit.
        for worker in list(self._seed_workers):
            worker.wait()
        shutil.rmtree(self._plot_cache_dir, ignore_errors=True)
        shutil.rmtree(self._basis_cache_dir, ignore_errors=True)
        super().closeEvent(event)

    # output folder
    def _initial_root(self) -> Path:
        """The folder this session starts in.

        An explicit ``QALI_PROJECT_ROOT`` (or the old ``LINAK_PROJECT_ROOT``) wins; otherwise the folder picked
        with File > Open output folder... in an earlier session, if it still
        exists; otherwise Qali's per-user data folder.
        """
        saved = self.settings.value(ROOT_SETTING, "")
        if saved and not project_root_from_env() and Path(saved).is_dir():
            return Path(saved)
        return default_workspace().root

    def _seed_root(self) -> None:
        """Copy the bundled data into the current folder, then refresh."""
        worker = _SeedWorker(self.root, self)
        worker.failed.connect(self._on_seed_failed)
        worker.finished.connect(self._on_seed_finished)
        self._seed_workers.add(worker)
        worker.start()

    def _on_seed_failed(self, message: str) -> None:
        self._log(message)
        QMessageBox.warning(self, "Bundled data not copied", message)

    def _on_seed_finished(self) -> None:
        worker = self.sender()
        self._seed_workers.discard(worker)
        worker.deleteLater()
        self._refresh_all()

    def _build_menu(self):
        file_menu = self.menuBar().addMenu("&File")

        open_act = QAction("Open output folder...", self)
        open_act.triggered.connect(self._choose_root)
        file_menu.addAction(open_act)

        default_act = QAction("Use default output folder", self)
        default_act.triggered.connect(self._use_default_root)
        file_menu.addAction(default_act)

        file_menu.addSeparator()

        quit_act = QAction("Quit", self)
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        edit_menu = self.menuBar().addMenu("&Edit")

        new_dataset_action = QAction("New...", self)
        new_dataset_action.triggered.connect(self._new_dataset_dialog)
        edit_menu.addAction(new_dataset_action)

        open_existing_action = QAction("Open existing...", self)
        open_existing_action.triggered.connect(
            self._open_existing_dataset
        )
        edit_menu.addAction(open_existing_action)

        properties_action = QAction("Species properties...", self)
        properties_action.triggered.connect(lambda: self._open_properties())
        edit_menu.addAction(properties_action)
        delete_species_action = QAction("Delete species...", self)
        delete_species_action.triggered.connect(lambda: self._delete_species())
        edit_menu.addAction(delete_species_action)

        edit_menu.addSeparator()

        prefs_action = QAction("Preferences...", self)
        prefs_action.setShortcut("Ctrl+,")
        prefs_action.triggered.connect(lambda _checked=False: self._open_preferences())
        edit_menu.addAction(prefs_action)

        edit_menu.addSeparator()

        json_menu = edit_menu.addMenu("JSON datasets")

        for name in [
            "feshbach",
            "transitions",
            "lifetimes",
            "polarizability",
            "blackbody",
            "hyperfine",
            "rydberg",
        ]:
            action = QAction(name.replace("_", " ").title(), self)
            action.triggered.connect(
                lambda _checked=False, dataset_name=name:
                self._open_dataset_editor("json", dataset_name)
            )
            json_menu.addAction(action)

        csv_menu = edit_menu.addMenu("CSV datasets")

        for name in [
            "lines",
            "levels",
            "nuclear_data",
            "transitions",
            "lifetimes",
        ]:
            action = QAction(name.replace("_", " ").title(), self)
            action.triggered.connect(
                lambda _checked=False, dataset_name=name:
                self._open_dataset_editor("csv", dataset_name)
            )
            csv_menu.addAction(action)

        view_menu = self.menuBar().addMenu("&View")

        theme_menu = view_menu.addMenu("Theme")
        theme_group = QActionGroup(self)
        self._theme_actions = {}
        for name, label in THEME_LABELS.items():
            action = QAction(label, self, checkable=True)
            action.triggered.connect(lambda _checked=False, theme=name: self._set_theme(theme))
            theme_group.addAction(action)
            theme_menu.addAction(action)
            self._theme_actions[name] = action

        view_menu.addSeparator()
        self._panel_actions = []
        self._panel_sizes = {}
        self._add_panel_action(view_menu, "Pipeline panel", "Ctrl+B",
                               self.main_splitter, self.sidebar)
        self._add_panel_action(view_menu, "Options panel", "Ctrl+\\",
                               self.main_splitter, self.inspector)
        self.bottom_action = self._add_panel_action(
            view_menu, "Console panel", "Ctrl+`", self.center_splitter, self.bottom_panel)
        self.panel_on_top_action = QAction("Console panel on top", self, checkable=True)
        self.panel_on_top_action.setShortcut("Ctrl+Shift+`")
        self.panel_on_top_action.toggled.connect(self._set_console_panel_on_top)
        view_menu.addAction(self.panel_on_top_action)

        run_menu = self.menuBar().addMenu("&Run")
        run_action = QAction("Run stage", self)
        run_action.setShortcut("F5")
        run_action.triggered.connect(self._run_stage)
        run_menu.addAction(run_action)
        stop_action = QAction("Stop", self)
        stop_action.setShortcut("Shift+F5")
        stop_action.triggered.connect(self._stop_stage)
        run_menu.addAction(stop_action)
        run_menu.addSeparator()
        tests_action = QAction("Tests...", self)
        tests_action.triggered.connect(self._open_tests)
        run_menu.addAction(tests_action)

        help_menu = self.menuBar().addMenu("&Help")

        help_act = QAction("Help contents...", self)
        help_act.triggered.connect(self._show_help)
        help_menu.addAction(help_act)

        help_menu.addSeparator()

        overview_act = QAction(
            "Open full Pipeline Overview (HTML)...",
            self,
        )
        overview_act.triggered.connect(
            lambda: open_pipeline_overview_html(self.root, self)
        )
        help_menu.addAction(overview_act)

        about_act = QAction("About", self)
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)

    def _show_help(self):
        dlg = HelpDialog(self.root, PIPELINE, STAGE_PHYSICS, self)
        dlg.exec()

    def _show_about(self):
        from qali import __version__

        QMessageBox.about(
            self, "About Qali",
            f"<h3>Qali {escape(__version__)}</h3>"
            "<p>Formerly LiNaK.</p>"
            "<p>An alkali-atom AMO pipeline: NIST / quantum-defect levels, transitions, lifetimes, "
            "polarizability, blackbody shifts, optical tweezers, hyperfine and Feshbach, with interactive "
            "plots and optional ORCA and ARC bridges. This panel runs the pipelines and displays their "
            "output; the physics lives in the pipelines.</p>"
            "<p>Author: Youssef Antoury<br>"
            "Licence: GNU GPL v3 or later<br>"
            "Source: <a href='https://github.com/joeantouri98-sketch/LiNaK'>github.com/joeantouri98-sketch/LiNaK</a><br>"
            "Cite: <a href='https://doi.org/10.5281/zenodo.21821149'>doi:10.5281/zenodo.21821149</a></p>"
            "<p>Plotly.js (MIT) is bundled for offline plots. See Help &gt; Help contents for the flags "
            "reference and short physics notes per stage.</p>",
        )

    def _new_dataset_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("New dataset")
        dialog.resize(360, 180)

        layout = QVBoxLayout(dialog)

        label = QLabel(
            "Choose the type of custom dataset to create:"
        )
        label.setWordWrap(True)
        layout.addWidget(label)

        button_row = QHBoxLayout()

        json_button = QPushButton("New JSON")
        csv_button = QPushButton("New CSV")

        button_row.addWidget(json_button)
        button_row.addWidget(csv_button)
        layout.addLayout(button_row)

        json_button.clicked.connect(
            lambda: self._open_new_dataset(dialog, "json")
        )
        csv_button.clicked.connect(
            lambda: self._open_new_dataset(dialog, "csv")
        )

        dialog.exec()

    def _open_new_dataset(self, chooser, dataset_type):
        chooser.accept()
        self._open_dataset_editor(dataset_type, "new")

    def _open_existing_dataset(self, path: str | Path | None = None):
        """Edit > Open existing...; with *path* the file dialog is skipped (right-click Edit on an output)."""
        if not path:
            path, _selected_filter = QFileDialog.getOpenFileName(
                self,
                "Open dataset file",
                str(self.root),
                "JSON files (*.json);;CSV files (*.csv);;All files (*)",
            )

        if not path:
            return

        opened_path = Path(path)
        suffix = opened_path.suffix.lower()

        try:
            if suffix == ".json":
                dataset_type = "json"
                initial_text = opened_path.read_text(encoding="utf-8")

            elif suffix == ".csv":
                dataset_type = "csv"

                with opened_path.open(
                    "r",
                    encoding="utf-8",
                    newline="",
                ) as handle:
                    initial_text = list(csv.reader(handle))

                if not initial_text:
                    initial_text = [["column_1"], [""]]

            else:
                QMessageBox.warning(
                    self,
                    "Unsupported file",
                    "Please choose a .json or .csv file.",
                )
                return
        except (OSError, UnicodeDecodeError) as exc:
            QMessageBox.warning(
                self,
                "Open failed",
                f"Could not read {opened_path.name}:\n\n{exc}",
            )
            return

        dialog = DatasetEditorDialog(
            parent=self,
            dataset_name=opened_path.stem,
            dataset_type=dataset_type,
            initial_text=initial_text,
            default_path=opened_path.parent,
            file_path=opened_path,
            root=self.root,
        )
        dialog.exec()

    def _open_dataset_editor(self, dataset_type, dataset_name):
        if dataset_type == "json":
            initial_text = JSON_DATASET_TEMPLATES.get(
                dataset_name,
                JSON_DATASET_TEMPLATES["new"],
            )
            default_path = Workspace(root=self.root).data_json_dir

        else:
            initial_text = CSV_DATASET_TEMPLATES.get(
                dataset_name,
                CSV_DATASET_TEMPLATES["new"],
            )
            default_path = self.root

        try:
            default_path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QMessageBox.warning(self, "Folder not writable", f"{default_path}:\n\n{exc}")
            return

        dialog = DatasetEditorDialog(
            parent=self,
            dataset_name=dataset_name,
            dataset_type=dataset_type,
            initial_text=initial_text,
            default_path=default_path,
            root=self.root,
        )
        dialog.exec()

    def _choose_root(self):
        """Pick the output folder the stage runs write into (any folder).

        The choice is remembered for later sessions.
        """
        d = QFileDialog.getExistingDirectory(self, "Output folder", str(self.root))
        if not d:
            return
        self.settings.setValue(ROOT_SETTING, d)
        self._switch_root(Path(d))

    def _use_default_root(self):
        """Go back to the per-user data folder and forget the chosen one."""
        self.settings.remove(ROOT_SETTING)
        self._switch_root(default_workspace().root)

    def _switch_root(self, root: Path) -> None:
        """Make *root* the current folder: forget the old folder's species."""
        self.root = root
        self.current_species = ""
        self.species_list.clear()
        self.species_input.clear()
        self._show_species_title("")
        # The --data-dir/--output-dir defaults depend on the folder.
        self._build_flags_form(self._selected_stage())
        self._update_stage_desc()
        self._seed_root()
        self._refresh_all()

    # UI
    def _build_ui(self):
        """The window: pipeline on the left, viewer in the middle, inspector right."""
        center = self._build_center()
        self.inspector = self._build_inspector()

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.sidebar = self._build_sidebar()
        self.main_splitter.addWidget(self.sidebar)
        self.main_splitter.addWidget(center)
        self.main_splitter.addWidget(self.inspector)
        self.main_splitter.setHandleWidth(SPLITTER_HANDLE_WIDTH)
        self.main_splitter.setCollapsible(CENTER_PANEL, False)
        self.main_splitter.setSizes([270, 850, 320])
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)
        self.main_splitter.splitterMoved.connect(self._sync_panel_actions)
        self.setCentralWidget(self.main_splitter)

        self.addToolBar(self._build_toolbar())
        self.setStatusBar(QStatusBar())
        self._update_status()

    def _build_toolbar(self):
        """Always-visible actions: what is selected, and how to run it."""
        toolbar = QToolBar("Main")
        toolbar.setObjectName("mainToolbar")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setContextMenuPolicy(Qt.PreventContextMenu)

        self.brand_label = QLabel()               # the wordmark; _set_theme paints it in the theme's colours
        self.brand_label.setObjectName("brand")
        self.brand_label.setAccessibleName("Qali")
        toolbar.addWidget(self.brand_label)

        divider = QFrame()
        divider.setObjectName("toolbarDivider")
        divider.setFixedSize(1, 22)
        toolbar.addWidget(divider)
        self.species_label = QLabel()
        self.species_label.setObjectName("speciesTitle")
        self._show_species_title("")
        toolbar.addWidget(self.species_label)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)

        open_btn = QPushButton("Open file...")
        open_btn.clicked.connect(self._open_file_dialog)
        toolbar.addWidget(open_btn)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setToolTip("Reload the species and output lists from the output folder")
        refresh_btn.clicked.connect(self._refresh_all)
        toolbar.addWidget(refresh_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.setToolTip("Stop the running stage (Shift+F5)")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_stage)
        toolbar.addWidget(self.stop_btn)
        self.run_btn = QPushButton("Run stage")
        self.run_btn.setObjectName("primary")
        self.run_btn.setToolTip("Run the selected stage (F5)")
        self.run_btn.clicked.connect(self._run_stage)
        toolbar.addWidget(self.run_btn)
        return toolbar

    def _section_title(self, text):
        label = QLabel(text.upper())
        label.setObjectName("sectionTitle")
        return label

    def _build_sidebar(self):
        """Species chips above the pipeline, grouped in running order."""
        sidebar = QWidget()
        sidebar.setMinimumWidth(210)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(10, 12, 10, 8)
        layout.setSpacing(6)

        layout.addWidget(self._section_title("Species"))
        self.species_list = QListWidget()
        self.species_list.setObjectName("speciesChips")
        self.species_list.setFlow(QListView.LeftToRight)
        self.species_list.setWrapping(True)
        self.species_list.setResizeMode(QListView.Adjust)
        self.species_list.setSpacing(2)
        self.species_list.setIconSize(QSize(9, 13))
        self.species_list.setFixedHeight(72)
        self.species_list.itemClicked.connect(lambda it: self._set_species(it.text()))
        self.species_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.species_list.customContextMenuRequested.connect(self._species_context_menu)
        delete_key = QShortcut(QKeySequence(QKeySequence.StandardKey.Delete), self.species_list)
        delete_key.setContext(Qt.WidgetShortcut)
        delete_key.activated.connect(self._delete_highlighted_species)
        layout.addWidget(self.species_list)

        row = QHBoxLayout()
        row.setSpacing(6)
        self.species_input = QLineEdit()
        self.species_input.setPlaceholderText("Add a species: Fe, Mg_c1")
        self.species_input.returnPressed.connect(self._set_species_from_input)
        set_btn = QPushButton("Set")
        set_btn.clicked.connect(self._set_species_from_input)
        row.addWidget(self.species_input)
        row.addWidget(set_btn)
        layout.addLayout(row)

        layout.addSpacing(8)
        layout.addWidget(self._section_title("Pipeline"))
        self.stage_list = QListWidget()
        self.stage_list.setItemDelegate(StageDelegate(self.stage_list))
        self.stage_list.setMouseTracking(True)
        self.stage_list.setVerticalScrollMode(QListView.ScrollPerPixel)
        for heading, stage_ids in STAGE_GROUPS:
            header = QListWidgetItem(heading)
            header.setFlags(Qt.NoItemFlags)
            self.stage_list.addItem(header)
            for stage_id in stage_ids:
                stage = STAGE_BY_ID[stage_id]
                item = QListWidgetItem(stage["label"])
                item.setData(Qt.UserRole, stage["id"])
                item.setData(SYMBOL_ROLE, stage["symbol"])
                item.setToolTip(stage["desc"])
                self.stage_list.addItem(item)
        self.stage_list.itemDoubleClicked.connect(lambda _it: self._run_stage())
        self.stage_list.currentItemChanged.connect(self._on_stage_selected)
        layout.addWidget(self.stage_list, stretch=1)
        return sidebar

    def _build_center(self):
        """The viewer, as large as possible, with the console folding away below."""
        self.center_tabs = QTabWidget()
        self.center_tabs.setDocumentMode(True)

        self.viewer_stack = QStackedWidget()
        if HAS_WEBENGINE:
            self.web_view = QWebEngineView()
            attach_web_downloads(self.web_view, self)
            self.web_view.loadFinished.connect(lambda ok: ok and self._sync_page_theme())
            self.viewer_stack.addWidget(self.web_view)
        else:
            self.web_view = None
            self.viewer_stack.addWidget(QLabel("(unused)"))

        self.json_view = QPlainTextEdit()
        self.json_view.setObjectName("jsonView")
        self.json_view.setReadOnly(True)
        self.json_view.setFont(mono_font())
        self.json_view_box = with_find_bar(self.json_view)
        self.viewer_stack.addWidget(self.json_view_box)  # index 1

        fallback = QWidget()
        fb = QVBoxLayout(fallback)
        self.fallback_label = QLabel(
            "QtWebEngine isn't available in this Python environment, so "
            "HTML plots can't be rendered inline.\n\n"
            "Install it with:  pip install PySide6-Addons\n"
            "or open the file in your system browser instead."
        )
        self.fallback_label.setWordWrap(True)
        fb.addWidget(self.fallback_label)
        open_browser_btn = QPushButton("Open in system browser")
        open_browser_btn.clicked.connect(self._open_fallback_in_browser)
        fb.addWidget(open_browser_btn)
        fb.addStretch(1)
        self.viewer_stack.addWidget(fallback)  # index 2

        placeholder = QWidget()                   # the empty viewer: a faint sprig and one line
        placeholder.setObjectName("emptyViewer")
        empty = QVBoxLayout(placeholder)
        empty.setSpacing(14)
        empty.addStretch(1)
        self.empty_sprig = QLabel()
        self.empty_sprig.setAlignment(Qt.AlignCenter)
        empty.addWidget(self.empty_sprig)
        empty_text = QLabel("Run a stage to see its plot here.")
        empty_text.setObjectName("emptyState")
        empty_text.setAlignment(Qt.AlignCenter)
        empty.addWidget(empty_text)
        empty.addStretch(1)
        self.viewer_stack.addWidget(placeholder)  # index 3
        self.compare_view = CompareTableView()
        self.compare_view.add_requested.connect(self._add_compare_elements)
        self.compare_view.raw_requested.connect(self._show_compare_json)
        self.viewer_stack.addWidget(self.compare_view)  # index 4
        self.viewer_stack.setCurrentIndex(3)
        self.center_tabs.addTab(self.viewer_stack, "Viewer")

        self.overview_view = SpeciesOverviewWidget()
        self.center_tabs.addTab(self.overview_view, "Overview")

        self.console = QPlainTextEdit()
        self.console.setObjectName("console")
        self.console.setReadOnly(True)
        self.console.setFont(mono_font())
        self.bottom_overview = SpeciesOverviewWidget()
        self.bottom_tabs = QTabWidget()
        self.bottom_tabs.setDocumentMode(True)
        self.bottom_tabs.addTab(self.console, "Console")
        self.bottom_tabs.addTab(self.bottom_overview, "Overview")
        self.move_panel_btn = QToolButton()
        self.move_panel_btn.setText("\u21c5")
        self.move_panel_btn.clicked.connect(
            lambda: self._set_console_panel_on_top(not self._console_panel_on_top()))
        self.bottom_tabs.setCornerWidget(self.move_panel_btn)
        self.bottom_tabs.tabBar().setToolTip(
            "Drag up or down to move this panel above or below the viewer")
        self.bottom_tabs.tabBar().installEventFilter(self)

        self.bottom_panel = QWidget()
        self.bottom_panel.setObjectName("bottomPanel")
        self.bottom_panel.setAttribute(Qt.WA_StyledBackground)
        bottom_layout = QVBoxLayout(self.bottom_panel)
        bottom_layout.setContentsMargins(0, 1, 0, 0)
        bottom_layout.addWidget(self.bottom_tabs)

        # Overview can sit in either place; each view refreshes when it is showing.
        self._overview_places = (
            (self.center_tabs, self.overview_view),
            (self.bottom_tabs, self.bottom_overview),
        )
        self._stale_overviews = set()
        self.center_tabs.currentChanged.connect(self._refresh_shown_overviews)
        self.bottom_tabs.currentChanged.connect(self._refresh_shown_overviews)

        self.center_splitter = QSplitter(Qt.Vertical)
        self.center_splitter.setObjectName("centerSplitter")
        self.center_splitter.addWidget(self.center_tabs)
        self.center_splitter.addWidget(self.bottom_panel)
        self.center_splitter.setHandleWidth(SPLITTER_HANDLE_WIDTH)
        self.center_splitter.splitterMoved.connect(self._sync_panel_actions)
        self.center_splitter.setStretchFactor(0, 1)
        self.center_splitter.setStretchFactor(1, 0)
        self.center_splitter.setSizes([660, 170])
        self.center_splitter.setCollapsible(0, False)
        return self.center_splitter

    def _build_inspector(self):
        """Right panel: the selected stage's options, then what it produced."""
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.setMinimumWidth(280)

        options = QWidget()
        layout = QVBoxLayout(options)
        layout.setContentsMargins(12, 12, 8, 10)
        layout.setSpacing(8)
        self.stage_desc = QLabel("")
        self.stage_desc.setObjectName("muted")
        self.stage_desc.setWordWrap(True)
        layout.addWidget(self.stage_desc)

        self.flags_form = QFormLayout()
        self.flags_form.setRowWrapPolicy(QFormLayout.WrapAllRows)
        self.flags_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.flags_form.setVerticalSpacing(4)
        flags_holder = QWidget()
        flags_holder.setLayout(self.flags_form)
        flags_scroll = QScrollArea()
        flags_scroll.setWidget(flags_holder)
        flags_scroll.setWidgetResizable(True)
        flags_scroll.setFrameShape(QFrame.NoFrame)
        flags_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)   # fields shrink to the panel instead
        layout.addWidget(flags_scroll, stretch=1)

        layout.addWidget(self._section_title("Command"))
        self.cmd_preview = QLabel("")
        self.cmd_preview.setObjectName("commandPreview")
        self.cmd_preview.setWordWrap(True)
        self.cmd_preview.setFont(mono_font())
        self.cmd_preview.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.cmd_preview)
        tabs.addTab(options, "Options")

        self.species_html_list = QListWidget()
        self.species_html_list.itemClicked.connect(self._on_output_clicked)
        tabs.addTab(self.species_html_list, "Plots")

        self.species_json_list = QListWidget()
        self.species_json_list.itemClicked.connect(self._on_output_clicked)
        tabs.addTab(self.species_json_list, "Data")

        self.shared_outputs_list = QListWidget()
        self.shared_outputs_list.itemClicked.connect(self._on_output_clicked)
        tabs.addTab(self.shared_outputs_list, "Shared")

        for output_list in (self.species_html_list, self.species_json_list, self.shared_outputs_list):
            output_list.setContextMenuPolicy(Qt.CustomContextMenu)
            output_list.customContextMenuRequested.connect(
                lambda pos, lst=output_list: self._output_context_menu(lst, pos))
        return tabs

    def _set_theme(self, name: str, save: bool = True) -> None:
        """Apply theme *name* to the app and the web view's blank background."""
        self.theme_name = name
        tokens = apply_theme(QApplication.instance(), name)
        self.brand_label.setPixmap(brand.wordmark(40, tokens.text, tokens.accent))   # the ember takes the
        self.empty_sprig.setPixmap(brand.sprig(150, tokens.raised, tokens.accent))   # theme's accent colour
        if self.web_view is not None:
            self.web_view.page().setBackgroundColor(QColor(tokens.bg))
            self._sync_page_theme()
        if save:
            self.settings.setValue(THEME_SETTING, name)

    # The redesigned pages have a 'Theme: auto / light / dark' button whose 'auto' follows the
    # browser's colour scheme, which inside the app is not the app's theme. Step the page's own button
    # to the app's light or dark (so the page redraws its plots); other pages get the attribute only.
    _PAGE_THEME_JS = (
        "(function(t){var b=document.getElementById('theme');"
        "if(b&&/^Theme: /.test(b.textContent.trim())){for(var i=0;i<3&&b.textContent.trim()!=='Theme: '+t;i++)b.click();}"
        "else{document.documentElement.setAttribute('data-theme',t);}})(%s)")

    def _sync_page_theme(self) -> None:
        """Show the page in the viewer in the app's light or dark theme."""
        if self.web_view is None:
            return
        import json as _json

        page_theme = "dark" if is_dark(self.theme_name) else "light"     # the pages know these two only
        self.web_view.page().runJavaScript(self._PAGE_THEME_JS % _json.dumps(page_theme))

    def _open_preferences(self, start_tab: str | None = None, species: str | None = None) -> None:
        PreferencesDialog(self, self.theme_name, self.settings,
                          lambda: self._set_theme(self.theme_name, save=False),
                          root=self.root, species=species or self.current_species or None,
                          start_tab=start_tab, set_theme=self._set_theme_from_prefs).exec()

    def _set_theme_from_prefs(self, name: str) -> None:
        self._set_theme(name)
        self._theme_actions[name].setChecked(True)

    def _on_system_scheme_changed(self, _scheme) -> None:
        if self.theme_name == SYSTEM:
            self._set_theme(SYSTEM, save=False)

    def _show_console(self) -> None:
        self.bottom_action.setChecked(True)
        self.bottom_tabs.setCurrentWidget(self.console)

    def _console_panel_on_top(self) -> bool:
        return self.center_splitter.indexOf(self.bottom_panel) == 0

    def _set_console_panel_on_top(self, on_top: bool, save: bool = True) -> None:
        """Put the Console / Overview panel above or below the viewer."""
        splitter = self.center_splitter
        if on_top != self._console_panel_on_top():
            viewer_height, panel_height = (
                reversed(splitter.sizes()) if self._console_panel_on_top()
                else splitter.sizes())
            splitter.insertWidget(0, self.bottom_panel if on_top else self.center_tabs)
            splitter.setSizes(
                [panel_height, viewer_height] if on_top else [viewer_height, panel_height])
            for widget, collapsible, stretch in (
                    (self.center_tabs, False, 1), (self.bottom_panel, True, 0)):
                splitter.setCollapsible(splitter.indexOf(widget), collapsible)
                splitter.setStretchFactor(splitter.indexOf(widget), stretch)
        self.bottom_panel.setProperty("onTop", on_top)
        self.bottom_panel.style().unpolish(self.bottom_panel)
        self.bottom_panel.style().polish(self.bottom_panel)
        self.move_panel_btn.setToolTip(
            f"Move this panel {'below' if on_top else 'above'} the viewer")
        self.panel_on_top_action.blockSignals(True)
        self.panel_on_top_action.setChecked(on_top)
        self.panel_on_top_action.blockSignals(False)
        if save:
            self.settings.setValue(CONSOLE_ON_TOP_SETTING, on_top)

    def eventFilter(self, watched, event):
        """Dragging the panel's tab bar moves the panel past the viewer."""
        if (watched is self.bottom_tabs.tabBar()
                and event.type() == QEvent.MouseMove
                and event.buttons() & Qt.LeftButton):
            y = self.center_splitter.mapFromGlobal(event.globalPosition().toPoint()).y()
            fraction = y / max(self.center_splitter.height(), 1)
            if not self._console_panel_on_top() and fraction < MOVE_PANEL_UP_BELOW:
                self._set_console_panel_on_top(True)
            elif self._console_panel_on_top() and fraction > MOVE_PANEL_DOWN_ABOVE:
                self._set_console_panel_on_top(False)
        return super().eventFilter(watched, event)

    def _add_panel_action(self, menu, label, shortcut, splitter, panel):
        """A View menu tick box that opens and closes one splitter panel."""
        action = QAction(label, self, checkable=True, checked=True)
        action.setShortcut(shortcut)
        action.toggled.connect(
            lambda shown: self._set_panel_open(splitter, panel, shown))
        menu.addAction(action)
        self._panel_actions.append((action, splitter, panel))
        return action

    def _set_panel_open(self, splitter, panel, shown) -> None:
        """Open or collapse a panel; reopening restores the size it had."""
        index = splitter.indexOf(panel)
        sizes = splitter.sizes()
        if shown == (sizes[index] > 0):
            return
        if shown:
            sizes[index] = self._panel_sizes.get(panel, PANEL_REOPEN_SIZE)
        else:
            self._panel_sizes[panel] = sizes[index]
            sizes[index] = 0
        splitter.setSizes(sizes)

    def _sync_panel_actions(self, *_position) -> None:
        """Keep the View menu ticks true after a panel is dragged shut or open."""
        for action, splitter, panel in self._panel_actions:
            action.blockSignals(True)
            action.setChecked(splitter.sizes()[splitter.indexOf(panel)] > 0)
            action.blockSignals(False)

    #species
    def _set_species_from_input(self):
        species = self.species_input.text().strip()
        # Naming a removed species brings it back, with its bundled data.
        if species and restore_species(Workspace(root=self.root), species):
            self._seed_root()
        self._set_species(species)
        self._offer_setup(species)

    def _offer_setup(self, species: str):
        """A species with no bundled levels or lines: open the dialog that imports them and shows what they enable."""
        if species and species_data.needs_setup(species):
            NewSpeciesDialog(self.root, species, self).exec()
            self._refresh_stage_markers()
            self._refresh_default_hints()

    def _species_context_menu(self, position):
        item = self.species_list.itemAt(position)
        if item is None:
            return
        menu = QMenu(self.species_list)
        props_action = menu.addAction("Properties...")
        delete_action = menu.addAction(f"Delete {item.text()}...")
        chosen = menu.exec(self.species_list.mapToGlobal(position))
        if chosen is props_action:
            self._open_properties(item.text())
        elif chosen is delete_action:
            self._delete_species(item.text())

    def _open_properties(self, species: str = ""):
        """The built-in numbers and data files of one species, for viewing, editing and adding."""
        species = species or self.current_species
        if species:
            PropertiesDialog(self.root, species, self, settings=self.settings).exec()
            self._refresh_species_icons()

    def _delete_highlighted_species(self):
        item = self.species_list.currentItem()
        if item is not None:
            self._delete_species(item.text())

    def _delete_species(self, species: str = ""):
        """Take *species* off the list, optionally deleting its files."""
        species = species or self.current_species
        if not species:
            QMessageBox.information(self, "Pick a species", "Select a species to delete first.")
            return
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            QMessageBox.information(self, "Busy", "Wait for the running stage to finish first.")
            return
        if self._seed_workers:
            QMessageBox.information(
                self, "Busy", "Wait for the bundled data to finish copying first.")
            return

        workspace = Workspace(root=self.root)
        files = species_files(workspace, species)
        dialog = DeleteSpeciesDialog(species, files, self)
        if dialog.exec() != QDialog.Accepted:
            return
        trees = dialog.selected_trees()
        kept: list[Path] = []
        if trees:
            kept = self._delete_paths([path for tree in trees for path in files[tree]])
        # Only what is really gone stays out of seeding; a kept tree stays as it was.
        gone = tuple(tree for tree in trees if not set(files[tree]) & set(kept))
        try:
            if not trees:
                mark_removed(workspace, species, hide=True)
            elif gone:
                mark_removed(workspace, species, no_seed=gone)
        except OSError as exc:
            QMessageBox.warning(self, "Removal not remembered", f"{species}:\n\n{exc}")
        if kept:
            QMessageBox.warning(
                self, "Some files were not deleted",
                "\n".join(str(path) for path in kept))
        self._drop_species_from_list(species)

    def _delete_paths(self, paths: list[Path]) -> list[Path]:
        """Delete *paths*, asking first if the Recycle Bin can't take them.

        Returns the paths that still exist.
        """
        stuck = move_to_trash(paths)
        if not stuck:
            return []
        answer = QMessageBox.question(
            self, "Recycle Bin not available",
            f"{len(stuck)} of {len(paths)} items can't go to the Recycle Bin, which "
            "some drives don't have.\n\nDelete them permanently? This can't be undone.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return delete_permanently(stuck) if answer == QMessageBox.Yes else stuck

    def _drop_species_from_list(self, species: str) -> None:
        """Refresh after a removal; a species with files left stays listed."""
        for item in self.species_list.findItems(species, Qt.MatchExactly):
            self.species_list.takeItem(self.species_list.row(item))
        self._refresh_species_list()
        if self.current_species == species and not self.species_list.findItems(
                species, Qt.MatchExactly):
            self.current_species = ""
            self.species_input.clear()
            self._show_species_title("")
            self.viewer_stack.setCurrentIndex(3)
        self._refresh_all()

    def _set_species(self, species: str):
        species = species.strip()
        if not species:
            return
        self.current_species = species
        self.species_input.clear()
        self._show_species_title(species)
        if not self.species_list.findItems(species, Qt.MatchExactly):
            self._add_species_item(species)
        self.species_list.setCurrentItem(
            self.species_list.findItems(species, Qt.MatchExactly)[0])
        self._refresh_stage_markers()
        self._refresh_species_outputs()
        self._refresh_overview()
        self._update_stage_desc()
        self._refresh_default_hints()
        self._update_cmd_preview()

    # stage selection
    def _on_stage_selected(self, current, _previous):
        self._update_stage_desc()
        self._build_flags_form(self._selected_stage())

    def _update_stage_desc(self):
        stage = self._selected_stage()
        text = "Select a stage from the pipeline." if stage is None else stage["desc"].replace(
            "{el}", self.current_species or "<el>")
        self.stage_desc.setText(text)

    def _build_flags_form(self, stage):
        """Rebuild the options form for *stage*: one labelled, tooltipped
        widget per CLI flag (checkbox / dropdown / text field / file-browse
        field) from the stage table in :mod:`.stages`, which the tests check
        against the pipeline's own argparse parser."""
        while self.flags_form.rowCount():
            self.flags_form.removeRow(0)
        # removeRow() deleted every widget, the basis picker's included.
        self._flag_widgets = {}
        self._flag_specs = {}
        self._save_as_edit = None
        self._basis_picker_checkbox = None
        self._basis_picker_combo = None
        if stage is None:
            self.cmd_preview.setText("")
            return

        rows: dict[str, list] = {key: [] for key, _title in FLAG_SECTIONS}
        for spec in stage.get("flags", []):
            kind = spec["kind"]
            help_text = spec.get("help", "")
            default = spec.get("default", "")
            if callable(default):
                # e.g. workspace_data_dir: resolved against the current folder.
                default = default(self.root)

            if kind in ("flag", "invflag"):
                value_widget = QCheckBox()
                value_widget.setChecked(kind == "invflag")      # "invflag": ticked = on, the flag is sent when unticked
                value_widget.stateChanged.connect(self._update_cmd_preview)
                row_widget = value_widget
            elif kind == "choice":
                value_widget = QComboBox()
                # a long choice must not make the whole panel wider than its window
                value_widget.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
                value_widget.setMinimumContentsLength(8)
                for choice in spec["choices"]:
                    cval, shown = choice if isinstance(choice, tuple) else (choice, choice)
                    value_widget.addItem(shown, userData=cval)
                default = spec.get("default")
                if default is not None:
                    idx = value_widget.findData(default)
                    if idx >= 0:
                        value_widget.setCurrentIndex(idx)
                value_widget.currentIndexChanged.connect(self._update_cmd_preview)
                row_widget = value_widget
            elif kind == "path":
                value_widget = QLineEdit(default)
                value_widget.textChanged.connect(self._update_cmd_preview)
                browse_btn = QPushButton("Browse...")
                mode = spec.get("browse")
                browse_btn.clicked.connect(
                    lambda _checked=False, le=value_widget, m=mode: self._browse_for_path(le, mode=m))
                container = QWidget()
                hl = QHBoxLayout(container)
                hl.setContentsMargins(0, 0, 0, 0)
                hl.addWidget(value_widget, stretch=1)
                hl.addWidget(browse_btn)
                row_widget = container
            elif kind == "pathlist":
                value_widget = QLineEdit(default)
                value_widget.setPlaceholderText("files separated by ;")
                value_widget.textChanged.connect(self._update_cmd_preview)
                browse_btn = QPushButton("Add files...")
                browse_btn.clicked.connect(
                    lambda _checked=False, le=value_widget: self._add_files_to(le))
                container = QWidget()
                hl = QHBoxLayout(container)
                hl.setContentsMargins(0, 0, 0, 0)
                hl.addWidget(value_widget, stretch=1)
                hl.addWidget(browse_btn)
                row_widget = container
            elif kind == "extrapolation":
                value_widget = QLineEdit(default)
                value_widget.setReadOnly(True)
                value_widget.setPlaceholderText("not set")
                value_widget.textChanged.connect(self._update_cmd_preview)
                open_btn = QPushButton("Extrapolation...")
                open_btn.clicked.connect(self._open_extrapolation_dialog)
                container = QWidget()
                hl = QHBoxLayout(container)
                hl.setContentsMargins(0, 0, 0, 0)
                hl.addWidget(value_widget, stretch=1)
                hl.addWidget(open_btn)
                row_widget = container
            else:  # int, float, str, strlist
                value_widget = QLineEdit(default)
                value_widget.textChanged.connect(self._update_cmd_preview)
                row_widget = value_widget

            row_widget.setToolTip(help_text or spec["label"])
            value_widget.setToolTip(help_text or spec["label"])
            self._flag_widgets[spec["name"]] = (kind, value_widget)
            self._flag_specs[spec["name"]] = spec
            label = _OptionLabel(spec["label"])
            label.setToolTip(help_text or spec["label"])
            rows[flag_section(spec)].append((label, row_widget))

        if stage["id"] == "runorca":
            self._basis_picker_checkbox = QCheckBox("Basis picker")
            self._basis_picker_checkbox.setToolTip(
                "GUI-only basis picker. It fills the existing --basis-file field "
                "without using --pick-basis or terminal input."
            )
            self._basis_picker_checkbox.stateChanged.connect(self._toggle_basis_picker)

            self._basis_picker_combo = QComboBox()
            self._basis_picker_combo.setEnabled(False)
            self._basis_picker_combo.setPlaceholderText("Select basis set")
            self._basis_picker_combo.currentIndexChanged.connect(self._apply_selected_basis)

            rows["input"].append((QLabel("Basis picker"), self._basis_picker_checkbox))
            rows["input"].append((QLabel("Basis set"), self._basis_picker_combo))

        if stage["id"] not in NO_SAVE_AS:          # the ORCA stages have their own output name suffix
            self._save_as_edit = QLineEdit()
            self._save_as_edit.setPlaceholderText("optional")
            rows["output"].append((QLabel("Save as:"), self._save_as_edit))

        # Model / Fit to NIST (stages that ask for them), Input, Output, then the rest; a heading only
        # where there is more than one kind of row.
        titles = FLAG_SECTIONS
        shown = [key for key, _title in titles if rows[key]]
        for key, title in titles:
            if not rows[key]:
                continue
            if len(shown) > 1:
                self.flags_form.addRow(self._form_heading(title))
            for label, widget in rows[key]:
                self.flags_form.addRow(label, widget)

        self._refresh_default_hints()
        self._update_cmd_preview()

    def _refresh_default_hints(self):
        """Grey hints in blank fields: the files the stage reads and writes by default for the current species."""
        stage = self._selected_stage()
        species = self.current_species
        for name, (kind, widget) in getattr(self, "_flag_widgets", {}).items():
            hint = input_hint(self._flag_specs[name], species) if kind == "path" else ""
            if hint:
                widget.setPlaceholderText(hint)
        if self._save_as_edit is not None and stage is not None:
            hint = output_hint(stage, species)
            self._save_as_edit.setPlaceholderText(
                hint or "optional")

    @staticmethod
    def _form_heading(text: str) -> QWidget:
        """A section title with a line beneath it, like the Theme colors title in Preferences."""
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 8, 0, 2)
        lay.setSpacing(2)
        title = QLabel(text)
        font = title.font()
        font.setPointSizeF(font.pointSizeF() + 1)
        font.setBold(True)
        title.setFont(font)
        lay.addWidget(title)
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Plain)
        lay.addWidget(line)
        return box

    def _open_extrapolation_dialog(self):
        element = self.current_species
        if not element:
            QMessageBox.information(self, "Extrapolation", "Select a species first.")
            return
        data_dir = Workspace(root=self.root).data_json_dir
        ryd_file = data_dir / f"{element}_rydberg.json"
        if not ryd_file.is_file():
            QMessageBox.information(self, "Extrapolation",
                                    f"Run the Rydberg series stage for {element} first; the dialog needs its levels.")
            return
        import json as _json
        from qali.physics.polarizability_core import load_nist_fvalues
        rydberg = _json.loads(ryd_file.read_text(encoding="utf-8"))
        nist = load_nist_fvalues(element, verbose=False)
        if open_extrapolation_dialog(self, self.root, element, nist, rydberg):
            entry = self._flag_widgets.get("--extrapolation")
            if entry is not None:
                entry[1].setText(active_config_path(self.root, element))
            self._update_cmd_preview()

    def _toggle_basis_picker(self, checked):
        if self._basis_picker_combo is None:
            return

        if not checked:
            self._basis_picker_combo.clear()
            self._basis_picker_combo.setEnabled(False)
            return

        if not self.current_species:
            QMessageBox.warning(
                self,
                "No species selected",
                "Select a species before using the ORCA basis picker.",
            )
            self._basis_picker_checkbox.setChecked(False)
            return

        try:
            import basis_set_exchange as bse
        except ImportError:
            self._basis_picker_combo.clear()
            self._basis_picker_combo.addItem("Install basis-set-exchange")
            self._basis_picker_combo.setEnabled(False)
            QMessageBox.warning(
                self,
                "Missing dependency",
                "basis_set_exchange is not installed.\n\nInstall it with:\n"
                "pip install basis-set-exchange",
            )
            self._basis_picker_checkbox.setChecked(False)
            return

        try:
            z = resolve_species(self.current_species)["Z"]

            names = []
            for basis_name in bse.get_all_basis_names():
                try:
                    bse.get_basis(basis_name, elements=[z])
                    names.append(basis_name)
                except Exception:
                    continue

            self._basis_picker_combo.clear()
            for name in sorted(set(names)):
                self._basis_picker_combo.addItem(name)
            self._basis_picker_combo.setEnabled(True)

        except Exception as exc:
            self._basis_picker_combo.clear()
            self._basis_picker_combo.addItem("Lookup failed")
            self._basis_picker_combo.setEnabled(False)
            QMessageBox.warning(self, "Basis lookup failed", str(exc))

    def _apply_selected_basis(self, index):
        if self._basis_picker_combo is None:
            return
        if index < 0:
            return

        selected = self._basis_picker_combo.currentText()
        if not selected or selected in ("Lookup failed", "Install basis-set-exchange"):
            return

        try:
            import basis_set_exchange as bse

            species = self.current_species.strip()
            z = resolve_species(species)["Z"]

            basis_text = None
            for fmt in ("orca", "gamess_us", "gaussian94"):
                try:
                    value = bse.get_basis(selected, elements=[z], fmt=fmt)
                    if value:
                        basis_text = str(value)
                        break
                except Exception:
                    continue

            if basis_text is None:
                raise RuntimeError(f"Could not export {selected} from Basis Set Exchange.")

            safe_name = "".join(ch for ch in selected if ch.isalnum() or ch in "._-")
            basis_path = self._basis_cache_dir / f"{species}_{safe_name}.basis"
            basis_path.write_text(basis_text, encoding="utf-8")

            basis_entry = self._flag_widgets.get("--basis-file")
            if basis_entry is not None:
                _kind, widget = basis_entry
                widget.setText(str(basis_path))

            self._update_cmd_preview()
            self.statusBar().showMessage(f"Selected basis: {selected}")

        except Exception as exc:
            QMessageBox.warning(self, "Basis export failed", str(exc))


    def _add_files_to(self, line_edit: QLineEdit):
        """Append the files picked in a dialog (several at once) to a ';'-separated field."""
        start = Workspace(root=self.root).data_json_dir
        paths, _filter = QFileDialog.getOpenFileNames(self, "Select files", str(start), "JSON (*.json);;All files (*)")
        if paths:
            current = [p.strip() for p in line_edit.text().split(";") if p.strip()]
            line_edit.setText("; ".join(current + [str(p) for p in paths if str(p) not in current]))

    def _browse_for_path(self, line_edit: QLineEdit, mode: str | None = None):
        """Fill *line_edit* from a dialog. *mode*: None = any file, "data" = a file from the data folder,
        "dir" = a folder, "orca_run" = a run folder of this species (the field takes its name)."""
        ws = Workspace(root=self.root)
        if mode in ("dir", "orca_run"):
            current = line_edit.text().strip()
            start = ws.orca_outputs_dir / (self.current_species or "") if mode == "orca_run" else self.root
            if mode == "dir" and current and Path(current).is_dir():
                start = Path(current)
            if not Path(start).is_dir():
                start = self.root
            path = QFileDialog.getExistingDirectory(self, "Select folder", str(start))
            if path and mode == "orca_run":
                path = Path(path).name
        else:
            start = ws.data_json_dir if mode == "data" else self.root
            path, _filter = QFileDialog.getOpenFileName(self, "Select file", str(start))
        if path:
            line_edit.setText(str(path))

    def _compare_elements_from_form(self):
        entry = self._flag_widgets.get("__elements")
        if entry is None:
            return []

        _kind, widget = entry
        return expand_element_families(widget.text())

    def _collect_extra_args(self):
        """Turn the current options-form values into an argv list."""
        args = []
        for name, (kind, w) in self._flag_widgets.items():
            # GUI-only field (not a CLI flag): the compare stage's element
            # expander; its values are appended separately in _run_stage.
            if name.startswith("__"):
                continue
            if kind == "extrapolation":
                path = active_config_path(self.root, self.current_species or "")
                if path:
                    args += [name, path]
            elif kind == "flag":
                if w.isChecked():
                    args.append(name)
            elif kind == "invflag":
                if not w.isChecked():
                    args.append(name)
            elif kind == "choice":
                value = w.currentData()
                if value not in (None, ""):
                    args += [name, value]
            elif kind == "strlist":
                text = w.text().strip()
                if text:
                    args.append(name)
                    args += text.split()
            elif kind == "pathlist":            # a repeatable option: --extra a --extra b
                for item in w.text().split(";"):
                    if item.strip():
                        args += [name, item.strip()]
            else:  # int, float, str, path
                text = w.text().strip()
                if text:
                    args += [name, text]
        return args

    def _stage_pipeline_args(self, stage, species):
        """``qali --pipeline`` arguments for *stage*, or None without elements."""
        extra = self._collect_extra_args()
        if stage["id"] == "compare":
            elements = self._compare_elements_from_form()
            if not elements:
                return None
            extra = ["--elements", *elements, *extra]
        return stage_argv(stage, species, extra)

    def _update_cmd_preview(self, *_unused):
        stage = self._selected_stage()
        if stage is None:
            self.cmd_preview.setText("")
            return
        args = self._stage_pipeline_args(stage, self.current_species or "<species>")
        if args is None:
            self.cmd_preview.setText(
                "$ qali --pipeline compare_elements --elements <enter elements>")
            return
        self.cmd_preview.setText("$ qali --pipeline " + " ".join(args))

    def _confirm_isotope_data(self, stage, pipeline_args) -> bool:
        """False if an explicit --isotope lacks data and the user cancels the run."""
        if stage["id"] not in ("hyperfine", "feshbach", "polarizability") or "--isotope" not in pipeline_args:
            return True
        try:
            iso_A = int(pipeline_args[pipeline_args.index("--isotope") + 1])
        except (ValueError, IndexError):
            return True
        if not self.current_species:
            return True
        issues = isotope_prerequisite_issues(self.root, stage["id"], self.current_species, iso_A)
        if not issues:
            return True
        body = "\n\n".join(("ERROR: " if sev == "error" else "Warning: ") + txt for sev, txt in issues)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle(f"Isotope {self.current_species}-{iso_A}: missing data")
        box.setText(body)
        run_btn = box.addButton("Run anyway", QMessageBox.AcceptRole)
        box.addButton("Cancel", QMessageBox.RejectRole)
        box.setDefaultButton(box.buttons()[-1])
        box.exec()
        return box.clickedButton() is run_btn

    def _selected_stage(self):
        item = self.stage_list.currentItem()
        if item is None:
            return None
        return STAGE_BY_ID[item.data(Qt.UserRole)]

    def _run_stage(self):
        stage = self._selected_stage()
        if stage is None:
            QMessageBox.information(self, "Pick a stage", "Select a pipeline stage first.")
            return
        if stage["id"] != "compare" and not self.current_species:
            QMessageBox.information(self, "Pick a species", "Set a species first.")
            return
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            QMessageBox.information(self, "Busy", "A stage is already running.")
            return

        pipeline_args = self._stage_pipeline_args(stage, self.current_species)
        if pipeline_args is None:
            QMessageBox.warning(
                self,
                "No elements",
                "Enter elements or a family, for example: alkali or Li,Na,K.",
            )
            return
        if not self._confirm_isotope_data(stage, pipeline_args):
            return
        save_name = sanitize(self._save_as_edit.text()) if self._save_as_edit is not None else ""
        if save_name and "--all" in pipeline_args:
            QMessageBox.information(self, "Save as",
                                    "Save as works for one species at a time; untick \"Run all available elements\" "
                                    "or clear the name.")
            return
        program, args = stage_program(pipeline_args)
        self._snapshot, self._snapshot_name = None, save_name
        if save_name:
            ws = Workspace(root=self.root)
            self._snapshot = OutputSnapshot(
                [ws.output_path(t.format(el=self.current_species or "")) for t in stage["outputs"]])

        self.console.clear()
        self._show_console()
        self._log(f"$ {program} {' '.join(args)}")
        self.statusBar().showMessage(f"Running {stage['label']} for {self.current_species}...")
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        proc = QProcess(self)
        proc.setWorkingDirectory(str(self.root))

        env = QProcessEnvironment.systemEnvironment()
        for name, value in stage_env_values(self.root).items():
            env.insert(name, value)
        proc.setProcessEnvironment(env)
        # A pipeline never reads stdin; a stray input() must fail, not hang.
        proc.setStandardInputFile(QProcess.nullDevice())
        start_in_own_session(proc)

        proc.setProgram(program)
        proc.setArguments(args)
        decoders = {
            False: codecs.getincrementaldecoder("utf-8")(errors="replace"),
            True: codecs.getincrementaldecoder("utf-8")(errors="replace"),
        }
        proc.readyReadStandardOutput.connect(
            lambda: self._log(decoders[False].decode(bytes(proc.readAllStandardOutput())),
                              newline=False))
        proc.readyReadStandardError.connect(
            lambda: self._log(decoders[True].decode(bytes(proc.readAllStandardError())),
                              newline=False))
        proc.started.connect(lambda: self._start_stage_log_tail(proc))
        proc.finished.connect(lambda code, status: self._stop_stage_log_tail())
        proc.finished.connect(lambda code, status: self._on_finished(stage, code, status))
        proc.finished.connect(proc.deleteLater)
        proc.errorOccurred.connect(self._on_process_error)

        self.process = proc
        self.current_stage = stage
        proc.start()

    def _stop_stage(self):
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            kill_process_tree(self.process)

    # ------------------------------------------------------------------
    # Frozen-mode CLI logs
    #
    # The bundle is a GUI-subsystem executable built with Nuitka's
    # --disable-console: stdout/stderr are not pipes but forced to files in
    # the OS temp dir ({TEMP}/Qali_stdout_{PID}.log etc., see
    # pysidedeploy.spec). QProcess therefore sees no output for frozen stage
    # runs; the Console tab is fed by tailing those files instead.
    # ------------------------------------------------------------------

    def _start_stage_log_tail(self, proc: QProcess) -> None:
        """Begin tailing the frozen child's CLI log files (no-op in dev).

        Also prunes log files from stage runs older than a day, so the temp
        dir does not grow unboundedly.
        """
        if not is_frozen_app():
            return
        stdout_log, stderr_log = cli_log_paths(int(proc.processId()))
        cutoff = time.time() - 24 * 3600
        for pattern in ("Qali_stdout_*.log", "Qali_stderr_*.log"):
            for stale in stdout_log.parent.glob(pattern):
                try:
                    if stale.is_file() and stale.stat().st_mtime < cutoff:
                        stale.unlink()
                except OSError:
                    pass
        # path -> [utf-8 decoder, byte offset]
        self._stage_log_tail = {
            path: [codecs.getincrementaldecoder("utf-8")(errors="replace"), 0]
            for path in (stdout_log, stderr_log)
        }
        timer = QTimer(self)
        timer.timeout.connect(self._poll_stage_logs)
        timer.start(200)
        self._stage_log_timer = timer
        self._poll_stage_logs()

    def _poll_stage_logs(self) -> None:
        """Append new bytes of the child's log files to the Console tab."""
        for path, entry in self._stage_log_tail.items():
            decoder, offset = entry
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size <= offset:
                continue
            try:
                with path.open("rb") as fh:
                    fh.seek(offset)
                    data = fh.read()
            except OSError:
                continue
            entry[1] = size
            self._log(decoder.decode(data), newline=False)

    def _stop_stage_log_tail(self) -> None:
        """Drain the tail once more and stop it."""
        timer = getattr(self, "_stage_log_timer", None)
        if timer is None:
            return
        self._poll_stage_logs()
        timer.stop()
        timer.deleteLater()
        self._stage_log_timer = None
        self._stage_log_tail = {}

    def _reset_process_state(self):
        """Return the run controls to idle once no child is running."""
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.process = None

    def _on_process_error(self, error):
        proc = self.process
        if proc is None or error == QProcess.ProcessError.Crashed:
            return  # a crash (or Stop) is reported by finished()
        message = proc.errorString()
        self._log(f"\n[process error: {message}]")
        self._stop_stage_log_tail()
        if proc.state() == QProcess.NotRunning:
            # FailedToStart emits no finished() signal, so the run controls
            # must be reset here or the UI stays stuck on "Busy".
            self.statusBar().showMessage(f"{self.current_stage['label']}: {message}")
            self._file_named_run(False)
            self._reset_process_state()

    def _file_named_run(self, ok: bool) -> None:
        """After a run with a Save as name: keep its outputs under the name, put the usual files back."""
        snapshot, self._snapshot = self._snapshot, None
        if snapshot is None:
            return
        kept = snapshot.finish(self._snapshot_name, ok=ok)
        if kept:
            self._log("Saved as: " + ", ".join(p.name for p in kept))
        elif ok:
            self._log("Save as: the stage wrote no output files.")

    def _on_finished(self, stage, code, _status):
        ok = code == 0
        self._file_named_run(ok)
        self._log(f"\n{'done.' if ok else f'exited with code {code}.'}")
        self.statusBar().showMessage(
            f"{stage['label']}: {'finished' if ok else f'failed (exit {code})'}"
        )
        self._reset_process_state()
        self._refresh_all()
        if ok and stage["id"] == "compare":
            self._show_compare_table()

    def _compare_json_path(self) -> Path:
        return Workspace(root=self.root).output_path("data/comparison_table.json")

    def _show_compare_table(self):
        path = self._compare_json_path()
        if path.is_file():
            self._load_output(path, "JSON")

    def _show_compare_json(self):
        path = self._compare_json_path()
        self._load_output(path, "JSON", raw=True)

    def _add_compare_elements(self, text: str):
        """Run the Compare stage for *text* (typed in the table's Add box)."""
        for i in range(self.stage_list.count()):
            if self.stage_list.item(i).data(Qt.UserRole) == "compare":
                self.stage_list.setCurrentRow(i)
                break
        entry = self._flag_widgets.get("__elements")
        if entry is None:
            return
        entry[1].setText(text)
        self._run_stage()

    def _log(self, text: str, newline=True):
        cursor = self.console.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.console.setTextCursor(cursor)
        self.console.insertPlainText(text + ("\n" if newline else ""))
        self.console.ensureCursorVisible()

    # output lists / viewer
    def _refresh_stage_markers(self):
        ws = Workspace(root=self.root)
        for i in range(self.stage_list.count()):
            item = self.stage_list.item(i)
            stage_id = item.data(Qt.UserRole)
            if stage_id is None:  # a group heading
                continue
            item.setData(DONE_ROLE, bool(self.current_species) and stage_has_output(
                STAGE_BY_ID[stage_id], ws, self.current_species))

    def _refresh_species_outputs(self):
        self.species_html_list.clear()
        self.species_json_list.clear()

        if not self.current_species:
            self.species_html_list.addItem("(no species selected)")
            self.species_json_list.addItem("(no species selected)")
            return

        html_count = 0
        json_count = 0

        for path, kind in species_outputs(self.root, self.current_species):
            item = QListWidgetItem(path.name)
            item.setData(Qt.UserRole, (str(path), kind))

            if kind == "PLOT":
                self.species_html_list.addItem(item)
                html_count += 1
            elif kind == "JSON":
                self.species_json_list.addItem(item)
                json_count += 1

        if html_count == 0:
            item = QListWidgetItem(
                "(no HTML plots generated for this species yet)"
            )
            item.setFlags(Qt.NoItemFlags)
            self.species_html_list.addItem(item)

        if json_count == 0:
            item = QListWidgetItem(
                "(no JSON data generated for this species yet)"
            )
            item.setFlags(Qt.NoItemFlags)
            self.species_json_list.addItem(item)

    def _refresh_shared_outputs(self):
        self.shared_outputs_list.clear()
        for path, kind in shared_outputs(self.root):
            item = QListWidgetItem(f"[{kind}]  {path.name}")
            item.setData(Qt.UserRole, (str(path), kind))
            self.shared_outputs_list.addItem(item)

    def _refresh_species_list(self):
        found = workspace_species(self.root)
        current_texts = {self.species_list.item(i).text() for i in range(self.species_list.count())}
        for sp in found:
            if sp not in current_texts:
                self._add_species_item(sp)

    def _show_species_title(self, species: str) -> None:
        """The current species next to the wordmark, or a muted hint when there is none."""
        self.species_label.setText(species or "No species selected")
        self.species_label.setProperty("empty", not species)
        self.species_label.style().unpolish(self.species_label)
        self.species_label.style().polish(self.species_label)

    def _add_species_item(self, species: str) -> None:
        """Put *species* on the list, with an ember in its colour."""
        self.species_list.addItem(QListWidgetItem(brand.ember_icon(brand.species_colour(species), 13), species))

    def _refresh_species_icons(self) -> None:
        """Repaint the species list's embers after a species' colour changed."""
        for i in range(self.species_list.count()):
            item = self.species_list.item(i)
            item.setIcon(brand.ember_icon(brand.species_colour(item.text()), 13))

    def _refresh_overview(self):
        self._stale_overviews.update(view for _tabs, view in self._overview_places)
        self._refresh_shown_overviews()

    def _refresh_shown_overviews(self, *_index) -> None:
        """Refresh the Overviews on show; a hidden one waits, as reading is slow."""
        for tabs, view in self._overview_places:
            if tabs.currentWidget() is view and view in self._stale_overviews:
                self._stale_overviews.discard(view)
                view.refresh(self.root, self.current_species)

    def _refresh_all(self):
        self._update_status()
        self._refresh_species_list()
        self._refresh_stage_markers()
        self._refresh_species_outputs()
        self._refresh_shared_outputs()
        self._refresh_overview()

    def _update_status(self):
        self.statusBar().showMessage(f"Output folder: {self.root}")

    def _open_tests(self) -> None:
        """Run > Tests...: one window, kept so a running suite survives closing and reopening it."""
        from .tests_dialog import TestsDialog

        dlg = getattr(self, "_tests_dialog", None)
        if dlg is None:
            dlg = self._tests_dialog = TestsDialog(self)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _output_context_menu(self, output_list: QListWidget, position) -> None:
        """Right-click on a Plots / Data / Shared entry: Edit (json, csv) and Delete."""
        item = output_list.itemAt(position)
        data = item.data(Qt.UserRole) if item is not None else None
        if not data:
            return
        path, kind = Path(data[0]), data[1]
        actions = output_actions(path, kind)
        menu = QMenu(output_list)
        edit_action = menu.addAction("Edit...")
        edit_action.setEnabled(actions["edit"])
        if not actions["edit"]:
            edit_action.setToolTip("Only JSON and CSV files can be edited.")
        delete_action = menu.addAction(f"Delete {path.name}...")
        delete_action.setEnabled(actions["delete"])
        if not actions["delete"]:
            delete_action.setText("Delete (protected)")
            delete_action.setToolTip(f"{path.name} is {actions['protected_reason']}.")
        chosen = menu.exec(output_list.viewport().mapToGlobal(position))
        if chosen is edit_action:
            self._open_existing_dataset(path)
            self._refresh_all()
        elif chosen is delete_action:
            self._delete_output(path)

    def _delete_output(self, path: Path) -> None:
        """Delete one output file after asking; the Recycle Bin is used when there is one."""
        if protected_reason_text := output_actions(path, "")["protected_reason"]:
            QMessageBox.information(self, "Protected file", f"{path.name} is {protected_reason_text}.")
            return
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            QMessageBox.information(self, "Busy", "Wait for the running stage to finish first.")
            return
        extra = ""
        if path.suffix.lower() == ".json" and "_" not in path.stem.replace("_c1", ""):
            extra = "\n\nThis is the ORCA result for the species; getting it back needs a new ORCA run."
        answer = QMessageBox.question(
            self, "Delete file", f"Delete {path.name}?\n\n{path}{extra}",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        kept = self._delete_paths([path])
        if kept:
            QMessageBox.warning(self, "Not deleted", "\n".join(str(p) for p in kept))
        if getattr(self, "_shown_path", None) == path and not kept:
            self.viewer_stack.setCurrentIndex(3)
            self._shown_path = None
        self._refresh_all()

    def _on_output_clicked(self, item: QListWidgetItem):
        data = item.data(Qt.UserRole)
        if not data:
            return
        path_str, kind = data
        path = Path(path_str)
        self._load_output(path, kind)

    def _load_output(self, path: Path, kind: str, raw: bool = False):
        self.center_tabs.setCurrentWidget(self.viewer_stack)
        self._shown_path = path
        if kind == "JSON" and path.name == "comparison_table.json" and not raw:
            self.viewer_stack.setCurrentIndex(4)
            self.compare_view.load(path)
        elif kind == "PLOT":
            if HAS_WEBENGINE:
                self.viewer_stack.setCurrentIndex(0)
                self.web_view.load(
                    QUrl.fromLocalFile(str(local_plot_copy(path, self._plot_cache_dir))))
            else:
                self.viewer_stack.setCurrentIndex(2)
                self.fallback_path = path
        else:  # JSON
            self.viewer_stack.setCurrentIndex(1)
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self.json_view.setPlainText(json.dumps(data, indent=2, ensure_ascii=False))
            except Exception as exc:  # noqa: BLE001 - show any parse error to the user
                self.json_view.setPlainText(f"Failed to parse {path.name}:\n{exc}")

    def _open_fallback_in_browser(self):
        if self.fallback_path is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.fallback_path)))

    def _open_file_dialog(self):
        start_dir = str(Workspace(root=self.root).plots_dir)
        path_str, _filter = QFileDialog.getOpenFileName(
            self, "Open plot or JSON file", start_dir,
            "Plots and data (*.html *.json);;All files (*)",
        )
        if not path_str:
            return
        path = Path(path_str)
        kind = "PLOT" if path.suffix.lower() == ".html" else "JSON"
        self._load_output(path, kind)


WINDOWS_APP_ID = "Qali.Qali"


def _set_windows_app_id() -> None:
    """Give the process its own taskbar identity on Windows.

    Without it Windows groups the window under the Python (or uv launcher) executable and shows that
    program's icon in the taskbar instead of the app's. Must run before the first window is created.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(WINDOWS_APP_ID)
    except (AttributeError, OSError):
        pass                                     # an older Windows: the taskbar keeps the Python icon


def main():
    from qali import __version__

    _set_windows_app_id()
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)
    app.setWindowIcon(brand.icon())             # every window and dialog without a species
    from .preferences_dialog import QALI_BASE_KEY

    base = open_settings().value(QALI_BASE_KEY, "")
    splash = QSplashScreen(brand.splash(__version__, SPLASH_TAGLINE,
                                        background=base if isinstance(base, str) and base else brand.SPLASH_BG))
    splash.show()
    app.processEvents()
    window = MainWindow()
    window.show()
    splash.finish(window)
    return app.exec()
