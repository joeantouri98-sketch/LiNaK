"""
LiNaK Desktop Control Panel
----------------------------
A plain, native desktop GUI (PySide6/Qt) for the LiNaK alkali-atom AMO
pipeline: run pipeline stages (including ORCA) for a species and browse
both freshly produced and already-on-disk plots/JSON, with the Plotly
HTML plots rendered directly inside the window.

This does not reimplement any physics: each stage runs the pipeline as a
``linak --pipeline <name> ...`` subprocess with the same flags you'd use
from the command line, then loads the JSON/HTML files it produces.

Run with ``linak-gui`` (or ``uv run linak-gui`` in a checkout).

Outputs go to LiNaK's per-user data folder unless another folder is picked
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
from pathlib import Path

from PySide6.QtCore import (
    QProcess,
    QProcessEnvironment,
    QSettings,
    Qt,
    QThread,
    QTimer,
    QUrl,
    Signal,
)
from PySide6.QtGui import QAction, QDesktopServices, QFont, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from linak.core.elements import expand_element_families
from linak.core.paths import PROJECT_ROOT_ENV, is_frozen_app
from linak.core.species import resolve_species
from linak.core.workspace import Workspace, default_workspace, seed_bundled_data

from .dataset_editor import DatasetEditorDialog
from .dataset_templates import CSV_DATASET_TEMPLATES, JSON_DATASET_TEMPLATES
from .help_dialog import HelpDialog
from .overview import SpeciesOverviewWidget
from .plot_cache import local_plot_copy
from .project_io import (
    open_pipeline_overview_html,
    rydberg_species,
    shared_outputs,
    species_outputs,
)
from .stage_process import (
    cli_log_paths,
    kill_process_tree,
    stage_env_values,
    stage_program,
    start_in_own_session,
)
from .stages import (
    PIPELINE,
    STAGE_BY_ID,
    STAGE_PHYSICS,
    isotope_prerequisite_issues,
    stage_argv,
    stage_has_output,
)

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    HAS_WEBENGINE = True
except ImportError:
    HAS_WEBENGINE = False


ORG_NAME = "LiNaK"
APP_NAME = "ControlPanel"
# QSettings key remembering the folder picked with File > Open output folder...
ROOT_SETTING = "root_path"


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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LiNaK Control Panel")
        self.resize(1320, 840)

        self.settings = QSettings(ORG_NAME, APP_NAME)
        self.root = self._initial_root()
        self.current_species = ""
        self.process: QProcess | None = None
        self.current_stage = None
        self.fallback_path: Path | None = None
        self._flag_widgets: dict = {}
        self._basis_picker_checkbox = None
        self._basis_picker_combo = None
        self._basis_cache_dir = Path(tempfile.mkdtemp(prefix="linak_gui_basis_"))
        self._plot_cache_dir = Path(tempfile.mkdtemp(prefix="linak_gui_"))
        self._seed_workers: set[_SeedWorker] = set()

        self._build_menu()
        self._build_ui()
        self._seed_root()
        self._refresh_all()

    def closeEvent(self, event):
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            kill_process_tree(self.process)
            self.process.waitForFinished(3000)
        self.process = None
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

        An explicit ``LINAK_PROJECT_ROOT`` wins; otherwise the folder picked
        with File > Open output folder... in an earlier session, if it still
        exists; otherwise LiNaK's per-user data folder.
        """
        saved = self.settings.value(ROOT_SETTING, "")
        if saved and not os.environ.get(PROJECT_ROOT_ENV) and Path(saved).is_dir():
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
        QMessageBox.about(
            self, "About LiNaK Control Panel",
            "LiNaK Desktop Control Panel\n\n"
            "A plain Qt front-end for the LiNaK alkali-atom AMO pipeline. "
            "Runs your existing scripts as subprocesses and displays their "
            "output; no physics is reimplemented here.\n\n"
            "See Help > Help contents for a flags reference and short "
            "physics notes per stage.",
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

    def _open_existing_dataset(self):
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
        self.species_label.setText("No species selected.")
        # The --data-dir/--output-dir defaults depend on the folder.
        self._build_flags_form(self._selected_stage())
        self._update_stage_desc()
        self._seed_root()
        self._refresh_all()

    # UI 
    def _build_ui(self):
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.Monospace)
        mono.setPointSize(9)

        #Left: species, stages, options form, console
        left = QWidget()
        left_layout = QVBoxLayout(left)

        species_box = QGroupBox("Species")
        sb = QVBoxLayout(species_box)
        self.species_list = QListWidget()
        self.species_list.setMaximumHeight(110)
        self.species_list.itemClicked.connect(lambda it: self._set_species(it.text()))
        sb.addWidget(self.species_list)
        row = QHBoxLayout()
        self.species_input = QLineEdit()
        self.species_input.setPlaceholderText("e.g. Na, Cs, Mg_c1")
        self.species_input.returnPressed.connect(self._set_species_from_input)
        set_btn = QPushButton("Set")
        set_btn.clicked.connect(self._set_species_from_input)
        row.addWidget(self.species_input)
        row.addWidget(set_btn)
        sb.addLayout(row)
        self.species_label = QLabel("No species selected.")
        self.species_label.setWordWrap(False)
        self.species_label.setMaximumHeight(24)
        self.species_label.setMinimumHeight(20)
        sb.addWidget(self.species_label)
        species_box.setMaximumHeight(205)
        left_layout.addWidget(species_box, stretch=0)

        stage_box = QGroupBox("Pipeline stages")
        st = QVBoxLayout(stage_box)
        self.stage_list = QListWidget()
        self.stage_list.setMinimumHeight(260)
        self.stage_list.setMaximumHeight(420)
        for stage in PIPELINE:
            item = QListWidgetItem(f"[{stage['symbol']}]  {stage['label']}")
            item.setData(Qt.UserRole, stage["id"])
            item.setToolTip(stage["desc"])
            self.stage_list.addItem(item)
        self.stage_list.itemDoubleClicked.connect(lambda _it: self._run_stage())
        st.addWidget(self.stage_list)
        self.stage_desc = QLabel("")
        self.stage_desc.setWordWrap(True)
        self.stage_list.currentItemChanged.connect(self._on_stage_selected)
        st.addWidget(self.stage_desc)

        st.addWidget(QLabel("Options for this stage:"))
        self.flags_form = QFormLayout()
        self.flags_form.setLabelAlignment(Qt.AlignRight)
        flags_holder = QWidget()
        flags_holder.setLayout(self.flags_form)
        flags_scroll = QScrollArea()
        flags_scroll.setWidget(flags_holder)
        flags_scroll.setWidgetResizable(True)
        flags_scroll.setMinimumHeight(180)
        flags_scroll.setMaximumHeight(360)
        flags_scroll.setFrameShape(QScrollArea.StyledPanel)
        st.addWidget(flags_scroll)

        self.cmd_preview = QLabel("")
        self.cmd_preview.setWordWrap(True)
        self.cmd_preview.setFont(mono)
        st.addWidget(self.cmd_preview)

        run_row = QHBoxLayout()
        self.run_btn = QPushButton("Run selected stage")
        self.run_btn.clicked.connect(self._run_stage)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_stage)
        run_row.addWidget(self.run_btn)
        run_row.addWidget(self.stop_btn)
        st.addLayout(run_row)
        left_layout.addWidget(stage_box, stretch=3)

        # Right: output browser + viewer
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(6, 6, 6, 6)
        right_layout.setSpacing(6)

        tabs = QTabWidget()

        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        console_font = QFont("Consolas")
        console_font.setStyleHint(QFont.Monospace)
        console_font.setPointSize(9)
        self.console.setFont(console_font)
        tabs.addTab(self.console, "Console")

        self.overview_view = SpeciesOverviewWidget()
        tabs.addTab(self.overview_view, "Overview")

        self.species_html_list = QListWidget()
        self.species_html_list.itemClicked.connect(self._on_output_clicked)
        tabs.addTab(self.species_html_list, "Species HTML plots")

        self.species_json_list = QListWidget()
        self.species_json_list.itemClicked.connect(self._on_output_clicked)
        tabs.addTab(self.species_json_list, "Species JSON data")

        self.shared_outputs_list = QListWidget()
        self.shared_outputs_list.itemClicked.connect(self._on_output_clicked)
        tabs.addTab(self.shared_outputs_list, "Shared / cross-element")

        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.addWidget(tabs)

        open_row = QHBoxLayout()
        open_btn = QPushButton("Open file...")
        open_btn.clicked.connect(self._open_file_dialog)
        refresh_btn = QPushButton("Refresh lists")
        refresh_btn.clicked.connect(self._refresh_all)
        open_row.addWidget(open_btn)
        open_row.addWidget(refresh_btn)
        open_row.addStretch(1)
        right_layout.addLayout(open_row)

        viewer_box = QGroupBox("Viewer")
        vb = QVBoxLayout(viewer_box)
        self.viewer_stack = QStackedWidget()

        if HAS_WEBENGINE:
            self.web_view = QWebEngineView()
            self.viewer_stack.addWidget(self.web_view) 
        else:
            self.web_view = None
            self.viewer_stack.addWidget(QLabel("(unused)"))

        self.json_view = QPlainTextEdit()
        self.json_view.setReadOnly(True)
        self.json_view.setFont(mono)
        self.viewer_stack.addWidget(self.json_view)  # index 1

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

        placeholder = QLabel("Select a species and an output file to view it here.")
        placeholder.setAlignment(Qt.AlignCenter)
        self.viewer_stack.addWidget(placeholder)  # index 3
        self.viewer_stack.setCurrentIndex(3)

        vb.addWidget(self.viewer_stack)
        right_splitter.addWidget(viewer_box)
        right_splitter.setChildrenCollapsible(False)
        right_splitter.setHandleWidth(8)
        right_splitter.setStretchFactor(0, 0)
        right_splitter.setStretchFactor(1, 1)
        right_splitter.setSizes([260, 620])
        right_layout.addWidget(right_splitter, stretch=1)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([420, 1100])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        self.setStatusBar(QStatusBar())
        self._update_status()

    #species
    def _set_species_from_input(self):
        self._set_species(self.species_input.text())

    def _set_species(self, species: str):
        species = species.strip()
        if not species:
            return
        self.current_species = species
        self.species_input.setText(species)
        self.species_label.setText(f"Current species: {species}")
        existing = self.species_list.findItems(species, Qt.MatchExactly)
        if not existing:
            self.species_list.addItem(species)
        self._refresh_stage_markers()
        self._refresh_species_outputs()
        self._refresh_overview()
        self._update_stage_desc()
        self._update_cmd_preview()

    # stage selection
    def _on_stage_selected(self, current, _previous):
        self._update_stage_desc()
        self._build_flags_form(self._selected_stage())

    def _update_stage_desc(self):
        stage = self._selected_stage()
        text = "" if stage is None else stage["desc"].replace(
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
        self._basis_picker_checkbox = None
        self._basis_picker_combo = None
        if stage is None:
            self.cmd_preview.setText("")
            return

        for spec in stage.get("flags", []):
            kind = spec["kind"]
            help_text = spec.get("help", "")
            default = spec.get("default", "")
            if callable(default):
                # e.g. workspace_data_dir: resolved against the current folder.
                default = default(self.root)

            if kind == "flag":
                value_widget = QCheckBox()
                value_widget.stateChanged.connect(self._update_cmd_preview)
                row_widget = value_widget
            elif kind == "choice":
                value_widget = QComboBox()
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
                browse_btn.clicked.connect(
                    lambda _checked=False, le=value_widget: self._browse_for_path(le))
                container = QWidget()
                hl = QHBoxLayout(container)
                hl.setContentsMargins(0, 0, 0, 0)
                hl.addWidget(value_widget, stretch=1)
                hl.addWidget(browse_btn)
                row_widget = container
            else:  # int, float, str, strlist
                value_widget = QLineEdit(default)
                value_widget.textChanged.connect(self._update_cmd_preview)
                row_widget = value_widget

            row_widget.setToolTip(help_text or spec["label"])
            value_widget.setToolTip(help_text or spec["label"])
            self._flag_widgets[spec["name"]] = (kind, value_widget)
            label = QLabel(spec["label"])
            label.setToolTip(help_text or spec["label"])
            self.flags_form.addRow(label, row_widget)

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

            self.flags_form.addRow(QLabel("Basis picker"), self._basis_picker_checkbox)
            self.flags_form.addRow(QLabel("Basis set"), self._basis_picker_combo)

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


    def _browse_for_path(self, line_edit: QLineEdit):
        path, _filter = QFileDialog.getOpenFileName(self, "Select file", str(self.root))
        if path:
            line_edit.setText(path)

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
            if kind == "flag":
                if w.isChecked():
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
            else:  # int, float, str, path
                text = w.text().strip()
                if text:
                    args += [name, text]
        return args

    def _stage_pipeline_args(self, stage, species):
        """``linak --pipeline`` arguments for *stage*, or None without elements."""
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
                "$ linak --pipeline compare_elements --elements <enter elements>")
            return
        self.cmd_preview.setText("$ linak --pipeline " + " ".join(args))

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
        program, args = stage_program(pipeline_args)

        self.console.clear()
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
    # the OS temp dir ({TEMP}/LiNaK_stdout_{PID}.log etc., see
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
        for pattern in ("LiNaK_stdout_*.log", "LiNaK_stderr_*.log"):
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
            self._reset_process_state()

    def _on_finished(self, stage, code, _status):
        ok = code == 0
        self._log(f"\n{'done.' if ok else f'exited with code {code}.'}")
        self.statusBar().showMessage(
            f"{stage['label']}: {'finished' if ok else f'failed (exit {code})'}"
        )
        self._reset_process_state()
        self._refresh_all()

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
            stage = STAGE_BY_ID[item.data(Qt.UserRole)]
            done = bool(self.current_species) and stage_has_output(
                stage, ws, self.current_species)
            mark = "\u2713 " if done else "\u00b7  "
            item.setText(f"{mark}[{stage['symbol']}]  {stage['label']}")

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
        found = rydberg_species(self.root)
        current_texts = {self.species_list.item(i).text() for i in range(self.species_list.count())}
        for sp in found:
            if sp not in current_texts:
                self.species_list.addItem(sp)

    def _refresh_overview(self):
        self.overview_view.refresh(self.root, self.current_species)

    def _refresh_all(self):
        self._update_status()
        self._refresh_species_list()
        self._refresh_stage_markers()
        self._refresh_species_outputs()
        self._refresh_shared_outputs()
        self._refresh_overview()

    def _update_status(self):
        self.statusBar().showMessage(f"Output folder: {self.root}")

    def _on_output_clicked(self, item: QListWidgetItem):
        data = item.data(Qt.UserRole)
        if not data:
            return
        path_str, kind = data
        path = Path(path_str)
        self._load_output(path, kind)

    def _load_output(self, path: Path, kind: str):
        if kind == "PLOT":
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


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)
    window = MainWindow()
    window.show()
    return app.exec()
