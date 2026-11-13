"""Extrapolator settings: fit settings (applied at once), model parameters (need a table rebuild), and presets for both.

Used as the "Extrapolator" tab of Edit > Preferences, and standalone when no Preferences window is available.
"""

from __future__ import annotations

from pathlib import Path

import json
from copy import deepcopy

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QFrame,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from qali.core.workspace import Workspace
from qali.physics.extrapolation import fit_settings as fs
from qali.physics.extrapolation import model_settings as ms
from qali.physics.extrapolation import userform
from qali.physics.extrapolation.exact import Cancelled, ExactBackendUnavailable
from qali.physics.extrapolation.stage import rebuild_tables, table_state
from qali.physics.extrapolation.params import model_params, supported_species

class _Spin(QDoubleSpinBox):
    """A type-in value field: no arrow buttons and the mouse wheel does not change it, so scrolling never edits a value."""

    def __init__(self, *args):
        super().__init__(*args)
        self.setButtonSymbols(QDoubleSpinBox.NoButtons)

    def wheelEvent(self, event):
        event.ignore()

    def flat(self):
        """For use inside a table cell: no frame of its own, so the cell does not hold a box inside a box."""
        self.setFrame(False)
        self.lineEdit().setStyleSheet("border: none; background: transparent; padding: 0 4px;")
        return self


class _Combo(QComboBox):
    def wheelEvent(self, event):
        event.ignore()


# Buttons inside a table cell: no frame of their own (the cell already has one), a light hover only.
CELL_BUTTON = ("QPushButton { border: none; background: transparent; padding: 0 8px; }"
               "QPushButton:hover { background: rgba(127, 127, 127, 60); }"
               "QPushButton:pressed { background: rgba(127, 127, 127, 100); }")

_NOTE_UNSOURCED = "not yet sourced"
_STATE_TEXT = {
    "shipped": "Tables: the shipped tables match these parameters.",
    "rebuilt": "Tables: rebuilt for these parameters.",
    "exact": "Tables: none match these parameters, so the exact calculation is used (slower). Rebuild to make it fast.",
}


def settings_file(root: Path) -> Path:
    return Workspace(root=root).runtime_root / "extrapolation" / "fit_settings.json"


class _RebuildWorker(QThread):
    progress = Signal(int, int, str)
    failed = Signal(str)

    def __init__(self, element: str, config_path: Path, model: dict, parent=None):
        super().__init__(parent)
        self.element, self.config_path, self.model = element, config_path, model
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            rebuild_tables(self.element, self.config_path, self.model,
                           progress=lambda d, n, m: self.progress.emit(d, n, m), cancelled=lambda: self._cancel)
        except Cancelled:
            pass
        except ExactBackendUnavailable as exc:
            self.failed.emit(str(exc))
        except Exception as exc:                                  # a numerical failure must not kill the GUI
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class ExtrapolatorSettingsPage(QWidget):
    """Edit the fit settings and model parameters; ``apply()`` writes ``fit_settings.json`` and ``model_settings.json``
    (only values that differ from the shipped ones)."""

    ROW_H = 34

    def __init__(self, root: Path, species: str | None = None, parent=None):
        super().__init__(parent)
        self.root = Path(root)
        self.path = settings_file(self.root)
        self.presets = fs.presets_dir(self.path)
        self.model_path = ms.settings_path(self.path)
        self.model_presets = ms.presets_dir(self.path)
        self._defaults = fs.defaults()
        self._model_edits: dict[str, dict] = ms.load(self.model_path)
        self.notes_path = self.model_path.with_name("model_notes.json")
        self._notes: dict[str, dict[str, str]] = self._load_notes()
        self._filling = False
        self._worker: _RebuildWorker | None = None
        inner = QWidget()
        outer = QVBoxLayout(inner)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        page = QVBoxLayout(self)
        page.setContentsMargins(0, 0, 0, 0)
        page.addWidget(scroll)

        box = QWidget()
        box.setToolTip("Take effect at once; no table rebuild needed.")
        boxlay = QVBoxLayout(box)
        boxlay.setContentsMargins(0, 0, 0, 0)
        self._section_header(boxlay, "Fit settings")
        grid = QGridLayout()
        boxlay.addLayout(grid)
        self.spins: dict[str, QDoubleSpinBox] = {}
        self.labels: dict[str, QLabel] = {}
        saved = fs.load(self.path)
        for row, spec in enumerate(fs.FIELDS):
            label = QLabel(spec["label"])
            label.setToolTip(spec["tip"])
            spin = _Spin()
            spin.setRange(spec["lo"], spec["hi"])
            spin.setDecimals(spec["dec"])
            spin.setSingleStep(spec["step"])
            spin.setValue(saved.get(spec["key"], self._defaults[spec["key"]]))
            spin.setToolTip(spec["tip"])
            spin.valueChanged.connect(self._mark)
            reset = QPushButton("Default")
            reset.setToolTip(f"Back to the packaged value ({self._defaults[spec['key']]:g}).")
            reset.clicked.connect(lambda _c=False, k=spec["key"]: self._reset(k))
            grid.addWidget(label, row, 0)
            grid.addWidget(spin, row, 1)
            grid.addWidget(reset, row, 2)
            self.spins[spec["key"]], self.labels[spec["key"]] = spin, label
        outer.addWidget(box)
        self.preset_combo = self._preset_row(outer, self._load_preset, self._save_preset, self._delete_preset,
                                             "the fit settings above")
        self._refresh_presets()

        mbox = QWidget()
        mlay = QVBoxLayout(mbox)
        mlay.setContentsMargins(0, 0, 0, 0)
        self._section_header(mlay, "Model parameters")
        srow = QHBoxLayout()
        srow.addWidget(QLabel("Species"))
        self.species_combo = _Combo()
        names = supported_species()
        self.species_combo.addItems(names)
        if species in names:
            self.species_combo.setCurrentText(species)
        self.species_combo.currentTextChanged.connect(self._fill_model)
        srow.addWidget(self.species_combo)
        reset_species = QPushButton("Default")
        reset_species.setToolTip("Back to the shipped parameters of this species.")
        reset_species.clicked.connect(self._reset_species)
        srow.addWidget(reset_species)
        srow.addStretch(1)
        mlay.addLayout(srow)
        self.model_table = QTableWidget(0, 4)
        self.model_table.setHorizontalHeaderLabels(["Parameter", "Value", "Default", "Note"])
        self.model_table.horizontalHeader().setStretchLastSection(True)
        self.model_table.verticalHeader().setVisible(False)
        self.model_table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.model_table.verticalHeader().setDefaultSectionSize(self.ROW_H)
        self.model_table.setEditTriggers(QTableWidget.DoubleClicked | QTableWidget.EditKeyPressed)
        self.model_table.itemChanged.connect(self._note_edited)
        self.model_table.setSelectionBehavior(QTableWidget.SelectItems)
        self.model_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.model_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.model_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.model_table.setObjectName("settingsTable")
        self.model_table.setShowGrid(True)
        hdr = self.model_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        mlay.addWidget(self.model_table)
        self._build_user_box(mlay)
        self.model_state = QLabel()
        self.model_state.setVisible(False)          # kept for the Rebuild button's state; not shown
        rrow = QHBoxLayout()
        self.rebuild_btn = QPushButton("Rebuild tables")
        self.rebuild_btn.clicked.connect(self._rebuild)
        rrow.addWidget(self.rebuild_btn)
        rrow.addStretch(1)
        mlay.addLayout(rrow)
        outer.addWidget(mbox)
        outer.addStretch(1)
        self.model_preset_combo = self._preset_row(outer, self._load_model_preset, self._save_model_preset,
                                                   self._delete_model_preset, "the model parameters (all species)")
        self._refresh_model_presets()
        self.form_combo: QComboBox | None = None
        self.model_spins: dict[str, QDoubleSpinBox] = {}
        self.model_labels: dict[str, QLabel] = {}
        self._fill_model()
        self._mark()

    @staticmethod
    def _section_header(layout, text: str) -> None:
        title = QLabel(text)
        font = title.font()
        font.setPointSizeF(font.pointSizeF() + 3)
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Plain)
        layout.addWidget(line)

    def _preset_row(self, layout, load, save, delete, what: str) -> QComboBox:
        row = QHBoxLayout()
        row.addWidget(QLabel("Presets"))
        combo = _Combo()
        combo.setMinimumWidth(160)
        row.addWidget(combo, stretch=1)
        for text, slot, tip in (("Load", load, "Put the preset's values into the fields (OK keeps them)."),
                                ("Save as...", save, f"Save {what} as a named preset."),
                                ("Delete", delete, "Remove the selected preset file.")):
            btn = QPushButton(text)
            btn.setToolTip(tip)
            btn.clicked.connect(slot)
            row.addWidget(btn)
        layout.addLayout(row)
        return combo

    # ---- fit settings ------------------------------------------------------------------------------------
    def values(self) -> dict[str, float]:
        """The fields as ``{key: value}``, only those that differ from the defaults."""
        return fs.validate({k: s.value() for k, s in self.spins.items()})[0]

    def set_values(self, fit: dict) -> None:
        for key, spin in self.spins.items():
            spin.setValue(fit.get(key, self._defaults[key]))

    def _mark(self, *_):
        """Bold label = the value differs from the packaged default."""
        for key, spin in self.spins.items():
            changed = abs(spin.value() - self._defaults[key]) > 1e-12
            font = self.labels[key].font()
            font.setBold(changed)
            self.labels[key].setFont(font)

    def _reset(self, key: str) -> None:
        self.spins[key].setValue(self._defaults[key])

    def reset_all(self) -> None:
        self.set_values({})
        self._model_edits = {}
        self._shown = None            # the widgets show old values: do not store them back
        self._fill_model()

    def apply(self) -> list[str]:
        """Save both files; returns what was left out and why (e.g. an invalid user-defined formula)."""
        fs.save(self.path, self.values())
        self._store_current()
        problems = ms.validate(self._model_edits)[1]
        ms.save(self.model_path, self.model_values())
        self._save_notes()
        return problems

    # ---- user notes (free text in the empty Note cells) ----------------------------------------------------
    def _load_notes(self) -> dict:
        try:
            data = json.loads(self.notes_path.read_text(encoding="utf-8"))
            return {str(k): {str(a): str(b) for a, b in v.items()} for k, v in data.items()}
        except (OSError, ValueError, AttributeError):
            return {}

    def _save_notes(self) -> None:
        clean = {el: {k: t for k, t in d.items() if t.strip()} for el, d in self._notes.items()}
        clean = {el: d for el, d in clean.items() if d}
        try:
            if clean:
                self.notes_path.parent.mkdir(parents=True, exist_ok=True)
                self.notes_path.write_text(json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8")
            elif self.notes_path.exists():
                self.notes_path.unlink()
        except OSError:
            pass

    def _note_edited(self, item) -> None:
        if self._filling or item.column() != 3:
            return
        key = item.data(Qt.UserRole)
        if key:
            self._notes.setdefault(self.species_combo.currentText(), {})[key] = item.text()

    # ---- fit presets -------------------------------------------------------------------------------------
    def _refresh_presets(self, select: str | None = None) -> None:
        names = fs.list_presets(self.presets)
        self.preset_combo.clear()
        self.preset_combo.addItems(names)
        if select in names:
            self.preset_combo.setCurrentText(select)

    def _load_preset(self) -> None:
        name = self.preset_combo.currentText()
        if name:
            self.set_values(fs.load_preset(self.presets, name))

    def _save_preset(self) -> None:
        name, ok = QInputDialog.getText(self, "Save preset", "Preset name:")
        if ok and name.strip():
            fs.save_preset(self.presets, name, self.values())
            self._refresh_presets(select=fs._safe(name))

    def _delete_preset(self) -> None:
        name = self.preset_combo.currentText()
        if name and QMessageBox.question(self, "Delete preset", f"Delete the preset '{name}'?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes:
            fs.delete_preset(self.presets, name)
            self._refresh_presets()

    # ---- model parameters --------------------------------------------------------------------------------
    def model_values(self) -> dict[str, dict]:
        """Edits of all species that differ from the shipped parameters: ``{species: {key: value}}``."""
        self._store_current()
        return ms.validate(self._model_edits)[0]

    def set_model_values(self, model: dict) -> None:
        self._model_edits = {k: dict(v) for k, v in (model or {}).items()}
        self._shown = None
        self._fill_model()

    def _store_current(self) -> None:
        el = getattr(self, "_shown", None)
        if el and self.model_spins:
            self._model_edits[el] = {k: s.value() for k, s in self.model_spins.items()}
            if self.form_combo is not None:
                self._model_edits[el]["form"] = self.form_combo.currentData()
                self._model_edits[el]["user_formula"] = self.user_formula.text()
                self._model_edits[el]["user_params"] = deepcopy(self._user_state)

    def _fill_model(self, *_):
        self._store_current()
        el = self.species_combo.currentText()
        self._shown = el
        self._filling = True
        mp = model_params()
        sp = mp["species"].get(el, {})
        base = ms.defaults(el) if el in mp["species"] else {}
        mine = {**base, **self._model_edits.get(el, {})}
        self.model_spins = {}
        self.model_labels = {}
        rows = [(f["html"], f, f["tip"]) for f in ms.FIELDS]
        extra = [("np cutoff rule", mp.get("np_cutoff"), "")]
        self.model_table.setRowCount(len(rows) + len(extra) + 1)
        for i, (name, spec, tip) in enumerate(rows):
            item = QLabel(name)                      # rich text: real subscripts
            item.setContentsMargins(4, 0, 4, 0)
            item.setToolTip(tip)
            self.model_table.setCellWidget(i, 0, item)
            self.model_labels[spec["key"]] = item
            spin = _Spin()
            spin.setRange(spec["lo"], spec["hi"])
            spin.setDecimals(spec["dec"])
            spin.setSingleStep(spec["step"])
            spin.setValue(mine.get(spec["key"], 0.0))
            spin.setToolTip(tip)
            spin.flat()
            spin.valueChanged.connect(self._model_changed)
            self.model_table.setCellWidget(i, 1, spin)
            self.model_spins[spec["key"]] = spin
            shipped = QTableWidgetItem("off" if spec["key"] == "wcore_eV" and not base.get(spec["key"]) else f"{base.get(spec['key'], 0):g}")
            shipped.setFlags(Qt.ItemIsEnabled)
            self.model_table.setItem(i, 2, shipped)
            note = ""
            if spec["key"] == "singor_rc" and sp.get("note"):
                note = _NOTE_UNSOURCED
            note_item = QTableWidgetItem(note)
            self._note_cell(note_item, spec["key"], el, used=bool(note))
            self.model_table.setItem(i, 3, note_item)
        for j, (name, value, note) in enumerate(extra, start=len(rows)):
            for col, text in enumerate((name, "-" if value is None else str(value), "", note)):
                it = QTableWidgetItem(text)
                it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                if col == 3:
                    self._note_cell(it, "extra:" + name, el, used=bool(note))
                self.model_table.setItem(j, col, it)
        frow = len(rows) + len(extra)
        item = QLabel("Potential form")
        item.setContentsMargins(4, 0, 4, 0)
        item.setToolTip("Shape of the core polarization correction in the dipole operator.")
        self.model_table.setCellWidget(frow, 0, item)
        self.model_labels["form"] = item
        combo = _Combo()
        for fid, label, formula in ms.FORMS:
            combo.addItem(label, fid)
            combo.setItemData(combo.count() - 1, formula, Qt.ToolTipRole)
        shipped_form = mp.get("form")
        combo.setCurrentIndex(max(combo.findData(self._model_edits.get(el, {}).get("form", shipped_form)), 0))
        combo.setStyleSheet("QComboBox { border: none; background: transparent; padding: 0 4px; }")
        combo.currentIndexChanged.connect(self._model_changed)
        self.model_table.setCellWidget(frow, 1, combo)
        self.form_combo = combo
        names = {fid: (label, formula) for fid, label, formula in ms.FORMS}
        shipped_item = QTableWidgetItem(names.get(shipped_form, ("", ""))[0])
        shipped_item.setFlags(Qt.ItemIsEnabled)
        self.model_table.setItem(frow, 2, shipped_item)
        self._load_user(el)
        self.form_note = QTableWidgetItem()
        self.form_note.setFlags(Qt.ItemIsEnabled)
        self.model_table.setItem(frow, 3, self.form_note)
        self.model_table.setFixedHeight(self.model_table.horizontalHeader().height() + self.model_table.rowCount() * self.ROW_H + 4)
        self._filling = False
        self._model_changed()

    def _note_cell(self, item, key: str, el: str, used: bool) -> None:
        """An unused Note cell takes the user's own text; one that already carries a note stays fixed."""
        if used:
            item.setFlags(Qt.ItemIsEnabled)
            return
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable)
        item.setData(Qt.UserRole, key)
        item.setText(self._notes.get(el, {}).get(key, ""))

    def _model_changed(self, *_):
        el = self.species_combo.currentText()
        base = ms.defaults(el) if el in model_params()["species"] else {}
        for key, spin in self.model_spins.items():
            changed = abs(spin.value() - base.get(key, 0.0)) > 1e-12
            item = self.model_labels[key]
            font = item.font()
            font.setBold(changed)
            item.setFont(font)
        if self.form_combo is not None:
            self.form_note.setText({f[0]: f[2] for f in ms.FORMS}[self.form_combo.currentData()])
            item = self.model_labels["form"]
            font = item.font()
            font.setBold(self.form_combo.currentData() != model_params()["form"])
            item.setFont(font)
            self.user_box.setVisible(self.form_combo.currentData().startswith("U"))
            self._validate_user()
        self._store_current()
        self._update_state()

    def _reset_species(self) -> None:
        self._model_edits.pop(self.species_combo.currentText(), None)
        self._shown = None
        self._fill_model()

    def _update_state(self) -> None:
        el = self.species_combo.currentText()
        mine = ms.validate({el: self._model_edits.get(el, {})})[0]
        try:
            state = table_state(el, self.path.parent / "x", mine)["state"]
        except Exception as exc:                                  # a missing table must not break the page
            self.model_state.setText(f"Tables: could not be checked ({exc}).")
            return
        self.model_state.setText(_STATE_TEXT[state])
        self.rebuild_btn.setEnabled(state == "exact")

    @staticmethod
    def _arc_installed() -> bool:
        try:
            import arc  # noqa: F401
        except ImportError:
            return False
        return True

    def _rebuild(self) -> None:
        if self._worker is not None:
            return
        if not self._arc_installed():
            QMessageBox.information(self, "Rebuild tables", "ARC is not installed.")
            return
        el = self.species_combo.currentText()
        mine = ms.validate({el: self._model_edits.get(el, {})})[0]
        dlg = QProgressDialog(f"Rebuilding the {el} tables...", "Cancel", 0, 100, self)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        worker = _RebuildWorker(el, self.path.parent / "x", mine, self)
        self._worker = worker
        worker.progress.connect(lambda d, n, m: (dlg.setMaximum(n), dlg.setValue(d), dlg.setLabelText(f"Rebuilding the {el} tables... {m}")))
        worker.failed.connect(lambda msg: QMessageBox.warning(self, "Rebuild tables", msg))
        dlg.canceled.connect(worker.cancel)
        worker.finished.connect(lambda: self._rebuild_done(dlg))
        worker.start()

    def _rebuild_done(self, dlg) -> None:
        dlg.close()
        self._worker = None
        self._update_state()

    # ---- user-defined form -------------------------------------------------------------------------------
    def _build_user_box(self, layout) -> None:
        self._user_state: list[dict] = []
        box = QGroupBox("User-defined g(r)")
        self.user_box = box
        lay = QVBoxLayout(box)
        row = QHBoxLayout()
        row.addWidget(QLabel("g(r) ="))
        self.user_formula = QLineEdit()
        self.user_formula.setPlaceholderText("e.g. alpha_D / r**3 * (1 - exp(-(r/rc)**3))")
        self.user_formula.textChanged.connect(self._model_changed)
        row.addWidget(self.user_formula, 1)
        lay.addLayout(row)
        hint = QLabel("Write g(r) as an expression in r (the distance from the nucleus, in a₀). You can use alpha_D, rc, "
                      "pi, e, these functions: " + ", ".join(userform.FUNCTIONS)
                      + ", and any parameters you add below. The transition dipole becomes r × (1 − λ g(r)).")
        hint.setWordWrap(True)
        hint.setObjectName("muted")
        lay.addWidget(hint)
        self.user_table = QTableWidget(0, 4)
        self.user_table.setHorizontalHeaderLabels(["Parameter", "Value", "", ""])
        self.user_table.verticalHeader().setVisible(False)
        self.user_table.setObjectName("settingsTable")
        self.user_table.setShowGrid(True)
        self.user_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.user_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.user_table.setEditTriggers(QTableWidget.NoEditTriggers)
        hdr = self.user_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        for col, width in ((2, 150), (3, 44)):            # the buttons' columns: fixed so they never clip
            hdr.setSectionResizeMode(col, QHeaderView.Fixed)
            self.user_table.setColumnWidth(col, width)
        lay.addWidget(self.user_table)
        brow = QHBoxLayout()
        self.user_add = QPushButton("+")
        self.user_add.setFixedWidth(40)
        self.user_add.setToolTip("Add a parameter")
        self.user_add.clicked.connect(self._add_user_param)
        brow.addWidget(self.user_add)
        brow.addStretch(1)
        lay.addLayout(brow)
        self.user_msg = QLabel()
        self.user_msg.setWordWrap(True)
        lay.addWidget(self.user_msg)
        box.setVisible(False)
        layout.addWidget(box)

    def _load_user(self, el: str) -> None:
        edits = self._model_edits.get(el, {})
        self.user_formula.blockSignals(True)
        self.user_formula.setText(edits.get("user_formula", ""))
        self.user_formula.blockSignals(False)
        self._user_state = deepcopy(edits.get("user_params") or [])
        self._render_user()

    def _render_user(self) -> None:
        self.user_table.setRowCount(len(self._user_state))
        for i, p in enumerate(self._user_state):
            name = QLineEdit(p.get("name", ""))
            name.setPlaceholderText("identifier")
            name.setFrame(False)
            name.setStyleSheet("border: none; background: transparent; padding: 0 4px;")
            name.textChanged.connect(lambda text, i=i: self._user_edit(i, "name", text))
            self.user_table.setCellWidget(i, 0, name)
            if p.get("table"):
                pts = len(p["table"]["r"])
                val = QPushButton(f"dataset, {pts} points (click for a constant)")
                val.setStyleSheet(CELL_BUTTON)
                val.setToolTip("Back to a constant value")
                val.clicked.connect(lambda _c=False, i=i: self._user_to_constant(i))
            else:
                val = _Spin()
                val.setRange(-1e9, 1e9)
                val.setDecimals(6)
                val.flat()
                val.setValue(float(p.get("value", 0.0)))
                val.valueChanged.connect(lambda v, i=i: self._user_edit(i, "value", v))
            self.user_table.setCellWidget(i, 1, val)
            imp = QPushButton("Import dataset...")
            imp.setStyleSheet(CELL_BUTTON)
            imp.setToolTip("Use a table of values against r (two columns: r in a0, value) instead of a constant.")
            imp.clicked.connect(lambda _c=False, i=i: self._import_dataset(i))
            self.user_table.setCellWidget(i, 2, imp)
            rem = QPushButton("−")
            rem.setStyleSheet(CELL_BUTTON)
            rem.setToolTip("Remove this parameter")
            rem.clicked.connect(lambda _c=False, i=i: self._remove_user_param(i))
            self.user_table.setCellWidget(i, 3, rem)
        self.user_table.resizeRowsToContents()
        self.user_table.setFixedHeight(34 + 36 * max(len(self._user_state), 1))      # fits its rows, no empty space

    def _user_edit(self, i: int, key: str, value) -> None:
        if 0 <= i < len(self._user_state):
            self._user_state[i][key] = value
            self._model_changed()

    def _add_user_param(self) -> None:
        taken = {p.get("name", "").lower() for p in self._user_state}
        k = 1
        while f"p{k}" in taken:
            k += 1
        self._user_state.append({"name": f"p{k}", "value": 0.0})
        self._render_user()
        self._model_changed()

    def _remove_user_param(self, i: int) -> None:
        if 0 <= i < len(self._user_state):
            del self._user_state[i]
            self._render_user()
            self._model_changed()

    def _user_to_constant(self, i: int) -> None:
        self._user_state[i].pop("table", None)
        self._render_user()
        self._model_changed()

    def _import_dataset(self, i: int, path: str | None = None) -> None:
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "Import dataset", "", "Data files (*.csv *.txt *.dat);;All files (*)")
            if not path:
                return
        try:
            self._user_state[i]["table"] = userform.parse_dataset(path)
        except (userform.FormulaError, OSError) as exc:
            self.user_msg.setText(f"Dataset not imported: {exc}")
            return
        self._render_user()
        self._model_changed()

    def _validate_user(self) -> None:
        if not self.user_box.isVisible() and not (self.form_combo and self.form_combo.currentData().startswith("U")):
            return
        try:
            userform.validate_user(self.user_formula.text().strip(), self._user_state)
            self.user_msg.setText("")
        except userform.FormulaError as exc:
            # an empty formula is just unfinished; anything else says what is wrong
            self.user_msg.setText(str(exc).capitalize() if self.user_formula.text().strip() else "")

    # ---- model presets -----------------------------------------------------------------------------------
    def _refresh_model_presets(self, select: str | None = None) -> None:
        names = ms.list_presets(self.model_presets)
        self.model_preset_combo.clear()
        self.model_preset_combo.addItems(names)
        if select in names:
            self.model_preset_combo.setCurrentText(select)

    def _load_model_preset(self) -> None:
        name = self.model_preset_combo.currentText()
        if name:
            self.set_model_values(ms.load_preset(self.model_presets, name))

    def _save_model_preset(self) -> None:
        name, ok = QInputDialog.getText(self, "Save preset", "Preset name:")
        if ok and name.strip():
            ms.save_preset(self.model_presets, name, self.model_values())
            self._refresh_model_presets(select=ms._safe(name))

    def _delete_model_preset(self) -> None:
        name = self.model_preset_combo.currentText()
        if name and QMessageBox.question(self, "Delete preset", f"Delete the preset '{name}'?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes:
            ms.delete_preset(self.model_presets, name)
            self._refresh_model_presets()
