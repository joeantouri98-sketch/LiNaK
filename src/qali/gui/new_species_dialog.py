"""Set up a species Qali has no data for: import its NIST files, type its numbers, see which stages that enables."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from qali.core import species_data as sd, user_tables
from qali.core.workspace import Workspace

from . import brand
from .hyperfine_dialog import HyperfineDialog
from .tables_page import AddRowDialog

_SECTIONS = (("required", "Needed to calculate"), ("recommended", "Improves results"), ("optional", "Optional"))


class NewSpeciesDialog(QDialog):
    def __init__(self, root: Path, species: str, parent=None):
        super().__init__(parent)
        self.root = Path(root)
        self.species_id, self.element = sd.resolve(species)
        self.setWindowTitle(f"Set up {self.species_id}")
        self.setWindowIcon(brand.species_icon(self.species_id))
        self.resize(900, 560)
        self._pending: set[str] = set()                      # numbers and records typed but not saved
        self._record: dict | None = None
        self._hf: tuple[int, list] | None = None
        self.data_dir = Workspace(root=self.root).data_json_dir
        self._boxes: dict[str, QLineEdit] = {}
        self._status: dict[str, QLabel] = {}
        self._buttons: dict[str, QPushButton] = {}

        outer = QVBoxLayout(self)
        intro = QLabel(f"No data for {self.species_id} is bundled. Add what you have; whatever you skip can be added later "
                       "from right-click > Properties.")
        intro.setWordWrap(True)
        outer.addWidget(intro)

        body = QHBoxLayout()
        outer.addLayout(body, 1)
        grid = QGridLayout()
        body.addLayout(grid, 3)
        row = 0
        for section, title in _SECTIONS:
            head = QLabel(f"<b>{title}</b>")
            grid.addWidget(head, row, 0, 1, 3)
            row += 1
            for item in (i for i in sd.ITEMS if i.section == section):
                grid.addWidget(QLabel(item.label), row, 0)
                self._status[item.key] = QLabel("")
                self._status[item.key].setObjectName("muted")
                if item.kind == "number":
                    box = QLineEdit()
                    unit = user_tables.BY_ID[item.table_id].unit
                    box.setPlaceholderText(unit or "number")
                    box.textChanged.connect(lambda _t, k=item.key: self._number_typed(k))
                    self._boxes[item.key] = box
                    grid.addWidget(box, row, 1)
                else:
                    btn = QPushButton("Import..." if item.kind == "file" else "Enter values...")
                    btn.clicked.connect(lambda _c=False, k=item.key: self._pick(k))
                    self._buttons[item.key] = btn
                    grid.addWidget(btn, row, 1)
                grid.addWidget(self._status[item.key], row, 2)
                if item.hint:
                    grid.itemAtPosition(row, 0).widget().setToolTip(item.hint)
                row += 1
        grid.setRowStretch(row, 1)
        grid.setColumnStretch(2, 1)

        right = QVBoxLayout()
        body.addLayout(right, 2)
        right.addWidget(QLabel("<b>What this enables</b>"))
        self.stages = QListWidget()
        self.stages.setSelectionMode(QListWidget.NoSelection)
        right.addWidget(self.stages, 1)

        self.message = QLabel("")
        self.message.setWordWrap(True)
        outer.addWidget(self.message)
        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Close)
        bb.button(QDialogButtonBox.Close).setText("Later")
        bb.accepted.connect(self._save)
        bb.rejected.connect(self.reject)
        outer.addWidget(bb)
        self._refresh()

    # --- state ---------------------------------------------------------------------------
    def _have(self, key: str) -> bool:
        return sd.present(sd.BY_KEY[key], self.species_id, self.element, self._pending, self.data_dir)

    def _refresh(self) -> None:
        for item in sd.ITEMS:
            have = self._have(item.key)
            saved = sd.present(item, self.species_id, self.element, data_dir=self.data_dir)
            label = self._status[item.key]
            label.setText("✓ present" if saved else "✓ ready to save" if have else "missing")
            if item.key in self._boxes:
                self._boxes[item.key].setEnabled(not saved)
                if saved:
                    self._boxes[item.key].setPlaceholderText("already set")
            if item.kind == "file" and saved:
                self._buttons[item.key].setText("Replace...")
            elif item.kind == "hyperfine" and saved:
                self._buttons[item.key].setText("Edit...")
        self.stages.clear()
        for _sid, name, ready, why in sd.stage_states(self.species_id, self.data_dir, self._pending):
            entry = QListWidgetItem(("✓ " if ready else "✗ ") + name + ("" if ready else f"  –  {why}"))
            entry.setToolTip(why)
            self.stages.addItem(entry)

    def _number_typed(self, key: str) -> None:
        table_id = sd.BY_KEY[key].table_id
        text = self._boxes[key].text().strip()
        try:
            user_tables.check_value(table_id, text)
            (self._pending.add if text else self._pending.discard)(key)
        except ValueError:
            self._pending.discard(key)
        self._refresh()

    # --- actions -------------------------------------------------------------------------
    def _pick(self, key: str) -> None:
        item = sd.BY_KEY[key]
        if item.kind == "record":
            dlg = AddRowDialog(item.table_id, {}, self, element=self.element)
            if dlg.exec() and isinstance(dlg.result_row[1], dict):
                self._record = {"record": dlg.result_row[1], "source": dlg.result_row[2]}
                self._pending.add(key)
                self._refresh()
            return
        if item.kind == "hyperfine":
            dlg = HyperfineDialog(self.species_id, self.data_dir, self)
            if dlg.exec() and dlg.result_rows:
                self._hf = dlg.result_rows
                self._pending.add(key)
                self._refresh()
            return
        path, _ = QFileDialog.getOpenFileName(self, f"Import {item.label}", "", "NIST export (*.csv *.txt);;All files (*)")
        if not path:
            return
        try:
            summary = sd.import_file(key, self.species_id, path)
        except ValueError as exc:
            self.message.setText(f"<span style='color:#d9534f'>{exc}</span>")
            return
        self.message.setText(f"{item.label}: {summary}")
        self._refresh()

    def _save(self) -> None:
        additions = user_tables.load_additions()
        try:
            for key in sorted(self._pending):
                item = sd.BY_KEY[key]
                if item.kind == "number":
                    value = user_tables.check_value(item.table_id, self._boxes[key].text())
                    additions.setdefault(item.table_id, {})[self.element] = {"value": value, "source": ""}
                elif item.kind == "record" and self._record:
                    additions.setdefault(item.table_id, {})[self.element] = self._record
            user_tables.save_additions(additions)
            if "hf" in self._pending and self._hf:
                sd.write_hf_constants(self.data_dir, self.species_id, *self._hf)
            for table_id in additions:
                user_tables.reload(table_id)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Set up", f"Could not save: {exc}")
            return
        self.accept()
