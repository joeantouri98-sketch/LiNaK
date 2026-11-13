"""Enter the hyperfine constants of one isotope: a state per row."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QHeaderView,
)

from qali.core import species_data as sd

from . import brand


class HyperfineDialog(QDialog):
    HEADERS = ("State", "A (MHz)", "B (MHz)", "C (kHz)")

    def __init__(self, species: str, data_dir: Path, parent=None):
        super().__init__(parent)
        self.species = species
        self.setWindowTitle(f"Hyperfine constants of {species}")
        self.setWindowIcon(brand.species_icon(species))
        self.resize(520, 380)
        self.result_rows: tuple[int, list] | None = None
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel("Mass number"))
        self.mass = QLineEdit()
        self.mass.setPlaceholderText("e.g. 25")
        self.mass.setMaximumWidth(90)
        top.addWidget(self.mass)
        top.addStretch(1)
        lay.addLayout(top)
        self.grid = QTableWidget(0, 4)
        self.grid.setHorizontalHeaderLabels(self.HEADERS)
        self.grid.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        lay.addWidget(self.grid, 1)
        row = QHBoxLayout()
        add, remove = QPushButton("Add state"), QPushButton("Remove state")
        add.clicked.connect(lambda: self.grid.insertRow(self.grid.rowCount()))
        remove.clicked.connect(self._remove)
        row.addWidget(add)
        row.addWidget(remove)
        row.addStretch(1)
        lay.addLayout(row)
        self.error = QLabel("")
        self.error.setWordWrap(True)
        self.error.setStyleSheet("color: #d9534f;")
        lay.addWidget(self.error)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

        sid, _el = sd.resolve(species)
        mass, rows = sd.read_hf_constants(data_dir, sid)             # edit what is already there
        if mass:
            self.mass.setText(str(mass))
        for r in rows or [("", 0, 0, 0)]:
            self.grid.insertRow(self.grid.rowCount())
            if r[0]:
                for c, v in enumerate((r[0], f"{r[1]:g}", f"{r[2]:g}", f"{r[3]:g}")):
                    self.grid.setItem(self.grid.rowCount() - 1, c, QTableWidgetItem(v))

    def _remove(self) -> None:
        row = self.grid.currentRow()
        if row >= 0:
            self.grid.removeRow(row)

    def _text(self, r: int, c: int) -> str:
        item = self.grid.item(r, c)
        return item.text() if item else ""

    def accept(self) -> None:
        rows = [tuple(self._text(r, c) for c in range(4)) for r in range(self.grid.rowCount())]
        try:
            self.result_rows = sd.check_hf(self.species, self.mass.text(), rows)
        except ValueError as exc:
            self.error.setText(str(exc))
            return
        super().accept()
