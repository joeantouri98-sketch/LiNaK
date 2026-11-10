"""Viewer page that shows ``comparison_table.json`` as a live, sortable table.

The Doppler columns follow the temperature box; "Add" asks the main window to
run the Compare stage for more elements; "Export CSV" writes the rows as shown.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QPainter
from PySide6.QtWidgets import (
    QAbstractItemView, QDoubleSpinBox, QMenu, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from linak.core import comparison_rows as cr
from linak.core.fileio import atomic_write_text, write_json


class _Cell(QTableWidgetItem):
    """A cell that sorts by its value (numbers as numbers, blanks last)."""

    def __init__(self, text: str, value, row_values=None):
        super().__init__(text)
        self.value = value
        self.row_values = row_values or {}
        self.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)

    def __lt__(self, other):
        a, b = self.value, getattr(other, "value", None)
        if a is None or b is None:
            return a is not None and b is None
        try:
            return a < b
        except TypeError:
            return str(a) < str(b)


_GROUP_TINTS = (QColor(0, 0, 0, 0), QColor(52, 152, 219, 28), QColor(155, 89, 182, 28),
                QColor(46, 204, 113, 28), QColor(243, 156, 18, 28), QColor(231, 76, 60, 28))
_KEYS = [k for k, *_ in cr.COLUMNS]
_GROUP_OF = {}
for _g, (_t, _a, _b) in enumerate(cr.GROUPS):
    for _k in _KEYS[_KEYS.index(_a):_KEYS.index(_b) + 1]:
        _GROUP_OF[_k] = _g


class _GroupBand(QWidget):
    """The row of section titles above the column headers, following the table's columns."""

    def __init__(self, table: QTableWidget, view):
        super().__init__()
        self._table, self._view = table, view
        self.setFixedHeight(24)
        header = table.horizontalHeader()
        header.sectionResized.connect(self.update)
        table.horizontalScrollBar().valueChanged.connect(self.update)

    def paintEvent(self, _event):
        p = QPainter(self)
        header = self._table.horizontalHeader()
        offset = self._table.verticalHeader().width() if self._table.verticalHeader().isVisible() else 0
        pal = self.palette()
        for g, (title, first, last) in enumerate(cr.GROUPS):
            a, b = _KEYS.index(first), _KEYS.index(last)
            x0 = header.sectionViewportPosition(a) + offset
            x1 = header.sectionViewportPosition(b) + header.sectionSize(b) + offset
            rect = QRect(x0, 0, x1 - x0, self.height())
            p.fillRect(rect, pal.window())
            p.fillRect(rect, _GROUP_TINTS[g] if g else QColor(128, 128, 128, 30))
            p.setPen(pal.windowText().color())
            p.drawText(rect, Qt.AlignCenter, title.format(T=f"{self._view.temperature.value():g}"))
            p.setPen(pal.mid().color())
            p.drawLine(x1 - 1, 0, x1 - 1, self.height())


class CompareTableView(QWidget):
    add_requested = Signal(str)
    raw_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._records: list[dict] = []
        self._path: Path | None = None

        bar = QHBoxLayout()
        bar.addWidget(QLabel("Temperature (K)"))
        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.001, 100000.0)
        self.temperature.setDecimals(1)
        self.temperature.setValue(300.0)
        self.temperature.setToolTip("Doppler columns are recomputed at this temperature.")
        self.temperature.valueChanged.connect(self._fill)
        bar.addWidget(self.temperature)
        bar.addSpacing(16)
        self.add_edit = QLineEdit()
        self.add_edit.setPlaceholderText("Add elements, e.g. Mg or alkaline_earth")
        self.add_edit.returnPressed.connect(self._add)
        bar.addWidget(self.add_edit, 1)
        self.add_btn = QPushButton("Add")
        self.add_btn.clicked.connect(self._add)
        bar.addWidget(self.add_btn)
        bar.addSpacing(16)
        self.remove_btn = QPushButton("Remove selected")
        self.remove_btn.clicked.connect(self._remove_selected)
        bar.addWidget(self.remove_btn)
        self.export_btn = QPushButton("Export CSV...")
        self.export_btn.clicked.connect(self._export)
        bar.addWidget(self.export_btn)
        self.raw_btn = QPushButton("Raw JSON")
        self.raw_btn.clicked.connect(self.raw_requested.emit)
        bar.addWidget(self.raw_btn)

        self.table = QTableWidget(0, len(cr.COLUMNS))
        self.table.setHorizontalHeaderLabels([h for _k, h, _c, _d in cr.COLUMNS])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setSortIndicator(1, Qt.AscendingOrder)

        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._menu)
        self.band = _GroupBand(self.table, self)
        self.note = QLabel("")
        lay = QVBoxLayout(self)
        lay.setSpacing(2)
        lay.addLayout(bar)
        lay.addWidget(self.band)
        lay.addWidget(self.table, 1)
        lay.addWidget(self.note)

    @property
    def path(self) -> Path | None:
        return self._path

    def load(self, path: Path) -> None:
        """Read *path* (comparison_table.json) and show it."""
        self._path = Path(path)
        try:
            t_k, self._records = cr.load_table(self._path)
        except (OSError, ValueError) as exc:
            self._records, t_k = [], self.temperature.value()
            self.note.setText(f"Could not read {self._path.name}: {exc}")
        else:
            self.note.setText("")
        self.temperature.blockSignals(True)
        self.temperature.setValue(t_k)
        self.temperature.blockSignals(False)
        self._fill()

    def rows(self) -> list[dict]:
        """The table's rows in the order they are shown."""
        by_name = {r["element"]: r for r in cr.build_rows(self._records, self.temperature.value())}
        names = [self.table.item(i, 0).text() for i in range(self.table.rowCount())]
        return [by_name[n] for n in names if n in by_name]

    def _fill(self, *_):
        rows = cr.build_rows(self._records, self.temperature.value())
        header = self.table.horizontalHeader()
        col, order = header.sortIndicatorSection(), header.sortIndicatorOrder()
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, (key, _h, _c, digits) in enumerate(cr.COLUMNS):
                value = row.get(key)
                item = _Cell(cr.cell_text(key, value, digits), value, row)
                if key == "element":
                    item.setToolTip(f"Source: {row.get('source', '')}")
                if j:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(i, j, item)
        self.table.setSortingEnabled(True)
        self.table.sortByColumn(col, order)
        self.table.resizeColumnsToContents()
        self.band.update()
        self._paint()
        if not rows and not self.note.text():
            self.note.setText("No elements in the table yet. Add some above.")

    def _paint(self, *_):
        """Each column section gets its own faint tint."""
        for i in range(self.table.rowCount()):
            for j, k in enumerate(_KEYS):
                self.table.item(i, j).setBackground(QBrush(_GROUP_TINTS[_GROUP_OF[k]]))
        self.band.update()

    def _menu(self, pos):
        row = self.table.rowAt(pos.y())
        if row < 0:
            return
        self.table.selectRow(row)
        menu = QMenu(self)
        act = QAction(f"Remove {self.table.item(row, 0).text()}", menu)
        act.triggered.connect(self._remove_selected)
        menu.addAction(act)
        menu.exec(self.table.viewport().mapToGlobal(pos))

    def _remove_selected(self):
        """Drop the selected elements from the saved table."""
        names = {self.table.item(i.row(), 0).text() for i in self.table.selectedIndexes()}
        if not names or self._path is None:
            return
        keep = [r for r in self._records if r.get("element") not in names]
        try:
            write_json(self._path, {"T_K": self.temperature.value(), "elements": keep})
        except OSError as exc:
            QMessageBox.warning(self, "Remove", f"Could not write {self._path.name}:\n{exc}")
            return
        self._records = keep
        self._fill()

    def _add(self):
        text = self.add_edit.text().strip()
        if text:
            self.add_requested.emit(text)

    def _export(self):
        start = str(self._path.parent / cr.CSV_FILENAME) if self._path else cr.CSV_FILENAME
        name, _f = QFileDialog.getSaveFileName(self, "Export CSV", start, "CSV (*.csv)")
        if not name:
            return
        try:
            atomic_write_text(name, cr.csv_text(self.rows()))
        except OSError as exc:
            QMessageBox.warning(self, "Export CSV", f"Could not write the file:\n{exc}")
