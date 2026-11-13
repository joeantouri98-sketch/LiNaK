"""Preferences > Tables and variables: Qali's built-in numbers, viewable and editable.

Constant tables (values written in the code) are edited in place and saved as overrides on Apply/OK; packaged data
files (NIST lines and levels, moments, extrapolation anchors) open read-only with Ctrl+F and are edited as a copy.
The packaged originals are never changed (see :mod:`qali.core.user_tables`).
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from qali.core import species_data, user_tables
from qali.core.paths import OVERRIDABLE_TREES, resource_root

from .dataset_editor import DatasetEditorDialog
from .find_bar import with_find_bar
from .theme import mono_font

_FILE_SUFFIXES = {".csv", ".json", ".txt"}
_SKIP = {("extrapolation", "model_params.json")}          # has its own page (Extrapolator)
_MARK = QColor(255, 213, 79, 70)


def _fmt(v: float) -> str:
    return f"{v:.10g}"


def data_files() -> dict[str, list[tuple[str, ...]]]:
    """Packaged files the user may edit, by tree: ``{"f_values": [("f_values", "Cs_lines.csv"), ...]}``."""
    out: dict[str, list[tuple[str, ...]]] = {}
    for tree in OVERRIDABLE_TREES:
        base = resource_root() / tree
        files = []
        for p in sorted(base.rglob("*")) if base.is_dir() else []:
            rel = p.relative_to(resource_root()).parts
            if p.is_file() and p.suffix.lower() in _FILE_SUFFIXES and rel[1:2] != ("tables",) and rel not in _SKIP:
                files.append(rel)
        if files:
            out[tree] = files
    return out


class AddRowDialog(QDialog):
    """Name and value of one new row; refuses a row the table already has."""

    def __init__(self, table_id: str, existing: dict[str, float], parent=None, element: str | None = None):
        super().__init__(parent)
        self.table_id, self.existing = table_id, existing
        info = user_tables.BY_ID[table_id]
        self.setWindowTitle(f"Add to {info.title}")
        form = QFormLayout(self)
        self.boxes = [QLineEdit() for _ in user_tables.key_fields(table_id)]
        for label, box in zip(user_tables.key_fields(table_id), self.boxes):
            form.addRow(label, box)
        if element:
            self.boxes[0].setText(element)
            self.boxes[0].setReadOnly(True)
        self.value = QLineEdit()
        self.fields: dict[str, QLineEdit] = {}
        if user_tables.ADDABLE[table_id].kind == "record":
            holder = QWidget()
            inner = QFormLayout(holder)
            for name, kind in user_tables.record_fields(table_id):
                box = QLineEdit("user" if name == "source" else "")
                unit = user_tables.unit(table_id, name)
                inner.addRow(name + (f" ({unit})" if unit else ""), box)
                self.fields[name] = box
            area = QScrollArea()
            area.setWidgetResizable(True)
            area.setWidget(holder)
            area.setMinimumSize(420, 320)
            form.addRow(area)
        else:
            self.value.setPlaceholderText(info.unit or "number")
            form.addRow("Value" + (f" ({info.unit})" if info.unit else ""), self.value)
        self.source = QLineEdit()
        self.source.setPlaceholderText("optional, e.g. paper or measurement")
        form.addRow("Source", self.source)
        self.use = QCheckBox("Use instead of the computed value")
        if table_id in user_tables.FLAGGABLE:
            form.addRow(self.use)
        self.error = QLabel("")
        self.error.setWordWrap(True)
        self.error.setStyleSheet("color: #d9534f;")
        form.addRow(self.error)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)
        self.result_row: tuple[str, float | dict, str] | None = None

    def accept(self) -> None:
        try:
            key, _path = user_tables.make_key(self.table_id, [b.text() for b in self.boxes])
            if key.lower() in {k.lower() for k in self.existing}:
                have = self.existing[next(k for k in self.existing if k.lower() == key.lower())]
                raise ValueError(f"{key} is already in this table" + (f" ({_fmt(have)})" if have is not None else "") + ". Edit its rows instead.")
            if self.fields:
                value = user_tables.check_record(self.table_id, key, {n: b.text() for n, b in self.fields.items()})
            else:
                value = user_tables.check_value(self.table_id, self.value.text())
        except ValueError as exc:
            self.error.setText(str(exc))
            return
        self.result_row = (key, value, self.source.text().strip())
        super().accept()


class TablesPage(QWidget):
    def __init__(self, root: Path, parent=None, element: str | None = None, species: str | None = None):
        super().__init__(parent)
        self.root = Path(root)
        self.species_id = species or element        # the id the data files are named after (Na_c1.csv)
        self.element = element                      # None: every table; a symbol: only that element's rows and files
        self._overrides: dict[str, dict] = {t: dict(r) for t, r in user_tables.load_overrides().items()}
        self._additions: dict[str, dict] = {t: {k: dict(r) for k, r in rows.items()} for t, rows in user_tables.load_additions().items()}
        self._flags: dict[str, dict[str, bool]] = {t: dict(r) for t, r in user_tables.load_flags().items()}
        self._table_id: str | None = None
        self._file: tuple[str, ...] | None = None
        self._building = False

        outer = QHBoxLayout(self)
        split = QSplitter(Qt.Horizontal)
        outer.addWidget(split)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter tables and files")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter_tree)
        lv.addWidget(self.search)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.currentItemChanged.connect(self._picked)
        lv.addWidget(self.tree, 1)
        split.addWidget(left)

        self.stack = QStackedWidget()
        self.intro = QLabel("Pick a table or a file.")
        self.intro.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.intro.setTextFormat(Qt.RichText)
        self.intro.setWordWrap(True)
        self.stack.addWidget(self.intro)                                              # 0
        split.addWidget(self.stack)
        split.setSizes([280, 760])
        self._build_table_view()                                                       # 1
        self._build_file_view()                                                        # 2
        self._fill_tree()
        if self.element:
            self.intro.setText(self.summary())

    # --- tree --------------------------------------------------------------------------
    def _fill_tree(self) -> None:
        self.tree.clear()
        groups: dict[str, QTreeWidgetItem] = {}
        for info in user_tables.TABLES:
            have = bool(self.element) and bool(user_tables.element_rows(info.table_id, self.element))
            if self.element and not have and info.table_id not in user_tables.ADDABLE:
                continue
            g = groups.get(info.group) or QTreeWidgetItem(self.tree, [info.group])
            groups[info.group] = g
            tag = "  ·  edited" if self._is_edited(info.table_id) else "  ·  missing" if self.element and not have else ""
            item = QTreeWidgetItem(g, [info.title + tag])
            item.setData(0, Qt.UserRole, ("table", info.table_id))
        files = QTreeWidgetItem(self.tree, ["Data files"])
        pattern = user_tables.element_file_pattern(self.element) if self.element else None
        for tree, paths in data_files().items():
            paths = [q for q in paths if pattern is None or pattern.match(q[-1])]
            if not paths:
                continue
            node = QTreeWidgetItem(files, [tree])
            for parts in paths:
                edited = user_tables.file_override(parts) is not None
                leaf = QTreeWidgetItem(node, ["/".join(parts[1:]) + ("  ·  edited" if edited else "")])
                leaf.setData(0, Qt.UserRole, ("file", parts))
        if self.element and not files.childCount():
            self.tree.takeTopLevelItem(self.tree.indexOfTopLevelItem(files))
        for i in range(self.tree.topLevelItemCount()):
            self.tree.topLevelItem(i).setExpanded(self.tree.topLevelItem(i).text(0) != "Data files" or bool(self.element))
        self._filter_tree()

    def summary(self) -> str:
        """What this element has and lacks, for the page shown before anything is picked."""
        el = self.element
        lines = [f"<b>{el}</b>", "", "<b>Values</b>"]
        for info in user_tables.TABLES:
            n = len([r for r in user_tables.packaged_rows(info.table_id) if user_tables.element_of(info.table_id, r[0]) == el])
            for akey, rec in self._additions.get(info.table_id, {}).items():
                if user_tables.element_of(info.table_id, akey) == el:
                    n += len(user_tables.flatten({akey: rec["record"]})) if "record" in rec else 1
            if n:
                lines.append(f"&#10003; {info.title}: {n} value{'s' if n != 1 else ''}")
            elif info.table_id in user_tables.ADDABLE:
                lines.append(f"&#10007; {info.title}: missing, use Add...")
        lines += ["", "<b>Data files</b>"]
        for item in (i for i in species_data.ITEMS if i.kind == "file"):
            if species_data.has_file(item, self.species_id):
                lines.append(f"&#10003; {item.label}")
            else:
                lines.append(f"&#10007; {item.label}: missing (use Set up missing data...)")
        return "<br>".join(lines)

    def _filter_tree(self, *_) -> None:
        needle = self.search.text().strip().lower()

        def show(item: QTreeWidgetItem) -> bool:
            kids = [show(item.child(i)) for i in range(item.childCount())]
            hit = (not needle) or needle in item.text(0).lower() or any(kids)
            item.setHidden(not hit)
            if needle and any(kids):
                item.setExpanded(True)
            return hit

        for i in range(self.tree.topLevelItemCount()):
            show(self.tree.topLevelItem(i))

    def _picked(self, item, _prev=None) -> None:
        data = item.data(0, Qt.UserRole) if item is not None else None
        if not data:
            self.stack.setCurrentIndex(0)
        elif data[0] == "table":
            self._show_table(data[1])
        else:
            self._show_file(tuple(data[1]))

    # --- constant tables -----------------------------------------------------------------
    def _build_table_view(self) -> None:
        page = QWidget()
        v = QVBoxLayout(page)
        head = QHBoxLayout()
        self.table_title = QLabel("")
        f = self.table_title.font()
        f.setBold(True)
        self.table_title.setFont(f)
        head.addWidget(self.table_title)
        head.addStretch(1)
        self.row_filter = QLineEdit()
        self.row_filter.setPlaceholderText("Filter rows")
        self.row_filter.setClearButtonEnabled(True)
        self.row_filter.textChanged.connect(self._filter_rows)
        head.addWidget(self.row_filter)
        self.ovr_all = QPushButton("Override all")
        self.ovr_all.setToolTip("Tick Override on the rows shown")
        self.ovr_all.clicked.connect(lambda: self._tick_all(True))
        self.ovr_none = QPushButton("Override none")
        self.ovr_none.setToolTip("Untick Override on the rows shown")
        self.ovr_none.clicked.connect(lambda: self._tick_all(False))
        head.addWidget(self.ovr_all)
        head.addWidget(self.ovr_none)
        self.add_btn = QPushButton("Add...")
        self.add_btn.clicked.connect(self._add_row)
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.setToolTip("Remove the selected row you added")
        self.remove_btn.clicked.connect(self._remove_row)
        head.addWidget(self.add_btn)
        head.addWidget(self.remove_btn)
        reset = QPushButton("Reset table")
        reset.clicked.connect(self.reset_table)
        head.addWidget(reset)
        v.addLayout(head)
        self.grid = QTableWidget(0, 5)
        self.grid.setHorizontalHeaderLabels(["Name", "Packaged", "Value", "Unit", "Override"])
        self.grid.horizontalHeaderItem(4).setToolTip("Use this value instead of the computed one")
        self.grid.setColumnHidden(4, True)
        self.grid.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.grid.verticalHeader().setVisible(False)
        self.grid.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.grid.itemChanged.connect(self._cell_edited)
        self.grid.itemSelectionChanged.connect(self._update_buttons)
        v.addWidget(with_find_bar(self.grid), 1)
        self.stack.addWidget(page)

    def _show_table(self, table_id: str) -> None:
        self._table_id, self._file = table_id, None
        info = user_tables.BY_ID[table_id]
        self.table_title.setText(info.title)
        self.row_filter.clear()
        self._building = True
        rows = user_tables.packaged_rows(table_id)
        if self.element:
            rows = [r for r in rows if user_tables.element_of(table_id, r[0]) == self.element]
        mine = self._overrides.get(table_id, {})
        added = self._additions.get(table_id, {})
        added_rows: list[tuple[str, float, str]] = []              # (flat key, value, source) of what the user added
        for akey, rec in added.items():
            if self.element and user_tables.element_of(table_id, akey) != self.element:
                continue
            if "record" in rec:
                added_rows += [(k, v, rec.get("source", "")) for k, (_p, v) in user_tables.flatten({akey: rec["record"]}).items()]
            else:
                added_rows.append((akey, rec["value"], rec.get("source", "")))
        self.grid.setRowCount(len(rows) + len(added_rows))
        for r, (key, packaged, _now) in enumerate(rows):
            name, pk = QTableWidgetItem(key), QTableWidgetItem(_fmt(packaged))
            for it in (name, pk):
                it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self._put_row(r, key, name, pk, _fmt(mine.get(key, packaged)), key in mine)
        for r, (key, value, source) in enumerate(added_rows, start=len(rows)):
            name, pk = QTableWidgetItem(key), QTableWidgetItem("added")
            for it in (name, pk):
                it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            if source:
                name.setToolTip(source)
            self._put_row(r, key, name, pk, _fmt(value), True)
        self.grid.resizeColumnToContents(1)
        self.grid.resizeColumnToContents(2)
        self._building = False
        flaggable = table_id in user_tables.FLAGGABLE
        self.grid.setColumnHidden(4, not flaggable)
        self.ovr_all.setVisible(flaggable)
        self.ovr_none.setVisible(flaggable)
        self.add_btn.setVisible(table_id in user_tables.ADDABLE)
        self.remove_btn.setVisible(table_id in user_tables.ADDABLE)
        self._update_buttons()
        self.stack.setCurrentIndex(1)

    def _put_row(self, r: int, key: str, name, pk, value_text: str, marked: bool) -> None:
        self.grid.setItem(r, 0, name)
        self.grid.setItem(r, 1, pk)
        self.grid.setItem(r, 2, QTableWidgetItem(value_text))
        unit = QTableWidgetItem(user_tables.unit(self._table_id, key))
        unit.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        self.grid.setItem(r, 3, unit)
        if self._table_id in user_tables.FLAGGABLE:
            box = QCheckBox()                          # a real check box, centred, drawn by the app's own style
            box.setChecked(self._flags.get(self._table_id, {}).get(key, False))
            box.toggled.connect(lambda on, k=key: self._ticked(k, on))
            holder = QWidget()
            lay = QHBoxLayout(holder)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setAlignment(Qt.AlignCenter)
            lay.addWidget(box)
            self.grid.setCellWidget(r, 4, holder)
        self._style(r, marked)

    def _is_edited(self, table_id: str) -> bool:
        default = {k: True for k in user_tables.DEFAULT_FLAGS.get(table_id, ())}
        return bool(self._overrides.get(table_id) or self._additions.get(table_id)
                    or self._flags.get(table_id, {}) != default)

    def _set_flag(self, key: str, on: bool) -> None:
        rows = self._flags.setdefault(self._table_id, {})
        if on:
            rows[key] = True
        else:
            rows.pop(key, None)
        if not rows:
            self._flags.pop(self._table_id, None)

    def override_box(self, row: int) -> QCheckBox | None:
        holder = self.grid.cellWidget(row, 4)
        return holder.findChild(QCheckBox) if holder is not None else None

    def _ticked(self, key: str, on: bool) -> None:
        if not self._building and self._table_id is not None:
            self._set_flag(key, on)
            self._mark_tree_item()

    def _tick_all(self, on: bool) -> None:
        for r in range(self.grid.rowCount()):
            box = self.override_box(r)
            if box is not None and not self.grid.isRowHidden(r):
                box.setChecked(on)

    def _owner(self, table_id: str | None, key: str | None) -> str | None:
        """The added entry a grid row belongs to (a whole element for a record table), or None."""
        if not table_id or not key or table_id not in user_tables.ADDABLE:
            return None
        added = self._additions.get(table_id, {})
        name = key.split(" / ")[0] if user_tables.ADDABLE[table_id].kind == "record" else key
        return name if name in added else None

    def _update_buttons(self) -> None:
        row = self.grid.currentRow()
        key = self.grid.item(row, 0).text() if row >= 0 and self.grid.item(row, 0) else None
        self.remove_btn.setEnabled(self._owner(self._table_id, key) is not None)

    def _add_row(self) -> None:
        tid = self._table_id
        if tid not in user_tables.ADDABLE:
            return
        if user_tables.ADDABLE[tid].kind == "record":
            existing = {el: None for el in user_tables.defaults_top(tid)}
            existing.update({el: None for el in self._additions.get(tid, {})})
        else:
            existing = {k: now for k, _pk, now in user_tables.packaged_rows(tid)}
            existing.update({k: r["value"] for k, r in self._additions.get(tid, {}).items()})
        dlg = AddRowDialog(tid, existing, self, element=self.element)
        if dlg.exec() and dlg.result_row:
            key, value, source = dlg.result_row
            self._additions.setdefault(tid, {})[key] = {("record" if isinstance(value, dict) else "value"): value, "source": source}
            if dlg.use.isChecked() and tid in user_tables.FLAGGABLE:
                self._table_id = tid
                self._set_flag(key, True)
            self._show_table(tid)
            self._mark_tree_item()
            self.grid.selectRow(self.grid.rowCount() - 1)

    def _remove_row(self) -> None:
        row, tid = self.grid.currentRow(), self._table_id
        key = self._owner(tid, self.grid.item(row, 0).text() if row >= 0 else None)
        if tid and key is not None:
            del self._additions[tid][key]
            if tid in user_tables.FLAGGABLE:
                self._set_flag(self.grid.item(row, 0).text(), False)
            if not self._additions[tid]:
                del self._additions[tid]
            self._show_table(tid)
            self._mark_tree_item()

    def _style(self, row: int, edited: bool) -> None:
        item = self.grid.item(row, 2)
        font = QFont(item.font())
        font.setBold(edited)
        item.setFont(font)
        item.setBackground(_MARK if edited else QColor(0, 0, 0, 0))

    def _cell_edited(self, item) -> None:
        if self._building or self._table_id is None:
            return
        if item.column() != 2:
            return
        key = self.grid.item(item.row(), 0).text()
        owner = self._owner(self._table_id, key)
        if owner is not None:                                         # a row the user added
            rec = self._additions[self._table_id][owner]
            node, *path = (rec["record"], *key.split(" / ")[1:]) if "record" in rec else (rec, "value")
            for k in path[:-1]:
                node = node[k]
            try:
                if "record" in rec:
                    new = user_tables.check_record_value(node[path[-1]], item.text())
                else:
                    new = user_tables.check_value(self._table_id, item.text())
                node[path[-1]] = new
            except ValueError:
                pass
            self._building = True
            item.setText(_fmt(node[path[-1]]))
            self._building = False
            return
        packaged = float(self.grid.item(item.row(), 1).text())
        try:
            value = float(item.text())
            if not math.isfinite(value):
                raise ValueError
        except ValueError:
            self._building = True
            item.setText(_fmt(self._overrides.get(self._table_id, {}).get(key, packaged)))
            self._building = False
            return
        rows = self._overrides.setdefault(self._table_id, {})
        if value == packaged:
            rows.pop(key, None)
        else:
            rows[key] = value
        if not rows:
            self._overrides.pop(self._table_id, None)
        self._style(item.row(), value != packaged)
        self._mark_tree_item()

    def _mark_tree_item(self) -> None:
        item = self.tree.currentItem()
        if item is not None and self._table_id:
            item.setText(0, user_tables.BY_ID[self._table_id].title + ("  ·  edited" if self._is_edited(self._table_id) else ""))
            if self.element:
                self.intro.setText(self.summary())

    def _filter_rows(self, text: str) -> None:
        needle = text.strip().lower()
        for r in range(self.grid.rowCount()):
            self.grid.setRowHidden(r, bool(needle) and needle not in self.grid.item(r, 0).text().lower())

    def reset_table(self) -> None:
        """Drop the edits and added rows of the table on screen (of this element only, in a Properties dialog)."""
        tid = self._table_id
        if not tid:
            return
        flags_before = dict(self._flags.get(tid, {}))
        if self.element:
            def mine(key: str) -> bool:
                return user_tables.element_of(tid, key) == self.element

            changed = False
            for store in (self._overrides, self._additions, self._flags):
                rows = store.get(tid, {})
                for k in [k for k in rows if mine(k)]:
                    del rows[k]
                    changed = True
                if tid in store and not store[tid]:
                    del store[tid]
        else:
            changed = (self._overrides.pop(tid, None) is not None) | (self._additions.pop(tid, None) is not None) \
                | (self._flags.pop(tid, None) is not None)
        for k in user_tables.DEFAULT_FLAGS.get(tid, ()):          # the rows that start ticked are ticked again
            if not self.element or user_tables.element_of(tid, k) == self.element:
                self._flags.setdefault(tid, {})[k] = True
        changed = changed or self._flags.get(tid, {}) != flags_before
        if changed:
            self._show_table(tid)
            self._mark_tree_item()

    # --- data files ------------------------------------------------------------------------
    def _build_file_view(self) -> None:
        page = QWidget()
        v = QVBoxLayout(page)
        head = QHBoxLayout()
        self.file_title = QLabel("")
        f = self.file_title.font()
        f.setBold(True)
        self.file_title.setFont(f)
        head.addWidget(self.file_title)
        head.addStretch(1)
        self.edit_btn = QPushButton("Edit...")
        self.edit_btn.clicked.connect(self._edit_file)
        self.revert_btn = QPushButton("Restore original")
        self.revert_btn.clicked.connect(self._revert_file)
        head.addWidget(self.edit_btn)
        head.addWidget(self.revert_btn)
        v.addLayout(head)
        self.viewer = QPlainTextEdit()
        self.viewer.setReadOnly(True)
        self.viewer.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.viewer.setFont(mono_font())
        v.addWidget(with_find_bar(self.viewer), 1)
        self.stack.addWidget(page)

    def _effective(self, parts: tuple[str, ...]) -> Path:
        return user_tables.file_override(parts) or resource_root().joinpath(*parts)

    def _show_file(self, parts: tuple[str, ...]) -> None:
        self._file, self._table_id = parts, None
        path = self._effective(parts)
        edited = user_tables.file_override(parts) is not None
        self.file_title.setText("/".join(parts[1:]) + ("  (edited copy)" if edited else ""))
        try:
            self.viewer.setPlainText(path.read_text(encoding="utf-8", errors="replace"))
        except OSError as exc:
            self.viewer.setPlainText(f"Could not read {path.name}: {exc}")
        self.edit_btn.setEnabled(path.suffix.lower() in (".json", ".csv"))
        self.edit_btn.setToolTip("" if self.edit_btn.isEnabled() else "Only JSON and CSV files can be edited.")
        self.revert_btn.setEnabled(edited)
        self.stack.setCurrentIndex(2)

    def _edit_file(self) -> None:
        parts = self._file
        if parts is None:
            return
        src, target = self._effective(parts), user_tables.override_file_path(parts)
        kind = "json" if src.suffix.lower() == ".json" else "csv"
        try:
            if kind == "json":
                text = src.read_text(encoding="utf-8")
            else:
                with src.open("r", encoding="utf-8", newline="") as fh:
                    text = list(csv.reader(fh))
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QMessageBox.warning(self, "Edit", f"Could not open {src.name}:\n\n{exc}")
            return
        DatasetEditorDialog(self, dataset_name=src.stem, dataset_type=kind, initial_text=text,
                            default_path=target.parent, file_path=target, root=self.root).exec()
        self._after_file_change(parts)

    def _revert_file(self) -> None:
        parts = self._file
        if parts is None or user_tables.file_override(parts) is None:
            return
        if QMessageBox.question(self, "Restore original", f"Discard your edited copy of {parts[-1]}?") == QMessageBox.Yes:
            user_tables.revert_file(parts)
            self._after_file_change(parts)

    def _after_file_change(self, parts: tuple[str, ...]) -> None:
        clear_caches()
        item = self.tree.currentItem()
        if item is not None:
            item.setText(0, "/".join(parts[1:]) + ("  ·  edited" if user_tables.file_override(parts) else ""))
        self._show_file(parts)

    # --- Preferences hooks -----------------------------------------------------------------
    def apply(self) -> list[str]:
        """Save the constant-table edits and put them in force in this process; returns what could not be saved."""
        before = set(user_tables.load_overrides()) | set(user_tables.load_additions())
        try:
            user_tables.save_overrides(self._overrides)
            user_tables.save_additions(self._additions)
            user_tables.save_flags(self._flags)
        except (OSError, ValueError) as exc:
            return [f"Tables and variables: {exc}"]
        for table_id in before | set(self._overrides) | set(self._additions):
            try:
                user_tables.reload(table_id)
            except Exception:
                pass
        return []

    def reset_all(self) -> None:
        """Restore Defaults: the table on screen, or the edited copy of the file on screen."""
        if self._file is not None:
            self._revert_file()
        else:
            self.reset_table()


def clear_caches() -> None:
    """Forget cached reads of the packaged files so an edited copy is picked up at once."""
    try:
        from qali.physics.extrapolation import params
        params._anchor_config_base.cache_clear()
    except Exception:
        pass
