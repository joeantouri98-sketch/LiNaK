"""Qt editor for user-maintained JSON and CSV datasets."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from .dataset_templates import CSV_DATASET_TEMPLATES


class DatasetEditorDialog(QDialog):
    """Edit a JSON document or a rectangular CSV dataset."""

    def __init__(
        self,
        parent=None,
        dataset_name="new",
        dataset_type="json",
        initial_text=None,
        default_path=None,
        file_path=None,
    ):
        super().__init__(parent)
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.default_path = Path(default_path or ".")
        self.file_path = Path(file_path) if file_path else None

        display_name = dataset_name.replace("_", " ").title()
        self.setWindowTitle(f"Edit {display_name} ({dataset_type.upper()})")
        self.resize(950, 650)

        layout = QVBoxLayout(self)
        description = QLabel(
            "Edit the dataset below. Use Save As... to choose a target file."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        if dataset_type == "json":
            self.editor = QPlainTextEdit()
            self.editor.setLineWrapMode(QPlainTextEdit.NoWrap)
            self.editor.setPlainText(initial_text if initial_text is not None else "{}")
            layout.addWidget(self.editor, stretch=1)
        else:
            self.table = QTableWidget()
            self.table.setAlternatingRowColors(True)
            self.table.setSortingEnabled(False)
            self.table.setEditTriggers(
                QTableWidget.DoubleClicked
                | QTableWidget.EditKeyPressed
                | QTableWidget.AnyKeyPressed
            )
            self.table.horizontalHeader().setSectionResizeMode(
                QHeaderView.Interactive
            )
            self._populate_csv_table(
                initial_text if initial_text is not None else CSV_DATASET_TEMPLATES["new"]
            )
            layout.addWidget(self.table, stretch=1)
            self._build_table_buttons(layout)

        self._build_action_buttons(layout)

    def _build_table_buttons(self, layout):
        table_buttons = QHBoxLayout()
        for label, handler in (
            ("Add row", self._add_row),
            ("Remove selected row", self._remove_selected_row),
            ("Add column", self._add_column),
            ("Remove selected column", self._remove_selected_column),
        ):
            button = QPushButton(label)
            button.clicked.connect(handler)
            table_buttons.addWidget(button)
        table_buttons.addStretch(1)
        layout.addLayout(table_buttons)

    def _build_action_buttons(self, layout):
        bottom_buttons = QHBoxLayout()
        for label, handler in (
            ("Save", self._save),
            ("Save As...", self._save_as),
            ("Open existing...", self._open_existing_file),
        ):
            button = QPushButton(label)
            button.clicked.connect(handler)
            bottom_buttons.addWidget(button)
        bottom_buttons.addStretch(1)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.reject)
        bottom_buttons.addWidget(close_button)
        layout.addLayout(bottom_buttons)

    def _populate_csv_table(self, rows):
        normalized = []
        for row in rows or [["column_1"], [""]]:
            normalized.append(
                [str(value) for value in row]
                if isinstance(row, (list, tuple))
                else [str(row)]
            )
        column_count = max(len(row) for row in normalized)
        self.table.setColumnCount(column_count)
        self.table.setRowCount(len(normalized))
        for row_index, row in enumerate(normalized):
            for column_index in range(column_count):
                value = row[column_index] if column_index < len(row) else ""
                self.table.setItem(row_index, column_index, QTableWidgetItem(value))
        self.table.resizeColumnsToContents()

    def _add_row(self):
        row_index = self.table.rowCount()
        self.table.insertRow(row_index)
        for column_index in range(self.table.columnCount()):
            self.table.setItem(row_index, column_index, QTableWidgetItem(""))

    def _remove_selected_row(self):
        selected_rows = sorted(
            {index.row() for index in self.table.selectionModel().selectedRows()},
            reverse=True,
        )
        for row_index in selected_rows:
            self.table.removeRow(row_index)

    def _add_column(self):
        column_index = self.table.columnCount()
        self.table.insertColumn(column_index)
        for row_index in range(self.table.rowCount()):
            self.table.setItem(row_index, column_index, QTableWidgetItem(""))

    def _remove_selected_column(self):
        selected_columns = sorted(
            {
                index.column()
                for index in self.table.selectionModel().selectedColumns()
            },
            reverse=True,
        )
        for column_index in selected_columns:
            self.table.removeColumn(column_index)

    def _csv_rows(self):
        rows = []
        for row_index in range(self.table.rowCount()):
            rows.append([
                "" if (item := self.table.item(row_index, column_index)) is None else item.text()
                for column_index in range(self.table.columnCount())
            ])
        while rows and all(value.strip() == "" for value in rows[-1]):
            rows.pop()
        return rows

    @staticmethod
    def _validate_csv(rows):
        if not rows:
            raise ValueError("CSV dataset is empty.")
        header = rows[0]
        if not any(value.strip() for value in header):
            raise ValueError("CSV header cannot be empty.")
        if len(set(header)) != len(header):
            raise ValueError("CSV column names must be unique.")
        width = len(header)
        for row_number, row in enumerate(rows[1:], start=2):
            if len(row) != width:
                raise ValueError(
                    f"Row {row_number} has {len(row)} columns, but the header has {width}."
                )

    def _open_existing_file(self):
        file_filter = (
            "JSON files (*.json);;All files (*)"
            if self.dataset_type == "json"
            else "CSV files (*.csv);;All files (*)"
        )
        path, _selected_filter = QFileDialog.getOpenFileName(
            self, "Open dataset", str(self.default_path), file_filter
        )
        if not path:
            return
        opened_path = Path(path)
        try:
            if self.dataset_type == "json":
                self.editor.setPlainText(opened_path.read_text(encoding="utf-8"))
            else:
                with opened_path.open("r", encoding="utf-8", newline="") as handle:
                    rows = list(csv.reader(handle))
                self._populate_csv_table(rows or [["column_1"], [""]])
            self.file_path = opened_path
            self.dataset_name = opened_path.stem
            self.setWindowTitle(f"Edit {opened_path.name} ({self.dataset_type.upper()})")
        except Exception as exc:
            QMessageBox.warning(self, "Open failed", f"Could not open dataset:\n\n{exc}")

    def _save(self):
        if self.file_path is None:
            self._save_as()
        else:
            self._write_current()

    def _save_as(self):
        self.default_path.mkdir(parents=True, exist_ok=True)
        base_name = "new_dataset" if self.dataset_name == "new" else self.dataset_name
        suffix = "json" if self.dataset_type == "json" else "csv"
        title = f"Save {suffix.upper()} dataset"
        suggested = self.default_path / f"{base_name}.{suffix}"
        path, _selected_filter = QFileDialog.getSaveFileName(
            self, title, str(suggested), f"{suffix.upper()} files (*.{suffix});;All files (*)"
        )
        if path:
            self.file_path = Path(path)
            self._write_current()

    def _write_current(self):
        if self.file_path is None:
            return
        try:
            if self.dataset_type == "json":
                content = self.editor.toPlainText().strip() or "{}"
                payload = json.loads(content)
                if not isinstance(payload, (dict, list)):
                    raise ValueError("Top-level JSON must be an object or an array.")
                self.file_path.write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
            else:
                rows = self._csv_rows()
                self._validate_csv(rows)
                output = io.StringIO(newline="")
                csv.writer(output, lineterminator="\n").writerows(rows)
                self.file_path.write_text(output.getvalue(), encoding="utf-8")
        except Exception as exc:
            QMessageBox.warning(self, "Save failed", f"Could not save dataset:\n\n{exc}")
            return
        QMessageBox.information(self, "Dataset saved", f"Saved to:\n\n{self.file_path}")
        self.accept()
