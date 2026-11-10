"""Ctrl+F find bar for the JSON viewer and the dataset editor (text and table views)."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QKeySequence, QShortcut, QTextCharFormat, QTextCursor, QTextDocument
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLineEdit,
    QTableWidget,
    QTextEdit,
    QToolButton,
    QLabel,
    QVBoxLayout,
    QWidget,
)

MAX_HIGHLIGHTS = 2000          # a search for "6p" in a 10 MB file must not stall the window


class FindBar(QWidget):
    """Find box with next/previous, a case toggle and an "n of N" count; works on a text edit or a table."""

    def __init__(self, target, parent=None):
        super().__init__(parent)
        self.target = target
        self._is_table = isinstance(target, QTableWidget)
        self._hits: list = []
        self._at = -1
        self._saved: list = []      # (table item, its background before it was marked)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 0)
        self.box = QLineEdit()
        self.box.setPlaceholderText("Find")
        self.box.setClearButtonEnabled(True)
        lay.addWidget(self.box, 1)
        self.case = QToolButton()
        self.case.setText("Aa")
        self.case.setCheckable(True)
        self.case.setToolTip("Match case")
        self.rows_only = QToolButton()
        self.rows_only.setText("Rows")
        self.rows_only.setCheckable(True)
        self.rows_only.setToolTip("Show only the rows that contain a match")
        self.rows_only.setVisible(self._is_table)
        prev, nxt, close = QToolButton(), QToolButton(), QToolButton()
        prev.setText("▲")
        prev.setToolTip("Previous (Shift+F3)")
        nxt.setText("▼")
        nxt.setToolTip("Next (F3)")
        close.setText("✕")
        close.setToolTip("Close (Esc)")
        self.count = QLabel("")
        self.count.setObjectName("muted")
        for w in (self.case, self.rows_only, prev, nxt, self.count, close):
            lay.addWidget(w)
        self.box.textChanged.connect(self._search)
        self.box.returnPressed.connect(self.next)
        self.case.toggled.connect(self._search)
        self.rows_only.toggled.connect(self._search)
        nxt.clicked.connect(self.next)
        prev.clicked.connect(self.previous)
        close.clicked.connect(self.hide_bar)
        QShortcut(QKeySequence("Esc"), self.box, activated=self.hide_bar, context=Qt.WidgetShortcut)
        self.hide()

    # --- showing ----------------------------------------------------------
    def show_bar(self) -> None:
        self.show()
        if not self._is_table:
            sel = self.target.textCursor().selectedText()
            if sel and " " not in sel:
                self.box.setText(sel)
        self.box.setFocus()
        self.box.selectAll()
        self._search()

    def hide_bar(self) -> None:
        self.hide()
        self._clear_marks()
        self.target.setFocus()

    # --- searching --------------------------------------------------------
    def _doc_flags(self):
        return QTextDocument.FindCaseSensitively if self.case.isChecked() else QTextDocument.FindFlag(0)

    def _search(self, *_):
        text = self.box.text()
        self._clear_marks()
        self._hits, self._at = [], -1
        if not text:
            self.count.setText("")
            return
        if self._is_table:
            flags = Qt.MatchContains | (Qt.MatchCaseSensitive if self.case.isChecked() else Qt.MatchFlag(0))
            self._hits = sorted(self.target.findItems(text, flags), key=lambda i: (i.row(), i.column()))
            self._mark_table()
        else:
            doc, cur = self.target.document(), QTextCursor(self.target.document())
            while len(self._hits) < MAX_HIGHLIGHTS:
                cur = doc.find(text, cur, self._doc_flags())
                if cur.isNull():
                    break
                self._hits.append(QTextCursor(cur))
            self._mark()
        self.count.setText("" if self._hits else "No matches")
        if self._hits:
            self._goto(self._first_from_cursor())

    def _first_from_cursor(self) -> int:
        if self._is_table:
            cur = self.target.currentItem()
            pos = (cur.row(), cur.column()) if cur else (-1, -1)
            return next((i for i, h in enumerate(self._hits) if (h.row(), h.column()) >= pos), 0)
        here = self.target.textCursor().selectionStart()
        return next((i for i, h in enumerate(self._hits) if h.selectionStart() >= here), 0)

    def _goto(self, i: int) -> None:
        self._at = i % len(self._hits)
        hit = self._hits[self._at]
        if self._is_table:
            self.target.setCurrentItem(hit)
            self.target.scrollToItem(hit)
        else:
            self.target.setTextCursor(hit)
            self.target.ensureCursorVisible()
        n = len(self._hits)
        self.count.setText(f"{self._at + 1} of {n}{'+' if n >= MAX_HIGHLIGHTS else ''}")

    def next(self) -> None:
        if self._hits:
            self._goto(self._at + 1)

    def previous(self) -> None:
        if self._hits:
            self._goto(self._at - 1)

    # --- highlighting (text only) -------------------------------------------
    def _mark(self) -> None:
        fmt = QTextCharFormat()
        fmt.setBackground(QColor(255, 213, 79, 140))
        fmt.setForeground(QColor(0, 0, 0))
        marks = []
        for h in self._hits:
            sel = QTextEdit.ExtraSelection()
            sel.cursor, sel.format = h, fmt
            marks.append(sel)
        self.target.setExtraSelections(marks)

    def _mark_table(self) -> None:
        """Every matching cell gets a yellow background; "Rows" hides the rows without a match."""
        for item in self._hits:
            self._saved.append((item, item.background()))
            item.setBackground(QBrush(QColor(255, 213, 79, 140)))
        if self.rows_only.isChecked():
            keep = {item.row() for item in self._hits}
            for r in range(self.target.rowCount()):
                self.target.setRowHidden(r, r not in keep)

    def _clear_marks(self) -> None:
        if not self._is_table:
            self.target.setExtraSelections([])
            return
        for item, brush in self._saved:
            try:
                item.setBackground(brush)
            except RuntimeError:        # the item was deleted meanwhile
                pass
        self._saved = []
        for r in range(self.target.rowCount()):
            self.target.setRowHidden(r, False)


def with_find_bar(target) -> QWidget:
    """*target* (a QPlainTextEdit or QTableWidget) in a container with a find bar; Ctrl+F opens it, F3 / Shift+F3 step."""
    box = QWidget()
    lay = QVBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(0)
    lay.addWidget(target, 1)
    bar = FindBar(target)
    lay.addWidget(bar)
    box.find_bar = bar
    for keys, fn in (("Ctrl+F", bar.show_bar), ("F3", bar.next), ("Shift+F3", bar.previous)):
        QShortcut(QKeySequence(keys), box, activated=fn, context=Qt.WidgetWithChildrenShortcut)
    return box
