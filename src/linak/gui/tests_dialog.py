"""Run > Tests: run single tests, ticked tests or all of them, with live progress."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from . import test_runner as R
from .find_bar import with_find_bar
from .theme import mono_font

GREEN, RED, GREY, AMBER = QColor(46, 160, 67), QColor(218, 54, 51), QColor(140, 140, 140), QColor(210, 150, 30)
LABEL = {"PASSED": ("✓ passed", GREEN), "FAILED": ("✗ failed", RED), "ERROR": ("! error", RED),
         "SKIPPED": ("– skipped", GREY), "XFAIL": ("– xfail", GREY), "XPASS": ("! xpass", AMBER)}
ID_ROLE = Qt.UserRole


class TestsDialog(QDialog):
    __test__ = False                                         # not a pytest class

    def __init__(self, parent=None, tests_dir: Path | None = None, python: str | None = None, collect: bool = True):
        super().__init__(parent)
        self.setWindowTitle("Tests")
        self.resize(980, 720)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.python = python or sys.executable
        self._avail = R.availability(tests_dir)
        self.tests_dir = self._avail.tests_dir
        self._ids: list[str] = []
        self._items: dict[str, QTreeWidgetItem] = {}
        self._files: dict[str, QTreeWidgetItem] = {}
        self._status: dict[str, str] = {}
        self._run_ids: list[str] = []
        self._buf, self._log = "", []
        self._proc: QProcess | None = None
        self._collecting = False
        self._t0 = 0.0
        self._building = False

        lay = QVBoxLayout(self)
        if not self._avail.ok:
            msg = QLabel(self._avail.reason)
            msg.setWordWrap(True)
            lay.addWidget(msg)
            lay.addStretch(1)
            return

        top = QHBoxLayout()
        self.run_all_btn = QPushButton("Run all")
        self.run_sel_btn = QPushButton("Run selected")
        self.run_failed_btn = QPushButton("Run failed")
        self.stop_btn = QPushButton("Stop")
        self.refresh_btn = QPushButton("Refresh")
        self.run_all_btn.clicked.connect(self.run_all)
        self.run_sel_btn.clicked.connect(self.run_selected)
        self.run_failed_btn.clicked.connect(self.run_failed)
        self.stop_btn.clicked.connect(self.stop)
        self.refresh_btn.clicked.connect(self.collect)
        for b in (self.run_all_btn, self.run_sel_btn, self.run_failed_btn, self.stop_btn):
            top.addWidget(b)
        top.addStretch(1)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter tests")
        self.filter.setClearButtonEnabled(True)
        self.filter.textChanged.connect(self._apply_filter)
        top.addWidget(self.filter)
        self.only_failed = QCheckBox("Failed only")
        self.only_failed.toggled.connect(self._apply_filter)
        top.addWidget(self.only_failed)
        top.addWidget(self.refresh_btn)
        lay.addLayout(top)

        prog = QHBoxLayout()
        self.bar = QProgressBar()
        self.bar.setTextVisible(True)
        prog.addWidget(self.bar, 1)
        self.counts = QLabel("")
        prog.addWidget(self.counts)
        lay.addLayout(prog)
        self.state = QLabel("")
        self.state.setObjectName("muted")
        lay.addWidget(self.state)

        split = QSplitter(Qt.Vertical)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Test", "Status"])
        self.tree.setColumnWidth(0, 640)
        self.tree.itemChanged.connect(self._ticks_changed)
        self.tree.itemDoubleClicked.connect(self._double_clicked)
        self.tree.currentItemChanged.connect(self._show_details)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._menu)
        split.addWidget(self.tree)
        tabs = QTabWidget()
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setFont(mono_font())
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setFont(mono_font())
        self.output.setLineWrapMode(QPlainTextEdit.NoWrap)
        tabs.addTab(with_find_bar(self.details), "Details")
        tabs.addTab(with_find_bar(self.output), "Output")
        split.addWidget(tabs)
        split.setSizes([430, 220])
        lay.addWidget(split, 1)

        bottom = QHBoxLayout()
        for text, fn in (("Check all", lambda: self._check_all(True)), ("Clear checks", lambda: self._check_all(False))):
            b = QPushButton(text)
            b.clicked.connect(fn)
            bottom.addWidget(b)
        bottom.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        bottom.addWidget(close)
        lay.addLayout(bottom)

        self.clock = QTimer(self)
        self.clock.setInterval(500)
        self.clock.timeout.connect(self._tick)
        self._update_buttons()
        if collect:
            self.collect()

    # --- process plumbing -----------------------------------------------------------
    def _new_process(self, args: list[str]) -> QProcess:
        p = QProcess(self)
        p.setProgram(self.python)
        p.setArguments(args)
        p.setWorkingDirectory(str(self.tests_dir.parent))
        p.setProcessChannelMode(QProcess.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONUTF8", "1")
        p.setProcessEnvironment(env)
        return p

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.state() != QProcess.NotRunning

    def collect(self) -> None:
        if self.running:
            return
        self._collecting = True
        self.state.setText("Collecting tests...")
        self._buf, self._log = "", []
        self._proc = self._new_process(R.collect_args())
        self._proc.readyReadStandardOutput.connect(self._read)
        self._proc.finished.connect(self._finished)
        self._proc.start()
        self._update_buttons()

    def _start(self, targets: list[str], run_ids: list[str]) -> None:
        if self.running or not targets:
            return
        self._collecting = False
        self._run_ids = list(run_ids)
        for i in run_ids:
            self._set_status(i, "")
            self._status.pop(i, None)
        self._buf, self._log = "", []
        self.output.clear()
        self.details.clear()
        self.bar.setRange(0, max(len(run_ids), 1))
        self.bar.setValue(0)
        self._t0 = time.monotonic()
        self.state.setText(f"Running {len(run_ids)} test{'s' if len(run_ids) != 1 else ''}...")
        self._proc = self._new_process(R.run_args(targets))
        self._proc.readyReadStandardOutput.connect(self._read)
        self._proc.finished.connect(self._finished)
        self._proc.start()
        self.clock.start()
        self._refresh_counts()
        self._update_buttons()

    def stop(self) -> None:
        if self.running:
            self.state.setText("Stopping...")
            self._proc.terminate()
            QTimer.singleShot(2000, lambda: self._proc.kill() if self.running else None)

    def shutdown(self) -> None:
        """Kill a running suite (the main window is closing)."""
        if self.running:
            self._proc.kill()
            self._proc.waitForFinished(1500)

    def _read(self) -> None:
        self._buf += bytes(self._proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        *lines, self._buf = self._buf.split("\n")
        for ln in lines:
            self._on_line(ln.rstrip("\r"))

    def _on_line(self, line: str) -> None:
        self._log.append(line)
        if self._collecting:
            return
        self.output.appendPlainText(line)
        got = R.parse_result(line)
        if got:
            self._set_status(got[0], got[1])
            self.bar.setValue(min(self.bar.value() + 1, self.bar.maximum()))
            self._refresh_counts()

    def _finished(self, code: int, _status) -> None:
        if self._buf:
            self._on_line(self._buf.rstrip("\r"))
            self._buf = ""
        self.clock.stop()
        text = "\n".join(self._log)
        if self._collecting:
            self._collecting = False
            ids = R.parse_collected(text)
            if ids:
                self._populate(ids)
                self.state.setText(f"{len(ids)} tests in {len(self._files)} files.")
            else:
                self.state.setText("No tests were collected.")
                self.details.setPlainText(text.strip())
        else:
            self._failures = R.split_failures(text)
            summary = R.parse_summary(text)
            stopped = code not in (0, 1) and not summary
            self.state.setText("Stopped." if stopped else (summary or f"Finished (exit code {code})."))
            if not stopped:
                self.bar.setValue(self.bar.maximum())
            self._refresh_counts()
            self._show_details(self.tree.currentItem())
        self._update_buttons()

    # --- tree -----------------------------------------------------------------------
    def _populate(self, ids: list[str]) -> None:
        self._building = True
        self.tree.clear()
        self._ids, self._items, self._files, self._status = list(ids), {}, {}, {}
        for path, tests in R.by_file(ids).items():
            f = QTreeWidgetItem(self.tree, [Path(path).name, ""])
            f.setFlags(f.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate)
            f.setCheckState(0, Qt.Unchecked)
            f.setData(0, ID_ROLE, path)
            self._files[path] = f
            for tid in tests:
                self._add_test(f, tid)
        self._building = False
        self.bar.setRange(0, max(len(ids), 1))
        self.bar.setValue(0)
        self._refresh_counts()

    def _add_test(self, file_item: QTreeWidgetItem, tid: str) -> None:
        t = QTreeWidgetItem(file_item, ["::".join(tid.split("::")[1:]), ""])
        t.setFlags(t.flags() | Qt.ItemIsUserCheckable)
        t.setCheckState(0, Qt.Unchecked)
        t.setData(0, ID_ROLE, tid)
        self._items[tid] = t

    def _set_status(self, tid: str, outcome: str) -> None:
        item = self._items.get(tid)
        if item is None:                                     # a parametrized id that collection did not list
            path = tid.split("::", 1)[0]
            parent = self._files.get(path)
            if parent is None:
                return
            self._building = True
            self._add_test(parent, tid)
            self._building = False
            self._ids.append(tid)
            item = self._items[tid]
        if outcome:
            self._status[tid] = outcome
        text, colour = LABEL.get(outcome, ("", GREY))
        item.setText(1, text)
        item.setForeground(1, QBrush(colour))
        if outcome in ("FAILED", "ERROR"):
            item.setForeground(0, QBrush(RED))
            item.parent().setExpanded(True)
        else:
            item.setData(0, Qt.ForegroundRole, None)
        self._paint_file(item.parent())

    def _paint_file(self, f: QTreeWidgetItem) -> None:
        res = [self._status.get(f.child(i).data(0, ID_ROLE)) for i in range(f.childCount())]
        bad = sum(r in ("FAILED", "ERROR") for r in res)
        done = sum(r is not None for r in res)
        text = "" if not done else (f"{bad} failed" if bad else f"✓ {done} passed" if done == len(res) else f"{done}/{len(res)}")
        f.setText(1, text)
        f.setForeground(1, QBrush(RED if bad else GREEN))

    def _apply_filter(self, *_) -> None:
        needle, failed = self.filter.text().strip().lower(), self.only_failed.isChecked()
        for f in self._files.values():
            shown = 0
            for i in range(f.childCount()):
                t = f.child(i)
                tid = t.data(0, ID_ROLE)
                hide = (needle and needle not in tid.lower()) or (failed and self._status.get(tid) not in ("FAILED", "ERROR"))
                t.setHidden(bool(hide))
                shown += not hide
            f.setHidden(shown == 0)
            if needle and shown:
                f.setExpanded(True)

    def _check_all(self, on: bool) -> None:
        self._building = True
        for f in self._files.values():
            for i in range(f.childCount()):
                if not f.child(i).isHidden():
                    f.child(i).setCheckState(0, Qt.Checked if on else Qt.Unchecked)
            f.setCheckState(0, Qt.Checked if on else Qt.Unchecked)
        self._building = False
        self._update_buttons()

    def _ticks_changed(self, *_) -> None:
        if not self._building:
            self._update_buttons()

    def checked_ids(self) -> list[str]:
        return [tid for tid, it in self._items.items() if it.checkState(0) == Qt.Checked]

    def _double_clicked(self, item, _col) -> None:
        tid = item.data(0, ID_ROLE)
        if tid and "::" in tid and not self.running:
            self.run_ids([tid])

    def _menu(self, pos) -> None:
        item = self.tree.itemAt(pos)
        if item is None or self.running:
            return
        tid = item.data(0, ID_ROLE)
        menu = QMenu(self.tree)
        if "::" in tid:
            act = menu.addAction("Run this test")
            if menu.exec(self.tree.viewport().mapToGlobal(pos)) is act:
                self.run_ids([tid])
        else:
            act = menu.addAction("Run this file")
            if menu.exec(self.tree.viewport().mapToGlobal(pos)) is act:
                self.run_ids([i for i in self._ids if i.split("::", 1)[0] == tid])

    def _show_details(self, item, _prev=None) -> None:
        if item is None or not hasattr(self, "details"):
            return
        tid = item.data(0, ID_ROLE) or ""
        if "::" not in tid:
            return
        fails = getattr(self, "_failures", {})
        key = R.failure_key(tid)
        text = fails.get(key) or next((v for k, v in fails.items() if k.endswith(key) or key.endswith(k)), "")
        self.details.setPlainText(text or {"PASSED": "Passed.", "SKIPPED": "Skipped."}.get(self._status.get(tid, ""), ""))

    # --- running ----------------------------------------------------------------------
    def run_all(self) -> None:
        self._start([str(self.tests_dir.relative_to(self.tests_dir.parent))], list(self._ids))

    def run_selected(self) -> None:
        self.run_ids(self.checked_ids())

    def run_failed(self) -> None:
        self.run_ids([t for t, s in self._status.items() if s in ("FAILED", "ERROR")])

    def run_ids(self, ids: list[str]) -> None:
        self._start(R.plan_args(ids, self._ids), ids)

    # --- status line --------------------------------------------------------------------
    def _refresh_counts(self) -> None:
        n = {k: 0 for k in ("PASSED", "FAILED", "SKIPPED")}
        for s in self._status.values():
            n["FAILED" if s in ("FAILED", "ERROR") else "SKIPPED" if s in ("SKIPPED", "XFAIL") else "PASSED"] += 1
        left = max(len([i for i in self._run_ids if i not in self._status]), 0) if self.running else 0
        secs = f"  ·  {time.monotonic() - self._t0:.0f} s" if self.running else ""
        self.counts.setText(f"Passed {n['PASSED']}  ·  Failed {n['FAILED']}  ·  Skipped {n['SKIPPED']}"
                            + (f"  ·  Left {left}" if self.running else "") + secs)

    def _tick(self) -> None:
        self._refresh_counts()

    def _update_buttons(self) -> None:
        if not hasattr(self, "run_all_btn"):
            return
        idle, have = not self.running, bool(self._ids)
        self.run_all_btn.setEnabled(idle and have)
        self.run_sel_btn.setEnabled(idle and bool(self.checked_ids()))
        self.run_failed_btn.setEnabled(idle and any(s in ("FAILED", "ERROR") for s in self._status.values()))
        self.stop_btn.setEnabled(not idle)
        self.refresh_btn.setEnabled(idle)
