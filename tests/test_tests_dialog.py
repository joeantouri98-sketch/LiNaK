"""Run > Tests: pytest output parsing, the command it builds, and a real run through the dialog on a tiny project."""
from __future__ import annotations

import os
import sys
import textwrap

import pytest

from qali.gui import test_runner as R

COLLECT = """tests/test_a.py::test_ok
tests/test_a.py::test_bad
tests/test_b.py::TestX::test_p[1]
tests/test_b.py::TestX::test_p[2]

4 tests collected in 0.02s
"""


def test_parse_collected_keeps_only_node_ids():
    assert R.parse_collected(COLLECT) == ["tests/test_a.py::test_ok", "tests/test_a.py::test_bad",
                                          "tests/test_b.py::TestX::test_p[1]", "tests/test_b.py::TestX::test_p[2]"]
    assert R.parse_collected("ERROR: file or directory not found: x") == []


def test_parse_result_lines():
    assert R.parse_result("tests/test_a.py::test_ok PASSED                          [ 25%]") == ("tests/test_a.py::test_ok", "PASSED")
    assert R.parse_result("tests/test_b.py::TestX::test_p[a b] FAILED [ 75%]") == ("tests/test_b.py::TestX::test_p[a b]", "FAILED")
    assert R.parse_result("tests/test_a.py::test_s SKIPPED (reason)  [100%]") == ("tests/test_a.py::test_s", "SKIPPED")
    assert R.parse_result("collecting ... collected 4 items") is None and R.parse_result("") is None


def test_plan_args_uses_a_whole_file_when_all_its_tests_are_chosen():
    ids = R.parse_collected(COLLECT)
    assert R.plan_args(ids[:2], ids) == ["tests/test_a.py"]
    assert R.plan_args([ids[0], ids[2]], ids) == [ids[0], ids[2]]
    assert R.plan_args([ids[2], "tests/test_b.py::TestX::test_p[3]"], ids) == [ids[2], "tests/test_b.py::TestX::test_p[3]"]
    assert R.plan_args([], ids) == []


def test_run_args_never_pick_up_the_project_addopts():
    a = R.run_args(["tests/test_a.py"])
    assert a[:3] == ["-u", "-m", "pytest"] and "addopts=" in a and "no:cacheprovider" in a and "--color=no" in a


def test_failures_and_summary_are_read_from_the_report():
    out = textwrap.dedent("""\
        ============================= FAILURES =============================
        _________________________ TestX.test_p[2] _________________________
        tests/test_b.py:9: in test_p
            assert 0
        E   assert 0
        _____________________________ test_bad ____________________________
        E   boom
        =========================== short test summary info ===========================
        FAILED tests/test_a.py::test_bad - boom
        ======================== 1 failed, 2 passed in 0.05s ========================
        """)
    f = R.split_failures(out)
    assert set(f) == {"TestX.test_p[2]", "test_bad"} and "assert 0" in f["TestX.test_p[2]"]
    assert R.failure_key("tests/test_b.py::TestX::test_p[2]") == "TestX.test_p[2]"
    assert R.parse_summary(out) == "1 failed, 2 passed in 0.05s"


def test_unavailable_when_there_is_no_tests_folder(tmp_path):
    assert not R.availability(tmp_path / "nope").ok


# --- the dialog, on a throw-away project -----------------------------------------------------------------
pytest.importorskip("PySide6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QCoreApplication, QElapsedTimer, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from qali.gui.tests_dialog import TestsDialog  # noqa: E402


@pytest.fixture()
def project(tmp_path):
    t = tmp_path / "tests"
    t.mkdir()
    (t / "test_a.py").write_text("import pytest\n\ndef test_ok():\n    assert True\n\ndef test_bad():\n    assert 1 == 2\n\n"
                                 "@pytest.mark.skip(reason='x')\ndef test_skip():\n    pass\n", encoding="utf-8")
    (t / "test_b.py").write_text("import pytest\n\n@pytest.mark.parametrize('n', [1, 2])\ndef test_p(n):\n    assert n\n", encoding="utf-8")
    return t


def _wait(app, cond, ms=60000):
    clock = QElapsedTimer()
    clock.start()
    while not cond() and clock.elapsed() < ms:
        QCoreApplication.processEvents()
        app.processEvents()
    assert cond(), "timed out"


@pytest.fixture()
def dlg(project):
    app = QApplication.instance() or QApplication([])
    d = TestsDialog(None, tests_dir=project, python=sys.executable)
    _wait(app, lambda: d._ids and not d.running)
    yield app, d
    d.shutdown()


def test_collects_and_lists_tests_by_file(dlg):
    _, d = dlg
    assert len(d._ids) == 5 and set(d._files) == {"tests/test_a.py", "tests/test_b.py"}
    assert d.run_all_btn.isEnabled() and not d.run_sel_btn.isEnabled() and not d.stop_btn.isEnabled()


def test_run_all_reports_progress_and_failures(dlg):
    app, d = dlg
    d.run_all()
    _wait(app, lambda: not d.running and d._status)
    assert {t: d._status[t] for t in d._ids} == {"tests/test_a.py::test_ok": "PASSED", "tests/test_a.py::test_bad": "FAILED",
                                                  "tests/test_a.py::test_skip": "SKIPPED", "tests/test_b.py::test_p[1]": "PASSED",
                                                  "tests/test_b.py::test_p[2]": "PASSED"}
    assert d.bar.value() == d.bar.maximum() == 5
    assert "1 failed" in d.state.text() and "Failed 1" in d.counts.text()
    d.tree.setCurrentItem(d._items["tests/test_a.py::test_bad"])
    assert "assert 1 == 2" in d.details.toPlainText()
    assert d.run_failed_btn.isEnabled()


def test_run_selected_and_single_test(dlg):
    app, d = dlg
    d._items["tests/test_a.py::test_ok"].setCheckState(0, Qt.Checked)
    assert d.checked_ids() == ["tests/test_a.py::test_ok"] and d.run_sel_btn.isEnabled()
    d.run_selected()
    _wait(app, lambda: not d.running and d._status)
    assert d._status == {"tests/test_a.py::test_ok": "PASSED"}
    d.run_ids(["tests/test_b.py::test_p[2]"])
    _wait(app, lambda: not d.running and "tests/test_b.py::test_p[2]" in d._status)
    assert d._status["tests/test_b.py::test_p[2]"] == "PASSED"


def test_run_failed_reruns_only_the_failures(dlg):
    app, d = dlg
    d.run_all()
    _wait(app, lambda: not d.running and d._status)
    d.run_failed()
    _wait(app, lambda: not d.running and d.bar.maximum() == 1 and d.bar.value() == 1)
    assert d._status["tests/test_a.py::test_bad"] == "FAILED" and d._status["tests/test_a.py::test_ok"] == "PASSED"   # others kept


def test_stop_kills_a_running_suite(project):
    app = QApplication.instance() or QApplication([])
    (project / "test_slow.py").write_text("import time\n\ndef test_slow():\n    time.sleep(30)\n", encoding="utf-8")
    d = TestsDialog(None, tests_dir=project, python=sys.executable)
    _wait(app, lambda: d._ids and not d.running)
    d.run_ids(["tests/test_slow.py::test_slow"])
    _wait(app, lambda: d.running)
    d.stop()
    _wait(app, lambda: not d.running, ms=15000)
    assert "Stopped" in d.state.text() and d.run_all_btn.isEnabled()
    d.shutdown()


def test_unavailable_dialog_shows_the_reason(tmp_path):
    app = QApplication.instance() or QApplication([])
    d = TestsDialog(None, tests_dir=tmp_path / "missing", collect=False)
    assert not hasattr(d, "run_all_btn") and app is not None
