"""Pure helpers behind Run > Tests: finding the tests, building the pytest command, reading its output."""

from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass
from pathlib import Path

from linak.core.paths import is_frozen_app

OUTCOMES = ("PASSED", "FAILED", "ERROR", "SKIPPED", "XFAIL", "XPASS")
_RESULT = re.compile(r"^(?P<id>\S.*?::\S.*?) (?P<res>PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\b")
_FAIL_HEAD = re.compile(r"^_{3,} (?P<name>.+?) _{3,}$")
_SUMMARY = re.compile(r"(?P<body>(?:\d+ \w+(?:, )?)+(?: \(\d+:\d+:\d+\))?) in (?P<secs>[\d.]+)s")
BASE_ARGS = ("-v", "--tb=short", "--color=no", "-p", "no:cacheprovider", "-o", "addopts=")


@dataclass(frozen=True)
class Availability:
    ok: bool
    reason: str = ""
    tests_dir: Path | None = None


def find_tests_dir() -> Path | None:
    """The checkout's ``tests`` folder (next to ``src``), or None in an installed or bundled build."""
    import linak

    root = Path(linak.__file__).resolve().parents[2]
    return root / "tests" if (root / "tests").is_dir() and (root / "pyproject.toml").is_file() else None


def availability(tests_dir: Path | None = None) -> Availability:
    if is_frozen_app():
        return Availability(False, "The tests are not part of the packaged app. Run them from a source checkout.")
    tests_dir = tests_dir or find_tests_dir()
    if tests_dir is None or not Path(tests_dir).is_dir():
        return Availability(False, "No tests folder was found next to this install.")
    if importlib.util.find_spec("pytest") is None:
        return Availability(False, "pytest is not installed in this environment (uv sync --group dev).")
    return Availability(True, "", Path(tests_dir))


def collect_args() -> list[str]:
    return ["-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "-o", "addopts="]


def parse_collected(output: str) -> list[str]:
    """Node ids from ``--collect-only -q`` output (lines of the form ``tests/x.py::name``)."""
    return [ln.strip() for ln in output.splitlines() if "::" in ln and not ln.startswith((" ", "=", "-")) and " " not in ln.split("::")[0]]


def by_file(ids: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for i in ids:
        out.setdefault(i.split("::", 1)[0], []).append(i)
    return out


def plan_args(selected: list[str], universe: list[str]) -> list[str]:
    """pytest targets for *selected*: a whole file when all its tests are chosen, else the single ids."""
    chosen, files = set(selected), by_file(universe)
    targets: list[str] = []
    for path, ids in files.items():
        mine = [i for i in ids if i in chosen]
        if mine and len(mine) == len(ids):
            targets.append(path)
        else:
            targets += mine
    targets += [i for i in selected if i not in set(universe)]          # ids found only at run time (parametrized)
    return targets


def run_args(targets: list[str]) -> list[str]:
    return ["-u", "-m", "pytest", *targets, *BASE_ARGS]


def parse_result(line: str) -> tuple[str, str] | None:
    """``(node id, outcome)`` from a ``-v`` result line, else None."""
    m = _RESULT.match(line.strip())
    return (m.group("id"), m.group("res")) if m else None


def split_failures(output: str) -> dict[str, str]:
    """Traceback text per failed test, keyed by the test name pytest prints in the FAILURES section."""
    blocks: dict[str, list[str]] = {}
    name = None
    in_section = False
    for ln in output.splitlines():
        if re.match(r"^=+ (FAILURES|ERRORS) =+$", ln):
            in_section = True
            continue
        if in_section and re.match(r"^=+ .* =+$", ln):
            in_section = False
            name = None
            continue
        if not in_section:
            continue
        head = _FAIL_HEAD.match(ln)
        if head:
            name = head.group("name")
            blocks[name] = []
        elif name is not None:
            blocks[name].append(ln)
    return {k: "\n".join(v).strip() for k, v in blocks.items()}


def failure_key(node_id: str) -> str:
    """The name pytest prints in a failure header: class and test joined by '.', file left off."""
    return ".".join(node_id.split("::")[1:])


def parse_summary(output: str) -> str:
    """``"3 passed, 1 failed in 0.41s"`` from the last summary line, or ''."""
    last = ""
    for ln in output.splitlines():
        m = _SUMMARY.search(ln)
        if m and ln.startswith("="):
            last = f"{m.group('body')} in {m.group('secs')}s"
    return last
