# -*- coding: utf-8 -*-
"""``tools/compare_outputs.py``: the refactor safety check must not pass silently."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from compare_outputs import compare  # noqa: E402


def _write(root: Path, name: str, text: str) -> None:
    path = root / ".linak" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_identical_outputs_compare_clean(tmp_path):
    for side in ("a", "b"):
        _write(tmp_path / side, "data/Na_rydberg.json", '{"n": 1}')
    assert compare(tmp_path / "a", tmp_path / "b") == []


def test_plotly_div_ids_are_ignored_but_content_is_not(tmp_path):
    _write(tmp_path / "a", "plots/Na/x.html", '<div id="0f8fad5b-d9cb-469f-a165-70867728950e">1</div>')
    _write(tmp_path / "b", "plots/Na/x.html", '<div id="7c9e6679-7425-40de-944b-e07fc1f90ae7">1</div>')
    assert compare(tmp_path / "a", tmp_path / "b") == []
    _write(tmp_path / "b", "plots/Na/x.html", '<div id="7c9e6679-7425-40de-944b-e07fc1f90ae7">2</div>')
    assert compare(tmp_path / "a", tmp_path / "b") == ["changed:       plots/Na/x.html"]


def test_json_must_match_byte_for_byte_and_file_sets_must_agree(tmp_path):
    _write(tmp_path / "a", "data/Na_rydberg.json", '{"n": 1}')
    _write(tmp_path / "b", "data/Na_rydberg.json", '{"n": 1.0}')
    _write(tmp_path / "a", "data/old.json", "{}")
    _write(tmp_path / "b", "data/new.json", "{}")
    assert compare(tmp_path / "a", tmp_path / "b") == [
        "missing after: data/old.json",
        "new after:     data/new.json",
        "changed:       data/Na_rydberg.json",
    ]
