# -*- coding: utf-8 -*-
"""Unit tests for the GUI path guards and ``core.fileio.atomic_write_text``.

These are the pure, widget-free functions: ``is_within_project``,
``ensure_project_safe_path`` and ``atomic_write_text`` (plus the checkout
detection they rely on, ``core.paths.is_checkout``).
Importing the module pulls in PySide6 but never instantiates a ``QApplication``
or widget, so everything below runs offscreen.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from linak.core.fileio import atomic_write_text
from linak.core.paths import is_checkout
from linak.gui.project_io import ensure_project_safe_path, is_within_project


def leftovers(directory: Path, target: Path) -> list[Path]:
    """Every file in *directory* except the expected *target* itself."""
    return [p for p in directory.iterdir() if p != target]


def test_is_checkout_detects_checkout_shape(tmp_path: Path) -> None:
    """A dir with pyproject.toml + src/linak counts as a checkout."""
    assert not is_checkout(tmp_path)
    (tmp_path / "pyproject.toml").write_text("# marker\n", encoding="utf-8")
    (tmp_path / "src" / "linak").mkdir(parents=True)
    assert is_checkout(tmp_path)


# --- is_within_project ------------------------------------------------------


def test_is_within_project_accepts_root_itself(tmp_path: Path) -> None:
    assert is_within_project(tmp_path, tmp_path)


def test_is_within_project_accepts_nested_path(tmp_path: Path) -> None:
    assert is_within_project(tmp_path, tmp_path / "data" / "out.json")


def test_is_within_project_rejects_sibling_path(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}_outside"
    assert not is_within_project(tmp_path, outside)


# --- ensure_project_safe_path -----------------------------------------------


def test_ensure_project_safe_path_returns_target_when_inside(tmp_path: Path) -> None:
    target = tmp_path / "data" / "Mg_c1.json"
    assert ensure_project_safe_path(tmp_path, target) == target


def test_ensure_project_safe_path_rejects_traversal(tmp_path: Path) -> None:
    """``<root>/../evil.txt`` resolves outside the root and must raise."""
    target = tmp_path / ".." / "evil.txt"
    with pytest.raises(ValueError, match="Refusing to write outside"):
        ensure_project_safe_path(tmp_path, target)


def test_ensure_project_safe_path_rejects_absolute_outside(tmp_path: Path) -> None:
    other = tmp_path.parent / "somewhere_else"
    with pytest.raises(ValueError, match="Refusing to write outside"):
        ensure_project_safe_path(tmp_path, other / "evil.txt")


def test_ensure_project_safe_path_rejects_home_expansion(tmp_path: Path) -> None:
    """``~`` expands to the user home, which is outside any project root."""
    with pytest.raises(ValueError, match="Refusing to write outside"):
        ensure_project_safe_path(tmp_path, Path("~") / "evil.txt")


# --- atomic_write_text ------------------------------------------------------


def test_atomic_write_text_creates_parent_and_exact_utf8_content(tmp_path: Path) -> None:
    target = tmp_path / "data" / "Mg_c1.json"
    text = '{"label": "Mg+", "note": "café π"}\n'
    atomic_write_text(target, text)
    assert target.read_text(encoding="utf-8") == text
    # No temp file may survive next to the destination.
    assert leftovers(target.parent, target) == []


def test_atomic_write_text_overwrites_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "out.json"
    atomic_write_text(target, "first")
    atomic_write_text(target, "second\n")
    assert target.read_text(encoding="utf-8") == "second\n"
    assert leftovers(target.parent, target) == []


def test_atomic_write_text_preserves_bytes_with_newline_eq_empty(tmp_path: Path) -> None:
    """``newline=""`` must keep the text byte-exact (no CRLF translation)."""
    target = tmp_path / "out.json"
    text = "a\r\nb\n"
    atomic_write_text(target, text)
    assert target.read_bytes() == text.encode("utf-8")


def test_atomic_write_text_cleanup_on_replace_failure(tmp_path: Path, monkeypatch) -> None:
    """If the final rename fails, no temp file survives and the target is absent."""
    target = tmp_path / "out.json"

    def boom(src: str, dst: str) -> None:
        raise OSError("simulated rename failure")

    monkeypatch.setattr("linak.core.fileio.os.replace", boom)
    with pytest.raises(OSError, match="simulated rename failure"):
        atomic_write_text(target, "payload")
    assert not target.exists()
    assert leftovers(tmp_path, target) == []


def test_atomic_write_text_failure_leaves_existing_target_intact(
    tmp_path: Path, monkeypatch
) -> None:
    """An overwrite whose rename fails must not corrupt the existing file."""
    target = tmp_path / "out.json"
    atomic_write_text(target, "original")

    def boom(src: str, dst: str) -> None:
        raise OSError("simulated rename failure")

    monkeypatch.setattr("linak.core.fileio.os.replace", boom)
    with pytest.raises(OSError, match="simulated rename failure"):
        atomic_write_text(target, "corrupted")
    assert target.read_text(encoding="utf-8") == "original"
    assert leftovers(tmp_path, target) == []


def test_atomic_write_text_failure_cleans_temp_on_fsync_error(
    tmp_path: Path, monkeypatch
) -> None:
    """A failure inside the write step also cleans up the temp file."""
    target = tmp_path / "out.json"

    def bad_fsync(fd) -> None:
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(os, "fsync", bad_fsync)
    with pytest.raises(OSError, match="simulated fsync failure"):
        atomic_write_text(target, "payload")
    assert not target.exists()
    assert leftovers(tmp_path, target) == []
