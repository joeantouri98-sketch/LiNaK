# -*- coding: utf-8 -*-
"""Small helpers shared by test modules (kept out of conftest for clean imports)."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def write_result_summary(results_dir: Path, name: str, payload: dict) -> Path:
    """Write a small JSON artifact under tests/results/ only."""
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"{name}.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.write("\n")
    return out


def make_workspace(root: Path) -> Path:
    """Turn *root* into a checkout-shaped project dir so pipeline writes stay local.

    ``linak.core.paths.runtime_root_for`` only uses ``<root>/.linak`` when the
    root looks like a checkout, so give it the two markers it checks.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text("# isolated test workspace\n", encoding="utf-8")
    (root / "src" / "linak").mkdir(parents=True, exist_ok=True)
    return root


def pipeline_env(workspace: Path) -> dict:
    """Environment that points LiNaK at an isolated workspace and the source tree."""
    env = os.environ.copy()
    env["LINAK_PROJECT_ROOT"] = str(workspace)
    src = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    return env


def run_pipeline(args: list[str], workspace: Path, timeout: float = 900.0) -> subprocess.CompletedProcess:
    """Run ``linak --pipeline <args...>`` as a subprocess against *workspace*."""
    return subprocess.run(
        [sys.executable, "-m", "linak", "--pipeline", *args],
        cwd=str(workspace),
        env=pipeline_env(workspace),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def assert_json_is_finite(payload, path: tuple = ()) -> None:
    """Fail if *payload* (a parsed JSON document) contains NaN or Infinity."""
    if isinstance(payload, float):
        assert math.isfinite(payload), f"non-finite number at {path}: {payload!r}"
    elif isinstance(payload, dict):
        for key, value in payload.items():
            assert_json_is_finite(value, (*path, key))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            assert_json_is_finite(value, (*path, index))


def load_json(path: Path) -> dict:
    """Read a UTF-8 JSON file and assert it is finite and well-formed."""
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text, f"{path} contains a NaN token"
    assert "Infinity" not in text, f"{path} contains an Infinity token"
    payload = json.loads(text)
    assert_json_is_finite(payload)
    return payload
