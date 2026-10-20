# -*- coding: utf-8 -*-
"""Shared fixtures for pipeline regression tests.

All paths stay under tests/ or the package's bundled resources
(src/linak/resources/). Artifacts go to tests/results/ only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from linak.core.paths import PROJECT_ROOT_ENV, reference_path

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
RESULTS = TESTS / "results"
ANCHORS = TESTS / "anchors" / "na_regression.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))


@pytest.fixture(scope="session", autouse=True)
def isolated_user_data(tmp_path_factory) -> Path:
    """Point LiNaK's per-user data home at a temp dir for the whole session.

    Unconfigured runs default to the per-user home, so no test may seed or
    write the developer's real one; pipeline subprocesses inherit the
    override through the environment.
    """
    home = tmp_path_factory.mktemp("user_data")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("XDG_DATA_HOME", str(home))
        mp.setenv("LOCALAPPDATA", str(home))
        if sys.platform == "darwin":
            mp.setenv("HOME", str(home))
        mp.delenv(PROJECT_ROOT_ENV, raising=False)
        yield home


@pytest.fixture(scope="session")
def root_dir() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """The shipped (bundled, read-only) JSON datasets the anchors validate."""
    return reference_path("data_json")


@pytest.fixture(scope="session")
def results_dir() -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    return RESULTS


@pytest.fixture(scope="session")
def anchors() -> dict:
    with ANCHORS.open(encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def na_polarizability(data_dir: Path) -> dict:
    path = data_dir / "Na_polarizability.json"
    assert path.is_file(), f"missing {path}"
    with path.open(encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def na_lifetimes(data_dir: Path) -> dict:
    path = data_dir / "Na_lifetimes.json"
    assert path.is_file(), f"missing {path}"
    with path.open(encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def na_blackbody(data_dir: Path) -> dict:
    path = data_dir / "Na_blackbody.json"
    assert path.is_file(), f"missing {path}"
    with path.open(encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def na_feshbach_params(data_dir: Path) -> dict:
    path = data_dir / "Na_feshbach.json"
    assert path.is_file(), f"missing {path}"
    with path.open(encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def lifetime_by_state(na_lifetimes: dict) -> dict:
    return {row["state"]: row for row in na_lifetimes["lifetimes"]}


@pytest.fixture(scope="session")
def workspace(tmp_path_factory) -> Path:
    """An isolated, checkout-shaped project root for end-to-end pipeline runs.

    ``set_active``/``runtime_root_for`` only use ``<root>/.linak`` when *root*
    looks like a checkout, so the two markers they check are created here. This
    keeps every pipeline write inside the temporary directory.
    """
    root = tmp_path_factory.mktemp("workspace")
    (root / "pyproject.toml").write_text("# isolated test workspace\n", encoding="utf-8")
    (root / "src" / "linak").mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture(scope="session")
def feshbach_run(workspace) -> object:
    """One shared end-to-end ``feshbach Na`` run, reused by every suite.

    The subprocess keeps file writes inside *workspace*; the single run keeps
    the prepared-suites' cost to one pipeline execution per session.
    """
    from helpers import run_pipeline

    result = run_pipeline(["feshbach", "Na"], workspace)
    assert result.returncode == 0, f"feshbach Na failed:\n{result.stderr}"
    return result
