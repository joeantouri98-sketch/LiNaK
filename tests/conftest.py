# -*- coding: utf-8 -*-
"""Shared fixtures for pipeline regression tests.

All paths stay under tests/ or the repo root data_json/. Artifacts go to
tests/results/ only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
RESULTS = TESTS / "results"
ANCHORS = TESTS / "anchors" / "na_regression.json"
DATA_JSON = ROOT / "data_json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))


@pytest.fixture(scope="session")
def root_dir() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA_JSON


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
