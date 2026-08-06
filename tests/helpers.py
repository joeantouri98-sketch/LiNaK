# -*- coding: utf-8 -*-
"""Small helpers shared by test modules (kept out of conftest for clean imports)."""
from __future__ import annotations

import json
from pathlib import Path


def write_result_summary(results_dir: Path, name: str, payload: dict) -> Path:
    """Write a small JSON artifact under tests/results/ only."""
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"{name}.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return out
