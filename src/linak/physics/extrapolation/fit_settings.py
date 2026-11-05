"""User-adjustable fit settings: how strongly the ridge correction is regularised and how anchors are weighted.

These change the fit and the bands only, never the model tables, so they take effect immediately (no rebuild).
They are stored as a small JSON file next to the extrapolation configs, ``fit_settings.json``, holding only the
values that differ from the packaged defaults. Presets are named copies of such a file in ``presets/``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from linak.core.fileio import write_json

from .params import _anchor_config_base

# key, label, tooltip, lowest, highest, step, decimals
FIELDS: list[dict] = [
    {"key": "ridge.tau0", "label": "Offset prior (τ₀)", "lo": 0.01, "hi": 2.0, "step": 0.01, "dec": 3,
     "tip": "How far the overall correction to the model (ln f) may move. Smaller keeps the model closer to its own values."},
    {"key": "ridge.tau1", "label": "Slope prior (τ₁)", "lo": 0.1, "hi": 50.0, "step": 0.1, "dec": 2,
     "tip": "How strongly the correction may change with 1/nu^2 (toward the series limit). Smaller makes it flatter."},
    {"key": "ridge.tau2_threshold_aware", "label": "Threshold term prior (τ₂)", "lo": 0.01, "hi": 20.0, "step": 0.1, "dec": 2,
     "tip": "Experimental. Strength of the extra threshold-aware correction term (only used when that option is on)."},
    {"key": "ssys_by_type.nist", "label": "Systematic, NIST anchors", "lo": 0.0, "hi": 1.0, "step": 0.01, "dec": 3,
     "tip": "Relative systematic uncertainty added to every NIST anchor. It limits how much any one anchor can pull."},
    {"key": "ssys_by_type.precision", "label": "Systematic, precision anchors", "lo": 0.0, "hi": 1.0, "step": 0.01, "dec": 3,
     "tip": "Same, for precision (lifetime-derived) anchors."},
    {"key": "floor_unc", "label": "Uncertainty floor", "lo": 0.0, "hi": 1.0, "step": 0.005, "dec": 3,
     "tip": "Smallest relative uncertainty an anchor may have, whatever its quoted value."},
    {"key": "low_weight_unc", "label": "Low-weight uncertainty", "lo": 0.05, "hi": 5.0, "step": 0.05, "dec": 2,
     "tip": "Uncertainty used for anchors flagged low weight (spliced or extrapolated NIST rows)."},
]
_BY_KEY = {f["key"]: f for f in FIELDS}


def _get(cfg: dict, key: str):
    node = cfg
    for part in key.split("."):
        node = node[part]
    return node


def defaults() -> dict[str, float]:
    base = _anchor_config_base()
    return {f["key"]: float(_get(base, f["key"])) for f in FIELDS}


def validate(fit: dict | None) -> tuple[dict[str, float], list[str]]:
    """(clean values that differ from the defaults, problems). Unknown or out-of-range entries are dropped."""
    clean: dict[str, float] = {}
    problems: list[str] = []
    base = defaults()
    for key, raw in (fit or {}).items():
        spec = _BY_KEY.get(key)
        if spec is None:
            problems.append(f"unknown fit setting {key!r} ignored")
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            problems.append(f"{spec['label']}: {raw!r} is not a number, ignored")
            continue
        if not (spec["lo"] <= val <= spec["hi"]):
            problems.append(f"{spec['label']}: {val:g} is outside {spec['lo']:g} to {spec['hi']:g}, ignored")
            continue
        if abs(val - base[key]) > 1e-12:
            clean[key] = val
    return clean, problems


def to_overrides(fit: dict | None) -> dict:
    """Flat ``{'ridge.tau0': 0.2}`` -> nested ``{'ridge': {'tau0': 0.2}}`` for ``merge_overrides``."""
    out: dict = {}
    for key, val in (fit or {}).items():
        node = out
        *parents, last = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[last] = val
    return out


def describe(fit: dict | None) -> str:
    """One line naming the settings in use, for the dialog warning and the JSON."""
    return ", ".join(f"{_BY_KEY[k]['label']} = {v:g}" for k, v in (fit or {}).items() if k in _BY_KEY)


def settings_path(config_path: str | Path) -> Path:
    """The fit-settings file that belongs to a per-species config file (same folder)."""
    return Path(config_path).parent / "fit_settings.json"


def load(path: str | Path) -> dict[str, float]:
    """Saved fit settings, validated. A missing or unreadable file means the defaults."""
    try:
        with Path(path).open(encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    clean, _problems = validate(data.get("fit", data) if isinstance(data, dict) else {})
    return clean


def save(path: str | Path, fit: dict | None) -> None:
    clean, _ = validate(fit)
    path = Path(path)
    if not clean:                         # nothing differs from the defaults: no file
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(str(path), {"fit": clean})


# ---- presets ---------------------------------------------------------------------------------------------

def presets_dir(config_path: str | Path) -> Path:
    return Path(config_path).parent / "presets"


def _safe(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", name.strip()).strip("._") or "preset"


def list_presets(folder: str | Path) -> list[str]:
    return sorted(p.stem for p in Path(folder).glob("*.json")) if Path(folder).is_dir() else []


def save_preset(folder: str | Path, name: str, fit: dict | None) -> Path:
    clean, _ = validate(fit)
    path = Path(folder) / f"{_safe(name)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(str(path), {"name": name.strip(), "fit": clean})
    return path


def load_preset(folder: str | Path, name: str) -> dict[str, float]:
    return load(Path(folder) / f"{_safe(name)}.json")


def delete_preset(folder: str | Path, name: str) -> None:
    (Path(folder) / f"{_safe(name)}.json").unlink(missing_ok=True)
