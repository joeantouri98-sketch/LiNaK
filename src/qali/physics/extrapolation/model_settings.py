"""User-adjustable model parameters, per species, and the tables rebuilt for them.

Unlike the fit settings these change the model f-values themselves, so the shipped tables no longer match: the
extrapolation then uses the exact (slow) calculation until the tables are rebuilt for the new values. Only values that
differ from the shipped ones are stored, in ``model_settings.json`` next to the extrapolation configs; rebuilt tables
live in ``extrapolation_tables/`` beside it. Presets are named copies in ``presets/model/``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from qali.core.fileio import write_json

from .params import packaged_species_params, supported_species

# key, label, tooltip, lowest, highest, step, decimals ("wcore_eV": 0 means no dynamic core factor)
FIELDS: list[dict] = [
    {"key": "singor_alpha_D", "label": "Core polarizability α_D (a.u.)", "html": "Core polarizability α<sub>D</sub> (a.u.)", "lo": 0.0, "hi": 60.0, "step": 0.01, "dec": 4,
     "tip": "Dipole polarizability of the ion core in the polarization correction to the dipole operator (Singor et al.)."},
    {"key": "singor_rc", "label": "Core cutoff radius r_c (a₀)", "html": "Core cutoff radius r<sub>c</sub> (a<sub>0</sub>)", "lo": 0.3, "hi": 12.0, "step": 0.01, "dec": 4,
     "tip": "Cutoff radius of the core polarization term (Singor et al.)."},
    {"key": "lam", "label": "Core polarization scale λ", "html": "Core polarization scale λ", "lo": 0.0, "hi": 2.5, "step": 0.01, "dec": 4,
     "tip": "Scale of the core polarization correction; 1 is the unscaled physical value."},
    {"key": "wcore_eV", "label": "Core frequency ω_core (eV)", "html": "Core frequency ω<sub>core</sub> (eV)", "lo": 0.0, "hi": 60.0, "step": 0.1, "dec": 2,
     "tip": "Dynamic core factor 1/(1−(ω/ω_core)²). 0 switches it off."},
    {"key": "so_scale", "label": "Spin-orbit scale", "html": "Spin-orbit scale", "lo": 0.0, "hi": 4.0, "step": 0.01, "dec": 3,
     "tip": "Scale of the screened spin-orbit term of the np states."},
]
_BY_KEY = {f["key"]: f for f in FIELDS}

# Core polarization correction g(r) in the dipole operator r [1 - lam g(r)]. The name's first letter selects the form.
FORMS: list[tuple[str, str, str]] = [
    ("E_sqrt(1-exp(-(r/rc)^6))", "Square-root cutoff", "a/r^3 sqrt(1-exp(-(r/rc)^6))"),
    ("N_1-exp(-(r/rc)^3)", "Cubic cutoff", "a/r^3 (1-exp(-(r/rc)^3))"),
    ("G_1-exp(-(r/rc)^6)", "Sixth-power cutoff", "a/r^3 (1-exp(-(r/rc)^6))"),
    ("P_plummer", "Plummer", "a/(r^2+rc^2)^(3/2)"),
    ("U_user", "User-defined", ""),
]
FORM_IDS = [f[0] for f in FORMS]


def defaults(element: str) -> dict[str, float]:
    sp = packaged_species_params(element)
    return {f["key"]: float(sp.get(f["key"]) or 0.0) for f in FIELDS}


def default_form() -> str:
    from .params import model_params
    return model_params()["form"]


def validate(model: dict | None) -> tuple[dict[str, dict], list[str]]:
    """({species: {field: value}} of the values that differ from the shipped ones, problems)."""
    clean: dict[str, dict] = {}
    problems: list[str] = []
    for element, vals in (model or {}).items():
        if element not in supported_species():
            problems.append(f"{element}: no model settings for this species, ignored")
            continue
        base = defaults(element)
        for key, raw in (vals or {}).items():
            if key in ("user_formula", "user_params"):
                continue                                   # checked together with the form below
            if key == "form":
                if raw not in FORM_IDS:
                    problems.append(f"{element}: unknown potential form {raw!r} ignored")
                elif raw.startswith("U"):
                    from .userform import FormulaError, validate_user
                    formula, uparams = (vals.get("user_formula") or "").strip(), list(vals.get("user_params") or [])
                    try:
                        validate_user(formula, uparams)
                    except FormulaError as exc:
                        problems.append(f"{element}: user-defined form not used: {exc}")
                    else:
                        clean.setdefault(element, {}).update(form=raw, user_formula=formula, user_params=uparams)
                elif raw != default_form():
                    clean.setdefault(element, {})["form"] = raw
                continue
            spec = _BY_KEY.get(key)
            if spec is None:
                problems.append(f"{element}: unknown model setting {key!r} ignored")
                continue
            try:
                val = float(raw if raw is not None else 0.0)
            except (TypeError, ValueError):
                problems.append(f"{element} {spec['label']}: {raw!r} is not a number, ignored")
                continue
            if not (spec["lo"] <= val <= spec["hi"]):
                problems.append(f"{element} {spec['label']}: {val:g} is outside {spec['lo']:g} to {spec['hi']:g}, ignored")
                continue
            if abs(val - base[key]) > 1e-12:
                clean.setdefault(element, {})[key] = val
    return clean, problems


def to_overrides(model: dict | None) -> dict[str, dict]:
    """The form ``model_overrides`` wants: a zero core frequency means None (no dynamic factor)."""
    return {el: {k: (None if k == "wcore_eV" and v == 0 else v) for k, v in vals.items()} for el, vals in (model or {}).items()}


def describe(model: dict | None, element: str | None = None) -> str:
    parts = []
    for el, vals in (model or {}).items():
        if element is None or el == element:
            parts += [f"{el} {_BY_KEY[k]['label'].split(' (')[0]} = {v:g}" for k, v in vals.items() if k in _BY_KEY]
            if "form" in vals:
                parts.append(f"{el} potential form = " + (f"g(r) = {vals['user_formula']}" if vals["form"].startswith("U") else vals["form"].split("_")[0]))
    return ", ".join(parts)


def settings_path(config_path: str | Path) -> Path:
    return Path(config_path).parent / "model_settings.json"


def tables_dir(config_path: str | Path) -> Path:
    return Path(config_path).parent / "extrapolation_tables"


def presets_dir(config_path: str | Path) -> Path:
    return Path(config_path).parent / "presets" / "model"


def load(path: str | Path) -> dict[str, dict]:
    try:
        with Path(path).open(encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    clean, _problems = validate(data.get("model", data) if isinstance(data, dict) else {})
    return clean


def save(path: str | Path, model: dict | None) -> None:
    clean, _ = validate(model)
    path = Path(path)
    if not clean:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(str(path), {"model": clean})


def _safe(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", name.strip()).strip("._") or "preset"


def list_presets(folder: str | Path) -> list[str]:
    return sorted(p.stem for p in Path(folder).glob("*.json")) if Path(folder).is_dir() else []


def save_preset(folder: str | Path, name: str, model: dict | None) -> Path:
    clean, _ = validate(model)
    path = Path(folder) / f"{_safe(name)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(str(path), {"name": name.strip(), "model": clean})
    return path


def load_preset(folder: str | Path, name: str) -> dict[str, dict]:
    return load(Path(folder) / f"{_safe(name)}.json")


def delete_preset(folder: str | Path, name: str) -> None:
    (Path(folder) / f"{_safe(name)}.json").unlink(missing_ok=True)
