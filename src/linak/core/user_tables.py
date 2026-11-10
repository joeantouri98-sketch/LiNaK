"""User overrides for LiNaK's built-in numbers (Preferences > Tables and variables).

Two kinds, both stored under ``<runtime root>/tables`` of the workspace and applied on top of the packaged values:

* constants written in the code (``ALPHA_CORE_AU`` and the like): ``overrides.json`` holds ``{table id: {key: value}}``;
  each module calls :func:`apply` right after defining its dict, so a stage subprocess sees them too;
* rows the user adds to the open lists (a new element, a new lifetime): ``additions.json`` holds
  ``{table id: {key: {"value": x, "source": "text"}}}``; they are put in the same dict, never over a packaged key;
* packaged data files (NIST lines and levels, moments, extrapolation anchors): an edited copy at
  ``files/<tree>/<name>``, which :func:`linak.core.paths.reference_path` returns in place of the original.

Nothing here writes to the packaged files. Importing this module reads nothing.
"""

from __future__ import annotations

import copy
import importlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

OVERRIDES_FILE = "overrides.json"
ADDITIONS_FILE = "additions.json"
FLAGS_FILE = "flags.json"


@dataclass(frozen=True)
class TableInfo:
    table_id: str
    group: str
    title: str
    module: str
    attr: str
    unit: str = ""


# One entry per editable constant table; the module calls apply(table_id, <dict>) after defining it.
TABLES = (
    TableInfo("polarizability.alpha_core", "Polarizability", "Ionic core polarizability", "linak.physics.polarizability_core", "ALPHA_CORE_AU", "a.u."),
    TableInfo("polarizability.core_electrons", "Polarizability", "Core electrons", "linak.physics.polarizability_core", "CORE_ELECTRONS", "electrons"),
    TableInfo("polarizability.precision_fosc", "Polarizability", "Precision D-line f-values", "linak.physics.polarizability_core", "PRECISION_FOSC_J"),
    TableInfo("constants.alpha0_expt", "Polarizability", "Experimental static polarizability", "linak.core.constants", "ALPHA0_EXPT", "a.u."),
    TableInfo("constants.c6_expt", "Polarizability", "Experimental C6", "linak.core.constants", "C6_EXPT", "a.u."),
    TableInfo("uncertainty.accuracy_grade", "Uncertainty model", "Relative uncertainty by NIST accuracy grade", "linak.physics.polarizability_core", "ACC_UNCERTAINTY", "relative"),
    TableInfo("uncertainty.source", "Uncertainty model", "Relative uncertainty by source", "linak.physics.polarizability_core", "SOURCE_REL_UNCERTAINTY", "relative"),
    TableInfo("blackbody.qd_diff", "Blackbody", "Quantum-defect differences", "linak.physics.blackbody_core", "QD_DIFF"),
    TableInfo("blackbody.a_l", "Blackbody", "A_L scaling", "linak.physics.blackbody_core", "A_L_TABLE"),
    TableInfo("lifetimes.known", "Lifetimes", "Known lifetimes", "linak.pipelines.lifetimes", "KNOWN_LIFETIMES", "ns"),
    TableInfo("nist.fscale", "Lifetimes", "NIST f-value scale (decays of a level)", "linak.physics.polarizability_core", "NIST_F_SCALE"),
    TableInfo("scattering.fallback", "Scattering", "D-line parameters (fallback)", "linak.pipelines.scattering_rate", "FALLBACK"),
    TableInfo("tweezer.default_wavelength", "Tweezer", "Default wavelength", "linak.pipelines.tweezer", "DEFAULT_WL_NM", "nm"),
    TableInfo("constants.alkali_mass", "Species constants", "Atomic mass", "linak.core.constants", "ALKALI_MASS_AMU", "u"),
    TableInfo("constants.abundance", "Species constants", "Natural abundance", "linak.core.constants", "NATURAL_ABUNDANCE_PCT", "%"),
    TableInfo("spectroscopy.dline", "Species constants", "D-line literature values", "linak.core.spectroscopy", "LITERATURE"),
)
BY_ID = {t.table_id: t for t in TABLES}


@dataclass(frozen=True)
class AddSpec:
    """How a row of a table is named, and what its value may be."""

    kind: str                      # "element" | "element_state" | "isotope" | "element_l" | "precision" | "record"
    integer: bool = False
    max_value: float | None = None
    allow_zero: bool = False
    letters: str = "spdfgh"        # orbital letters accepted by "element_l"


# Tables the user may add rows to. The ones that are lists of measured or tabulated numbers per element.
ADDABLE = {
    "polarizability.alpha_core": AddSpec("element"),
    "polarizability.core_electrons": AddSpec("element", integer=True),
    "constants.alpha0_expt": AddSpec("element"),
    "constants.c6_expt": AddSpec("element"),
    "constants.alkali_mass": AddSpec("element"),
    "constants.abundance": AddSpec("isotope", max_value=100.0),
    "tweezer.default_wavelength": AddSpec("element"),
    "lifetimes.known": AddSpec("element_state"),
    "blackbody.a_l": AddSpec("element_l"),                                  # any l: beterov_A_L looks the letter up
    "blackbody.qd_diff": AddSpec("element_l", allow_zero=True, letters="spd"),   # the model reads only s, p and d
    "polarizability.precision_fosc": AddSpec("precision"),
    "spectroscopy.dline": AddSpec("record"),                                # a whole new element
}
_STATE_RE = re.compile(r"^[1-9]\d?[spdfgh]$")
_J_RE = re.compile(r"^\d{1,2}/2$")

# Tables whose rows have different units: by the last part of the key.
_SCATTERING_UNITS = {"mass": "u", "wl": "nm", "A": "s\u207b\u00b9", "Isat": "mW/cm\u00b2", "tau": "ns", "vrec": "cm/s", "Erec": "nK"}


def unit(table_id: str, key: str) -> str:
    """Unit of one row; empty for a dimensionless number."""
    last = key.rsplit(" / ", 1)[-1]
    if table_id == "scattering.fallback":
        return _SCATTERING_UNITS.get(last, "")
    if table_id == "spectroscopy.dline":
        if last == "mass_amu":
            return "u"
        if last.endswith("_nm"):
            return "nm"
        if last.endswith("_eV"):
            return "eV"
        if last.endswith("_meV"):
            return "meV"
        return "ns" if last.startswith("tau_") and last.endswith("_ns") else ""
    return BY_ID[table_id].unit

_DEFAULTS: dict[str, dict] = {}        # table id -> {flat key: default}, recorded the first time apply() sees the table


# --- keys ------------------------------------------------------------------
def _part(key) -> str:
    return "-".join(str(k) for k in key) if isinstance(key, tuple) else str(key)


def flatten(table: dict) -> dict[str, tuple[list, float]]:
    """``{"Fr / 7s-7p / 1/2": (path, value)}`` for every numeric leaf of a (nested) dict."""
    out: dict[str, tuple[list, float]] = {}

    def walk(node, path):
        for k, v in node.items():
            if isinstance(v, dict):
                walk(v, path + [k])
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                out[" / ".join(_part(p) for p in path + [k])] = (path + [k], v)

    walk(table, [])
    return out


# --- where things live -------------------------------------------------------
def tables_dir() -> Path:
    """``<runtime root>/tables`` of the workspace in effect; nothing is created or read."""
    from linak.core import workspace
    from linak.core.paths import PROJECT_ROOT_ENV, runtime_root_for, user_data_root

    ws = workspace.active()
    if ws is not None:
        root = ws.root
    elif os.environ.get(PROJECT_ROOT_ENV):
        root = Path(os.environ[PROJECT_ROOT_ENV]).expanduser().resolve()
    else:
        root = user_data_root(create=False)
    return runtime_root_for(root) / "tables"


def overrides_path() -> Path:
    return tables_dir() / OVERRIDES_FILE


def load_overrides() -> dict[str, dict]:
    try:
        data = json.loads(overrides_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict)} if isinstance(data, dict) else {}


def save_overrides(data: dict[str, dict]) -> None:
    from linak.core.fileio import write_json

    clean = {t: {k: v for k, v in rows.items()} for t, rows in data.items() if rows}
    path = overrides_path()
    if clean:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, clean)
    elif path.is_file():
        path.unlink()


# --- constant tables -----------------------------------------------------------
def _set(table: dict, path: list, value) -> None:
    node = table
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value


def _set_new(table: dict, path: list, value) -> None:
    node = table
    for k in path[:-1]:
        node = node.setdefault(k, {})
    node[path[-1]] = value


def _drop(table: dict, path: list) -> None:
    """Remove the leaf at *path* and any parent left empty."""
    chain = [table]
    for k in path[:-1]:
        chain.append(chain[-1][k])
    chain[-1].pop(path[-1], None)
    for node, k in zip(reversed(chain[:-1]), reversed(path[:-1])):
        if not node[k]:
            del node[k]


def apply(table_id: str, table: dict) -> None:
    """Record *table*'s defaults, then overwrite the values the user changed. Never raises."""
    try:
        flat = flatten(table)
        _DEFAULTS.setdefault(table_id, {k: v for k, (_p, v) in flat.items()})
        for key, value in load_overrides().get(table_id, {}).items():
            if key in flat and isinstance(value, (int, float)) and not isinstance(value, bool):
                path, default = flat[key]
                _set(table, path, int(value) if isinstance(default, int) and float(value).is_integer() else float(value))
        for key, rec in load_additions().get(table_id, {}).items():
            if key in flat or key in table:
                continue                                  # a packaged row always wins
            try:
                _set_new(table, make_key(table_id, key_parts(table_id, key))[1],
                         copy.deepcopy(rec["record"]) if "record" in rec else rec["value"])
            except (ValueError, KeyError, TypeError):
                continue
    except Exception:
        pass


def defaults(table_id: str) -> dict:
    """``{flat key: packaged value}`` of a table (imports its module on first use)."""
    table = live_table(table_id)
    return dict(_DEFAULTS.get(table_id) or {k: v for k, (_p, v) in flatten(table).items()})


def live_table(table_id: str) -> dict:
    info = BY_ID[table_id]
    return getattr(importlib.import_module(info.module), info.attr)


def reload(table_id: str) -> None:
    """Put the module's dict back to the packaged values, then apply the saved overrides (the GUI calls this after Save)."""
    table = live_table(table_id)
    flat = flatten(table)
    base = _DEFAULTS.get(table_id)
    if base is not None:
        for key, (path, _v) in flat.items():
            if key in base:
                _set(table, path, base[key])
            else:
                _drop(table, path)                       # a row the user added
        if ADDABLE.get(table_id) and ADDABLE[table_id].kind == "record":
            tops = {k.split(" / ")[0] for k in base}
            for el in [e for e in table if e not in tops]:
                del table[el]                             # a whole element the user added
    apply(table_id, table)


def values(table_id: str) -> list[tuple[str, float, float]]:
    """``[(key, packaged value, value in use)]`` in table order."""
    table = live_table(table_id)
    base = defaults(table_id)
    return [(k, base.get(k, v), v) for k, (_p, v) in flatten(table).items()]


def packaged_rows(table_id: str) -> list[tuple[str, float, float]]:
    """:func:`values` without the rows the user added."""
    base = defaults(table_id)
    return [r for r in values(table_id) if r[0] in base]


# --- added rows ----------------------------------------------------------------
def additions_path() -> Path:
    return tables_dir() / ADDITIONS_FILE


def load_additions() -> dict[str, dict]:
    try:
        data = json.loads(additions_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict[str, dict] = {}
    for tid, rows in (data.items() if isinstance(data, dict) else []):
        if tid in ADDABLE and isinstance(rows, dict):
            good = {k: r for k, r in rows.items() if isinstance(r, dict)
                    and (isinstance(r.get("value"), (int, float)) or isinstance(r.get("record"), dict))}
            if good:
                out[tid] = good
    return out


def save_additions(data: dict[str, dict]) -> None:
    from linak.core.fileio import write_json

    clean = {t: rows for t, rows in data.items() if rows}
    path = additions_path()
    if clean:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, clean)
    elif path.is_file():
        path.unlink()


def key_fields(table_id: str) -> tuple[str, ...]:
    """Labels of the boxes that name a new row of *table_id*."""
    return {"element": ("Element",), "element_state": ("Element", "State"), "isotope": ("Element", "Mass number"),
            "element_l": ("Element", "Orbital (" + " ".join(ADDABLE[table_id].letters) + ")"),
            "precision": ("Element", "Lower state", "Upper state", "J (e.g. 1/2)"), "record": ("Element",)}[ADDABLE[table_id].kind]


def key_parts(table_id: str, key: str) -> list[str]:
    """Inverse of the flat key: ``"Mg / 3p"`` -> ``["Mg", "3p"]``, ``"Mg-24"`` -> ``["Mg", "24"]``."""
    kind = ADDABLE[table_id].kind
    if kind == "isotope":
        return key.split("-")
    parts = [p.strip() for p in key.split(" / ")]
    if kind == "precision" and len(parts) == 3:
        return [parts[0], *parts[1].split("-"), parts[2]]
    return parts


def make_key(table_id: str, parts: list[str]) -> tuple[str, list]:
    """``(flat key, path in the dict)`` for what the user typed; raises ValueError with a message for the user."""
    from linak.core.elements import ELEMENTS

    spec = ADDABLE[table_id]
    parts = [str(p).strip() for p in parts]
    if len(parts) != len(key_fields(table_id)) or not all(parts):
        raise ValueError("Fill in " + " and ".join(key_fields(table_id)).lower() + ".")
    el = parts[0][:1].upper() + parts[0][1:].lower()
    if el not in {sym for sym, _c, _m in ELEMENTS.values()}:
        raise ValueError(f"'{parts[0]}' is not an element symbol.")
    if spec.kind == "element":
        return el, [el]
    if spec.kind == "record":
        return el, [el]
    if spec.kind == "element_l":
        if parts[1].lower() not in spec.letters:
            raise ValueError("Orbital is one of " + ", ".join(spec.letters) + ".")
        return f"{el} / {parts[1].lower()}", [el, parts[1].lower()]
    if spec.kind == "precision":
        lo, up, j = parts[1].lower(), parts[2].lower(), parts[3]
        if not (_STATE_RE.match(lo) and _STATE_RE.match(up)):
            raise ValueError("States look like 7s and 7p.")
        if not _J_RE.match(j):
            raise ValueError("J looks like 1/2 or 3/2.")
        return f"{el} / {lo}-{up} / {j}", [el, (lo, up), j]
    if spec.kind == "element_state":
        state = parts[1].lower()
        if not _STATE_RE.match(state):
            raise ValueError("State looks like 3p, 4d or 10s.")
        return f"{el} / {state}", [el, state]
    try:
        a = int(parts[1])
    except ValueError:
        raise ValueError("Mass number is a whole number, such as 24.") from None
    if not 1 <= a <= 300:
        raise ValueError("Mass number must be between 1 and 300.")
    return f"{el}-{a}", [(el, a)]


def check_value(table_id: str, text) -> float | int:
    spec = ADDABLE[table_id]
    try:
        v = float(str(text).replace(",", "."))
    except ValueError:
        raise ValueError("Value must be a number.") from None
    if not math.isfinite(v) or not (v >= 0 if spec.allow_zero else v > 0):
        raise ValueError("Value must be " + ("zero or more." if spec.allow_zero else "greater than zero."))
    if spec.max_value is not None and v > spec.max_value:
        raise ValueError(f"Value must be at most {spec.max_value:g}.")
    if spec.integer:
        if not float(v).is_integer():
            raise ValueError("Value must be a whole number.")
        return int(v)
    return v


# --- packaged data files ------------------------------------------------------
def file_override(parts: tuple[str, ...]) -> Path | None:
    """The user's edited copy of resource ``parts`` if it exists."""
    path = tables_dir().joinpath("files", *parts)
    return path if path.is_file() else None


def override_file_path(parts: tuple[str, ...]) -> Path:
    return tables_dir().joinpath("files", *parts)


def revert_file(parts: tuple[str, ...]) -> bool:
    path = file_override(parts)
    if path is None:
        return False
    path.unlink()
    return True


def snapshot_defaults() -> dict[str, dict]:
    """Deep copy of the recorded packaged values (tests)."""
    return copy.deepcopy(_DEFAULTS)


# --- whole-element records (D-line literature) -----------------------------------
def record_template(table_id: str) -> dict:
    """The first element's record: the fields, their order and their types."""
    table = live_table(table_id)
    return copy.deepcopy(table[next(iter(defaults_top(table_id)), next(iter(table)))])


def defaults_top(table_id: str) -> list[str]:
    seen: list[str] = []
    for k in defaults(table_id):
        top = k.split(" / ")[0]
        if top not in seen:
            seen.append(top)
    return seen


def record_fields(table_id: str) -> list[tuple[str, str]]:
    """``[(field name, "int" | "float" | "str")]`` of a record, nested names joined by " / "."""
    out: list[tuple[str, str]] = []

    def walk(node, path):
        for k, v in node.items():
            if isinstance(v, dict):
                walk(v, path + [k])
            else:
                kind = "str" if isinstance(v, str) else "int" if isinstance(v, int) and not isinstance(v, bool) else "float"
                out.append((" / ".join(path + [k]), kind))

    walk(record_template(table_id), [])
    return out


def check_record(table_id: str, element: str, texts: dict[str, str]) -> dict:
    """The record for *element* from what the user typed; raises ValueError naming the field that is wrong."""
    from linak.core.elements import atomic_number

    rec: dict = {}
    for name, kind in record_fields(table_id):
        raw = str(texts.get(name, "")).strip()
        if not raw:
            raise ValueError(f"Fill in {name}.")
        value: str | float | int
        if kind == "str":
            value = raw
        else:
            try:
                number = float(raw.replace(",", "."))
            except ValueError:
                raise ValueError(f"{name} must be a number.") from None
            if not 0 <= number < float("inf"):
                raise ValueError(f"{name} must be zero or more.")
            value = number
            if kind == "int":
                if not number.is_integer():
                    raise ValueError(f"{name} must be a whole number.")
                value = int(number)
        node = rec
        *path, last = name.split(" / ")
        for k in path:
            node = node.setdefault(k, {})
        node[last] = value
    z = atomic_number(element)
    if "Z" in rec and z is not None and rec["Z"] != z:
        raise ValueError(f"Z for {element} is {z}.")
    return rec


def check_record_value(old, text) -> float | int:
    """A new number for one field of an added record, of the same type (whole or not) as the old one."""
    try:
        v = float(str(text).replace(",", "."))
    except ValueError:
        raise ValueError("Value must be a number.") from None
    if not 0 <= v < float("inf"):
        raise ValueError("Value must be zero or more.")
    if isinstance(old, int) and not isinstance(old, bool):
        if not v.is_integer():
            raise ValueError("Value must be a whole number.")
        return int(v)
    return v


# --- one element's slice ---------------------------------------------------------
def element_of(table_id: str, key: str) -> str:
    """The element a flat key belongs to (``"Mg / 3p"`` and ``"Mg-24"`` -> ``"Mg"``); keys of other tables give their first part."""
    spec = ADDABLE.get(table_id)
    return key.split("-")[0] if spec is not None and spec.kind == "isotope" else key.split(" / ")[0]


def element_rows(table_id: str, element: str) -> list[tuple[str, float, float]]:
    """:func:`values` of one element."""
    return [r for r in values(table_id) if element_of(table_id, r[0]) == element]


def element_file_pattern(element: str) -> re.Pattern:
    """Matches the packaged file names that belong to *element*: ``Na.csv``, ``Na_lines.csv``, ``Na_literature.json``."""
    return re.compile(rf"^{re.escape(element)}([._]|$)")


# --- "use instead of the computed value" flags ------------------------------------------
FLAGGABLE = {"lifetimes.known", "nist.fscale"}          # tables whose rows can be ticked to replace what the stage computes
# Rows that start ticked; flags.json then records an explicit False when the user unticks one.
DEFAULT_FLAGS = {"nist.fscale": ("Fr / 7d3/2", "Fr / 7d5/2")}


def flags_path() -> Path:
    return tables_dir() / FLAGS_FILE


def load_flags() -> dict[str, dict[str, bool]]:
    """The ticked rows per table: the default-ticked rows, changed by what flags.json says."""
    try:
        data = json.loads(flags_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    out = {}
    for tid in FLAGGABLE:
        state = {k: True for k in DEFAULT_FLAGS.get(tid, ())}
        rows = data.get(tid) if isinstance(data, dict) else None
        for k, v in (rows.items() if isinstance(rows, dict) else []):
            if v is True:
                state[k] = True
            elif v is False:
                state.pop(k, None)
        if state:
            out[tid] = state
    return out


def save_flags(data: dict[str, dict[str, bool]]) -> None:
    from linak.core.fileio import write_json

    clean: dict[str, dict[str, bool]] = {}
    for tid in FLAGGABLE:
        ticked = {k for k, v in data.get(tid, {}).items() if v}
        defaults = set(DEFAULT_FLAGS.get(tid, ()))
        rows = {k: True for k in ticked - defaults}
        rows.update({k: False for k in defaults - ticked})
        if rows:
            clean[tid] = rows
    path = flags_path()
    if clean:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, clean)
    elif path.is_file():
        path.unlink()


def lifetime_overrides(table: dict, element: str) -> dict[str, float]:
    """``{state: lifetime in ns}`` of the ticked rows of *element* in the known-lifetimes *table*."""
    return ticked_values("lifetimes.known", table, element)


def ticked_values(table_id: str, table: dict, element: str) -> dict[str, float]:
    """``{state: value}`` of the ticked rows of *element* in a flaggable *table* (``{element: {state: value}}``)."""
    out: dict[str, float] = {}
    for key in load_flags().get(table_id, {}):
        parts = key.split(" / ")
        if len(parts) == 2 and parts[0] == element:
            value = table.get(element, {}).get(parts[1])
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
                out[parts[1]] = float(value)
    return out
