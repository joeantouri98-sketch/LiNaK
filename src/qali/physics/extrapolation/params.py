"""Model settings and anchor data for the ns->np extrapolation, read from bundled JSON.

Nothing here computes physics. The packaged files are read-only; a caller may layer
overrides on top (``merge_overrides``) without touching them.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from copy import deepcopy
from functools import lru_cache

from qali.core.errors import InvalidSpeciesError
from qali.core.paths import reference_path


class UnsupportedSpeciesError(InvalidSpeciesError):
    """The extrapolation model has no validated parameters for this species."""


def _read(name: str) -> dict:
    with reference_path("extrapolation", name).open(encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def model_params() -> dict:
    return _read("model_params.json")


@lru_cache(maxsize=1)
def _anchor_config_base() -> dict:
    return _read("anchors.json")


_ACTIVE: dict | None = None      # merged anchor config while ``fit_overrides`` is active


def anchor_config() -> dict:
    """The packaged anchor settings, with the user's fit settings on top while ``fit_overrides`` is active."""
    return _ACTIVE if _ACTIVE is not None else _anchor_config_base()


@contextmanager
def fit_overrides(nested: dict | None):
    """Use the given nested overrides (see ``fit_settings.to_overrides``) for everything inside the block."""
    global _ACTIVE
    previous = _ACTIVE
    _ACTIVE = merge_overrides(_anchor_config_base(), nested) if nested else previous
    try:
        yield
    finally:
        _ACTIVE = previous


def supported_species() -> list[str]:
    return sorted(model_params()["species"])


_MODEL_ACTIVE: dict | None = None    # {species: {field: value}} while ``model_overrides`` is active
_USER_DIR = None                     # where rebuilt tables and the exact-calculation cache live


def user_dir():
    """Folder for rebuilt tables (None: only the packaged tables)."""
    return _USER_DIR


@contextmanager
def model_overrides(nested: dict | None, tables_dir=None):
    """Use the user's model settings (``{species: {field: value}}``) and rebuilt tables inside the block."""
    global _MODEL_ACTIVE, _USER_DIR
    previous = (_MODEL_ACTIVE, _USER_DIR)
    _MODEL_ACTIVE = {k: dict(v) for k, v in (nested or {}).items() if v} or None
    _USER_DIR = tables_dir
    try:
        yield
    finally:
        _MODEL_ACTIVE, _USER_DIR = previous


def packaged_species_params(element: str) -> dict:
    """The shipped settings of ``element`` (what the packaged tables were built with)."""
    mp = model_params()
    if element in mp["species"]:
        return mp["species"][element]
    species_params(element)  # raises the refusal
    raise AssertionError("unreachable")


def species_params(element: str) -> dict:
    """Parameters for ``element`` or a clear refusal (e.g. Ra); the user's model settings are applied when active."""
    mp = model_params()
    if element in mp["species"]:
        base = mp["species"][element]
        ov = (_MODEL_ACTIVE or {}).get(element)
        return {**base, **ov} if ov else base
    why = mp.get("unsupported", {}).get(element)
    if why:
        raise UnsupportedSpeciesError(f"Extrapolation is not available for {element}: {why}")
    raise UnsupportedSpeciesError(
        f"Extrapolation is not available for {element!r}; supported: {', '.join(supported_species())}"
    )


def params_hash(element: str) -> str:
    """Fingerprint of every setting that changes the model f-values of ``element``.

    Same recipe as the prototype so tables built there stay valid.
    """
    mp = model_params()
    sp = species_params(element)
    d = {
        "form": sp.get("form") or mp["form"],
        "singor": [sp["singor_alpha_D"], sp["singor_rc"]],
        "lam": sp["lam"],
        "wcore": sp["wcore_eV"],
        "sof": mp["sof"],
        "np": mp["np_cutoff"],
    }
    if str(d["form"]).startswith("U"):          # a user-written form: its formula and parameters define the model
        d["user"] = [sp.get("user_formula", ""), sorted(sp.get("user_params") or [], key=lambda q: q.get("name", ""))]
    base = mp["species"][element]
    if sp["so_scale"] != base["so_scale"]:   # shipped tables never carried it; only an edited value changes the hash
        d["so_scale"] = sp["so_scale"]
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


def merge_overrides(base: dict, overrides: dict | None) -> dict:
    """Return a copy of ``base`` with ``overrides`` merged in (dicts merge, everything else replaces)."""
    out = deepcopy(base)
    for key, val in (overrides or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = merge_overrides(out[key], val)
        else:
            out[key] = deepcopy(val)
    return out
