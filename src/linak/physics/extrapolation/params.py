"""Model settings and anchor data for the ns->np extrapolation, read from bundled JSON.

Nothing here computes physics. The packaged files are read-only; a caller may layer
overrides on top (``merge_overrides``) without touching them.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from functools import lru_cache

from linak.core.errors import InvalidSpeciesError
from linak.core.paths import reference_path


class UnsupportedSpeciesError(InvalidSpeciesError):
    """The extrapolation model has no validated parameters for this species."""


def _read(name: str) -> dict:
    with reference_path("extrapolation", name).open(encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def model_params() -> dict:
    return _read("model_params.json")


@lru_cache(maxsize=1)
def anchor_config() -> dict:
    return _read("anchors.json")


def supported_species() -> list[str]:
    return sorted(model_params()["species"])


def species_params(element: str) -> dict:
    """Parameters for ``element`` or a clear refusal (e.g. Fr)."""
    mp = model_params()
    if element in mp["species"]:
        return mp["species"][element]
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
        "form": mp["form"],
        "singor": [sp["singor_alpha_D"], sp["singor_rc"]],
        "lam": sp["lam"],
        "wcore": sp["wcore_eV"],
        "sof": mp["sof"],
        "np": mp["np_cutoff"],
    }
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
