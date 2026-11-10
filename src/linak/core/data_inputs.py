"""Choosing which upstream data file a stage reads (``--lifetimes-file`` and the like).

By default a stage reads the usual ``data/<species>_<kind>.json``. A run saved under a name (``Na_lifetimes__hot.json``)
or any other file of the same species can be chosen instead, so results built from different inputs stay independent.
"""

from __future__ import annotations

import os
from pathlib import Path

from linak.core.errors import MissingDataError
from linak.core.workspace import stem_belongs_to_species


def resolve_data_file(spec: str | None, element: str, data_dir: str | os.PathLike, what: str) -> str | None:
    """Path of the chosen file, or None when *spec* is blank (use the default file).

    *spec* may be a path, or a name under *data_dir* with or without ``.json`` and without the ``<species>_`` prefix.
    The file must belong to *element*: a Na file cannot feed a K run.
    """
    spec = (spec or "").strip()
    if not spec:
        return None
    candidates = [spec, os.path.join(data_dir, spec)]
    if not spec.lower().endswith(".json"):
        candidates += [os.path.join(data_dir, f"{spec}.json"), os.path.join(data_dir, f"{element}_{spec}.json")]
    for path in candidates:
        if os.path.isfile(path):
            if not stem_belongs_to_species(element, Path(path).stem):
                raise MissingDataError(f"{what} file '{Path(path).name}' is not a {element} file")
            return os.path.normpath(path)
    raise MissingDataError(f"{what} file '{spec}' not found for {element} (looked in {data_dir})")
