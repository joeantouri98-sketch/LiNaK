"""Level energies of neutral francium, for the model calculation (ARC has no francium).

Measured NIST levels are used where they exist; above them (and for any missing fine-structure
component) the quantum-defect Ritz expansion fitted to the n >= 9 levels of the same series.
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

from linak.core.paths import reference_path

RYDBERG_EV = 13.605693122994
_L = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}


@lru_cache(maxsize=1)
def _data() -> tuple[dict, dict]:
    with reference_path("data_json", "Fr_rydberg.json").open(encoding="utf-8") as fh:
        ryd = json.load(fh)
    levels = {(1 * e["n"], _L[e["l"]], float(eval(e["J"]))): e["energy_eV"]      # "1/2" -> 0.5
              for e in ryd["excitations"] if e["source"] == "NIST" and e["J"]}
    levels[(7, 0, 0.5)] = -ryd["ionization_energy_eV"]
    ritz = {}
    for l, j in {(k[1], k[2]) for k in levels}:
        pts = [(n, E) for (n, ll, jj), E in levels.items() if ll == l and jj == j and n >= 9]
        if len(pts) < 4:
            continue
        n = np.array([p[0] for p in pts], float)
        d = n - 1.0 / np.sqrt(-np.array([p[1] for p in pts]) / RYDBERG_EV)
        for _ in range(20):                       # delta(n) = d0 + d2 / (n - delta)^2
            A = np.vstack([np.ones_like(n), 1.0 / (n - d) ** 2]).T
            c, *_ = np.linalg.lstsq(A, d, rcond=None)
            d = A @ c
        ritz[(l, j)] = c
    return levels, ritz


class FrAtom:
    """The slice of the ARC atom interface the model calculation uses."""

    def getEnergy(self, n: int, l: int, j: float) -> float:       # noqa: N802 (ARC's name)
        levels, ritz = _data()
        key = (n, l, float(j))
        if key in levels:
            return levels[key]
        c = ritz.get((l, float(j))) if (l, float(j)) in ritz else ritz.get((l, 0.5))
        if c is None:
            raise KeyError(f"no Fr level for n={n} l={l} j={j}")
        return -RYDBERG_EV / (n - (c[0] + c[1] / (n - c[0]) ** 2)) ** 2
