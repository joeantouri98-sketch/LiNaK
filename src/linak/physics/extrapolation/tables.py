"""Precomputed model tables: F = f*nu^3 for ns->np_J on a grid of n, interpolated in s = 1/nu^2.

F is analytic in s, so a cubic spline in s reproduces the exact model to <0.1% (see tests).
Tables carry the parameter hash they were built with; a stale table is never used silently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.interpolate import CubicSpline

from linak.core.errors import MissingDataError
from linak.core.paths import reference_path

from .params import params_hash, species_params

SIGMA_TH_FACTOR_MB = 4.0334  # sigma_th [Mb] = factor * sum_J F_inf


class OutOfTableRange(ValueError):
    """A requested effective quantum number lies outside the table; use the exact backend."""


@dataclass(frozen=True)
class Table:
    element: str
    table_hash: str
    nu_min: float
    nu_max: float
    splines: dict  # J -> CubicSpline in s

    @property
    def fresh(self) -> bool:
        return self.table_hash == params_hash(self.element)

    def contains(self, nu: float) -> bool:
        return self.nu_min * 0.99 <= nu <= self.nu_max * 1.001

    def F(self, nu: float, J: float) -> float:
        if not self.contains(nu):
            raise OutOfTableRange(
                f"{self.element} nu={nu:.3f} outside table range [{self.nu_min:.3f}, {self.nu_max:.3f}]"
            )
        return float(self.splines[J](1.0 / nu**2))

    def f(self, nu: float, J: float) -> float:
        return self.F(nu, J) / nu**3

    def sigma_th_Mb(self) -> float:
        return SIGMA_TH_FACTOR_MB * sum(float(self.splines[J](0.0)) for J in (0.5, 1.5))


@lru_cache(maxsize=16)
def load_table(element: str, variant: str = "") -> Table:
    species_params(element)  # refuses unsupported species before touching files
    path = reference_path("extrapolation", "tables", f"phys_table_{element}{variant}.json")
    if not path.is_file():
        raise MissingDataError(f"No extrapolation table for {element}: {path.name}")
    with path.open(encoding="utf-8") as fh:
        d = json.load(fh)
    splines = {}
    for J in (0.5, 1.5):
        rows = sorted((x for x in d["rows"] if x["J"] == J), key=lambda x: 1 / x["nu"] ** 2)
        s = np.array([1 / x["nu"] ** 2 for x in rows])
        splines[J] = CubicSpline(s, np.array([x["F"] for x in rows]), extrapolate=True)
    nus = [x["nu"] for x in d["rows"]]
    return Table(element, d.get("hash", ""), min(nus), max(nus), splines)


def table_status(element: str) -> dict:
    t = load_table(element)
    return {
        "fresh": t.fresh,
        "table_hash": t.table_hash,
        "current_hash": params_hash(element),
        "nu_min": t.nu_min,
        "nu_max": t.nu_max,
    }
