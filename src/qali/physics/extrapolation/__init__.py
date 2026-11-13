"""Physically motivated extrapolation of alkali ns->np oscillator strengths.

Pure functions, no GUI or file-writing; any pipeline stage can call :func:`extrapolate`.
Supported species: see :func:`supported_species` (others are refused with an explanation).
"""

from .api import extrapolate, flag_low_weight, precision_rows
from .params import UnsupportedSpeciesError, params_hash, supported_species
from .tables import OutOfTableRange, table_status

__all__ = [
    "OutOfTableRange",
    "UnsupportedSpeciesError",
    "extrapolate",
    "flag_low_weight",
    "params_hash",
    "precision_rows",
    "supported_species",
    "table_status",
]
