# -*- coding: utf-8 -*-
"""Unit tests for the species registry in ``linak.core.species``."""
from __future__ import annotations

import pytest

from linak.core.species import (
    is_ion_id,
    listed_ion_ids,
    parse_species_id,
    require_neutral_species,
    resolve_species,
    species_json_stem,
)


# --- neutrals ---------------------------------------------------------------


def test_resolve_species_neutral_sodium() -> None:
    sp = resolve_species("Na")
    assert sp["species_id"] == "Na"
    assert sp["symbol"] == "Na"
    assert sp["Z"] == 11
    assert sp["charge"] == 0
    assert sp["multiplicity"] == 2
    assert sp["label"] == "Na"
    assert sp["kind"] == "neutral"
    assert sp["soc_class"] is None


def test_resolve_species_neutral_rubidium() -> None:
    sp = resolve_species("Rb")
    assert sp["Z"] == 37
    assert sp["kind"] == "neutral"
    assert sp["multiplicity"] == 2


def test_resolve_species_accepts_atomic_number() -> None:
    sp = resolve_species("11")
    assert sp["symbol"] == "Na"
    assert sp["Z"] == 11
    assert sp["kind"] == "neutral"


def test_parse_species_id_normalises_bare_symbol_case() -> None:
    assert parse_species_id("na") == ("Na", "Na", None, None)


# --- ions -------------------------------------------------------------------


def test_resolve_species_alkaline_earth_cation_mg_c1() -> None:
    sp = resolve_species("Mg_c1")
    assert sp["kind"] == "ion"
    assert sp["Z"] == 12
    assert sp["charge"] == 1
    assert sp["multiplicity"] == 2
    assert sp["soc_class"] == "ALKALI_VALENCE"
    assert sp["label"] == "Mg+"


def test_resolve_species_closed_shell_cation_na_c1() -> None:
    sp = resolve_species("Na_c1")
    assert sp["kind"] == "ion"
    assert sp["charge"] == 1
    assert sp["multiplicity"] == 1
    assert sp["soc_class"] == "NOBLE"
    assert sp["label"] == "Na+"


def test_resolve_species_alkali_anion_na_a1() -> None:
    sp = resolve_species("Na_a1")
    assert sp["kind"] == "ion"
    assert sp["charge"] == -1
    assert sp["multiplicity"] == 1
    assert sp["soc_class"] == "ALKALINE_EARTH"
    assert sp["label"] == "Na-"


def test_resolve_species_halide_anion_f_a1() -> None:
    sp = resolve_species("F_a1")
    assert sp["kind"] == "ion"
    assert sp["charge"] == -1
    assert sp["soc_class"] == "NOBLE"
    assert sp["label"] == "F-"


def test_listed_ion_ids_are_sorted() -> None:
    ids = listed_ion_ids()
    assert ids == sorted(ids)
    assert "Mg_c1" in ids
    assert "Na_a1" in ids


def test_is_ion_id() -> None:
    assert is_ion_id("Mg_c1")
    assert is_ion_id("Na_a1")
    assert not is_ion_id("Na")
    assert not is_ion_id("Mg_cX")


# --- error branches ---------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        # Registered stem, but no such entry in the allow-list.
        ("Mg_a1", "is not in the allow-list"),
        ("Na_c2", "is not in the allow-list"),
        ("Xx_c1", "is not in the allow-list"),
    ],
)
def test_resolve_species_rejects_unregistered_ions(raw: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        resolve_species(raw)


@pytest.mark.parametrize(
    "raw",
    ["", "  ", "Na_x1", "Na_cc1", "na_c1", "3Na", "Na_"],
)
def test_resolve_species_rejects_malformed_ids(raw: str) -> None:
    with pytest.raises(ValueError):
        resolve_species(raw)


def test_parse_species_id_empty_string_message() -> None:
    with pytest.raises(ValueError, match="Empty species id"):
        parse_species_id("")


def test_parse_species_id_invalid_message() -> None:
    with pytest.raises(ValueError, match="Invalid species id 'Na_x1'"):
        parse_species_id("Na_x1")


def test_parse_species_id_none_returns_none() -> None:
    assert parse_species_id(None) is None


# --- require_neutral_species ------------------------------------------------


def test_require_neutral_species_passes_neutral() -> None:
    sp = require_neutral_species("Na", "feshbach.py")
    assert sp["kind"] == "neutral"
    assert sp["species_id"] == "Na"


def test_require_neutral_species_rejects_ion_with_a_typed_error() -> None:
    """Ions are refused with a typed error that ``cli.py`` maps to exit 1."""
    from linak.core.errors import InvalidSpeciesError, PipelineError

    with pytest.raises(InvalidSpeciesError) as excinfo:
        require_neutral_species("Na_c1", "feshbach.py")
    assert isinstance(excinfo.value, PipelineError)
    assert "ions unsupported" in str(excinfo.value)
    assert "Na_c1" in str(excinfo.value)


# --- species_json_stem ------------------------------------------------------


def test_species_json_stem_without_suffix() -> None:
    assert species_json_stem("Mg_c1") == "Mg_c1"


def test_species_json_stem_with_plain_suffix() -> None:
    assert species_json_stem("Mg_c1", "soc") == "Mg_c1_soc"


def test_species_json_stem_strips_json_suffix() -> None:
    assert species_json_stem("Mg_c1", "soc.json") == "Mg_c1_soc"


def test_species_json_stem_strips_underscores() -> None:
    assert species_json_stem("Mg_c1", "_soc_") == "Mg_c1_soc"


def test_species_json_stem_empty_suffix_returns_id() -> None:
    assert species_json_stem("Mg_c1", "") == "Mg_c1"
    assert species_json_stem("Mg_c1", ".json") == "Mg_c1"
