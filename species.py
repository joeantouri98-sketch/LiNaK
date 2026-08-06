"""
species.py  -  neutral atoms and registered ions (El_cN / El_aN)
------------------------------------------------------------------
Neutrals come from element_data.ELEMENTS. Ions are an allow-list only.

Naming:
  Na       neutral sodium
  Na_c1    Na+  (cation, |charge|=1)
  Na_a1    Na-  (anion,  |charge|=1)
  Mg_c1    Mg+  (alkali-like doublet)

SOC class hints (orca_templates.ElementClass strings):
  ALKALI_VALENCE   - alkali-like cations (ns1, mult=2)
  NOBLE            - closed-shell cations / halide anions (mult=1)
  ALKALINE_EARTH   - ns2 anions e.g. Na_a1 (mult=1)
"""

from __future__ import annotations

import re

from element_data import ELEMENTS

_SYMBOL_TO_Z = {sym: z for z, (sym, _c, _m) in ELEMENTS.items()}
_ION_RE = re.compile(r"^([A-Z][a-z]?)_([ca])(\d+)$")

# soc_class: ElementClass name string used by orca_templates
_ION_REGISTRY = {
    # Alkali-like cations (2S, mult=2) -> ALKALI_VALENCE CAS(1,n)
    "Be_c1": {"symbol": "Be", "charge": 1, "multiplicity": 2, "soc_class": "ALKALI_VALENCE"},
    "Mg_c1": {"symbol": "Mg", "charge": 1, "multiplicity": 2, "soc_class": "ALKALI_VALENCE"},
    "Ca_c1": {"symbol": "Ca", "charge": 1, "multiplicity": 2, "soc_class": "ALKALI_VALENCE"},
    "Sr_c1": {"symbol": "Sr", "charge": 1, "multiplicity": 2, "soc_class": "ALKALI_VALENCE"},
    "Ba_c1": {"symbol": "Ba", "charge": 1, "multiplicity": 2, "soc_class": "ALKALI_VALENCE"},
    # Closed-shell alkali cations (1S, mult=1) -> NOBLE multiplet
    "Li_c1": {"symbol": "Li", "charge": 1, "multiplicity": 1, "soc_class": "NOBLE"},
    "Na_c1": {"symbol": "Na", "charge": 1, "multiplicity": 1, "soc_class": "NOBLE"},
    "K_c1":  {"symbol": "K",  "charge": 1, "multiplicity": 1, "soc_class": "NOBLE"},
    "Rb_c1": {"symbol": "Rb", "charge": 1, "multiplicity": 1, "soc_class": "NOBLE"},
    "Cs_c1": {"symbol": "Cs", "charge": 1, "multiplicity": 1, "soc_class": "NOBLE"},
    # Halide anions (closed shell) -> NOBLE
    "F_a1":  {"symbol": "F",  "charge": -1, "multiplicity": 1, "soc_class": "NOBLE"},
    "Cl_a1": {"symbol": "Cl", "charge": -1, "multiplicity": 1, "soc_class": "NOBLE"},
    "Br_a1": {"symbol": "Br", "charge": -1, "multiplicity": 1, "soc_class": "NOBLE"},
    "I_a1":  {"symbol": "I",  "charge": -1, "multiplicity": 1, "soc_class": "NOBLE"},
    # Alkali anions (ns2) -> ALKALINE_EARTH-like CAS(2,n)
    "Li_a1": {"symbol": "Li", "charge": -1, "multiplicity": 1, "soc_class": "ALKALINE_EARTH"},
    "Na_a1": {"symbol": "Na", "charge": -1, "multiplicity": 1, "soc_class": "ALKALINE_EARTH"},
    "K_a1":  {"symbol": "K",  "charge": -1, "multiplicity": 1, "soc_class": "ALKALINE_EARTH"},
}


def _pretty_label(symbol: str, charge: int) -> str:
    if charge == 0:
        return symbol
    if charge == 1:
        return f"{symbol}+"
    if charge == -1:
        return f"{symbol}-"
    if charge > 1:
        return f"{symbol}{charge}+"
    return f"{symbol}{abs(charge)}-"


def parse_species_id(raw: str):
    """
    Parse a species id or bare symbol.
    Returns (species_id, symbol, charge_sign_char_or_None, charge_mag_or_None)
    for ions, or (symbol, symbol, None, None) for neutrals.
    Does not validate registry membership.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        raise ValueError("Empty species id")
    m = _ION_RE.match(text)
    if m:
        sym, kind, mag_s = m.group(1), m.group(2), m.group(3)
        return text, sym, kind, int(mag_s)
    # bare symbol (case-normalize known symbols)
    if len(text) <= 3:
        cand = text[0].upper() + text[1:].lower() if len(text) > 1 else text.upper()
        if cand in _SYMBOL_TO_Z:
            return cand, cand, None, None
    if text.isdigit():
        z = int(text)
        if z in ELEMENTS:
            sym = ELEMENTS[z][0]
            return sym, sym, None, None
    raise ValueError(
        f"Invalid species id {raw!r}. Use a symbol (Na), Z, or ion stem "
        f"(Mg_c1 cation, Na_a1 anion).")


def is_ion_id(species_id: str) -> bool:
    return bool(_ION_RE.match(str(species_id).strip()))


def listed_ion_ids():
    return sorted(_ION_REGISTRY.keys())


def resolve_species(raw: str) -> dict:
    """
    Resolve CLI target to a species record.

    Keys: species_id, symbol, Z, charge, multiplicity, label, kind,
          soc_class (optional str for ions).
    """
    species_id, symbol, kind_char, mag = parse_species_id(raw)

    if kind_char is None:
        # neutral
        z = _SYMBOL_TO_Z[symbol]
        _sym, charge, mult = ELEMENTS[z]
        return {
            "species_id": species_id,
            "symbol": symbol,
            "Z": z,
            "charge": charge,
            "multiplicity": mult,
            "label": symbol,
            "kind": "neutral",
            "soc_class": None,
        }

    if species_id not in _ION_REGISTRY:
        raise ValueError(
            f"Ion {species_id!r} is not in the allow-list. "
            f"Registered ions: {', '.join(listed_ion_ids())}")

    rec = _ION_REGISTRY[species_id]
    if rec["symbol"] != symbol:
        raise ValueError(f"Internal registry mismatch for {species_id}")
    sign = 1 if kind_char == "c" else -1
    expect = sign * mag
    if rec["charge"] != expect:
        raise ValueError(
            f"Ion stem {species_id} charge mismatch with registry "
            f"(stem implies {expect}, registry has {rec['charge']})")
    if symbol not in _SYMBOL_TO_Z:
        raise ValueError(f"Unknown element symbol {symbol} in ion {species_id}")

    return {
        "species_id": species_id,
        "symbol": symbol,
        "Z": _SYMBOL_TO_Z[symbol],
        "charge": rec["charge"],
        "multiplicity": rec["multiplicity"],
        "label": _pretty_label(symbol, rec["charge"]),
        "kind": "ion",
        "soc_class": rec["soc_class"],
    }


def require_neutral_species(raw: str, context: str = "this script") -> dict:
    """Resolve; ions unsupported as of yet for Tier-C (alkali-physics) tools."""
    sp = resolve_species(raw)
    if sp["kind"] == "ion":
        raise SystemExit(
            f"{context}: ions unsupported as of yet "
            f"({sp['species_id']} / {sp['label']}). "
            f"Use a neutral symbol (e.g. Na), or runorca/plotinteractive for "
            f"ion structure/plots. Future plans may include support.")
    return sp


def species_json_stem(species_id: str, suffix: str = None) -> str:
    """Build data_json basename without .json: Mg_c1, Mg_c1_soc, Mg_c1_UKS."""
    if not suffix:
        return species_id
    suffix = suffix.strip().strip("_")
    if suffix.lower().endswith(".json"):
        suffix = suffix[:-5]
    return f"{species_id}_{suffix}" if suffix else species_id
