"""Labels, colours and small helpers shared by the four interactive plots."""

from __future__ import annotations

import math

from qali.core.spectroscopy import SPECTRAL_REGIONS
from qali.physics.rydberg import label_to_pretty, label_to_unicode, unicode_digits_to_plain


def ionization_energy_eV(rd):
    """Ionization energy from the rydberg JSON. ONE fallback for every plot when it is missing
    (0.0 eV, with a warning): the old per-plot defaults (0.0 / 5.139 / 5.0 eV) were inconsistent,
    and 5.139 eV is sodium's value, wrong for every other species."""
    ie = rd.get('ionization_energy_eV') if rd else None
    if ie is None:
        print("  WARNING: no ionization_energy_eV in the rydberg data; using 0.0 eV "
              "(level positions in the diagrams are then relative, not absolute).")
        return 0.0
    return float(ie)


def _display_label(label, pretty_by_label):
    """Readable spectroscopic label (full-size digits: '3s 3P° J=2', '3p1/2').

    *pretty_by_label* maps ASCII keys to the rydberg/transitions JSON pretty labels.
    """
    if not label:
        return label
    # Prefer rydberg/JSON pretty (metals); else derive from ASCII key.
    # Always strip Unicode sub/sup — Plotly hover draws them too small.
    pretty = pretty_by_label.get(label)
    if pretty:
        return unicode_digits_to_plain(pretty)
    return label_to_pretty(label)


def _disp3d(label):
    """Classic Unicode label for 3D Grotrian / orbital viewer: '3s ³P°₂'."""
    if not label:
        return label
    return label_to_unicode(label)


def orca_line_strength(ex):
    """Total dipole strength for a (possibly merged) TD-DFT level."""
    if ex.get("total_oscillator_strength") is not None:
        return ex["total_oscillator_strength"]
    return ex.get("oscillator_strength", 0.0) * ex.get("n_degenerate", 1)


# Region colour map for transition arrows
REGION_COLORS = {
    'X-ray':      '#9400D3',
    'Vacuum UV':  '#8B00FF',
    'UV':         '#4B0082',
    'Visible':    '#FF6600',
    'Near-IR':    '#CC0000',
    'Mid/Far-IR': '#888888',
    'Microwave':  '#AAAAAA',
}

# Single source for the spectral-region display order, used by the Python
# sampling code and embedded into all three interactive panels via
# _region_order_js().
_REGION_ORDER = list(SPECTRAL_REGIONS)


def _region_order_js() -> str:
    """The panel-JS literal for the spectral-region display order."""
    return "[" + ",".join(f"'{r}'" for r in _REGION_ORDER) + "]"


def _fnv_hash(s: str) -> int:
    h = 2166136261
    for c in s.encode('utf-8'):
        h ^= c
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def _fmt_f(f):
    """Oscillator strength with 4 significant digits (scientific below 1e-3, so small f is never shown as 0)."""
    if f >= 1e-3:
        return f"{f:.4g}"
    places = max(0, 4 - math.floor(math.log10(f))) if f > 0 else 0
    return f"{f:.3e} = {f:.{places}f}"
