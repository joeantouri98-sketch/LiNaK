"""Shared atomic-spectroscopy data tables (single source of truth).

The bundled literature tables pipelines used to re-declare inline live here.

- ``LITERATURE`` — per-element literature parameter record for the
  cross-element comparison table (consumed by
  ``pipelines.compare_elements``). Edit values here only.
- ``dline_transitions`` — the D lines of an element as labelled records,
  built from ``LITERATURE`` (never a second copy of numbers).

This module stays import-light: pure data and labels only — no IO, no
argument parsing, no plotting. The typed ``Transition``
record is defined here too: "absent" values are
``None``, never a magic number.
"""

from __future__ import annotations

from qali.core import user_tables as _user_tables
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# literature parameter set per alkali (compare_elements, spectra fallback)
# ─────────────────────────────────────────────────────────────────────────────
# Used when pipeline JSON is not available.
# Sources: Steck D-line data sheets, NIST ASD, Sansonetti 2011.
# fosc_* are J-resolved absorption oscillator strengths (not multiplet totals).

LITERATURE = {
    'Li': {
        'Z': 3, 'mass_amu': 6.941, 'gs': '2s', 'np': '2p',
        'IE_eV': 5.391714,
        'D1_nm': 670.976,   'D2_nm': 670.962,
        'tau_D1_ns': 27.10,  'tau_D2_ns': 27.10,
        'fosc_D1': 0.2490,   'fosc_D2': 0.4980,
        'g_upper_D1': 2, 'g_lower_D1': 2,
        'g_upper_D2': 4, 'g_lower_D2': 2,
        'delta_s': 0.3995,   'delta_p': 0.0472,
        'soc_meV': 0.043,
        'source': 'literature',
    },
    'Na': {
        'Z': 11, 'mass_amu': 22.990, 'gs': '3s', 'np': '3p',
        'IE_eV': 5.139076,
        'D1_nm': 589.7558, 'D2_nm': 589.1583,
        'tau_D1_ns': 16.299, 'tau_D2_ns': 16.249,
        'fosc_D1': 0.3200,   'fosc_D2': 0.6410,
        'g_upper_D1': 2, 'g_lower_D1': 2,
        'g_upper_D2': 4, 'g_lower_D2': 2,
        'delta_s': 1.3478,   'delta_p': 0.8551,
        'soc_meV': 2.19,
        'source': 'literature',
    },
    'K': {
        'Z': 19, 'mass_amu': 39.098, 'gs': '4s', 'np': '4p',
        'IE_eV': 4.340664,
        'D1_nm': 770.1083, 'D2_nm': 766.7000,
        'tau_D1_ns': 26.72,  'tau_D2_ns': 26.37,
        'fosc_D1': 0.3320,   'fosc_D2': 0.6660,
        'g_upper_D1': 2, 'g_lower_D1': 2,
        'g_upper_D2': 4, 'g_lower_D2': 2,
        'delta_s': 2.1800,   'delta_p': 1.7130,
        'soc_meV': 7.23,
        'source': 'literature',
    },
    'Rb': {
        'Z': 37, 'mass_amu': 85.468, 'gs': '5s', 'np': '5p',
        'IE_eV': 4.177128,
        'D1_nm': 794.9788, 'D2_nm': 780.2414,
        'tau_D1_ns': 27.70,  'tau_D2_ns': 26.24,
        'fosc_D1': 0.3424,   'fosc_D2': 0.6958,
        'g_upper_D1': 2, 'g_lower_D1': 2,
        'g_upper_D2': 4, 'g_lower_D2': 2,
        'delta_s': 3.1311,   'delta_p': 2.6549,
        'soc_meV': 29.43,
        'source': 'literature',
    },
    'Cs': {
        'Z': 55, 'mass_amu': 132.905, 'gs': '6s', 'np': '6p',
        'IE_eV': 3.893905,
        'D1_nm': 894.5930, 'D2_nm': 852.3471,
        'tau_D1_ns': 34.894, 'tau_D2_ns': 30.462,
        'fosc_D1': 0.3435,   'fosc_D2': 0.7142,
        'g_upper_D1': 2, 'g_lower_D1': 2,
        'g_upper_D2': 4, 'g_lower_D2': 2,
        'delta_s': 4.0493,   'delta_p': 3.5915,
        'soc_meV': 68.70,   # 554.04 cm^-1 NIST ASD
        'source': 'literature',
    },
    'Fr': {
        'Z': 87, 'mass_amu': 223.000, 'gs': '7s', 'np': '7p',
        'IE_eV': 4.072741,
        'D1_nm': 817.2,    'D2_nm': 718.2,
        'tau_D1_ns': 29.45, 'tau_D2_ns': 21.02,   # Simsarian 1999
        'fosc_D1': 0.340,   'fosc_D2': 0.736,
        'g_upper_D1': 2, 'g_lower_D1': 2,
        'g_upper_D2': 4, 'g_lower_D2': 2,
        'delta_s': 4.9731,  'delta_p': 4.4965,
        'soc_meV': 209.10,  # 1686.6 cm^-1 NIST ASD / Sansonetti 2007
        'source': 'literature',
    },
}
_user_tables.apply("spectroscopy.dline", LITERATURE)


def dline_transitions(element):
    """Spectral-bar records [{wl, fosc, label, src}] for the two fine-structure D lines of `element`
    (J-resolved f from LITERATURE), or [] if the element is not tabulated."""
    rec = LITERATURE.get(element)
    if not rec:
        return []
    out = []
    for tag, J in (('D2', '3/2'), ('D1', '1/2')):
        out.append({"wl": rec[f'{tag}_nm'], "fosc": rec[f"fosc_{tag}"],
                    "label": f"{rec['np']}{J}\u2192{rec['gs']}", "src": "lit"})
    return out


# spectra fallback: labelled D-line transitions for the spectral-bar viewer

# ─────────────────────────────────────────────────────────────────────────────
# typed transition record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Transition:
    """One signed transition of a polarizability source-state network.

    Field order is the target shape (and the order
    ``pipelines.polarizability`` serializes these records to JSON):

    - ``delta_E_au`` — level separation in atomic units (sign convention
      carried by ``sign``, never re-encoded here),
    - ``fosc`` — absorption oscillator strength (dimensionless),
    - ``sign`` — +1 upward from the summed state, −1 downward,
    - ``upper_label`` — the state at the *other* end of the line, as seen
      from the summed state (the engine's ``other_label``; it may lie below
      the summed state, despite the historical field name),
    - ``source`` — provenance of ``fosc`` ("NIST", "Coulomb", "core", …),
    - ``sigma`` — known 1σ uncertainty of ``fosc``; ``None`` means unknown,
      never ``0.0`` (PHYSICS_BUGS P11 tracks that σ is not yet propagated),
    - ``wl_nm`` — line wavelength in nm, when known.
    """

    delta_E_au: float
    fosc: float
    sign: int
    upper_label: str
    source: str
    sigma: float | None = None
    wl_nm: float | None = None


# Spectral regions by upper wavelength bound (nm), in display order.
_REGION_BOUNDS_NM = (
    ("X-ray", 10.0),
    ("Vacuum UV", 200.0),
    ("UV", 380.0),
    ("Visible", 750.0),
    ("Near-IR", 2500.0),
    ("Mid/Far-IR", 1e5),
)
SPECTRAL_REGIONS = (*(name for name, _ in _REGION_BOUNDS_NM), "Microwave")


def spectral_region(wavelength_nm: float) -> str:
    """The spectral region of a transition at *wavelength_nm*."""
    for name, upper_nm in _REGION_BOUNDS_NM:
        if wavelength_nm < upper_nm:
            return name
    return "Microwave"


__all__ = [
    "LITERATURE", "SPECTRAL_REGIONS", "Transition", "dline_transitions",
    "spectral_region",
]
