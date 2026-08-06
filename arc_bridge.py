#!/usr/bin/env python3
"""
arc_bridge.py
-------------
Standalone ARC (Alkali Rydberg Calculator) levels export.

Writes data_json/arc/<El>_rydberg.json in the same schema as the NIST
pipeline rydberg JSON. Never overwrites data_json/<El>_rydberg.json.

Usage:
  python arc_bridge.py Na --n-max 50
  python arc_bridge.py Sr --n-max 40
  python arc_bridge.py Rb --isotope 87 --n-max 50

To feed the pipeline (manual):
  copy data_json/arc/Na_rydberg.json -> data_json/Na_rydberg.json
  then run transitions.py / lifetimes.py as usual; restore NIST file after.
"""

from __future__ import annotations

import argparse
import json
import os
from fractions import Fraction

import numpy as np

try:
    import arc
    from arc import (
        Calcium40,
        Caesium,
        Lithium6,
        Lithium7,
        Potassium,
        Rubidium,
        Rubidium85,
        Rubidium87,
        Sodium,
        Strontium88,
    )
except ImportError as exc:
    raise SystemExit(
        "ARC is not installed. Run:\n"
        "  python -m pip install ARC-Alkali-Rydberg-Calculator\n"
        f"({exc})"
    ) from exc

_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ARC = os.path.join(_ROOT, "data_json", "arc")

# Match rydberg.py wavelength conversion (eV -> nm from ground)
HC_EV_NM = 1239.84
L_CHARS = ("s", "p", "d", "f", "g")
L_TO_INT = {c: i for i, c in enumerate(L_CHARS)}


def _ensure_dirs():
    os.makedirs(DATA_ARC, exist_ok=True)


def _j_to_str(j: float) -> str:
    frac = Fraction(j).limit_denominator(16)
    if frac.denominator == 1:
        return str(frac.numerator)
    return f"{frac.numerator}/{frac.denominator}"


def _state_label(n: int, l_char: str, j: float, *, divalent: bool = False,
                 s: float | None = None) -> str:
    j_str = _j_to_str(j)
    if divalent and s is not None:
        return f"{n}{l_char}_s{int(s)}_J{j_str.replace('/', '_')}"
    if l_char == "s":
        return f"{n}s" if abs(j - 0.5) < 1e-9 else f"{n}s{j_str}"
    return f"{n}{l_char}{j_str}"


def make_atom(element: str, isotope: str | None = None):
    """
    Map element symbol (+ optional isotope) to an ARC atom instance.
    Returns (atom, element_symbol, element_class).
    """
    el = element.strip()
    if el[:2].isalpha() and el[2:].isdigit():
        isotope = isotope or el[2:]
        el = el[:2]
    elif el[:1].isalpha() and len(el) > 1 and el[1:].isdigit():
        isotope = isotope or el[1:]
        el = el[:1]

    el = el[0].upper() + el[1:].lower() if len(el) > 1 else el.upper()

    if el == "Li":
        atom = Lithium6() if isotope in ("6", "Li6") else Lithium7()
        return atom, "Li", "alkali"
    if el == "Na":
        return Sodium(), "Na", "alkali"
    if el == "K":
        return Potassium(), "K", "alkali"
    if el == "Rb":
        if isotope in ("85", "Rb85"):
            return Rubidium85(), "Rb", "alkali"
        if isotope in ("87", "Rb87"):
            return Rubidium87(), "Rb", "alkali"
        return Rubidium(), "Rb", "alkali"
    if el == "Cs":
        return Caesium(), "Cs", "alkali"
    if el == "Sr":
        return Strontium88(), "Sr", "divalent"
    if el == "Ca":
        return Calcium40(), "Ca", "divalent"

    raise SystemExit(
        f"Unsupported element '{element}'. "
        f"Supported: Li, Na, K, Rb, Cs, Sr, Ca (optional --isotope)."
    )


def _get_energy(atom, n, l, j, s=None):
    """ARC getEnergy; continuum = 0 eV (negative when bound)."""
    kwargs = {}
    if s is not None:
        kwargs["s"] = s
    try:
        return float(atom.getEnergy(n, l, j, **kwargs))
    except Exception:
        return None


def _quantum_defects_summary(atom, el_class: str, n_ref: int = 20) -> dict:
    out = {}
    for l_char, l in L_TO_INT.items():
        vals = []
        if el_class == "divalent":
            for s in (0.0, 1.0):
                j = float(l) if l > 0 else 0.0
                try:
                    vals.append(float(atom.getQuantumDefect(n_ref, l, j, s=s)))
                except Exception:
                    pass
        else:
            js = [0.5] if l == 0 else [l - 0.5, l + 0.5]
            for j in js:
                try:
                    vals.append(float(atom.getQuantumDefect(n_ref, l, j)))
                except Exception:
                    pass
        out[l_char] = float(np.mean(vals)) if vals else 0.0
    return out


def build_levels(atom, element: str, el_class: str, n_max: int,
                 n_start: int | None = None) -> dict:
    """Build pipeline-compatible *_rydberg.json structure from ARC."""
    IE = float(atom.ionisationEnergy)
    gs_n = int(getattr(atom, "groundStateN", n_start or 1))
    if n_start is None:
        n_start = gs_n

    E_ground = -IE
    excitations = []
    state_num = 0

    if el_class == "divalent":
        spin_blocks = (0.0, 1.0)
        gs_j, gs_s = 0.0, 0.0
        gs_label = f"{gs_n}s"
    else:
        spin_blocks = (None,)
        gs_j, gs_s = 0.5, None
        gs_label = f"{gs_n}s"

    for n in range(n_start, n_max + 1):
        for l_char, l in L_TO_INT.items():
            if l >= n:
                continue
            for s in spin_blocks:
                if el_class == "divalent":
                    S = float(s)
                    j_min = abs(l - S)
                    j_max = l + S
                    js = []
                    j = j_min
                    while j <= j_max + 1e-9:
                        js.append(j)
                        j += 1.0
                else:
                    js = [0.5] if l == 0 else [l - 0.5, l + 0.5]

                for j in js:
                    if (n == gs_n and l == 0
                            and abs(j - gs_j) < 1e-9
                            and (s is None or abs(float(s) - gs_s) < 1e-9)):
                        continue
                    E = _get_energy(atom, n, l, j, s=s)
                    if E is None or E >= 0:
                        continue
                    delta_E = E - E_ground
                    if delta_E <= 0:
                        continue
                    state_num += 1
                    label = _state_label(
                        n, l_char, j, divalent=(el_class == "divalent"), s=s)
                    entry = {
                        "state_number": state_num,
                        "n": n,
                        "l": l_char,
                        "J": _j_to_str(j),
                        "energy_eV": round(E, 6),
                        "wavelength_nm": round(HC_EV_NM / delta_E, 4),
                        "label": label,
                        "source": "ARC",
                        "method": "ARC",
                        "theoretical": False,
                        "uncertainty_eV": None,
                    }
                    if el_class == "divalent":
                        entry["s"] = int(s)
                        entry["term_spin"] = "singlet" if s == 0 else "triplet"
                    excitations.append(entry)

    qd = _quantum_defects_summary(atom, el_class)
    data = {
        "element": element,
        "method": "ARC (Alkali Rydberg Calculator)",
        "arc_version": getattr(arc, "__version__", "unknown"),
        "element_class": el_class,
        "ionization_energy_eV": IE,
        "quantum_defects": qd,
        "qd_notes": "ARC getQuantumDefect averaged at n~20; not a NIST ASD fit.",
        "ground_state": {
            "n": gs_n,
            "l": "s",
            "J": _j_to_str(gs_j),
            "label": gs_label,
        },
        "n_start": n_start,
        "n_max": n_max,
        "num_states": len(excitations),
        "num_theoretical": 0,
        "excitations": excitations,
    }
    if el_class == "divalent":
        data["note"] = (
            "Divalent ARC levels. Pipeline Numerov/alpha/BBR paths are "
            "alkali-oriented; use this JSON for levels overlay or manual "
            "swap only, not as a full AMO claim."
        )
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Export ARC Rydberg levels to data_json/arc/<El>_rydberg.json "
            "(pipeline schema; does not overwrite NIST files)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("element", help="Element symbol: Li Na K Rb Cs Sr Ca")
    parser.add_argument("--isotope", default=None,
                        help="Isotope tag: 6/7 (Li), 85/87 (Rb)")
    parser.add_argument("--n-max", type=int, default=50)
    parser.add_argument("--n-start", type=int, default=None,
                        help="Default: ARC groundStateN")
    args = parser.parse_args(argv)

    _ensure_dirs()
    atom, element, el_class = make_atom(args.element, args.isotope)
    data = build_levels(atom, element, el_class, n_max=args.n_max,
                        n_start=args.n_start)
    out = os.path.join(DATA_ARC, f"{element}_rydberg.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {out}")
    print(f"  {element}: {data['num_states']} states  "
          f"IE={data['ionization_energy_eV']:.6f} eV  "
          f"n={data['n_start']}..{data['n_max']}  class={el_class}")
    for ex in data["excitations"][:4]:
        print(f"  {ex['label']}: E={ex['energy_eV']:.4f} eV  "
              f"wl={ex['wavelength_nm']:.3f} nm")


if __name__ == "__main__":
    main()
