"""
orca_to_json.py  -  standalone ORCA output -> data_json/<species_id>.json
-----------------------------------------------------------------------
Re-parses existing orca_outputs/<species_id>/[run_tag]/ files (no ORCA run)
using the shared parse_orca module and writes data_json/<species_id>.json in the
same format as runorca.py.

Layout (new):
    orca_outputs/<species_id>/<func>_<basis>[...]/<species_id>_ex.out
Legacy (still supported for neutrals):
    orca_outputs/<El>/<El>_ex.out

Usage:
    python orca_to_json.py Na
    python orca_to_json.py Mg_c1
    python orca_to_json.py Li Na K Rb Cs Fe Fr
    python orca_to_json.py Na --run CAM-B3LYP_aug-cc-pVTZ
    python orca_to_json.py Na --outname UKS
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from element_data import ELEMENTS
from species import resolve_species
from parse_orca import (
    parse_ground_energy, parse_excitations, deduplicate_excitations,
)

OUT = "data_json"
SYMBOL_TO_Z = {sym: z for z, (sym, _c, _m) in ELEMENTS.items()}


def _read(path):
    with open(path, "r", errors="ignore") as f:
        return f.read()


def _parse_xyz_header(text, symbol):
    """Return (charge, mult) from '* xyz q m' in an inp, else None."""
    m = re.search(
        rf"^\*\s*xyz\s+(-?\d+)\s+(\d+)\s*$",
        text, flags=re.MULTILINE | re.IGNORECASE,
    )
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _keyword_line(inp_text):
    for line in inp_text.splitlines():
        if line.strip().startswith("!"):
            return line.strip()[1:].strip()
    return ""


def _infer_basis(kw, inp_text):
    if 'NewGTO' in inp_text and 'SARC-DKH' in inp_text:
        return "SARC-DKH-TZVP"
    if 'NewGTO' in inp_text and 'ANO-RCC' in inp_text.upper():
        return "ANO-RCC (inline NewGTO)"
    if 'AutoAux' in kw and 'NewGTO' in inp_text:
        return "ANO-RCC-VTZP (inline NewGTO)"
    for name in ("aug-cc-pVTZ", "def2-TZVPD", "def2-TZVP", "DKH-def2-TZVP",
                 "SARC-DKH-TZVP", "X2C-TZVPall"):
        if name in kw or name in inp_text:
            return name
    # last token that looks like a basis (skip SCF / functionals)
    skip = {"UKS", "RKS", "ROHF", "UHF", "RHF", "DKH", "AutoAux", "TightSCF",
            "VeryTightSCF", "NormalSCF", "LooseSCF", "TDA", "TD"}
    toks = [t for t in kw.split() if t not in skip and "/" not in t]
    # drop known functionals
    funcs = {"B3LYP", "CAM-B3LYP", "PBE0", "PBE", "TPSS", "wB97X", "M06"}
    toks = [t for t in toks if t not in funcs]
    return toks[-1] if toks else "unknown"


def _infer_functional(kw):
    for name in ("CAM-B3LYP", "B3LYP", "PBE0", "wB97X", "M06", "TPSS", "PBE"):
        if name in kw:
            return name
    return "unknown"


def _infer_nroots(inp_text):
    m = re.search(r"nroots\s+(\d+)", inp_text, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def _list_run_dirs(species_id):
    """Return (legacy_flat_dir_or_None, [(tag, path), ...]) under orca_outputs/<id>."""
    base = os.path.join("orca_outputs", species_id)
    if not os.path.isdir(base):
        return None, []
    legacy = None
    if os.path.isfile(os.path.join(base, f"{species_id}_ex.out")):
        legacy = base
    runs = []
    for name in sorted(os.listdir(base)):
        path = os.path.join(base, name)
        if not os.path.isdir(path):
            continue
        if os.path.isfile(os.path.join(path, f"{species_id}_ex.out")):
            runs.append((name, path))
    return legacy, runs


def find_orca_run_dir(species_id, run_tag=None):
    """
    Resolve the directory containing <species_id>_ex.out.
    Prefers --run tag; else legacy flat layout; else the only subfolder run;
    if several subfolders exist, raise with a list.
    """
    legacy, runs = _list_run_dirs(species_id)
    if run_tag:
        path = os.path.join("orca_outputs", species_id, run_tag)
        if os.path.isfile(os.path.join(path, f"{species_id}_ex.out")):
            return path, run_tag
        available = [t for t, _ in runs]
        if legacy:
            available = ["(legacy flat)"] + available
        raise FileNotFoundError(
            f"No {species_id}_ex.out under orca_outputs/{species_id}/{run_tag}/. "
            f"Available: {', '.join(available) if available else '(none)'}"
        )
    if legacy is not None:
        return legacy, None
    if len(runs) == 1:
        return runs[0][1], runs[0][0]
    if not runs:
        raise FileNotFoundError(
            f"missing orca_outputs/{species_id}/[run_tag]/{species_id}_ex.out "
            f"(and no legacy flat {species_id}_ex.out)"
        )
    tags = ", ".join(t for t, _ in runs)
    raise FileNotFoundError(
        f"{species_id}: multiple ORCA runs found. Pass --run <tag>. Options: {tags}"
    )


def rebuild_one(raw_target, run_tag=None, outname=None):
    try:
        sp = resolve_species(raw_target)
    except ValueError as exc:
        print(f"  [FAIL] {exc}")
        return False

    species_id = sp["species_id"]
    symbol = sp["symbol"]
    z = sp["Z"]
    charge = sp["charge"]
    mult = sp["multiplicity"]

    try:
        base, resolved_tag = find_orca_run_dir(species_id, run_tag=run_tag)
    except FileNotFoundError as exc:
        print(f"  [FAIL] {exc}")
        return False

    ex_out = os.path.join(base, f"{species_id}_ex.out")
    gs_out = os.path.join(base, f"{species_id}_gs.out")
    ex_inp = os.path.join(base, f"{species_id}_ex.inp")
    gs_inp = os.path.join(base, f"{species_id}_gs.inp")

    inp_text = _read(ex_inp) if os.path.isfile(ex_inp) else ""
    gs_inp_text = _read(gs_inp) if os.path.isfile(gs_inp) else ""
    xyz = _parse_xyz_header(inp_text, symbol) or _parse_xyz_header(gs_inp_text, symbol)
    if xyz:
        charge, mult = xyz

    ex_text = _read(ex_out)
    if "ORCA TERMINATED NORMALLY" not in ex_text:
        print(f"  [WARN] {species_id}: EX out did not terminate normally; parsing anyway")

    raw = parse_excitations(ex_text, ground_mult=mult)
    if not raw:
        print(f"  [FAIL] {species_id}: no excitations parsed")
        return False
    unique = deduplicate_excitations(raw)

    ground_energy = None
    if os.path.isfile(gs_out):
        gs_text = _read(gs_out)
        ground_energy = parse_ground_energy(gs_text)

    kw_ex = _keyword_line(inp_text)
    kw_gs = _keyword_line(gs_inp_text)
    func_ex = _infer_functional(kw_ex) if kw_ex else "unknown"
    func_gs = _infer_functional(kw_gs) if kw_gs else func_ex
    basis = _infer_basis(kw_ex or kw_gs, inp_text or gs_inp_text)
    nroots = _infer_nroots(inp_text) or len(raw)
    tda = "tda" in inp_text.lower()
    rel = "auto"
    if "DKH" in (kw_ex + kw_gs + inp_text):
        rel = "dkh"
    elif "X2C" in (kw_ex + kw_gs + inp_text):
        rel = "x2c"

    alkali = (
        symbol in {"Li", "Na", "K", "Rb", "Cs", "Fr"}
        and charge == 0 and mult == 2
    )
    quality = "benchmark" if alkali else ("screening_only" if mult > 2 else "benchmark")

    data = {
        "species_id": species_id,
        "element": symbol,
        "label": sp["label"],
        "kind": sp["kind"],
        "Z": z,
        "charge": charge,
        "multiplicity": mult,
        "ground_energy_eV": ground_energy,
        "excitations": raw,
        "excitations_unique": unique,
        "method": "TD-DFT/TDA" if tda else "TD-DFT",
        "template": "excited_states_alkali" if alkali else "excited_states",
        "functional_gs": func_gs,
        "functional_ex": func_ex,
        "basis": basis,
        "basis_tddft": basis,
        "relativistic_mode": rel,
        "nroots": nroots,
        "orca_quality": quality,
        "orca_run_tag": resolved_tag,
        "orca_outputs_dir": base.replace("\\", "/"),
        "reparsed_from": ex_out.replace("\\", "/"),
        "reparsed_gs": gs_out.replace("\\", "/") if os.path.isfile(gs_out) else None,
    }
    if sp.get("soc_class"):
        data["soc_class"] = sp["soc_class"]

    os.makedirs(OUT, exist_ok=True)
    if outname:
        suffix = outname[:-5] if outname.lower().endswith(".json") else outname
        suffix = suffix.strip().strip("_")
        out_file = os.path.join(OUT, f"{species_id}_{suffix}.json")
    else:
        out_file = os.path.join(OUT, f"{species_id}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    bright = [e for e in unique if e["oscillator_strength"] > 0.001]
    print(
        f"  [OK] {species_id}: {len(raw)} raw / {len(unique)} unique / "
        f"{len(bright)} bright  -> {out_file}"
    )
    if resolved_tag:
        print(f"     run tag: {resolved_tag}")
    if bright:
        e0 = bright[0]
        print(
            f"     first bright {e0['energy_eV']:.3f} eV  "
            f"f={e0['oscillator_strength']:.4f}  "
            f"[{func_gs}/{func_ex}, {basis}, nroots={nroots}]"
        )
    return True


def main(argv):
    p = argparse.ArgumentParser(
        description="Re-parse ORCA outs into data_json/<species_id>.json")
    p.add_argument("elements", nargs="+",
                   help="Species id(s): symbol (Na) or ion stem (Mg_c1)")
    p.add_argument("--run", default=None, metavar="TAG",
                   help="orca_outputs/<species>/<TAG>/ folder "
                        "(required if several runs exist)")
    p.add_argument("--outname", default=None, metavar="SUFFIX",
                   help="Write data_json/<species>_<SUFFIX>.json "
                        "instead of <species>.json")
    ns = p.parse_args(argv[1:])
    ok = 0
    for el in ns.elements:
        print(f"\n=== {el} ===")
        if rebuild_one(el, run_tag=ns.run, outname=ns.outname):
            ok += 1
    print(f"\nDone: {ok}/{len(ns.elements)} written")
    if ok != len(ns.elements):
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv)
