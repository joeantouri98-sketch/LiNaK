"""
The ``orca_to_json`` pipeline: ORCA output -> data/<species_id>.json
--------------------------------------------------------------------
Re-parses existing orca_outputs/<species_id>/[run_tag]/ files (no ORCA run)
with ``linak.integrations.orca_parser`` and writes data/<species_id>.json in
the same format as the ``runorca`` pipeline.

Layout (new):
    orca_outputs/<species_id>/<func>_<basis>[...]/<species_id>_ex.out
Legacy (still supported for neutrals):
    orca_outputs/<El>/<El>_ex.out

Usage (``linak --pipeline orca_to_json ...``):
    orca_to_json Na
    orca_to_json Mg_c1
    orca_to_json Li Na K Rb Cs Fe Fr
    orca_to_json Na --run CAM-B3LYP_aug-cc-pVTZ
    orca_to_json Na --outname UKS
"""

import argparse
import os
from pathlib import Path
import re
import sys

from linak.core.fileio import is_safe_name, write_json
from linak.core.paths import reference_path
from linak.core.species import resolve_species
from linak.core.workspace import Workspace
from linak.integrations.orca_parser import (
    parse_ground_energy, parse_excitations, deduplicate_excitations,
)
from linak.integrations.orca_templates import is_alkali_neutral


def _out_dir() -> str:
    """Writable JSON data dir of the workspace in effect now."""
    return str(Workspace.current().data_json_dir)


def _safe_name(value, flag):
    """Reject a CLI value that could escape its directory (separators, '..', 'C:')."""
    name = value.strip()
    if not is_safe_name(name):
        raise ValueError(f"{flag} must be a single path component, got {value!r}")
    return name


def _read(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _parse_xyz_header(text, symbol):
    """Return (charge, mult) from '* xyz q m' in an inp, else None."""
    m = re.search(
        r"^\*\s*xyz\s+(-?\d+)\s+(\d+)\s*$",
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


def _species_runs_dir(species_id):
    """``orca_outputs/<id>`` in the output folder, else the bundled example runs.

    The bundled runs are read in place (never copied), so a fresh CLI output
    folder can still re-parse the shipped Li/Na/Na_c1 calculations.
    """
    for root in (Workspace.current().orca_outputs_dir, reference_path("orca_outputs")):
        candidate = os.path.join(root, species_id)
        if os.path.isdir(candidate):
            return candidate
    return None


def _as_recorded(path):
    """*path* as stored in the JSON: relative, e.g. ``orca_outputs/Na/Na_ex.out``."""
    parts = Path(path).parts
    start = max(i for i, part in enumerate(parts) if part == "orca_outputs")
    return "/".join(parts[start:])


def _list_run_dirs(species_id):
    """Return (legacy_flat_dir_or_None, [(tag, path), ...]) under orca_outputs/<id>."""
    base = _species_runs_dir(species_id)
    if base is None:
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
        path = os.path.join(_species_runs_dir(species_id) or "", run_tag)
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
        if run_tag is not None:
            run_tag = _safe_name(run_tag, "--run")
        if outname is not None:
            outname = _safe_name(outname, "--outname")
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

    alkali = is_alkali_neutral(symbol, charge, mult)
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
        "orca_outputs_dir": _as_recorded(base),
        "reparsed_from": _as_recorded(ex_out),
        "reparsed_gs": _as_recorded(gs_out) if os.path.isfile(gs_out) else None,
    }
    if sp.get("soc_class"):
        data["soc_class"] = sp["soc_class"]

    os.makedirs(_out_dir(), exist_ok=True)
    if outname:
        suffix = outname[:-5] if outname.lower().endswith(".json") else outname
        suffix = suffix.strip().strip("_")
        out_file = os.path.join(_out_dir(), f"{species_id}_{suffix}.json")
    else:
        out_file = os.path.join(_out_dir(), f"{species_id}.json")
    write_json(out_file, data)

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


# ============================================================================
# ARGPARSE
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="linak --pipeline orca_to_json",
        description="Re-parse ORCA outputs into data/<species_id>.json")
    p.add_argument("elements", nargs="+",
                   help="Species id(s): symbol (Na) or ion stem (Mg_c1)")
    p.add_argument("--run", default=None, metavar="TAG",
                   help="orca_outputs/<species>/<TAG>/ folder "
                        "(required if several runs exist)")
    p.add_argument("--outname", default=None, metavar="SUFFIX",
                   help="Write data/<species>_<SUFFIX>.json "
                        "instead of <species>.json")
    return p


def main(argv=None):
    """Run the re-parse pipeline; ``argv`` are user args after the program name."""
    ns = build_parser().parse_args(argv)
    ok = 0
    for el in ns.elements:
        print(f"\n=== {el} ===")
        if rebuild_one(el, run_tag=ns.run, outname=ns.outname):
            ok += 1
    print(f"\nDone: {ok}/{len(ns.elements)} written")
    return 1 if ok != len(ns.elements) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
