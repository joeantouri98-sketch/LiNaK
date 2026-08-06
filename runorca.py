"""
runorca.py  -  run ORCA calculations and save results to data_json/
-------------------------------------------------------------------
Usage:
    python runorca.py              -> run ALL neutrals (auto mode)
    python runorca.py Na           -> run only Sodium
    python runorca.py Mg_c1        -> registered ion (Mg+); dirs/files use Mg_c1
    python runorca.py 19           -> run only Z=19 (Potassium)
    python runorca.py Cs --mode soc --soc-recipe casscf --nroots 6
    python runorca.py Cs --mode soc --soc-recipe nevpt2 --soc-norb 10
    python runorca.py Na --mode casscf --nroots 20
    python runorca.py Fe --mode tddft --nroots 20
    python runorca.py Fr --scf-tightness tight --basis-tightness vtz
    python runorca.py Na --pick-basis
    python runorca.py Li --basis-file "Basis sets/Li/Li-cc-pCVTZ.txt"
    python runorca.py Fr --functional-gs B3LYP --functional-ex CAM-B3LYP
    python runorca.py Na --outname UKS          -> data_json/Na_UKS.json
                                           and orca_outputs/Na/<func>_<basis>_UKS/
    ORCA files land in orca_outputs/<species_id>/<run_tag>/ (always a subfolder);
    run_tag includes functional, basis, SCF keyword, optional mode/outname.
    JSON stays flat under data_json/.
"""

import argparse
import json
import os
import re
import subprocess
import sys

from element_data import ELEMENTS
from species import resolve_species, is_ion_id
from orca_templates import (
    DEFAULT_FUNCTIONAL,
    ground_state,
    excited_states,
    excited_states_alkali,
    excited_states_soc,
    excited_states_alkali_soc,
    excited_states_atomic_alkali,
    select_calculation_mode,
    calculation_metadata,
    is_alkali_neutral,
    is_alkali_like,
    default_soc_nroots,
    get_element_class,
    get_recipe_tddft,
    basis_label_tddft,
    basis_label_soc,
    _build_basis_block,
    basis_block_from_file,
)
from parse_orca import (
    parse_ground_energy, parse_excitations, parse_soc_excitations,
    deduplicate_excitations,
)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Run ORCA and save data_json output")
parser.add_argument("target", nargs="?", default=None,
                    help="Element symbol, Z, or ion species id "
                         "(e.g. Mg_c1, Na_a1); omit for all neutrals")
parser.add_argument("--mode", choices=["auto", "tddft", "alkali", "soc", "alkali-soc", "casscf"],
                    default="auto",
                    help="Calculation template (auto: alkali TD-DFT; soc: CASSCF/NEVPT2+SOC)")
parser.add_argument("--soc-recipe", choices=["casscf", "nevpt2"], default="nevpt2",
                    help="SOC correlation level (soc mode only): casscf=bare CASSCF+DoSOC, nevpt2=SC-NEVPT2+DoSOC")
parser.add_argument("--nroots", type=int, default=None,
                    help="Excited-state roots (default: 20)")
parser.add_argument("--soc-norb", type=int, default=None,
                    help="Override CAS norb for SOC (soc mode only)")
parser.add_argument("--functional", default=DEFAULT_FUNCTIONAL,
                    help=f"DFT functional for GS+EX (default: {DEFAULT_FUNCTIONAL})")
parser.add_argument("--functional-gs", default=None,
                    help="Override functional for the ground-state job only "
                         "(default: value of --functional)")
parser.add_argument("--functional-ex", default=None,
                    help="Override functional for the TD-DFT excited-state job only "
                         "(default: value of --functional)")
parser.add_argument("--relativistic", default="auto",
                    choices=["auto", "ecp", "dkh", "x2c"],
                    help="Relativistic/basis recipe")
parser.add_argument("--scf-tightness",
                    choices=["loose", "normal", "tight", "verytight"],
                    default=None,
                    help="Override SCF convergence preset on the ! line "
                         "(default: alkali GS+EX VeryTightSCF, other TD-DFT TightSCF, "
                         "SOC/CASSCF VeryTightSCF)")
parser.add_argument("--basis-tightness",
                    choices=["vdz", "vtz", "vqz"],
                    default=None,
                    help="ANO-RCC basis tier for Z>=87 (vdz/vtz/vqz); skips the "
                         "interactive BSE picker. Ignored for lighter elements.")
parser.add_argument("--pick-basis", action="store_true",
                    help="Interactive basis_set_exchange picker for every element "
                         "(default: Z>=87 only). Embeds the choice as inline NewGTO.")
parser.add_argument("--basis-file", default=None, metavar="PATH",
                    help="External basis file (GAMESS-US or ORCA format from "
                         "basissetexchange.org). Resolves under Basis sets/ "
                         "when given a bare name/stem. Overrides --pick-basis "
                         "and embeds as inline NewGTO + AutoAux.")
parser.add_argument("--outname", default=None, metavar="SUFFIX",
                    help="Suffix for data_json output: data_json/<El>_<SUFFIX>.json "
                         "(e.g. --outname UKS -> Na_UKS.json). Also appended to the "
                         "orca_outputs/<El>/<run_tag>/ folder name. Overrides default "
                         "JSON names including SOC auto tags. Strip a trailing .json if given.")


def normalize_outname(raw):
    """Return sanitized JSON suffix, or None when --outname is unused."""
    if raw is None:
        return None
    suffix = raw.strip()
    if suffix.lower().endswith(".json"):
        suffix = suffix[:-5]
    suffix = suffix.strip().strip("_")
    if not suffix or any(c in suffix for c in ("/", "\\")):
        raise SystemExit(
            f"Invalid --outname {raw!r}: use a simple suffix "
            f"(e.g. UKS), no path separators.")
    return suffix


def sanitize_run_token(s):
    """Make a string safe as one Windows path component."""
    if s is None:
        return "unknown"
    text = str(s).strip()
    text = re.sub(r"\s*\([^)]*\)", "", text).strip()  # drop "(inline NewGTO)" etc.
    text = text.replace(" ", "_")
    bad = '<>:"/\\|?*'
    text = "".join("_" if (c in bad or ord(c) < 32) else c for c in text)
    text = text.strip(" ._")
    return text or "unknown"


def basis_tag_label(chosen_basis, external_path, symbol, Z, relativistic_mode, calc_mode):
    """Short basis token for the run-tag folder name."""
    if external_path:
        stem = os.path.splitext(os.path.basename(external_path))[0]
        return sanitize_run_token(stem)
    if chosen_basis:
        return sanitize_run_token(chosen_basis)
    if calc_mode in ("soc",):
        label = basis_label_soc(symbol)
    else:
        label = basis_label_tddft(symbol, relativistic_mode)
    return sanitize_run_token(label)


def effective_scf_keyword(calc_mode):
    """ORCA SCF keyword that will appear on the ! line for this mode."""
    override = resolve_scf_keyword(calc_mode)
    if override is not None:
        return override
    if calc_mode in ("alkali", "soc", "casscf"):
        return default_gs_scf_keyword("alkali")  # VeryTightSCF
    if calc_mode == "tddft":
        return default_gs_scf_keyword("tddft")   # TightSCF
    return default_correlated_scf_keyword()


def build_run_tag(functional_gs, functional_ex, basis_tag, outname, calc_mode,
                  soc_recipe=None, soc_norb=None, scf_keyword=None):
    """
    Folder name under orca_outputs/<El>/: join all distinguishing pieces.
    Example: CAM-B3LYP_aug-cc-pVTZ_VeryTightSCF
             or B3LYP_CAM-B3LYP_def2-TZVPD_TightSCF_v13
    """
    parts = []
    if functional_gs == functional_ex:
        parts.append(sanitize_run_token(functional_gs))
    else:
        parts.append(sanitize_run_token(f"{functional_gs}_{functional_ex}"))
    parts.append(basis_tag)
    if scf_keyword:
        parts.append(sanitize_run_token(scf_keyword))
    if calc_mode == "soc":
        parts.append("soc")
        if soc_recipe and soc_recipe != "nevpt2":
            parts.append(sanitize_run_token(soc_recipe))
        if soc_norb is not None:
            parts.append(f"norb{soc_norb}")
    elif calc_mode == "casscf":
        parts.append("casscf")
    elif calc_mode == "tddft":
        parts.append("tddft")
    if outname:
        parts.append(sanitize_run_token(outname))
    return "_".join(parts)


def resolve_basis_choice(symbol, Z, relativistic_mode):
    """
    Resolve external / BSE / recipe basis once (before creating the run folder).
    Returns (chosen_basis_or_None, external_path_or_None).
    """
    if args.basis_file:
        external_path = resolve_basis_file(args.basis_file, symbol)
        if external_path is None:
            raise FileNotFoundError(
                f"--basis-file '{args.basis_file}' not found for {symbol}. "
                f"Tried paths under Basis sets/{symbol}/ and as given."
            )
        if args.pick_basis:
            print("  (--pick-basis ignored; --basis-file takes precedence)")
        return None, external_path
    chosen = resolve_bse_basis(symbol, Z, relativistic_mode)
    return chosen, None


# Parse real CLI only when executed as a script. On import use empty argv so
# helpers stay usable without launching the all-elements ORCA loop.
args = parser.parse_args() if __name__ == "__main__" else parser.parse_args([])

TARGET = args.target

FUNCTIONAL_GS = args.functional_gs or args.functional
FUNCTIONAL_EX = args.functional_ex or args.functional
SPLIT_FUNCTIONALS = FUNCTIONAL_GS != FUNCTIONAL_EX
OUTNAME_SUFFIX = normalize_outname(args.outname)

SCF_TIGHTNESS_KEYWORDS = {
    "loose": "LooseSCF",
    "normal": "NormalSCF",
    "tight": "TightSCF",
    "verytight": "VeryTightSCF",
}

ANO_RCC_BASIS_TIERS = {
    "vdz": "ANO-RCC-VDZ",
    "vtz": "ANO-RCC-VTZP",
    "vqz": "ANO-RCC-VQZP",
}


def resolve_scf_keyword(calc_mode):
    """Return an explicit ORCA SCF keyword, or None to keep mode defaults."""
    if args.scf_tightness is None:
        return None
    return SCF_TIGHTNESS_KEYWORDS[args.scf_tightness]


def default_gs_scf_keyword(calc_mode):
    return "VeryTightSCF" if calc_mode == "alkali" else "TightSCF"


def default_ex_scf_keyword(calc_mode):
    return "VeryTightSCF" if calc_mode == "alkali" else "TightSCF"


def default_correlated_scf_keyword():
    return "VeryTightSCF"


def is_neutral_target(Z, symbol):
    """Match TARGET against a neutral element when looping ELEMENTS."""
    if TARGET is None:
        return True
    if TARGET.isdigit():
        return int(TARGET) == Z
    if is_ion_id(TARGET):
        return False
    return TARGET.lower() == symbol.lower()


# ── Directories ───────────────────────────────────────────────────────────────
BASE_DIR = "orca_outputs"
JSON_DIR = "data_json"
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)


def orca_terminated_ok(text: str) -> bool:
    return "ORCA TERMINATED NORMALLY" in text


def run_orca(inp_path: str, out_path: str) -> None:
    """Run ORCA; raise if the process or normal termination fails."""
    inp_abs = os.path.abspath(inp_path)
    out_abs = os.path.abspath(out_path)
    work_dir = os.path.dirname(inp_abs) or "."
    inp_base = os.path.basename(inp_abs)
    result = subprocess.run(
        f'orca "{inp_base}"',
        shell=True,
        cwd=work_dir,
        capture_output=True,
        text=True,
        errors="ignore",
    )
    with open(out_abs, "w", encoding="utf-8", errors="ignore") as f:
        if result.stdout:
            f.write(result.stdout)
        if result.stderr:
            f.write(result.stderr)
    if result.returncode != 0:
        tail = ((result.stderr or "") + (result.stdout or ""))[-800:]
        raise RuntimeError(
            f"ORCA failed for {inp_path} (code {result.returncode})\n{tail}")
    if not orca_terminated_ok(open(out_abs, encoding="utf-8", errors="ignore").read()):
        raise RuntimeError(f"ORCA did not terminate normally: {out_path}")


def default_nroots(symbol, charge, mult, calc_mode, soc_class_hint=None):
    if args.nroots is not None:
        return args.nroots
    if calc_mode == "soc":
        el_class = get_element_class(
            symbol, charge, mult, soc_class_hint=soc_class_hint)
        if el_class:
            return default_soc_nroots(el_class, symbol=symbol)
        return 6
    if TARGET is not None:
        try:
            prompt = "Number of excited-state roots [default: 20]: "
            entered = input(prompt).strip()
            if entered:
                return int(entered)
        except (ValueError, EOFError):
            pass
    return 20


# ── BSE basis selector ────────────────────────────────────────────────────────
def recipe_default_basis(symbol, z, relativistic_mode="auto"):
    """Template default basis name for BSE picker default (Enter key)."""
    if args.basis_tightness is not None and z >= 87:
        return ANO_RCC_BASIS_TIERS[args.basis_tightness]
    recipe = get_recipe_tddft(symbol, relativistic_mode=relativistic_mode)
    if recipe.get("basis_keyword"):
        return recipe["basis_keyword"]
    if z >= 87:
        return "ANO-RCC-VTZP"
    if 55 <= z <= 86:
        return "SARC-DKH-TZVP"
    if z >= 19:
        return "def2-TZVPD"
    return "aug-cc-pVTZ"


def should_prompt_bse(z):
    return args.pick_basis or z >= 87


def select_bse_basis(symbol, z, default=None, include_nonrel=False):
    if default is None:
        default = recipe_default_basis(symbol, z, args.relativistic)
    try:
        import basis_set_exchange as bse
    except ImportError:
        print(f"  basis_set_exchange not installed - using {default}")
        return default

    def rel_type(name):
        n = name.upper()
        if "ECP" in n or "PP" in n:
            return "ECP"
        if any(x in n for x in ["DKH", "DK3", "X2C", "ZORA",
                                "ANO-RCC", "ANO-DK", "DYALL", "JORGE"]):
            return "Rel-AE"
        return "NonRel"

    print(f"\n  Fetching available basis sets for {symbol} (Z={z}) from BSE...")
    available = []
    for name in bse.get_all_basis_names():
        try:
            bse.get_basis(name, elements=[z])
            rt = rel_type(name)
            if include_nonrel or rt in ("Rel-AE", "ECP"):
                available.append((name, rt))
        except Exception:
            pass

    rel_ae = [(n, t) for n, t in available if t == "Rel-AE"]
    ecp = [(n, t) for n, t in available if t == "ECP"]
    nonrel = [(n, t) for n, t in available if t == "NonRel"] if include_nonrel else []

    print(f"\n  {'─'*60}")
    idx = 0
    all_bases = []

    def _print_group(title, items):
        nonlocal idx
        if not items:
            return
        print(f"  {title} ({len(items)}):")
        for name, _ in items:
            idx += 1
            marker = " ◀ default" if name == default else ""
            print(f"    {idx:>3}. {name}{marker}")
            all_bases.append(name)

    _print_group("Relativistic all-electron bases", rel_ae)
    _print_group("ECP bases", ecp)
    _print_group("Non-relativistic / general bases", nonrel)

    if not all_bases:
        print(f"  No matching bases found - using {default}")
        return default

    print(f"  {'─'*60}")
    print(f"  Enter number to select, or press Enter for default ({default}):")

    choice = input("  > ").strip()
    if not choice:
        print(f"  Using default: {default}")
        return default
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(all_bases):
            chosen = all_bases[idx]
            print(f"  Selected: {chosen}")
            return chosen
    except ValueError:
        if choice in all_bases:
            print(f"  Selected: {choice}")
            return choice
    print(f"  Not recognised - using default: {default}")
    return default


def resolve_bse_basis(symbol, z, relativistic_mode):
    """Return a BSE basis name to embed, or None to keep the template recipe."""
    if not should_prompt_bse(z):
        return None
    if args.basis_tightness is not None and z >= 87:
        chosen = ANO_RCC_BASIS_TIERS[args.basis_tightness]
        print(f"  --basis-tightness {args.basis_tightness} -> {chosen}")
        return chosen
    return select_bse_basis(
        symbol, z,
        default=recipe_default_basis(symbol, z, relativistic_mode),
        include_nonrel=args.pick_basis,
    )


def resolve_basis_file(spec, symbol, basis_root='Basis sets'):
    """
    Resolve --basis-file to an absolute path.
    Accepts absolute/relative paths, or bare names/stems under Basis sets/.
    Examples for Li: Li-cc-pCVTZ.txt, cc-pCVTZ, Li-cc-pCVTZ
    """
    if not spec:
        return None
    spec = spec.strip().strip('"').strip("'")
    if os.path.isfile(spec):
        return os.path.abspath(spec)
    stem = spec[:-5] if spec.lower().endswith('.txt') else spec
    stem = stem[:-5] if stem.lower().endswith('.bas') else stem
    candidates = [
        os.path.join(basis_root, symbol, spec),
        os.path.join(basis_root, symbol, f"{spec}.txt"),
        os.path.join(basis_root, symbol, f"{spec}.bas"),
        os.path.join(basis_root, symbol, f"{symbol}-{spec}"),
        os.path.join(basis_root, symbol, f"{symbol}-{spec}.txt"),
        os.path.join(basis_root, symbol, f"{symbol}-{stem}.txt"),
        os.path.join(basis_root, spec),
        os.path.join(basis_root, f"{spec}.txt"),
        f"{spec}.txt",
        f"{symbol}-{spec}.txt",
    ]
    # If user passed Li-cc-pCVTZ, also try as-is under Basis sets/Li/
    if not stem.lower().startswith(f"{symbol.lower()}-"):
        candidates.insert(0, os.path.join(basis_root, symbol, f"{symbol}-{stem}.txt"))
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    return None


def _ensure_autoaux(recipe):
    """Custom NewGTO bases lack ORCA's built-in AuxJ pairing; add AutoAux."""
    r = dict(recipe)
    sk = r.get("simple_keywords", "") or ""
    parts = sk.split()
    if "AutoAux" not in parts:
        r["simple_keywords"] = "AutoAux" if not sk else f"{sk} AutoAux"
    return r


def _patch_inline_basis_recipes(_ot, symbol, z, basis_block_text):
    """Monkey-patch recipe builders so one element uses a given NewGTO block."""
    _orig_tddft = _ot.get_recipe_tddft
    _orig_soc = _ot.get_recipe_soc
    _orig_ano = _ot._build_ano_rcc_block

    def _with_inline_basis(recipe):
        r = _ensure_autoaux(recipe)
        r["basis_keyword"] = ""
        r["basis_block"] = basis_block_text
        return r

    def get_recipe_tddft_patched(sym, relativistic_mode="auto"):
        r = _orig_tddft(sym, relativistic_mode=relativistic_mode)
        if sym == symbol:
            return _with_inline_basis(r)
        return r

    def get_recipe_soc_patched(sym):
        r = _orig_soc(sym)
        if sym == symbol:
            return _with_inline_basis(r)
        return r

    def build_ano_rcc_block_patched(sym, z_arg):
        if sym == symbol:
            return basis_block_text
        return _orig_ano(sym, z_arg)

    _ot.get_recipe_tddft = get_recipe_tddft_patched
    _ot.get_recipe = get_recipe_tddft_patched
    _ot.get_recipe_soc = get_recipe_soc_patched
    _ot._build_ano_rcc_block = build_ano_rcc_block_patched
    return (_orig_tddft, _orig_soc, _orig_ano)


def _patch_bse_recipes(_ot, symbol, z, basis_name):
    """Monkey-patch recipe builders so one element uses inline BSE NewGTO."""
    block = _build_basis_block(symbol, z, basis_name)
    return _patch_inline_basis_recipes(_ot, symbol, z, block)


def write_inputs(symbol, charge, mult, Z, element_dir, calc_mode,
                 functional_gs, functional_ex, nroots, relativistic_mode,
                 soc_recipe="nevpt2", soc_norb=None, scf_override=None,
                 chosen_basis=None, external_basis_path=None,
                 file_prefix=None, soc_class_hint=None):
    """Write ORCA input file(s); return (paths dict, gbw name for moinp, basis label).

    file_prefix: basename stem for .inp/.gbw (species_id, default=symbol).
    Atom line still uses chemical symbol; charge/mult go on * xyz.
    """
    import orca_templates as _ot

    prefix = file_prefix or symbol
    basis_ctx = None

    if external_basis_path is not None:
        block = basis_block_from_file(external_basis_path, symbol)
        basis_ctx = _patch_inline_basis_recipes(_ot, symbol, Z, block)
        print(f"  External basis: {external_basis_path}")
    elif chosen_basis is not None:
        basis_ctx = _patch_bse_recipes(_ot, symbol, Z, chosen_basis)
    elif args.basis_tightness is not None and Z < 87:
        print(f"  --basis-tightness ignored for {symbol} (Z={Z}; only applies to Z>=87)")

    paths = {}
    gbw_name = f"{prefix}_gs.gbw"
    correlated_scf = scf_override or default_correlated_scf_keyword()

    try:
        if calc_mode in ("casscf", "soc"):
            if calc_mode == "soc":
                suffix = "casscf_soc"
                paths[suffix] = os.path.join(element_dir, f"{prefix}_{suffix}.inp")
                text = excited_states_soc(
                    symbol, charge, mult, nroots=nroots,
                    soc_recipe=soc_recipe, norb_override=soc_norb,
                    scf_keyword=correlated_scf,
                    soc_class_hint=soc_class_hint)
            elif not is_alkali_like(symbol, charge, mult, soc_class_hint=soc_class_hint):
                raise ValueError(
                    f"--mode {calc_mode} only supports alkali / alkali-like species")
            else:
                suffix = "casscf"
                paths[suffix] = os.path.join(element_dir, f"{prefix}_{suffix}.inp")
                text = excited_states_atomic_alkali(
                    symbol, charge, mult, nroots=nroots,
                    relativistic_mode=relativistic_mode,
                    scf_keyword=correlated_scf)
            with open(paths[suffix], "w") as f:
                f.write(text)
        else:
            gs_scf = scf_override or default_gs_scf_keyword(calc_mode)
            ex_scf = scf_override or default_ex_scf_keyword(calc_mode)
            paths["gs"] = os.path.join(element_dir, f"{prefix}_gs.inp")
            paths["ex"] = os.path.join(element_dir, f"{prefix}_ex.inp")

            with open(paths["gs"], "w") as f:
                f.write(ground_state(
                    symbol, charge, mult, functional=functional_gs,
                    relativistic_mode=relativistic_mode, scf_keyword=gs_scf))

            # Reusing GS orbitals across different functionals is inconsistent;
            # with split functionals the EX job runs its own SCF (legacy behavior).
            moinp = None if functional_gs != functional_ex else gbw_name
            if moinp is None:
                print(f"  Split functionals ({functional_gs} GS / {functional_ex} EX): "
                      "EX job runs a fresh SCF (no %moinp)")

            if calc_mode == "alkali":
                ex_text = excited_states_alkali(
                    symbol, charge, mult, nroots=nroots,
                    functional=functional_ex, relativistic_mode=relativistic_mode,
                    moinp_gbw=moinp,
                    scf_keyword=ex_scf if scf_override else None,
                    soc_class_hint=soc_class_hint)
            else:
                ex_text = excited_states(
                    symbol, charge, mult, nroots=nroots,
                    functional=functional_ex, relativistic_mode=relativistic_mode,
                    moinp_gbw=moinp, scf_keyword=ex_scf)

            with open(paths["ex"], "w") as f:
                f.write(ex_text)
    finally:
        if basis_ctx is not None:
            _orig_tddft, _orig_soc, _orig_ano = basis_ctx
            _ot.get_recipe_tddft = _orig_tddft
            _ot.get_recipe = _orig_tddft
            _ot.get_recipe_soc = _orig_soc
            _ot._build_ano_rcc_block = _orig_ano

    # Tag external path for JSON metadata (basename already in chosen_basis)
    if external_basis_path is not None:
        chosen_basis = f"file:{external_basis_path}"
    return paths, gbw_name, chosen_basis


# ── Main loop (script only - never on import) ─────────────────────────────────
def main():
  if SPLIT_FUNCTIONALS:
    print(f"  functional = GS: {FUNCTIONAL_GS}  /  EX: {FUNCTIONAL_EX}")
  else:
    print(f"  functional = {FUNCTIONAL_GS}")
  print(f"  mode       = {args.mode}")
  print(f"  relativity = {args.relativistic}")
  if args.scf_tightness is not None:
    print(f"  scf        = {SCF_TIGHTNESS_KEYWORDS[args.scf_tightness]} "
          f"(from --scf-tightness {args.scf_tightness})")
  if args.basis_tightness is not None:
    print(f"  basis tier = {ANO_RCC_BASIS_TIERS[args.basis_tightness]} "
          f"(from --basis-tightness {args.basis_tightness}; Z>=87 only)")
  if args.pick_basis:
    print("  bse picker = on for all elements (--pick-basis)")
  if args.basis_file:
    print(f"  basis file = {args.basis_file} (--basis-file; overrides BSE picker)")
  if OUTNAME_SUFFIX:
    print(f"  outname    = {OUTNAME_SUFFIX}  (-> data_json/<species>_{OUTNAME_SUFFIX}.json "
          f"+ run-tag folder)")
  print(f"  ORCA outs  = {BASE_DIR}/<species>/<func>_<basis>_<SCF>[_<mode>][_<outname>]/")
  if args.mode in ("soc", "alkali-soc"):
    print(f"  soc-recipe = {args.soc_recipe}")
    if args.soc_norb is not None:
        print(f"  soc-norb   = {args.soc_norb}")

  species_list = []
  if TARGET is not None:
    try:
        species_list = [resolve_species(TARGET)]
    except ValueError as exc:
        raise SystemExit(str(exc))
  else:
    for Z, (symbol, charge, multiplicity) in ELEMENTS.items():
        if is_neutral_target(Z, symbol):
            species_list.append(resolve_species(symbol))

  for sp in species_list:
    _run_one_species(sp)

  print("\n=== All requested calculations finished ===")


def _run_one_species(sp):
    """Run ORCA + write JSON for one resolved species record."""
    from orca_templates import require_soc_class

    species_id = sp["species_id"]
    symbol = sp["symbol"]
    Z = sp["Z"]
    charge = sp["charge"]
    multiplicity = sp["multiplicity"]
    soc_hint = sp.get("soc_class")

    calc_mode = select_calculation_mode(
        symbol, charge, multiplicity, args.mode, soc_class_hint=soc_hint)
    if calc_mode == "alkali" and not is_alkali_like(
            symbol, charge, multiplicity, soc_class_hint=soc_hint):
        print(f"\n=== Skipping {species_id}: not alkali / alkali-like ===")
        return
    if calc_mode == "casscf" and not is_alkali_neutral(symbol, charge, multiplicity):
        print(f"\n=== Skipping {species_id}: --mode casscf is neutral alkali only ===")
        return
    if args.mode in ("soc", "alkali-soc") and calc_mode == "soc":
        try:
            require_soc_class(symbol, charge, multiplicity, soc_class_hint=soc_hint)
        except (ValueError, NotImplementedError) as exc:
            print(f"\n=== Skipping {species_id}: {exc} ===")
            return

    nroots = default_nroots(
        symbol, charge, multiplicity, calc_mode, soc_class_hint=soc_hint)

    print(f"\n=== Running {species_id} ({sp['label']}, Z={Z}, "
          f"q={charge}, mult={multiplicity})  mode={calc_mode}  nroots={nroots} ===")
    if calc_mode == "soc":
        print(f"  soc-recipe={args.soc_recipe}" +
              (f"  soc-norb={args.soc_norb}" if args.soc_norb else "") +
              (f"  soc_class={soc_hint}" if soc_hint else ""))
    if multiplicity > 2:
        print("  WARNING:  High-spin atom - TD-DFT oscillator strengths are screening-quality only")

    try:
        chosen_basis, external_basis_path = resolve_basis_choice(
            symbol, Z, args.relativistic)
    except FileNotFoundError as exc:
        print(f"  [X] {exc}")
        return

    basis_tag = basis_tag_label(
        chosen_basis, external_basis_path, symbol, Z, args.relativistic, calc_mode)
    scf_override = resolve_scf_keyword(calc_mode)
    scf_tag = effective_scf_keyword(calc_mode)
    run_tag = build_run_tag(
        FUNCTIONAL_GS, FUNCTIONAL_EX, basis_tag, OUTNAME_SUFFIX, calc_mode,
        soc_recipe=args.soc_recipe, soc_norb=args.soc_norb,
        scf_keyword=scf_tag)
    element_dir = os.path.join(BASE_DIR, species_id, run_tag)
    os.makedirs(element_dir, exist_ok=True)
    print(f"  run dir   = {element_dir}")
    paths, _gbw, bse_basis = write_inputs(
        symbol, charge, multiplicity, Z, element_dir, calc_mode,
        FUNCTIONAL_GS, FUNCTIONAL_EX, nroots, args.relativistic,
        soc_recipe=args.soc_recipe, soc_norb=args.soc_norb,
        scf_override=scf_override,
        chosen_basis=chosen_basis, external_basis_path=external_basis_path,
        file_prefix=species_id, soc_class_hint=soc_hint)

    ground_energy = None
    ex_text = ""

    try:
        if calc_mode == "soc":
            out_path = os.path.join(element_dir, f"{species_id}_casscf_soc.out")
            label = "CASSCF + SOC" if args.soc_recipe == "casscf" else "CASSCF/NEVPT2 + SOC"
            print(f"  -> {label}...")
            run_orca(paths["casscf_soc"], out_path)
            with open(out_path, "r", errors="ignore") as f:
                ex_text = f.read()
            ground_energy = parse_ground_energy(ex_text)
        elif calc_mode == "casscf":
            out_path = os.path.join(element_dir, f"{species_id}_casscf.out")
            print("  -> CASSCF + SC-NEVPT2...")
            run_orca(paths["casscf"], out_path)
            with open(out_path, "r", errors="ignore") as f:
                ex_text = f.read()
            ground_energy = parse_ground_energy(ex_text)
        else:
            gs_out = os.path.join(element_dir, f"{species_id}_gs.out")
            ex_out = os.path.join(element_dir, f"{species_id}_ex.out")
            print("  -> Ground state...")
            run_orca(paths["gs"], gs_out)
            with open(gs_out, "r", errors="ignore") as f:
                gs_text = f.read()
            ground_energy = parse_ground_energy(gs_text)

            print("  -> Excited states (TD-DFT)...")
            run_orca(paths["ex"], ex_out)
            with open(ex_out, "r", errors="ignore") as f:
                ex_text = f.read()
    except RuntimeError as exc:
        print(f"  [X] {exc}")
        return

    if calc_mode == "soc":
        raw_excitations = parse_soc_excitations(ex_text, ground_mult=multiplicity)
        if not raw_excitations:
            if "SOC CORRECTED ABSORPTION SPECTRUM" in ex_text:
                print("  [X] SOC spectrum present but no transitions parsed - check parse_orca.py")
            else:
                print("  [X] No SOC CORRECTED spectrum in ORCA output")
            return
    else:
        raw_excitations = parse_excitations(ex_text, ground_mult=multiplicity)
    excitations_unique = deduplicate_excitations(raw_excitations)

    print(f"  States: {len(raw_excitations)} raw -> {len(excitations_unique)} unique levels")
    bright = [e for e in excitations_unique if e["oscillator_strength"] > 0.001]
    for e in bright[:12]:
        print(f"    {e['energy_eV']:.3f} eV  fosc={e['oscillator_strength']:.4f}  "
              f"{e.get('from', '?')}->{e.get('to', '?')}")
    if len(bright) > 12:
        print(f"    ... and {len(bright) - 12} more bright lines")

    meta = calculation_metadata(
        symbol, charge, multiplicity, calc_mode,
        FUNCTIONAL_GS, nroots, args.relativistic,
        soc_recipe=args.soc_recipe, norb_override=args.soc_norb,
        soc_class_hint=soc_hint)
    if calc_mode in ("alkali", "tddft"):
        meta["functional_ex"] = FUNCTIONAL_EX
    if args.scf_tightness is not None:
        meta["scf_tightness"] = args.scf_tightness
        meta["scf_keyword"] = scf_override
    if bse_basis is not None:
        if isinstance(bse_basis, str) and bse_basis.startswith("file:"):
            _fpath = bse_basis[5:]
            meta["basis"] = f"{os.path.basename(_fpath)} (external file, inline NewGTO)"
            meta["basis_file"] = _fpath
        else:
            meta["basis"] = f"{bse_basis} (inline NewGTO)"
            meta["bse_basis"] = bse_basis
        meta["basis_tddft"] = meta["basis"]
        if calc_mode in ("soc", "alkali-soc"):
            meta["basis_soc"] = meta["basis"]

    result = {
        "species_id": species_id,
        "element": symbol,
        "label": sp["label"],
        "kind": sp["kind"],
        "Z": Z,
        "charge": charge,
        "multiplicity": multiplicity,
        "ground_energy_eV": ground_energy,
        "excitations": raw_excitations,
        "excitations_unique": excitations_unique,
        "orca_run_tag": run_tag,
        "orca_outputs_dir": element_dir.replace("\\", "/"),
        **meta,
    }
    if soc_hint:
        result["soc_class"] = soc_hint
    if calc_mode == "soc" and raw_excitations:
        result["soc_energy_source"] = raw_excitations[0].get("soc_energy_source", "casscf")

    if OUTNAME_SUFFIX:
        out_name = f"{species_id}_{OUTNAME_SUFFIX}.json"
    elif calc_mode == "soc":
        parts = [f"{species_id}_soc"]
        if args.soc_recipe != "nevpt2":
            parts.append(args.soc_recipe)
        if args.soc_norb is not None:
            parts.append(f"norb{args.soc_norb}")
        out_name = "_".join(parts) + ".json"
    else:
        out_name = f"{species_id}.json"
    out_file = os.path.join(JSON_DIR, out_name)
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    basis_disp = meta.get("basis_soc", meta.get("basis", "?")) if calc_mode == "soc" else meta.get("basis", "?")
    print(f"  [OK] Saved: {out_file}  [{meta['method']}, {basis_disp}]")
    if calc_mode == "soc" and sp["kind"] == "neutral":
        print(f"     (TD-DFT screening data stays in {JSON_DIR}/{species_id}.json if present)")


if __name__ == "__main__":
    main()
