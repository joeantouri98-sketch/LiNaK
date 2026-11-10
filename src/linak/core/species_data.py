"""What a species needs before the stages can run on it, and what the user can supply (no Qt).

Used by the dialog that opens when a species with no bundled data is set, and by Properties.
The needs below were checked by running the stages on a species that has only a NIST levels file and a NIST lines file.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from linak.core import user_tables
from linak.core.elements import ALKALI_SYMBOLS
from linak.core.errors import InvalidSpeciesError
from linak.core.paths import reference_path
from linak.core.species import parse_species_id


@dataclass(frozen=True)
class Item:
    key: str
    label: str
    section: str                 # "required" | "recommended" | "optional"
    kind: str                    # "file" | "number" | "record"
    table_id: str = ""           # number / record: the table in Tables and variables
    tree: str = ""               # file: resource tree
    suffix_name: str = ""        # file: name after the species id, e.g. "_lines"
    hint: str = ""


ITEMS = (
    Item("levels", "NIST energy levels", "required", "file", tree="NIST Levels", suffix_name="",
         hint="ASD levels export (.csv or .txt)"),
    Item("lines", "NIST lines and f-values", "required", "file", tree="f_values", suffix_name="_lines",
         hint="ASD lines export (.csv or .txt)"),
    Item("mass", "Atomic mass", "recommended", "number", table_id="constants.alkali_mass"),
    Item("alpha_core", "Ionic core polarizability", "recommended", "number", table_id="polarizability.alpha_core",
         hint="taken as 0 when missing"),
    Item("alpha0", "Static polarizability (experiment)", "optional", "number", table_id="constants.alpha0_expt",
         hint="only a comparison value"),
    Item("tweezer_wl", "Default tweezer wavelength", "optional", "number", table_id="tweezer.default_wavelength"),
    Item("dline", "D-line values", "optional", "record", table_id="spectroscopy.dline",
         hint="for Compare elements, Scattering rate and the spectrum fallback"),
    Item("hf", "Hyperfine constants", "optional", "hyperfine", hint="A and B (and C) of each state, for one isotope"),
)
BY_KEY = {i.key: i for i in ITEMS}

# (stage id as in gui.stages, label, items it needs). "orca" and "manual" are not items.
STAGES = (
    ("runorca", "ORCA (TD-DFT / SOC)", ()),
    ("rydberg", "Rydberg series", ("levels",)),
    ("transitions", "Transitions", ("levels",)),
    ("lifetimes", "Lifetimes", ("levels", "lines")),
    ("polarizability", "Polarizability", ("levels", "lines")),
    ("blackbody", "Blackbody (BBR)", ("levels", "lines")),
    ("tweezer", "Optical tweezer", ("levels", "lines")),
    ("spectra", "Spectrum", ("levels",)),
    ("grotrian", "Grotrian diagram and plots", ("orca",)),
    ("compare", "Compare elements", ("dline",)),
    ("scattering", "Scattering rate", ("dline",)),
    ("hyperfine", "Hyperfine / Zeeman", ("hf",)),
    ("feshbach", "Feshbach resonances", ("manual:feshbach",)),
)
_MANUAL = {"manual:feshbach": "needs a file of resonances that you provide"}

def resolve(species: str) -> tuple[str, str]:
    """``(species id, element symbol)``; raises InvalidSpeciesError for something that is not a species."""
    parsed = parse_species_id(species)
    if parsed is None:
        raise InvalidSpeciesError("Empty species id")
    return parsed[0], parsed[1]


def file_names(item: Item, species_id: str) -> tuple[str, str]:
    base = f"{species_id}{item.suffix_name}"
    return f"{base}.csv", f"{base}.txt"


def has_file(item: Item, species_id: str) -> bool:
    return any(reference_path(item.tree, n).is_file() for n in file_names(item, species_id))


def present(item: Item, species_id: str, element: str, pending: set[str] | frozenset[str] = frozenset(),
            data_dir: Path | None = None) -> bool:
    """Is the item there already (bundled, imported, in a table, or still waiting to be saved)?"""
    if item.key in pending:
        return True
    if item.kind == "file":
        return has_file(item, species_id)
    if item.kind == "hyperfine":
        return data_dir is not None and read_hf_constants(data_dir, species_id)[0] is not None
    if element in user_tables.live_table(item.table_id):
        return True
    return element in user_tables.load_additions().get(item.table_id, {})


def missing_items(species: str, pending=frozenset(), data_dir: Path | None = None) -> list[Item]:
    sid, el = resolve(species)
    return [i for i in ITEMS if not present(i, sid, el, pending, data_dir)]


def needs_setup(species: str) -> bool:
    """True when *species* is a species LiNaK has no levels and lines for (a required item is missing)."""
    try:
        sid, el = resolve(species)
    except (InvalidSpeciesError, KeyError, ValueError):
        return False
    return any(not present(i, sid, el) for i in ITEMS if i.section == "required")


def stage_states(species: str, data_dir: Path | None = None, pending=frozenset()) -> list[tuple[str, str, bool, str]]:
    """``[(stage id, label, ready, why not)]``; *data_dir* is where the ORCA results would be."""
    sid, el = resolve(species)
    out = []
    for stage_id, label, needs in STAGES:
        why, lacks = [], []
        for need in needs:
            if need == "orca":
                have = data_dir is not None and any((Path(data_dir) / f"{sid}{s}.json").is_file() for s in ("", "_soc"))
                if not have:
                    why.append("needs ORCA results (run the ORCA stage; the NIST levels are overlaid)")
            elif need.startswith("manual:"):
                why.append(_MANUAL[need])
            elif not present(BY_KEY[need], sid, el, pending, data_dir):
                lacks.append(BY_KEY[need].label)
        if lacks:
            why.insert(0, "needs " + " and ".join(lacks))
        out.append((stage_id, label, not why, "; ".join(why)))
    return out


def is_one_electron(species: str) -> bool:
    return resolve(species)[1] in ALKALI_SYMBOLS


# --- importing the two files --------------------------------------------------------
def _target(item: Item, species_id: str, source: Path) -> tuple[str, ...]:
    ext = ".csv" if source.suffix.lower() == ".csv" else ".txt"
    return (item.tree, f"{species_id}{item.suffix_name}{ext}")


def check_levels(path: Path) -> str:
    from linak.pipelines.rydberg import parse_nist_asd_term_levels

    try:
        ie, levels = parse_nist_asd_term_levels(str(path))
    except Exception as exc:
        raise ValueError(f"Could not read the levels: {exc}") from exc
    if not levels or ie is None:
        raise ValueError("No levels or no ionization limit found; use the NIST ASD levels export with the Limit line.")
    return f"{len(levels)} levels, ionization energy {ie:.4f} eV"


def check_lines(path: Path, species_id: str) -> str:
    from linak.physics.polarizability_core import load_nist_fvalues

    ext = ".csv" if path.suffix.lower() == ".csv" else ".txt"
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copyfile(path, Path(tmp) / f"{species_id}_lines{ext}")
        try:
            res = load_nist_fvalues(species_id, f_values_dir=tmp, verbose=False)
        except Exception as exc:
            raise ValueError(f"Could not read the lines: {exc}") from exc
    n = len(res.get("fosc_line") or {}) or len(res.get("fosc_j") or {}) or len(res.get("fosc") or {})
    if not n:
        raise ValueError("No usable lines found; use the NIST ASD lines export with the gA and fik columns.")
    return f"{n} lines with f-values"


def import_file(key: str, species: str, source: str | Path) -> str:
    """Check *source* and copy it where the stages look for the species' levels or lines; returns a short summary."""
    sid, _el = resolve(species)
    item, src = BY_KEY[key], Path(source)
    if not src.is_file():
        raise ValueError(f"{src} is not a file.")
    summary = check_levels(src) if key == "levels" else check_lines(src, sid)
    target = user_tables.override_file_path(_target(item, sid, src))
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, target)
    return summary


# --- hyperfine constants -----------------------------------------------------------------
_HF_LABEL = re.compile(r"^(\d+)([spdfgh])([\d/]+)?$")


def hf_path(data_dir: str | Path, species_id: str) -> Path:
    return Path(data_dir) / f"{species_id}_hf_constants.json"


def read_hf_constants(data_dir: str | Path, species_id: str) -> tuple[int | None, list[tuple[str, float, float, float]]]:
    """``(mass number, [(state, A_MHz, B_MHz, C_kHz)])`` of the file; ``(None, [])`` when it has no states (a template has none)."""
    try:
        raw = json.loads(hf_path(data_dir, species_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, []
    rows = [(k, float(v["A_MHz"]), float(v.get("B_MHz") or 0.0), float(v.get("C_kHz") or 0.0))
            for k, v in raw.items() if not k.startswith("_") and isinstance(v, dict) and isinstance(v.get("A_MHz"), (int, float))]
    if not rows:
        return None, []
    m = re.match(r"\s*[A-Za-z]+-(\d+)", str(raw.get("_isotope", "")))
    return (int(m.group(1)) if m else 0), rows


def nuclear_spin(element: str, mass_number: int) -> float | None:
    """Nuclear spin of an isotope from the bundled nuclear data; None if the isotope is not listed."""
    try:
        data = json.loads(reference_path("data_json", "nuclear_data.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    for iso in data.get("isotopes", []):
        if iso.get("symbol") == element and iso.get("A") == mass_number:
            return float(iso.get("I") or 0.0)
    return None


def check_hf(species: str, mass_text: str, rows: list[tuple[str, str, str, str]]) -> tuple[int, list[tuple[str, float, float, float]]]:
    """Validate what the user typed; raises ValueError naming what is wrong."""
    _sid, el = resolve(species)
    try:
        a = int(str(mass_text).strip())
    except ValueError:
        raise ValueError("Mass number is a whole number, such as 25.") from None
    spin = nuclear_spin(el, a)
    if spin is None:
        raise ValueError(f"{el}-{a} is not in the nuclear data (isotopes with spin 0 are not listed).")
    if spin == 0:
        raise ValueError(f"{el}-{a} has nuclear spin 0, so it has no hyperfine structure.")
    out, seen = [], set()
    for label, a_txt, b_txt, c_txt in rows:
        label = label.strip()
        if not label and not a_txt.strip():
            continue                                         # an empty row
        if not _HF_LABEL.match(label):
            raise ValueError(f"State '{label}' looks like 3s1/2 or 3p3/2.")
        if label in seen:
            raise ValueError(f"State {label} appears twice.")
        seen.add(label)
        nums = []
        for name, txt in (("A", a_txt), ("B", b_txt), ("C", c_txt)):
            try:
                v = float(txt.strip().replace(",", ".") or ("nan" if name == "A" else "0"))
            except ValueError:
                raise ValueError(f"{name} of {label} must be a number.") from None
            if not math.isfinite(v):
                raise ValueError(f"Fill in A of {label}." if name == "A" else f"{name} of {label} must be a number.")
            nums.append(v)
        out.append((label, *nums))
    if not out:
        raise ValueError("Enter at least one state.")
    return a, out


def write_hf_constants(data_dir: str | Path, species: str, mass_number: int, rows: list[tuple[str, float, float, float]]) -> Path:
    from linak.core.fileio import write_json

    sid, el = resolve(species)
    doc: dict = {"_source": "entered in LiNaK", "_isotope": f"{el}-{mass_number} (I={nuclear_spin(el, mass_number):g})"}
    for label, a, b, c in rows:
        doc[label] = {"A_MHz": a, "B_MHz": b, "C_kHz": c}
    path = hf_path(data_dir, sid)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, doc)
    return path
