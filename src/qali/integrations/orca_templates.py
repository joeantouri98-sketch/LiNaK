# Full periodic table, derived from core.elements (Z=1..118).
from __future__ import annotations

import re
from typing import cast

from qali.core.elements import ELEMENTS as _ELEMENTS_TABLE
ATOMIC_NUMBERS = {sym: z for z, (sym, _c, _m) in _ELEMENTS_TABLE.items()}

ALKALI_NEUTRAL_DOUBLET = {"Li", "Na", "K", "Rb", "Cs", "Fr"}

# Element classes for SOC (extend by adding to SOC_IMPLEMENTED_CLASSES) 
class ElementClass:
    ALKALI_VALENCE   = "ALKALI_VALENCE"
    HALOGEN          = "HALOGEN"
    CHALCOGEN        = "CHALCOGEN"
    ALKALINE_EARTH   = "ALKALINE_EARTH"
    NOBLE            = "NOBLE"
    TRANSITION_METAL = "TRANSITION_METAL"

SOC_IMPLEMENTED_CLASSES = {
    ElementClass.ALKALI_VALENCE,
    ElementClass.HALOGEN,
    ElementClass.CHALCOGEN,
    ElementClass.ALKALINE_EARTH,
    ElementClass.NOBLE,
    ElementClass.TRANSITION_METAL,
}

HALOGEN_SYMBOLS        = {"F", "Cl", "Br", "I", "At"}
CHALCOGEN_SYMBOLS      = {"O", "S", "Se", "Te", "Po"}
ALKALINE_EARTH_SYMBOLS = {"Be", "Mg", "Ca", "Sr", "Ba", "Ra"}
NOBLE_SYMBOLS          = {"He", "Ne", "Ar", "Kr", "Xe", "Rn"}
# First-row (3d) transition metals. 4d/5d are a later phase.
TRANSITION_METAL_3D    = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
}

# Per-element CAS spatial orbital count; recipe-dependent defaults
_ALKALI_SOC_NORB_NEVPT2 = {
    "Li": 6, "Na": 8, "K": 10, "Rb": 10, "Cs": 12, "Fr": 12,
}
_ALKALI_SOC_NORB_CASSCF = {
    "Li": 6, "Na": 8, "K": 8, "Rb": 8, "Cs": 8, "Fr": 8,
}
SOC_RECIPE_CHOICES = ("casscf", "nevpt2")
_DEFAULT_SOC_NROOTS = {
    ElementClass.ALKALI_VALENCE: 6,
    ElementClass.HALOGEN: 6,        # ground 2P (3 roots) + np4(n+1)s doublets
    ElementClass.CHALCOGEN: 6,      # 3P (3 roots) / 1D+1S (6 roots)
    ElementClass.ALKALINE_EARTH: 4, # 1S ground + 1P; 3P in the triplet block
    ElementClass.NOBLE: 5,          # 1S ground + np5(n+1)s excited manifold
    ElementClass.TRANSITION_METAL: 5,  # Fe 5D default; per-symbol via _SOC_TM_BY_SYMBOL
}

# CI multiplicity blocks state-averaged in the SOC CASSCF, per class.
# SOC mixes these blocks via QDPT (DoSOC); ground multiplicity listed first.
# TRANSITION_METAL mults are per-symbol (_SOC_TM_BY_SYMBOL), not listed here.
_SOC_CLASS_MULTS = {
    ElementClass.HALOGEN: (2, 4),         # 2P ground; 4P from np4(n+1)s
    ElementClass.CHALCOGEN: (3, 1),       # 3P ground; 1D, 1S
    ElementClass.ALKALINE_EARTH: (1, 3),  # 1S ground; 3P intercombination
    ElementClass.NOBLE: (1, 3),           # 1S ground; 3P from np5(n+1)s
}

# Active-space data per class: nel = valence electrons kept in the CAS,
# norb by recipe (casscf = minimal, nevpt2 = extended with Rydberg/d shells).
_SOC_CLASS_ACTIVE = {
    ElementClass.HALOGEN: {
        "nel": 7,                       # ns2 np5
        "norb_casscf": 5,               # ns + 3np + (n+1)s — (n+1)s is required:
                                        # the 4P block needs np4(n+1)s; with only
                                        # 4 orbitals mult=4 has zero CSFs (ORCA NORB==0)
        "norb_nevpt2": 8,               # + 3(n+1)p
        "description": "ns/np/(n+1)s/(n+1)p CAS",
    },
    ElementClass.CHALCOGEN: {
        "nel": 6,                       # ns2 np4
        "norb_casscf": 4,
        "norb_nevpt2": 8,
        "description": "ns/np/(n+1)s/(n+1)p CAS",
    },
    ElementClass.ALKALINE_EARTH: {
        "nel": 2,                       # ns2
        "norb_casscf": 4,               # ns + 3np
        "norb_nevpt2": 9,               # + 5 (n-1)d/nd
        "description": "ns/np/nd CAS",
    },
    ElementClass.NOBLE: {
        "nel": 8,                       # ns2 np6
        "norb_casscf": 5,               # ns + 3np + (n+1)s (minimal for excitations)
        "norb_nevpt2": 8,               # + 3(n+1)p
        "description": "ns/np/(n+1)s/(n+1)p CAS",
    },
    # Class-level TM fallback; real nel/norb/mults come from _SOC_TM_BY_SYMBOL.
    ElementClass.TRANSITION_METAL: {
        "nel": 8,
        "norb_casscf": 6,               # 4s + five 3d
        "norb_nevpt2": 6,
        "description": "4s/3d CAS",
    },
}

# 3d TM per-symbol CAS + state-average settings.
# nel = 4s+3d valence electrons; norb = 6 (4s + 5×3d) for casscf and nevpt2.
# Pilot: Fe — ground a⁵D only (mult 5, nroots 5). Wider SA (⁵F/³F) needs a
# larger CAS; extra roots in CAS(8,6) were dirty / NEVPT2-unstable.
# Others are ground-term heuristics, not yet smoke-tested.
_SOC_TM_BY_SYMBOL = {
    # symbol: nel, norb, mults (ground first), nroots (per mult block unless
    # nroots_per_mult is set)
    "Sc": {"nel": 3,  "norb": 6, "mults": (2,), "nroots": 5},   # ²D  3d1 4s2
    "Ti": {"nel": 4,  "norb": 6, "mults": (3,), "nroots": 7},   # ³F  3d2 4s2
    "V":  {"nel": 5,  "norb": 6, "mults": (4,), "nroots": 7},   # ⁴F  3d3 4s2
    # Cr: a⁷S ground + low a⁵S (same 3d5.4s) + a⁵D (3d4.4s2) in CAS(6,6).
    "Cr": {"nel": 6,  "norb": 6, "mults": (7, 5), "nroots": 1,
           "nroots_per_mult": (1, 6)},  # ⁷S×1 + ⁵S×1 + ⁵D×5
    "Mn": {"nel": 7,  "norb": 6, "mults": (6,), "nroots": 1},   # ⁶S  3d5 4s2
    "Fe": {"nel": 8,  "norb": 6, "mults": (5,), "nroots": 5},   # ⁵D  3d6 4s2  (pilot)
    "Co": {"nel": 9,  "norb": 6, "mults": (4,), "nroots": 7},   # ⁴F  3d7 4s2
    "Ni": {"nel": 10, "norb": 6, "mults": (3,), "nroots": 7},   # ³F  3d8 4s2
    "Cu": {"nel": 11, "norb": 6, "mults": (2,), "nroots": 1},   # ²S  3d10 4s1
    "Zn": {"nel": 12, "norb": 6, "mults": (1,), "nroots": 1},   # ¹S  3d10 4s2
}

# Per-symbol overrides where the generic class defaults don't fit.
_SOC_ACTIVE_SYMBOL_OVERRIDES = {
    "He": {"nel": 2, "norb_casscf": 5, "norb_nevpt2": 5},   # 1s + 2s + 3x2p
    "Be": {"norb_nevpt2": 5},                                # 2s + 3x2p + 3s (no low d)
}


def get_element_class(symbol: str, charge: int, mult: int, soc_class_hint=None):
    """
    Map (symbol, charge, mult) to an element class for SOC routing.

    Registered ions pass soc_class_hint (from species.resolve_species).
    Unregistered ions (charge != 0 without a hint) return None.
    """
    if charge != 0:
        if soc_class_hint and soc_class_hint in SOC_IMPLEMENTED_CLASSES:
            return soc_class_hint
        if soc_class_hint == ElementClass.TRANSITION_METAL:
            return ElementClass.TRANSITION_METAL
        return None
    if symbol in ALKALI_NEUTRAL_DOUBLET and mult == 2:
        return ElementClass.ALKALI_VALENCE
    if symbol in HALOGEN_SYMBOLS and mult == 2:
        return ElementClass.HALOGEN
    if symbol in CHALCOGEN_SYMBOLS and mult == 3:
        return ElementClass.CHALCOGEN
    if symbol in ALKALINE_EARTH_SYMBOLS and mult == 1:
        return ElementClass.ALKALINE_EARTH
    if symbol in NOBLE_SYMBOLS and mult == 1:
        return ElementClass.NOBLE
    if symbol in TRANSITION_METAL_3D:
        return ElementClass.TRANSITION_METAL
    return None


def require_soc_class(symbol: str, charge: int, mult: int, soc_class_hint=None) -> str:
    """Return element class or raise with a clear message."""
    el_class = get_element_class(symbol, charge, mult, soc_class_hint=soc_class_hint)
    if el_class is None:
        raise ValueError(
            f"No SOC element class for {symbol} (charge={charge}, mult={mult}). "
            f"Registered ions only (see species.py); unclassified atoms unsupported.")
    if el_class not in SOC_IMPLEMENTED_CLASSES:
        raise NotImplementedError(
            f"SOC recipe for class '{el_class}' is not implemented yet. "
            f"Implemented: {sorted(SOC_IMPLEMENTED_CLASSES)}")
    return el_class


def get_active_space(el_class: str, symbol: str, soc_recipe: str = "nevpt2",
                     norb_override=None) -> dict:
    """
    Active-space parameters for CASSCF/SOC by element class.

    soc_recipe:
      nevpt2 — larger CAS defaults for SC-NEVPT2 + DoSOC
      casscf — legacy v1 CAS(1,8)-style defaults for bare CASSCF + DoSOC
    """
    if el_class == ElementClass.ALKALI_VALENCE:
        if norb_override is not None:
            norb = norb_override
        elif soc_recipe == "casscf":
            norb = _ALKALI_SOC_NORB_CASSCF.get(symbol, 8)
        else:
            norb = _ALKALI_SOC_NORB_NEVPT2.get(symbol, 10)
        return {
            "nel": 1,
            "norb": norb,
            "description": f"valence ns/np/(n-1)d CAS(1,{norb})",
        }

    if el_class == ElementClass.TRANSITION_METAL:
        tm = _SOC_TM_BY_SYMBOL.get(symbol)
        if tm is None:
            raise NotImplementedError(
                f"No 3d TM SOC active space for {symbol}. "
                f"Supported: {sorted(_SOC_TM_BY_SYMBOL)}")
        norb = norb_override if norb_override is not None else tm["norb"]
        nel = tm["nel"]
        return {
            "nel": nel,
            "norb": norb,
            "description": f"4s/3d CAS({nel},{norb})",
        }

    if el_class in _SOC_CLASS_ACTIVE:
        base = dict(_SOC_CLASS_ACTIVE[el_class])
        base.update(_SOC_ACTIVE_SYMBOL_OVERRIDES.get(symbol, {}))
        nel = base["nel"]
        if norb_override is not None:
            norb = norb_override
        elif soc_recipe == "casscf":
            norb = base["norb_casscf"]
        else:
            norb = base["norb_nevpt2"]
        return {
            "nel": nel,
            "norb": norb,
            "description": f"{base['description']}({nel},{norb})",
        }

    raise NotImplementedError(f"No active space defined for class {el_class}")


def _soc_recipe_keywords(recipe: dict, soc_recipe: str) -> dict:
    """Return a copy of an SOC recipe dict adjusted for casscf vs nevpt2."""
    r = dict(recipe)
    if soc_recipe == "casscf":
        r["simple_keywords"] = " ".join(
            kw for kw in r["simple_keywords"].split() if kw != "SARC/J"
        )
    return r


def default_soc_nroots(el_class: str, symbol: str | None = None) -> int:
    """Default nroots for SOC SA. TM values are per-symbol when provided."""
    if el_class == ElementClass.TRANSITION_METAL and symbol:
        tm = _SOC_TM_BY_SYMBOL.get(symbol)
        if tm:
            return cast(int, tm["nroots"])
    return _DEFAULT_SOC_NROOTS.get(el_class, 6)


def _sarc_dkh_recipe(symbol: str) -> dict:
    """All-electron DKH + SARC-DKH-TZVP (shared by TD-DFT heavy atoms and SOC Z≥19)."""
    return {
        "simple_keywords": "DKH",
        "basis_keyword": "",
        "basis_block": f"""%basis
  NewGTO {symbol} "SARC-DKH-TZVP" end
end""",
        "rel_block": """%rel
  FiniteNuc true
end""",
    }


def _build_basis_block(symbol: str, z: int, basis_name: str = 'ANO-RCC-VTZP') -> str:
    """
    Build an inline ORCA NewGTO basis block for any BSE basis.
    Works for both:
      - General (ANO-type) contractions: all primitives contribute to every
        contracted function → emit all nprim primitives per contraction.
      - Segmented (cc-type) contractions: each contracted function uses only
        a subset of primitives (the rest have zero coefficients) → emit only
        the nonzero primitives. This is required for aug-cc-pVXZ-DK3 etc.;
        ORCA's AddNewGTOs rejects "S 38" when only 1 primitive is nonzero.

    The detection is automatic: if any coefficient in a contraction set has
    abs(coef) < 1e-14, that primitive is skipped for that contraction.
    For ANO bases all coefficients are nonzero so behaviour is unchanged.
    """
    try:
        import basis_set_exchange as bse
    except ImportError:
        raise ImportError(
            "basis_set_exchange is required for Z>=87 elements.\n"
            "Install with: pip install basis-set-exchange"
        )
    L_NAMES = {0: 'S', 1: 'P', 2: 'D', 3: 'F', 4: 'G', 5: 'H', 6: 'I'}
    raw    = bse.get_basis(basis_name, elements=[z])
    shells = raw['elements'][str(z)]['electron_shells']
    lines  = ['%basis', f'  NewGTO {symbol}']
    for shell in shells:
        ams   = shell['angular_momentum']
        exps  = shell['exponents']
        coefs = shell['coefficients']
        for k, coef_set in enumerate(coefs):
            # A combined SP shell lists angular_momentum [0, 1] with one coefficient set per l;
            # otherwise (general contraction) every set shares ams[0]. Never fall back to S silently.
            am = ams[k] if len(ams) == len(coefs) else ams[0]
            if am not in L_NAMES:
                raise ValueError(f"BSE basis {basis_name}: angular momentum l={am} not supported")
            lname = L_NAMES[am]
            # Keep only primitives with nonzero coefficients for this contraction
            nonzero = [(float(exps[i]), float(coef_set[i]))
                       for i in range(len(exps))
                       if abs(float(coef_set[i])) > 1e-14]
            if not nonzero:
                continue  # skip fully-zero contraction (shouldn't happen)
            lines.append(f'  {lname} {len(nonzero)}')
            for rank, (exp, coef) in enumerate(nonzero, 1):
                lines.append(f'    {rank}  {exp:>20.8f}  {coef:>18.10f}')
    lines.append('  end')
    lines.append('end')
    return '\n'.join(lines)


def basis_block_from_file(path: str, symbol: str) -> str:
    """
    Build an inline ORCA %basis / NewGTO block from an external basis file.

    Accepted formats (Basis Set Exchange downloads):
      - GAMESS-US: $DATA ... $END  (common BSE default)
      - ORCA:      NewGTO ... end (optionally already wrapped in %basis)

    The NewGTO element label is always rewritten to ``symbol`` so a Li file
    can be used for Li regardless of whether the download says LITHIUM or Li.
    """
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    if re.search(r'\$\s*DATA', text, re.IGNORECASE):
        return _basis_block_from_gamess(text, symbol)
    if re.search(r'\bNewGTO\b', text, re.IGNORECASE):
        return _basis_block_from_orca_text(text, symbol)
    raise ValueError(
        f"Unrecognized basis format in {path}. "
        "Download GAMESS-US or ORCA format from basissetexchange.org."
    )


def _basis_block_from_orca_text(text: str, symbol: str) -> str:
    """Normalize an ORCA-format basis download into a %basis block."""
    m = re.search(
        r'NewGTO\s+\S+(.*?)(?:\n\s*end\s*)',
        text, re.IGNORECASE | re.DOTALL)
    if not m:
        raise ValueError("ORCA basis file: could not find NewGTO ... end")
    body = m.group(1).strip('\n')
    # Drop a trailing 'end' that closed %basis if present inside capture
    lines_body = [ln.rstrip() for ln in body.splitlines() if ln.strip()]
    while lines_body and lines_body[-1].strip().lower() == 'end':
        lines_body.pop()
    out = ['%basis', f'  NewGTO {symbol}']
    for ln in lines_body:
        s = ln.strip()
        if not s or s.startswith('!'):
            continue
        out.append(f'  {s}' if not ln.startswith(' ') else ln.rstrip())
    out.append('  end')
    out.append('end')
    return '\n'.join(out)


def _basis_block_from_gamess(text: str, symbol: str) -> str:
    """
    Convert GAMESS-US / BSE $DATA basis text to ORCA NewGTO.

    Expected shell records:
        S   11
        1      5988.0000000              0.0001330
        ...
    """
    lines = text.splitlines()
    start = None
    end = None
    for i, ln in enumerate(lines):
        if re.match(r'^\s*\$\s*DATA\b', ln, re.IGNORECASE):
            start = i + 1
        elif start is not None and re.match(r'^\s*\$\s*END\b', ln, re.IGNORECASE):
            end = i
            break
    if start is None or end is None:
        raise ValueError("GAMESS basis file: missing $DATA ... $END")

    # Skip title / element-name lines until first shell header (S/P/D/...)
    shell_hdr = re.compile(
        r'^\s*([SPDFGHILspdfghil])\s+(\d+)\s*$')
    _num = r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[DdEe][+-]?\d+)?'
    # index, exponent, coefficient [, second coefficient for an L (SP) shell]
    prim_line = re.compile(rf'^\s*(\d+)\s+({_num})\s+({_num})(?:\s+({_num}))?\s*$')

    i = start
    # consume non-shell preamble (e.g. LITHIUM)
    while i < end and not shell_hdr.match(lines[i]):
        i += 1

    out = ['%basis', f'  NewGTO {symbol}']
    n_shells = 0
    while i < end:
        m = shell_hdr.match(lines[i])
        if not m:
            i += 1
            continue
        lname = m.group(1).upper()
        nprim = int(m.group(2))
        i += 1
        prims: list[tuple[float, float, float | None]] = []   # exponent, coefficient(s)
        while i < end and len(prims) < nprim:
            if not lines[i].strip() or lines[i].lstrip().startswith('!'):
                i += 1
                continue
            pm = prim_line.match(lines[i])
            if not pm:
                break
            _f = lambda x: float(x.replace('D', 'E').replace('d', 'e'))
            prims.append((_f(pm.group(2)), _f(pm.group(3)),
                          _f(pm.group(4)) if pm.group(4) else None))
            i += 1
        if len(prims) != nprim:
            raise ValueError(
                f"GAMESS basis: shell {lname} expected {nprim} primitives, "
                f"got {len(prims)}"
            )
        if lname == 'L':
            # GAMESS SP shell: columns exponent, s-coefficient, p-coefficient -> an S and a P shell
            if any(p[2] is None for p in prims):
                raise ValueError("GAMESS basis: L shell needs two coefficient columns")
            for lab, col in (('S', 1), ('P', 2)):
                out.append(f'  {lab} {len(prims)}')
                for rank, p in enumerate(prims, 1):
                    out.append(f'    {rank}  {p[0]:>20.8f}  {p[col] if col == 2 else p[1]:>18.10f}')
                n_shells += 1
        else:
            if any(p[2] is not None for p in prims):
                raise ValueError(f"GAMESS basis: shell {lname} has an unexpected extra column")
            out.append(f'  {lname} {len(prims)}')
            for rank, p in enumerate(prims, 1):
                out.append(f'    {rank}  {p[0]:>20.8f}  {p[1]:>18.10f}')
            n_shells += 1

    if n_shells == 0:
        raise ValueError("GAMESS basis file: no shells found inside $DATA")
    out.append('  end')
    out.append('end')
    return '\n'.join(out)


# Keep old name as alias for backward compatibility
def _build_ano_rcc_block(symbol: str, z: int) -> str:
    """Backward-compatible wrapper — builds ANO-RCC-VTZP block."""
    return _build_basis_block(symbol, z, 'ANO-RCC-VTZP')


def atomic_number(symbol: str) -> int:
    if symbol not in ATOMIC_NUMBERS:
        raise ValueError(f"Unsupported or misspelled element symbol: {symbol}")
    return ATOMIC_NUMBERS[symbol]


def ks_ref(mult: int) -> str:
    return "RKS" if mult == 1 else "UKS"


def join_nonempty(lines):
    return "\n".join(line for line in lines if line.strip()) + "\n"


def get_recipe_tddft(symbol: str, relativistic_mode: str = "auto") -> dict:
    """
    TD-DFT / screening basis recipe.

    relativistic_mode:
      - "auto" : light atoms use nonrelativistic basis; heavy atoms use def2-ECP route
      - "ecp"  : force def2-ECP route for heavy atoms
      - "x2c"  : force all-electron scalar relativistic X2C route for heavy atoms
      - "dkh"  : force all-electron scalar relativistic DKH/SARC route for heavy atoms
    """
    Z = atomic_number(symbol)

    # Light atoms
    if Z <= 18:
        return {
            "simple_keywords": "",
            "basis_keyword": "aug-cc-pVTZ",
            "basis_block": "",
            "rel_block": ""
        }

    # K-Kr: keep your def2-TZVPD choice
    if 19 <= Z <= 36:
        return {
            "simple_keywords": "",
            "basis_keyword": "def2-TZVPD",
            "basis_block": "",
            "rel_block": ""
        }

    # Rb (Z=37-54): def2-TZVPD with automatic ECP
    # ORCA 6 applies def2-ECP automatically for Z>=37 when def2-TZVPD is used.
    # The ECP implicitly encodes relativistic effects and keeps a good valence
    # basis for diffuse Rydberg excitations. Gives 28 meV error on Rb 5p D-line
    # vs 148 meV with all-electron SARC-DKH-TZVP.
    if relativistic_mode in ("auto", "ecp") and Z <= 54:
        return {
            "simple_keywords": "",
            "basis_keyword": "def2-TZVPD",
            "basis_block": "",
            "rel_block": ""
        }

    # Cs-Rn (Z=55-86): SARC-DKH-TZVP + DKH (all-electron relativistic)
    # def2-ECP for Cs gives worse level spacing than all-electron DKH treatment.
    # SARC-DKH-TZVP is explicitly contracted for the DKH Hamiltonian.
    # SARC-DKH-TZVP only covers up to Z=86 (Rn) in ORCA 6.1.
    if relativistic_mode in ("auto", "ecp", "dkh") and 55 <= Z <= 86:
        return {
            "simple_keywords": "DKH",
            "basis_keyword": "",
            "basis_block": f"""%basis
  NewGTO {symbol} "SARC-DKH-TZVP" end
end""",
            "rel_block": """%rel
  FiniteNuc true
end"""
        }

    # Fr and heavier (Z>=87): ANO-RCC-VTZP inline via NewGTO + DKH + AutoAux
    # No internal basis set in ORCA 6.1.1 covers Z>=87 (SARC/X2C/def2 all fail).
    # ANO-RCC-VTZP covers H-Cm (Z=1-96) and is fetched at runtime from the
    # basis_set_exchange Python package (pip install basis-set-exchange).
    # The basis is embedded directly in the input via NewGTO block with indexed
    # primitives. AutoAux generates the auxiliary basis automatically.
    # DKH via %rel block (not as simple keyword -- causes exit 4 with NewGTO).
    # Confirmed working for Fr: 87 electrons (all-electron), Dim=97.
    if relativistic_mode in ("auto", "ecp", "dkh") and Z >= 87:
        return {
            "simple_keywords": "AutoAux",
            "basis_keyword": "",
            "basis_block": _build_ano_rcc_block(symbol, Z),
            "rel_block": """%rel
  method DKH
  order 2
  FiniteNuc true
end"""
        }

    if relativistic_mode == "x2c":
        return {
            "simple_keywords": "DLU-X2C X2C/J",
            "basis_keyword": "X2C-TZVPall",
            "basis_block": "",
            "rel_block": """%rel
  FiniteNuc true
end"""
        }

    if relativistic_mode == "dkh":
        return {
            "simple_keywords": "DKH",
            "basis_keyword": "",
            "basis_block": f"""%basis
  NewGTO {symbol} "SARC-DKH-TZVP" end
end""",
            "rel_block": """%rel
  FiniteNuc true
end"""
        }

    raise ValueError(f"Unknown relativistic_mode: {relativistic_mode}")




def get_recipe_soc(symbol: str) -> dict:
    """
    All-electron relativistic recipe for CASSCF/NEVPT2 + SOC.

    Never uses def2-ECP. ORCA coverage:
      Li–Ar (Z≤18): aug-cc-pVTZ + SARC/J
      K–Kr (19–36): DKH-def2-TZVP + DKH + SARC/J
      Rb–Rn (37–86): SARC-DKH-TZVP + DKH + SARC/J
      Fr+ (≥87): ANO-RCC-VTZP inline + DKH + AutoAux (no SARC/J — fails on Fr)
    """
    Z = atomic_number(symbol)

    if Z <= 18:
        return {
            "simple_keywords": "SARC/J",
            "basis_keyword": "aug-cc-pVTZ",
            "basis_block": "",
            "rel_block": "",
        }

    if Z <= 36:
        return {
            "simple_keywords": "DKH SARC/J",
            "basis_keyword": "DKH-def2-TZVP",
            "basis_block": "",
            "rel_block": """%rel
  FiniteNuc true
end""",
        }

    if Z <= 86:
        r = _sarc_dkh_recipe(symbol)
        sk = r["simple_keywords"]
        r["simple_keywords"] = "SARC/J" if not sk else f"{sk} SARC/J"
        return r

    return {
        "simple_keywords": "AutoAux",
        "basis_keyword": "",
        "basis_block": _build_ano_rcc_block(symbol, Z),
        "rel_block": """%rel
  method DKH
  order 2
  FiniteNuc true
end""",
    }


DEFAULT_FUNCTIONAL = "CAM-B3LYP"


def common_scf_block(maxiter: int = 400) -> str:
    return f"""%scf
  MaxIter {maxiter}
end"""


def moinp_block(gbw_filename: str) -> str:
    """Reuse converged ground-state orbitals in the excited-state job."""
    # %moinp is a one-line directive in ORCA — no trailing 'end'
    return f'%moinp "{gbw_filename}"'


def _keyword_line(ref, functional, basis_keyword, simple_keywords,
                  extra=(), scf_keyword="TightSCF"):
    parts = [p for p in [ref, functional, basis_keyword, simple_keywords,
                         scf_keyword, *extra] if p and str(p).strip()]
    return "! " + " ".join(parts)


def basis_label_tddft(symbol: str, relativistic_mode: str = "auto") -> str:
    """Human-readable TD-DFT basis name for JSON metadata."""
    r = get_recipe_tddft(symbol, relativistic_mode=relativistic_mode)
    if r["basis_keyword"]:
        return r["basis_keyword"]
    # Basis-less recipes embed the basis in an inline %basis block; identify it
    # from the block so forced --relativistic dkh (Z<87) is labelled correctly.
    if "SARC-DKH-TZVP" in r["basis_block"]:
        return "SARC-DKH-TZVP"
    z = atomic_number(symbol)
    if z >= 87:
        return "ANO-RCC-VTZP (inline NewGTO)"
    return "def2-TZVPD" if z >= 19 else "aug-cc-pVTZ"


def basis_label_soc(symbol: str) -> str:
    """Human-readable SOC basis name for JSON metadata."""
    r = get_recipe_soc(symbol)
    if r["basis_keyword"]:
        return r["basis_keyword"]
    z = atomic_number(symbol)
    if z >= 87:
        return "ANO-RCC-VTZP (inline NewGTO)"
    return "SARC-DKH-TZVP"


basis_label = basis_label_tddft  # backward-compatible alias


def is_alkali_neutral(symbol: str, charge: int, mult: int) -> bool:
    return (symbol in ALKALI_NEUTRAL_DOUBLET and charge == 0 and mult == 2)


def is_alkali_like(symbol: str, charge: int, mult: int, soc_class_hint=None) -> bool:
    """Neutral alkali doublet or registered alkali-like ion (e.g. Mg_c1)."""
    if is_alkali_neutral(symbol, charge, mult):
        return True
    return (
        charge != 0 and mult == 2
        and soc_class_hint == ElementClass.ALKALI_VALENCE
    )


def select_calculation_mode(symbol: str, charge: int, mult: int,
                            mode: str = "auto", soc_class_hint=None) -> str:
    """
    mode:
      auto       — alkali doublets -> alkali TD-DFT; registered ions with
                   soc_class -> soc; others -> screening TD-DFT
      tddft      — force generic TD-DFT
      alkali     — force alkali TD-DFT (alkali / alkali-like only)
      soc        — CASSCF/NEVPT2 + SOC (element-class routing)
      alkali-soc — alias for soc (alkali only)
      casscf     — CASSCF + SC-NEVPT2 single job, no SOC (alkali only)
    """
    if mode == "alkali-soc":
        return "soc"
    if mode == "auto":
        if is_alkali_neutral(symbol, charge, mult):
            return "alkali"
        if charge != 0 and soc_class_hint:
            return "soc"
        return "tddft"
    return mode


def calculation_metadata(symbol, charge, mult, calc_mode, functional,
                         nroots, relativistic_mode="auto", soc_recipe="nevpt2",
                         norb_override=None, soc_class_hint=None):
    """Provenance block written into data/<element>.json."""
    el_class = get_element_class(symbol, charge, mult, soc_class_hint=soc_class_hint)

    if calc_mode in ("soc", "alkali-soc"):
        if soc_recipe == "casscf":
            method = "CASSCF/SOC"
            functional_ex = "CASSCF+SOC"
        else:
            method = "CASSCF/NEVPT2/SOC"
            functional_ex = "SC-NEVPT2+SOC"
        template = "excited_states_soc"
    elif calc_mode == "casscf":
        method = "CASSCF/SC-NEVPT2"
        template = "excited_states_atomic_alkali"
        functional_ex = "SC-NEVPT2"
    elif calc_mode == "alkali":
        method = "TD-DFT/TDA"
        template = "excited_states_alkali"
        functional_ex = functional
    else:
        method = "TD-DFT/TDA"
        template = "excited_states"
        functional_ex = functional

    quality = "benchmark"
    if calc_mode in ("soc", "alkali-soc"):
        quality = "soc_dline"
    elif calc_mode == "casscf":
        quality = "casscf_nevpt2"
    elif mult > 2:
        quality = "screening_only"

    meta = {
        "method": method,
        "template": template,
        "functional_gs": functional,
        "functional_ex": functional_ex,
        "basis": basis_label_tddft(symbol, relativistic_mode),
        "basis_tddft": basis_label_tddft(symbol, relativistic_mode),
        "relativistic_mode": relativistic_mode,
        "nroots": nroots,
        "orca_quality": quality,
    }
    if calc_mode in ("soc", "alkali-soc"):
        meta["spin_orbit"] = True
        meta["basis_soc"] = basis_label_soc(symbol)
        meta["element_class"] = el_class
        meta["soc_recipe"] = soc_recipe
        act = (get_active_space(el_class, symbol, soc_recipe=soc_recipe,
                                norb_override=norb_override)
               if el_class else None)
        if act:
            meta["active_space"] = act
    return meta


def xyz_block(symbol: str, charge: int, mult: int) -> str:
    return f"""* xyz {charge} {mult}
{symbol} 0.0 0.0 0.0
*"""


def ground_state(symbol, charge, mult,
                 functional=DEFAULT_FUNCTIONAL,
                 relativistic_mode="auto",
                 scf_keyword="TightSCF"):
    ref = ks_ref(mult)
    r = get_recipe_tddft(symbol, relativistic_mode=relativistic_mode)

    return join_nonempty([
        _keyword_line(ref, functional, r['basis_keyword'],
                      r['simple_keywords'], scf_keyword=scf_keyword),
        common_scf_block(500),
        r["rel_block"],
        r["basis_block"],
        xyz_block(symbol, charge, mult)
    ])


def excited_states(symbol, charge, mult, nroots=20,
                   functional=DEFAULT_FUNCTIONAL,
                   relativistic_mode="auto",
                   moinp_gbw=None,
                   scf_keyword="TightSCF"):
    """
    Generic screening excited-state template.
    """
    ref = ks_ref(mult)
    r = get_recipe_tddft(symbol, relativistic_mode=relativistic_mode)
    maxdim = max(5 * nroots, 100)

    tddft_block = f"""%tddft
  nroots {nroots}
  maxdim {maxdim}
  tda true
end"""

    blocks = [
        _keyword_line(ref, functional, r['basis_keyword'],
                      r['simple_keywords'], scf_keyword=scf_keyword),
        common_scf_block(500),
    ]
    if moinp_gbw:
        blocks.append(moinp_block(moinp_gbw))
    blocks += [r["rel_block"], r["basis_block"], tddft_block,
               xyz_block(symbol, charge, mult)]
    return join_nonempty(blocks)


def excited_states_alkali(symbol, charge, mult, nroots=20,
                          functional=DEFAULT_FUNCTIONAL,
                          relativistic_mode="auto",
                          moinp_gbw=None,
                          scf_keyword=None,
                          soc_class_hint=None):
    """
    Improved TD-DFT for alkali / alkali-like ns->np excitations.
    Same functional for GS+EX; reads GS orbitals via %moinp when provided.
    """
    if not is_alkali_like(symbol, charge, mult, soc_class_hint=soc_class_hint):
        raise ValueError(
            "excited_states_alkali requires a neutral alkali doublet "
            "or registered alkali-like ion (e.g. Mg_c1)")

    if scf_keyword is None:
        scf_keyword = "VeryTightSCF"

    return excited_states(
        symbol, charge, mult, nroots=nroots, functional=functional,
        relativistic_mode=relativistic_mode, moinp_gbw=moinp_gbw,
        scf_keyword=scf_keyword,
    )


def _casscf_soc_block(mults, nroots_per_mult, nel, norb, soc=False):
    """CASSCF block; mults/nroots_per_mult are sequences for SA over spin blocks."""
    mult_str = ",".join(str(m) for m in mults)
    nroots_str = ",".join(str(n) for n in nroots_per_mult)
    rel = "  rel\n    DoSOC true\n  end\n" if soc else ""
    return f"""%casscf
  nel {nel}
  norb {norb}
  mult {mult_str}
  nroots {nroots_str}
  MaxIter 200
{rel}end"""


def _casscf_alkali_block(mult, nroots, nel=1, norb=8, soc=False):
    return _casscf_soc_block([mult], [nroots], nel=nel, norb=norb, soc=soc)


def _build_soc_alkali_valence(symbol, charge, mult, nroots, soc_recipe="nevpt2",
                              norb_override=None, scf_keyword="VeryTightSCF"):
    """CASSCF(1,norb) + DoSOC; optional SC-NEVPT2 when soc_recipe='nevpt2'.

    Accepts neutral alkali doublets and registered alkali-like ions (e.g. Mg+).
    """
    if mult != 2 or (charge == 0 and symbol not in ALKALI_NEUTRAL_DOUBLET):
        raise ValueError(
            "_build_soc_alkali_valence requires mult=2 alkali / alkali-like "
            f"(got {symbol} charge={charge} mult={mult})")
    if charge != 0 and symbol in ALKALI_NEUTRAL_DOUBLET:
        raise ValueError(
            f"{symbol} ions are closed-shell (use NOBLE SOC), not ALKALI_VALENCE")

    act = get_active_space(
        ElementClass.ALKALI_VALENCE, symbol,
        soc_recipe=soc_recipe, norb_override=norb_override)
    r = _soc_recipe_keywords(get_recipe_soc(symbol), soc_recipe)
    extra = ("SC-NEVPT2",) if soc_recipe == "nevpt2" else ()

    return join_nonempty([
        _keyword_line("ROHF", "", r["basis_keyword"], r["simple_keywords"],
                      extra=extra, scf_keyword=scf_keyword),
        common_scf_block(500),
        r["rel_block"],
        r["basis_block"],
        _casscf_alkali_block(
            mult, nroots, nel=act["nel"], norb=act["norb"], soc=True),
        xyz_block(symbol, charge, mult),
    ])


def _soc_mults_for(el_class: str, symbol: str):
    """Spin blocks for SA-CASSCF+DoSOC (ground multiplicity first)."""
    if el_class == ElementClass.TRANSITION_METAL:
        tm = _SOC_TM_BY_SYMBOL.get(symbol)
        if tm is None:
            raise NotImplementedError(
                f"No 3d TM SOC multiplicity blocks for {symbol}. "
                f"Supported: {sorted(_SOC_TM_BY_SYMBOL)}")
        return tm["mults"]
    return _SOC_CLASS_MULTS[el_class]


def _build_soc_multiplet(symbol, charge, mult, el_class, nroots,
                         soc_recipe="nevpt2", norb_override=None,
                         scf_keyword="VeryTightSCF"):
    """
    Generic CASSCF(+SC-NEVPT2) + DoSOC builder for p-block / alkaline-earth /
    noble / 3d TM classes. State-averages over the class's spin blocks (e.g.
    3P+1D for chalcogens, ⁵D+⁵F/³F for Fe) so QDPT SOC can mix them.
    """
    mults = _soc_mults_for(el_class, symbol)
    if mults[0] != mult:
        raise ValueError(
            f"{symbol}: ground multiplicity {mult} does not match "
            f"class {el_class} (expected {mults[0]})")

    # Per-mult root counts (TM may set nroots_per_mult); else same nroots each.
    nroots_list = [nroots] * len(mults)
    if el_class == ElementClass.TRANSITION_METAL:
        tm = _SOC_TM_BY_SYMBOL.get(symbol) or {}
        npm = tm.get("nroots_per_mult")
        if npm is not None:
            if len(npm) != len(mults):
                raise ValueError(
                    f"{symbol}: nroots_per_mult length {len(npm)} != "
                    f"mults length {len(mults)}")
            nroots_list = list(npm)

    act = get_active_space(el_class, symbol,
                           soc_recipe=soc_recipe, norb_override=norb_override)
    r = _soc_recipe_keywords(get_recipe_soc(symbol), soc_recipe)
    extra = ("SC-NEVPT2",) if soc_recipe == "nevpt2" else ()
    ref = "RHF" if mult == 1 else "ROHF"

    return join_nonempty([
        _keyword_line(ref, "", r["basis_keyword"], r["simple_keywords"],
                      extra=extra, scf_keyword=scf_keyword),
        common_scf_block(500),
        r["rel_block"],
        r["basis_block"],
        _casscf_soc_block(mults, nroots_list,
                          nel=act["nel"], norb=act["norb"], soc=True),
        xyz_block(symbol, charge, mult),
    ])


def excited_states_soc(symbol, charge, mult, nroots=None, soc_recipe="nevpt2",
                       norb_override=None, scf_keyword="VeryTightSCF",
                       soc_class_hint=None):
    """
    Class-routed SOC template.

    Implemented classes:
      ALKALI_VALENCE   — CAS(1,n), single doublet block
      HALOGEN          — CAS(7,n), SA over 2P/4P blocks
      CHALCOGEN        — CAS(6,n), SA over 3P/1D-1S blocks
      ALKALINE_EARTH   — CAS(2,n), SA over 1S-1P/3P blocks
      NOBLE            — CAS(8,n), SA over 1S/3P blocks
      TRANSITION_METAL — CAS(4s+3d); Fe SA over ground a⁵D only (pilot)

    Registered ions pass soc_class_hint from species.resolve_species.

    soc_recipe:
      nevpt2 — CASSCF + SC-NEVPT2 + DoSOC (default)
      casscf — bare CASSCF + DoSOC (legacy v1-style)
    """
    if soc_recipe not in SOC_RECIPE_CHOICES:
        raise ValueError(f"soc_recipe must be one of {SOC_RECIPE_CHOICES}")
    el_class = require_soc_class(
        symbol, charge, mult, soc_class_hint=soc_class_hint)
    if nroots is None:
        nroots = default_soc_nroots(el_class, symbol=symbol)
    if el_class == ElementClass.ALKALI_VALENCE:
        return _build_soc_alkali_valence(
            symbol, charge, mult, nroots,
            soc_recipe=soc_recipe, norb_override=norb_override,
            scf_keyword=scf_keyword)
    if el_class == ElementClass.TRANSITION_METAL or el_class in _SOC_CLASS_MULTS:
        return _build_soc_multiplet(
            symbol, charge, mult, el_class, nroots,
            soc_recipe=soc_recipe, norb_override=norb_override,
            scf_keyword=scf_keyword)
    raise NotImplementedError(f"SOC builder missing for class {el_class}")


def excited_states_alkali_soc(symbol, charge, mult, nroots=6,
                              relativistic_mode="auto"):
    """Legacy alias — delegates to excited_states_soc."""
    del relativistic_mode
    return excited_states_soc(symbol, charge, mult, nroots=nroots)


def excited_states_atomic_alkali(symbol, charge, mult, nroots=20,
                                 relativistic_mode="auto",
                                 scf_keyword="VeryTightSCF"):
    """
    Higher-accuracy neutral alkali template: 1e CASSCF + SC-NEVPT2.
    Single-job calculation (no separate GS/TD-DFT step).
    """
    if not is_alkali_neutral(symbol, charge, mult):
        raise ValueError("atomic_alkali template is intended for neutral alkali doublets only")

    ref = ks_ref(mult)
    r = get_recipe_tddft(symbol, relativistic_mode=relativistic_mode)

    return join_nonempty([
        _keyword_line(ref, "", r['basis_keyword'], r['simple_keywords'],
                      extra=("SC-NEVPT2",), scf_keyword=scf_keyword),
        common_scf_block(500),
        r["rel_block"],
        r["basis_block"],
        _casscf_alkali_block(mult, nroots, soc=False),
        xyz_block(symbol, charge, mult)
    ])
