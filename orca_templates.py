# Full periodic table, derived from element_data.py (Z=1..118).
import re

from element_data import ELEMENTS as _ELEMENTS_TABLE
ATOMIC_NUMBERS = {sym: z for z, (sym, _c, _m) in _ELEMENTS_TABLE.items()}

ALKALI_NEUTRAL_DOUBLET = {"Li", "Na", "K", "Rb", "Cs", "Fr"}

# ── Element classes for SOC (extend by adding to SOC_IMPLEMENTED_CLASSES) ─────
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

# Per-element CAS spatial orbital count — recipe-dependent defaults
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


def default_soc_nroots(el_class: str, symbol: str = None) -> int:
    """Default nroots for SOC SA. TM values are per-symbol when provided."""
    if el_class == ElementClass.TRANSITION_METAL and symbol:
        tm = _SOC_TM_BY_SYMBOL.get(symbol)
        if tm:
            return tm["nroots"]
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
    L_NAMES = {0: 'S', 1: 'P', 2: 'D', 3: 'F', 4: 'G', 5: 'H'}
    raw    = bse.get_basis(basis_name, elements=[z])
    shells = raw['elements'][str(z)]['electron_shells']
    lines  = ['%basis', f'  NewGTO {symbol}']
    for shell in shells:
        am    = shell['angular_momentum'][0]
        exps  = shell['exponents']
        coefs = shell['coefficients']
        lname = L_NAMES.get(am, 'S')
        for coef_set in coefs:
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
        r'^\s*([SPDFGHIRspdfghir])\s+(\d+)\s*$')
    prim_line = re.compile(
        r'^\s*(\d+)\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+-]?\d+)?)\s+'
        r'([+-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+-]?\d+)?)\s*$')

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
        prims = []
        while i < end and len(prims) < nprim:
            if not lines[i].strip() or lines[i].lstrip().startswith('!'):
                i += 1
                continue
            pm = prim_line.match(lines[i])
            if not pm:
                break
            prims.append((float(pm.group(2)), float(pm.group(3))))
            i += 1
        if len(prims) != nprim:
            raise ValueError(
                f"GAMESS basis: shell {lname} expected {nprim} primitives, "
                f"got {len(prims)}"
            )
        out.append(f'  {lname} {len(prims)}')
        for rank, (exp, coef) in enumerate(prims, 1):
            out.append(f'    {rank}  {exp:>20.8f}  {coef:>18.10f}')
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

IC_NUMBERS = {
    'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Ne':10,
    'Na':11,'Mg':12,'Al':13,'Si':14,'P':15,'S':16,'Cl':17,'Ar':18,
    'K':19,'Ca':20,'Sc':21,'Ti':22,'V':23,'Cr':24,'Mn':25,'Fe':26,
    'Co':27,'Ni':28,'Cu':29,'Zn':30,'Ga':31,'Ge':32,'As':33,'Se':34,
    'Br':35,'Kr':36,'Rb':37,'Sr':38,'Cs':55,'Ba':56,'Fr':87,
}

ALKALI_NEUTRAL_DOUBLET = {"Li", "Na", "K", "Rb", "Cs", "Fr"}


# ANO-RCC-VTZP for Fr (Z=87) — embedded inline, no internal ORCA 6.1.1 basis covers Z>=87
_ANO_RCC_Fr = """%basis
  NewGTO Fr
  S 28
    1     54454690.90000000        0.0007877200
    2     13484523.60000000        0.0021009500
    3      4126956.97000000        0.0042211700
    4      1367732.73000000        0.0082924900
    5       484373.47600000        0.0154858800
    6       180062.99000000        0.0285646800
    7        69840.59720000        0.0518556700
    8        27990.41600000        0.0936044600
    9        11495.84880000        0.1647075900
    10         4830.57837000        0.2635391000
    11         2085.12306000        0.3272761300
    12          926.38513500        0.2274178100
    13          421.50108900        0.0478980000
    14          199.45202600        0.0009409100
    15           97.43309670       -0.0003190600
    16           48.88992000        0.0003704400
    17           24.70130010       -0.0003504600
    18           12.31757490        0.0002495800
    19            6.11916038       -0.0001455700
    20            2.83981136        0.0000768500
    21            1.29220895       -0.0000392200
    22            0.52733268        0.0000189900
    23            0.21185096       -0.0000104500
    24            0.08474038        0.0000060900
    25            0.03389615       -0.0000035600
    26            0.01355846        0.0000019700
    27            0.00542338       -0.0000009200
    28            0.00216935        0.0000002700
  S 28
    1     54454690.90000000       -0.0003175100
    2     13484523.60000000       -0.0008483900
    3      4126956.97000000       -0.0017093700
    4      1367732.73000000       -0.0033735400
    5       484373.47600000       -0.0063459800
    6       180062.99000000       -0.0118480100
    7        69840.59720000       -0.0219205500
    8        27990.41600000       -0.0408583600
    9        11495.84880000       -0.0757464800
    10         4830.57837000       -0.1328368500
    11         2085.12306000       -0.1887832400
    12          926.38513500       -0.1201563500
    13          421.50108900        0.2637668300
    14          199.45202600        0.5970955600
    15           97.43309670        0.2832169300
    16           48.88992000        0.0180680000
    17           24.70130010        0.0017790900
    18           12.31757490       -0.0010168900
    19            6.11916038        0.0003806200
    20            2.83981136       -0.0001752300
    21            1.29220895        0.0000937800
    22            0.52733268       -0.0000431900
    23            0.21185096        0.0000230300
    24            0.08474038       -0.0000134900
    25            0.03389615        0.0000078600
    26            0.01355846       -0.0000043500
    27            0.00542338        0.0000020400
    28            0.00216935       -0.0000005900
  S 28
    1     54454690.90000000        0.0001539500
    2     13484523.60000000        0.0004114700
    3      4126956.97000000        0.0008294700
    4      1367732.73000000        0.0016382200
    5       484373.47600000        0.0030856500
    6       180062.99000000        0.0057723700
    7        69840.59720000        0.0107161200
    8        27990.41600000        0.0200813500
    9        11495.84880000        0.0375908300
    10         4830.57837000        0.0670138900
    11         2085.12306000        0.0979209700
    12          926.38513500        0.0601636300
    13          421.50108900       -0.2090939800
    14          199.45202600       -0.6096233800
    15           97.43309670       -0.2464024500
    16           48.88992000        0.8075618500
    17           24.70130010        0.5539855400
    18           12.31757490        0.0355228400
    19            6.11916038        0.0075630500
    20            2.83981136       -0.0036981200
    21            1.29220895        0.0017075500
    22            0.52733268       -0.0008542800
    23            0.21185096        0.0004796200
    24            0.08474038       -0.0002780800
    25            0.03389615        0.0001627900
    26            0.01355846       -0.0000901900
    27            0.00542338        0.0000422400
    28            0.00216935       -0.0000121900
  S 28
    1     54454690.90000000       -0.0000788400
    2     13484523.60000000       -0.0002107400
    3      4126956.97000000       -0.0004248400
    4      1367732.73000000       -0.0008393100
    5       484373.47600000       -0.0015811400
    6       180062.99000000       -0.0029600600
    7        69840.59720000       -0.0054975700
    8        27990.41600000       -0.0103229400
    9        11495.84880000       -0.0193498600
    10         4830.57837000       -0.0346998700
    11         2085.12306000       -0.0508965400
    12          926.38513500       -0.0313544900
    13          421.50108900        0.1220050600
    14          199.45202600        0.3763156300
    15           97.43309670        0.1653714500
    16           48.88992000       -0.8994949300
    17           24.70130010       -0.7516404000
    18           12.31757490        0.9212417000
    19            6.11916038        0.7156981900
    20            2.83981136        0.0258557300
    21            1.29220895        0.0084519700
    22            0.52733268       -0.0039754900
    23            0.21185096        0.0021852400
    24            0.08474038       -0.0013117300
    25            0.03389615        0.0007703200
    26            0.01355846       -0.0004275100
    27            0.00542338        0.0002005300
    28            0.00216935       -0.0000578900
  S 28
    1     54454690.90000000        0.0000374000
    2     13484523.60000000        0.0000999800
    3      4126956.97000000        0.0002015700
    4      1367732.73000000        0.0003982300
    5       484373.47600000        0.0007503000
    6       180062.99000000        0.0014047300
    7        69840.59720000        0.0026097600
    8        27990.41600000        0.0049014700
    9        11495.84880000        0.0091949900
    10         4830.57837000        0.0165033800
    11         2085.12306000        0.0242595900
    12          926.38513500        0.0148907900
    13          421.50108900       -0.0597254600
    14          199.45202600       -0.1879923200
    15           97.43309670       -0.0839689500
    16           48.88992000        0.5226497900
    17           24.70130010        0.4741852400
    18           12.31757490       -0.8499127500
    19            6.11916038       -1.0212670200
    20            2.83981136        0.9572307500
    21            1.29220895        0.7467495500
    22            0.52733268       -0.0040546700
    23            0.21185096        0.0172972000
    24            0.08474038       -0.0103261100
    25            0.03389615        0.0061707400
    26            0.01355846       -0.0034535400
    27            0.00542338        0.0016239300
    28            0.00216935       -0.0004694900
  S 28
    1     54454690.90000000       -0.0000150800
    2     13484523.60000000       -0.0000403000
    3      4126956.97000000       -0.0000812600
    4      1367732.73000000       -0.0001605200
    5       484373.47600000       -0.0003024900
    6       180062.99000000       -0.0005662200
    7        69840.59720000       -0.0010523600
    8        27990.41600000       -0.0019755500
    9        11495.84880000       -0.0037096100
    10         4830.57837000       -0.0066513200
    11         2085.12306000       -0.0098028300
    12          926.38513500       -0.0059661900
    13          421.50108900        0.0241373700
    14          199.45202600        0.0769302800
    15           97.43309670        0.0336294000
    16           48.88992000       -0.2185129600
    17           24.70130010       -0.2084534600
    18           12.31757490        0.4058053400
    19            6.11916038        0.5296060300
    20            2.83981136       -0.6840105700
    21            1.29220895       -0.8198726700
    22            0.52733268        0.7228292600
    23            0.21185096        0.7604082800
    24            0.08474038        0.0116199200
    25            0.03389615        0.0228015300
    26            0.01355846       -0.0126661400
    27            0.00542338        0.0062823300
    28            0.00216935       -0.0018310000
  S 28
    1     54454690.90000000        0.0000038000
    2     13484523.60000000        0.0000101700
    3      4126956.97000000        0.0000205100
    4      1367732.73000000        0.0000405100
    5       484373.47600000        0.0000763400
    6       180062.99000000        0.0001429000
    7        69840.59720000        0.0002655700
    8        27990.41600000        0.0004986300
    9        11495.84880000        0.0009360900
    10         4830.57837000        0.0016791500
    11         2085.12306000        0.0024730300
    12          926.38513500        0.0015092800
    13          421.50108900       -0.0061075100
    14          199.45202600       -0.0194183200
    15           97.43309670       -0.0085722400
    16           48.88992000        0.0556471700
    17           24.70130010        0.0527765500
    18           12.31757490       -0.1039816400
    19            6.11916038       -0.1400547200
    20            2.83981136        0.1900161000
    21            1.29220895        0.2381343400
    22            0.52733268       -0.2599278700
    23            0.21185096       -0.3864847500
    24            0.08474038       -0.1162793900
    25            0.03389615        0.7327563600
    26            0.01355846        0.5199180500
    27            0.00542338       -0.0341290800
    28            0.00216935        0.0113310500
  S 28
    1     54454690.90000000       -0.0000174900
    2     13484523.60000000       -0.0000467600
    3      4126956.97000000       -0.0000942500
    4      1367732.73000000       -0.0001862500
    5       484373.47600000       -0.0003508200
    6       180062.99000000       -0.0006571400
    7        69840.59720000       -0.0012200300
    8        27990.41600000       -0.0022944800
    9        11495.84880000       -0.0042969400
    10         4830.57837000       -0.0077393800
    11         2085.12306000       -0.0113183900
    12          926.38513500       -0.0070866400
    13          421.50108900        0.0286010000
    14          199.45202600        0.0885133800
    15           97.43309670        0.0422473400
    16           48.88992000       -0.2673324400
    17           24.70130010       -0.2330105500
    18           12.31757490        0.4915553500
    19            6.11916038        0.7081291400
    20            2.83981136       -1.1859038900
    21            1.29220895       -1.0902992000
    22            0.52733268        3.2375338800
    23            0.21185096       -1.7366100400
    24            0.08474038       -0.7937565600
    25            0.03389615        0.5156313400
    26            0.01355846        0.2535325800
    27            0.00542338        0.0054370300
    28            0.00216935       -0.0006542100
  S 28
    1     54454690.90000000        0.0000215000
    2     13484523.60000000        0.0000576000
    3      4126956.97000000        0.0001157400
    4      1367732.73000000        0.0002298400
    5       484373.47600000        0.0004295600
    6       180062.99000000        0.0008149400
    7        69840.59720000        0.0014822000
    8        27990.41600000        0.0028792400
    9        11495.84880000        0.0051292100
    10         4830.57837000        0.0099812800
    11         2085.12306000        0.0126590700
    12          926.38513500        0.0122678500
    13          421.50108900       -0.0454018400
    14          199.45202600       -0.0833547900
    15           97.43309670       -0.1186449500
    16           48.88992000        0.4967448000
    17           24.70130010       -0.0258209300
    18           12.31757490       -0.1033674200
    19            6.11916038       -1.9448684000
    20            2.83981136        4.0799579300
    21            1.29220895       -2.7608152400
    22            0.52733268       -1.2026340400
    23            0.21185096        4.3296718100
    24            0.08474038       -4.3801785500
    25            0.03389615        1.8811105200
    26            0.01355846       -0.3262236100
    27            0.00542338        0.3373307600
    28            0.00216935       -0.0965346200
  P 25
    1     22059071.30000000        0.0000291900
    2      4330633.46000000        0.0000962000
    3      1041846.32000000        0.0002689600
    4       285439.05800000        0.0007259000
    5        86520.78280000        0.0019740800
    6        28615.02700000        0.0055005500
    7        10276.64610000        0.0155719100
    8         3994.98194000        0.0436412300
    9         1668.21525000        0.1139591800
    10          739.22152200        0.2489170000
    11          343.32211100        0.3809832200
    12          165.25013300        0.3086396500
    13           80.12487850        0.0860507300
    14           40.44458940        0.0012368200
    15           20.31220350        0.0012573000
    16           10.09863720       -0.0007988400
    17            4.97702486        0.0003764700
    18            2.28910027       -0.0001793000
    19            1.04199889        0.0000921800
    20            0.41679956       -0.0000420000
    21            0.16671982        0.0000227700
    22            0.06668793       -0.0000127500
    23            0.02667517        0.0000069500
    24            0.01067007       -0.0000033400
    25            0.00426803        0.0000010600
  P 25
    1     22059071.30000000       -0.0000152800
    2      4330633.46000000       -0.0000503900
    3      1041846.32000000       -0.0001411500
    4       285439.05800000       -0.0003819200
    5        86520.78280000       -0.0010432600
    6        28615.02700000       -0.0029241000
    7        10276.64610000       -0.0083668500
    8         3994.98194000       -0.0238229400
    9         1668.21525000       -0.0641126200
    10          739.22152200       -0.1461349800
    11          343.32211100       -0.2339379400
    12          165.25013300       -0.1171314800
    13           80.12487850        0.3593008300
    14           40.44458940        0.5742966900
    15           20.31220350        0.1999168100
    16           10.09863720        0.0103171800
    17            4.97702486        0.0010889700
    18            2.28910027       -0.0005180000
    19            1.04199889        0.0001645700
    20            0.41679956       -0.0000820600
    21            0.16671982        0.0000444800
    22            0.06668793       -0.0000247700
    23            0.02667517        0.0000134700
    24            0.01067007       -0.0000064700
    25            0.00426803        0.0000020500
  P 25
    1     22059071.30000000        0.0000078300
    2      4330633.46000000        0.0000258200
    3      1041846.32000000        0.0000723600
    4       285439.05800000        0.0001959300
    5        86520.78280000        0.0005357400
    6        28615.02700000        0.0015038800
    7        10276.64610000        0.0043141400
    8         3994.98194000        0.0123321600
    9         1668.21525000        0.0334416000
    10          739.22152200        0.0770748900
    11          343.32211100        0.1249023300
    12          165.25013300        0.0481043900
    13           80.12487850       -0.2964498400
    14           40.44458940       -0.4943753300
    15           20.31220350        0.1804130900
    16           10.09863720        0.7447941200
    17            4.97702486        0.2904547000
    18            2.28910027        0.0116038300
    19            1.04199889        0.0015725500
    20            0.41679956       -0.0007161500
    21            0.16671982        0.0003584200
    22            0.06668793       -0.0002040700
    23            0.02667517        0.0001118700
    24            0.01067007       -0.0000537600
    25            0.00426803        0.0000170900
  P 25
    1     22059071.30000000       -0.0000035500
    2      4330633.46000000       -0.0000117100
    3      1041846.32000000       -0.0000328100
    4       285439.05800000       -0.0000889300
    5        86520.78280000       -0.0002429200
    6        28615.02700000       -0.0006834800
    7        10276.64610000       -0.0019564200
    8         3994.98194000       -0.0056190800
    9         1668.21525000       -0.0151917800
    10          739.22152200       -0.0353388000
    11          343.32211100       -0.0568321900
    12          165.25013300       -0.0213326000
    13           80.12487850        0.1531412200
    14           40.44458940        0.2464333800
    15           20.31220350       -0.1425848600
    16           10.09863720       -0.6224094300
    17            4.97702486       -0.0316098700
    18            2.28910027        0.8022050900
    19            1.04199889        0.3668786100
    20            0.41679956        0.0159624700
    21            0.16671982        0.0021572500
    22            0.06668793       -0.0011968900
    23            0.02667517        0.0006706500
    24            0.01067007       -0.0003317300
    25            0.00426803        0.0001059100
  P 25
    1     22059071.30000000        0.0000012600
    2      4330633.46000000        0.0000041500
    3      1041846.32000000        0.0000116100
    4       285439.05800000        0.0000314700
    5        86520.78280000        0.0000859800
    6        28615.02700000        0.0002418600
    7        10276.64610000        0.0006926500
    8         3994.98194000        0.0019887600
    9         1668.21525000        0.0053821700
    10          739.22152200        0.0125158900
    11          343.32211100        0.0201670100
    12          165.25013300        0.0073964100
    13           80.12487850       -0.0552250700
    14           40.44458940       -0.0894764900
    15           20.31220350        0.0569737400
    16           10.09863720        0.2448799500
    17            4.97702486       -0.0039816900
    18            2.28910027       -0.4590096400
    19            1.04199889       -0.1577633700
    20            0.41679956        0.6041994400
    21            0.16671982        0.5401184300
    22            0.06668793        0.0550557800
    23            0.02667517        0.0068844500
    24            0.01067007       -0.0031539400
    25            0.00426803        0.0011310500
  P 25
    1     22059071.30000000       -0.0000002500
    2      4330633.46000000       -0.0000008300
    3      1041846.32000000       -0.0000023100
    4       285439.05800000       -0.0000062700
    5        86520.78280000       -0.0000171300
    6        28615.02700000       -0.0000481500
    7        10276.64610000       -0.0001380400
    8         3994.98194000       -0.0003957600
    9         1668.21525000       -0.0010732100
    10          739.22152200       -0.0024892600
    11          343.32211100       -0.0040282900
    12          165.25013300       -0.0014415200
    13           80.12487850        0.0109521000
    14           40.44458940        0.0180377600
    15           20.31220350       -0.0118588600
    16           10.09863720       -0.0487193700
    17            4.97702486        0.0002123800
    18            2.28910027        0.0986470700
    19            1.04199889        0.0270521200
    20            0.41679956       -0.1544786300
    21            0.16671982       -0.1317764300
    22            0.06668793       -0.0262879900
    23            0.02667517        0.5393594800
    24            0.01067007        0.5539876600
    25            0.00426803        0.0109589100
  P 25
    1     22059071.30000000        0.0000015800
    2      4330633.46000000        0.0000052000
    3      1041846.32000000        0.0000146000
    4       285439.05800000        0.0000394300
    5        86520.78280000        0.0001082900
    6        28615.02700000        0.0003022200
    7        10276.64610000        0.0008754900
    8         3994.98194000        0.0024758200
    9         1668.21525000        0.0068337200
    10          739.22152200        0.0154882500
    11          343.32211100        0.0260141600
    12          165.25013300        0.0074360000
    13           80.12487850       -0.0654502700
    14           40.44458940       -0.1238944900
    15           20.31220350        0.0975743500
    16           10.09863720        0.2800477100
    17            4.97702486        0.0467169700
    18            2.28910027       -0.8272493600
    19            1.04199889        0.1514249400
    20            0.41679956        1.4730372000
    21            0.16671982       -1.1372220700
    22            0.06668793       -0.3908675800
    23            0.02667517        0.1630407200
    24            0.01067007        0.2709877500
    25            0.00426803        0.0005711000
  P 25
    1     22059071.30000000       -0.0000018300
    2      4330633.46000000       -0.0000060100
    3      1041846.32000000       -0.0000169400
    4       285439.05800000       -0.0000453700
    5        86520.78280000       -0.0001264000
    6        28615.02700000       -0.0003450300
    7        10276.64610000       -0.0010317000
    8         3994.98194000       -0.0027970200
    9         1668.21525000       -0.0081455600
    10          739.22152200       -0.0172032000
    11          343.32211100       -0.0322544400
    12          165.25013300       -0.0028449700
    13           80.12487850        0.0627518700
    14           40.44458940        0.1781072700
    15           20.31220350       -0.1919058100
    16           10.09863720       -0.2100864100
    17            4.97702486       -0.2525964000
    18            2.28910027        1.6816487900
    19            1.04199889       -1.8748533300
    20            0.41679956        0.1907542400
    21            0.16671982        1.6114370900
    22            0.06668793       -1.8664756800
    23            0.02667517        0.3789983000
    24            0.01067007        0.3340070800
    25            0.00426803        0.0565541100
  D 17
    1        11495.84880000        0.0003474900
    2         4830.57837000        0.0009403700
    3         2085.12306000        0.0045951700
    4          926.38513500        0.0188745000
    5          421.50108900        0.0692602300
    6          199.45202600        0.1926672400
    7           97.43309670        0.3618653400
    8           48.88992000        0.3796069000
    9           24.70130010        0.1680958100
    10           12.31757490        0.0210179000
    11            6.11916038        0.0008812100
    12            2.83981136       -0.0000965800
    13            1.29220895        0.0000697400
    14            0.52733268       -0.0000325600
    15            0.21093307        0.0000179100
    16            0.08437323       -0.0000083000
    17            0.03374929        0.0000028000
  D 17
    1        11495.84880000       -0.0001896200
    2         4830.57837000       -0.0005130000
    3         2085.12306000       -0.0025254200
    4          926.38513500       -0.0104286900
    5          421.50108900       -0.0388758600
    6          199.45202600       -0.1099490800
    7           97.43309670       -0.2040885700
    8           48.88992000       -0.1598965100
    9           24.70130010        0.2037469100
    10           12.31757490        0.5302199100
    11            6.11916038        0.3559346300
    12            2.83981136        0.0609703500
    13            1.29220895       -0.0012504800
    14            0.52733268        0.0007782900
    15            0.21093307       -0.0004268900
    16            0.08437323        0.0001884700
    17            0.03374929       -0.0000633500
  D 17
    1        11495.84880000        0.0000768300
    2         4830.57837000        0.0002039900
    3         2085.12306000        0.0010258800
    4          926.38513500        0.0041820600
    5          421.50108900        0.0158126800
    6          199.45202600        0.0444375300
    7           97.43309670        0.0833310400
    8           48.88992000        0.0569389000
    9           24.70130010       -0.1140436600
    10           12.31757490       -0.2851284800
    11            6.11916038       -0.0836070700
    12            2.83981136        0.4409344400
    13            1.29220895        0.5472304300
    14            0.52733268        0.1799234600
    15            0.21093307        0.0005526200
    16            0.08437323        0.0044340200
    17            0.03374929       -0.0015031500
  D 17
    1        11495.84880000       -0.0000135000
    2         4830.57837000       -0.0000355200
    3         2085.12306000       -0.0001804300
    4          926.38513500       -0.0007305800
    5          421.50108900       -0.0027778800
    6          199.45202600       -0.0077728800
    7           97.43309670       -0.0146666700
    8           48.88992000       -0.0097590900
    9           24.70130010        0.0204313600
    10           12.31757490        0.0523334600
    11            6.11916038        0.0116339900
    12            2.83981136       -0.0949064900
    13            1.29220895       -0.1242969000
    14            0.52733268        0.1023353100
    15            0.21093307        0.2818822400
    16            0.08437323        0.1280116100
    17            0.03374929        0.7434517600
  D 17
    1        11495.84880000        0.0000372400
    2         4830.57837000        0.0000947700
    3         2085.12306000        0.0004982300
    4          926.38513500        0.0019701400
    5          421.50108900        0.0076258000
    6          199.45202600        0.0210078400
    7           97.43309670        0.0404410600
    8           48.88992000        0.0256531800
    9           24.70130010       -0.0534239100
    10           12.31757490       -0.1497543500
    11            6.11916038       -0.0223660900
    12            2.83981136        0.2575858300
    13            1.29220895        0.4073357300
    14            0.52733268       -0.6302430200
    15            0.21093307       -0.5652874300
    16            0.08437323        0.0961401100
    17            0.03374929        0.5050502900
  F 12
    1          739.22152200        0.0010793600
    2          343.32211100        0.0041783500
    3          165.25013300        0.0223589900
    4           80.12487850        0.0764942500
    5           40.44458940        0.1949113800
    6           20.31220350        0.3415474800
    7           10.09863720        0.3712998700
    8            4.97702486        0.2295645800
    9            2.28910027        0.0607676500
    10            0.91564011        0.0014121900
    11            0.36625604        0.0005312000
    12            0.14650242       -0.0001970700
  F 12
    1          739.22152200       -0.0001155600
    2          343.32211100       -0.0001234500
    3          165.25013300       -0.0021202700
    4           80.12487850       -0.0039584700
    5           40.44458940       -0.0175356400
    6           20.31220350       -0.0149335000
    7           10.09863720       -0.0406177600
    8            4.97702486        0.0358458200
    9            2.28910027       -0.0631723900
    10            0.91564011        0.3384240300
    11            0.36625604        0.7199148900
    12            0.14650242        0.0823335700
  F 12
    1          739.22152200        0.0001402000
    2          343.32211100       -0.0001844500
    3          165.25013300        0.0022954000
    4           80.12487850        0.0004625400
    5           40.44458940        0.0178244000
    6           20.31220350       -0.0031887100
    7           10.09863720        0.0555564000
    8            4.97702486       -0.0746113600
    9            2.28910027        0.2345141900
    10            0.91564011       -0.8278404600
    11            0.36625604        0.0639854300
    12            0.14650242        0.8595041100
  G 2
    1           12.62000000        1.0000000000
    2            0.45780000        0.0000000000
  G 2
    1           12.62000000        0.0000000000
    2            0.45780000        1.0000000000
  end
end"""


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
    TD-DFT / screening basis recipe (unchanged from legacy get_recipe).

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


get_recipe = get_recipe_tddft  # backward-compatible alias


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
    z = atomic_number(symbol)
    if z >= 87:
        return "ANO-RCC-VTZP (inline NewGTO)"
    if 55 <= z <= 86:
        return "SARC-DKH-TZVP"
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
    """Provenance block written into data_json/<element>.json."""
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
