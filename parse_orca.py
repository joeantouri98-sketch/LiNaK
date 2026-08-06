r"""
parse_orca.py  —  ORCA 6.x output parser
-----------------------------------------
Parses TD-DFT excitations and ground-state energy from ORCA output files.

ORCA output structure (critical to understand):
  - STATE blocks contain: excitation energy (eV) + orbital coefficients
  - Oscillator strengths live in a SEPARATE absorption spectrum table
  - The two sections must be parsed independently and joined on state number

Original parse_orca.py bugs (all three caused empty/wrong output):
  Bug 1: regex r'E=\s*([\d.]+)\s+eV' fails on 'E=   0.057766 au      1.572 eV'
         because the number before 'eV' is not directly after 'E='
         -> all states skipped, returns []
  Bug 2: searched for 'oscillator strength' inside STATE blocks
         -> ORCA never puts fosc there, always returns f=0.0
  Bug 3: orbital regex looked for % signs ('[\d.]+%')
         -> ORCA writes decimals (0.937327), never percentages
         -> orbital_contributions always empty
r"""

import re

from constants import HARTREE_TO_EV, HC_EV_NM


def parse_ground_energy(text):
    """Extract ground state SCF energy in eV from ORCA output."""
    m = re.search(r'FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)', text)
    if not m:
        return None
    return float(m.group(1)) * HARTREE_TO_EV


def parse_excitations(text, ground_mult=2):
    r"""
    Parse TD-DFT excitations from ORCA output.

    Strategy:
      1. Parse the ABSORPTION SPECTRUM table  ->  state_number: (energy_eV, fosc)
      2. Parse each STATE block               ->  state_number: (s2, mult, dominant orbital)
      3. Join on state number

    Returns list of dicts with keys:
      energy_eV, oscillator_strength, multiplicity, s2,
      spin_forbidden, from, to, dominant_coeff, raw_transition
    """

    # ── Step 1: Absorption spectrum table ────────────────────────────────────
    # Location: between the two dashed lines after
    #   'ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS'
    # Format of each row:
    #   '  0-2A  ->  1-2A    1.571640   12676.1   788.9   0.328219167   ...'
    #    ground->state_n   energy_eV   cm-1      nm      fosc

    fosc_by_state = {}

    table_start = text.find('ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS')
    table_end   = text.find('ABSORPTION SPECTRUM VIA TRANSITION VELOCITY', table_start + 1)

    if table_start != -1:
        table_text = text[table_start : table_end if table_end != -1 else table_start + 80000]
        for m in re.finditer(
            r'0-\w+\s*->\s*(\d+)-\w+\s+'   # state number
            r'([\d.]+)\s+'                   # energy eV
            r'[\d.]+\s+[\d.]+\s+'            # cm-1, nm
            r'([\d.Ee+\-]+)',                 # fosc
            table_text
        ):
            state_n = int(m.group(1))
            e_ev    = float(m.group(2))
            fosc    = float(m.group(3))
            fosc_by_state[state_n] = {'energy_eV': e_ev, 'oscillator_strength': fosc}

    # ── Step 2: STATE blocks ─────────────────────────────────────────────────
    # ORCA format:
    #   STATE  1:  E=   0.057766 au      1.572 eV ... <S**2> = 0.750056 Mult 2
    #        9a ->  10a  :     0.937327 (c=  0.96815660)
    #        9a ->  12a  :     0.041011 (c=  0.20251218)
    #
    # Energy is after 'au' on the STATE header line.
    # Orbital weight is the decimal (0.937327), NOT a percentage.
    # Dominant orbital = the one with the HIGHEST weight value.

    state_data = {}

    for m in re.finditer(
        r'STATE\s+(\d+):\s+'              # state number
        r'E=\s*[\d.]+\s+au\s+'            # Hartree (skip)
        r'([\d.]+)\s+eV'                  # energy eV (from STATE header)
        r'.*?<S\*\*2>\s*=\s*([\d.]+)'     # S^2
        r'\s+Mult\s+(\d+)'                # multiplicity
        r'(.*?)(?=STATE\s+\d+:|\Z)',       # body until next STATE or end
        text, re.S
    ):
        n    = int(m.group(1))
        s2   = float(m.group(3))
        mult = int(m.group(4))
        body = m.group(5)

        # Parse orbital lines: '9a ->  10a  :     0.937327 (c=  0.96815660)'
        transitions = []
        for t in re.finditer(
            r'(\w+)\s*->\s*(\w+)\s*:\s*'   # from -> to
            r'([\d.]+)\s*'                  # weight (decimal, not %)
            r'\(c=\s*([-\d.]+)\)',           # coefficient
            body
        ):
            transitions.append({
                'from':           t.group(1),
                'to':             t.group(2),
                'weight':         float(t.group(3)),
                'dominant_coeff': float(t.group(4)),
            })

        # Dominant = highest weight
        dominant = max(transitions, key=lambda x: x['weight']) if transitions else None

        state_data[n] = {
            's2':             s2,
            'multiplicity':   mult,
            'spin_forbidden': mult != ground_mult,
            'dominant':       dominant,
        }

    # ── Fallback: if no absorption table, use STATE block energies ────────────
    if not fosc_by_state and state_data:
        for m in re.finditer(
            r'STATE\s+(\d+):\s+E=\s*[\d.]+\s+au\s+([\d.]+)\s+eV',
            text
        ):
            n    = int(m.group(1))
            e_ev = float(m.group(2))
            fosc_by_state[n] = {'energy_eV': e_ev, 'oscillator_strength': 0.0}

    # ── Step 3: Merge ────────────────────────────────────────────────────────
    excitations = []
    for n in sorted(fosc_by_state.keys()):
        entry = {
            'energy_eV':           fosc_by_state[n]['energy_eV'],
            'oscillator_strength': fosc_by_state[n]['oscillator_strength'],
            'multiplicity':        2,
            's2':                  0.75,
            'spin_forbidden':      False,
            'from':                '',
            'to':                  '',
            'dominant_coeff':      0.0,
            'raw_transition':      '',
        }
        if n in state_data:
            sd = state_data[n]
            entry['s2']             = sd['s2']
            entry['multiplicity']   = sd['multiplicity']
            entry['spin_forbidden'] = sd['spin_forbidden']
            if sd['dominant']:
                dom = sd['dominant']
                entry['from']           = dom['from']
                entry['to']             = dom['to']
                entry['dominant_coeff'] = dom['dominant_coeff']
                entry['raw_transition'] = (
                    f"{dom['from']} ->  {dom['to']}  :     "
                    f"{dom['weight']:.6f} (c= {dom['dominant_coeff']:.8f})"
                )
        excitations.append(entry)

    return excitations


_SOC_ABSORPTION_MARKER = (
    "SOC CORRECTED ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS"
)
_QDPT_CASSCF_MARKER = "QDPT WITH CASSCF DIAGONAL ENERGIES"
_QDPT_NEVPT2_MARKER = "QDPT WITH NEVPT2 DIAGONAL ENERGIES"


def _resolve_soc_absorption_start(text, soc_energy_source="auto"):
    """
    Pick the SOC absorption table to parse.

    SC-NEVPT2+DoSOC jobs print two tables (CASSCF then NEVPT2 diagonal energies).
    For ``auto``, use the NEVPT2 block when that marker is present; otherwise the
    first absorption table (bare CASSCF or single QDPT pass).
    """
    if soc_energy_source == "auto":
        source = "nevpt2" if _QDPT_NEVPT2_MARKER in text else "casscf"
    else:
        source = soc_energy_source

    if source == "nevpt2":
        anchor = text.find(_QDPT_NEVPT2_MARKER)
        if anchor == -1:
            return -1, source
    elif source == "casscf":
        anchor = text.find(_QDPT_CASSCF_MARKER)
        if anchor == -1:
            anchor = 0
    else:
        raise ValueError(
            f"soc_energy_source must be 'auto', 'casscf', or 'nevpt2', got {source!r}"
        )

    start = text.find(_SOC_ABSORPTION_MARKER, anchor)
    return start, source


def parse_soc_excitations(text, ground_mult=2, soc_energy_source="auto"):
    """
    Parse CASSCF/MRCI spin-orbit corrected absorption spectrum.

    Reads the 'SOC CORRECTED ABSORPTION SPECTRUM' table and returns one entry
    per SOC excited-state index (from the ground component 0-*.0A), keeping the
    strongest dipole line for each target state. ORCA labels doublet states as
    e.g. '0-2.0A' not '0-2A'.

    When ORCA prints two tables (after QDPT with CASSCF and NEVPT2 diagonal
    energies), ``soc_energy_source='auto'`` selects the NEVPT2 block; bare
    CASSCF jobs keep the single (first) table. Each entry includes
    ``soc_energy_source`` ('casscf' or 'nevpt2') recording which block was used.
    """
    _state = r"[\w.]+"   # 2.0A, 2A, etc.
    start, resolved_source = _resolve_soc_absorption_start(text, soc_energy_source)
    if start == -1:
        return []

    end = text.find("SOC CORRECTED CD SPECTRUM", start + 1)
    table = text[start : end if end != -1 else start + 80000]

    by_state = {}
    row_re = re.compile(
        rf"0-{_state}\s*->\s*(\d+)-{_state}\s+"
        r"([\d.]+)\s+"
        r"[\d.]+\s+[\d.]+\s+"
        r"([\d.Ee+\-]+)"
    )
    for m in row_re.finditer(table):
        state_n = int(m.group(1))
        e_ev = float(m.group(2))
        fosc = float(m.group(3))
        if state_n == 0 or e_ev <= 1e-6:
            continue
        entry = by_state.get(state_n)
        if entry is None or fosc > entry["oscillator_strength"]:
            by_state[state_n] = {
                "energy_eV": e_ev,
                "oscillator_strength": fosc,
                "multiplicity": ground_mult,
                "s2": 0.75,
                "spin_forbidden": False,
                "from": "0",
                "to": str(state_n),
                "dominant_coeff": 0.0,
                "raw_transition": f"0 -> {state_n} (SOC)",
                "soc_state": state_n,
                "soc_energy_source": resolved_source,
            }

    return [by_state[k] for k in sorted(by_state)]


# TD-DFT roots closer than this (eV) are numerical duplicates (same printed energy).
TD_DFT_NUMERIC_TOLERANCE_EV = 0.0001
# Legacy alias — only used for TD-DFT post-processing, not SOC results.
TD_DFT_LEVEL_TOLERANCE_EV = TD_DFT_NUMERIC_TOLERANCE_EV


def deduplicate_excitations(raw_excitations, tolerance=TD_DFT_LEVEL_TOLERANCE_EV):
    """
    Collapse near-degenerate TD-DFT roots to one level per physical state.

    Keeps the brightest root as representative (from/to labels), sets
    n_degenerate to the number of merged roots, and total_oscillator_strength
    to the sum of their f values (for spectrum / arrow intensity).
    """
    if not raw_excitations:
        return []

    sorted_raw = sorted(raw_excitations, key=lambda x: x["energy_eV"])
    groups = []
    current = [sorted_raw[0]]
    for state in sorted_raw[1:]:
        if state["energy_eV"] - current[-1]["energy_eV"] <= tolerance:
            current.append(state)
        else:
            groups.append(current)
            current = [state]
    groups.append(current)

    result = []
    for grp in groups:
        rep = dict(max(grp, key=lambda x: (
            x["oscillator_strength"],
            abs(x.get("dominant_coeff", 0.0)),
        )))
        rep["n_degenerate"] = len(grp)
        rep["total_oscillator_strength"] = sum(
            x.get("oscillator_strength", 0.0) for x in grp
        )
        if len(grp) > 1:
            rep["energy_eV"] = sum(x["energy_eV"] for x in grp) / len(grp)
        result.append(rep)
    return result
