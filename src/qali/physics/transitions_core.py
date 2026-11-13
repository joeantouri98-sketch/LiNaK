"""Electric-dipole transitions between the levels of a rydberg JSON: the pure computation.

Alkali / ion (QDT) path: every pair of levels with

    Δl = ±1,   ΔJ = 0, ±1,   not J = 0 → 0

where ΔJ is enforced only when both levels carry a J (QDT model levels have none).

Metal path (no QDT): the NIST ASD line list itself, each endpoint matched to a known
level by label, else by a unique (LS term, J), else by the closest excitation energy
(within 0.05 eV) among levels sharing that (term, J).

Energies are relative to the ionization threshold (negative = bound).

Known issues (PHYSICS_BUGS.md): wavelengths use hc = 1239.84 eV nm (P33); the metal
path drops every line of a no-QDT alkali after ``rydberg --nist-only`` (P44). Kept as is
so results stay comparable until they are fixed deliberately.

No files or prints here: the ``transitions`` pipeline reads the rydberg JSON and the
NIST lines and passes them in.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from qali.core.spectroscopy import spectral_region
from qali.physics.polarizability_core import L_CHAR
from qali.physics.rydberg import (
    label_to_pretty,
    normalize_term,
    parse_j,
    split_asd_term,
    unicode_digits_to_plain,
)

HC_EV_NM_ROUNDED = 1239.84   # P33: rounded hc, kept so wavelengths stay comparable


# ============================================================================
# RECORDS
# ============================================================================

@dataclass(frozen=True)
class Level:
    """One level of the alkali / ion (QDT) path."""

    n: int
    l: str
    l_int: int            # -1 when l is not a known orbital letter
    J: str | None         # None for QDT model levels
    J_float: float | None
    term: str | None
    label: str
    energy: float         # eV, relative to the ionization threshold
    source: str


@dataclass(frozen=True)
class Transition:
    """One allowed E1 transition (a row of ``<El>_transitions.json``)."""

    upper_label: str
    lower_label: str
    upper_n: int
    lower_n: int
    upper_l: str
    lower_l: str
    upper_J: str | None
    lower_J: str | None
    delta_n: int
    upper_energy_eV: float
    lower_energy_eV: float
    delta_E_eV: float
    wavelength_nm: float
    region: str
    upper_source: str
    lower_source: str


@dataclass(frozen=True)
class MetalLevel:
    label: str
    label_pretty: str
    energy: float         # eV, relative to the ionization threshold
    n: int | None
    l: str | None
    J: str | None
    term: str
    source: str


@dataclass(frozen=True)
class MetalTransition:
    """One NIST ASD line between two known levels (metal path)."""

    upper_label: str
    lower_label: str
    upper_label_pretty: str
    lower_label_pretty: str
    upper_n: int | None
    lower_n: int | None
    upper_l: str | None
    lower_l: str | None
    upper_J: str | None
    lower_J: str | None
    delta_n: int | None
    upper_energy_eV: float
    lower_energy_eV: float
    delta_E_eV: float
    wavelength_nm: float
    region: str
    upper_source: str
    lower_source: str
    fik: float | None
    gA: float | None
    fosc: float | None
    Acc: str | None


@dataclass(frozen=True)
class MetalLines:
    transitions: list[MetalTransition]
    n_missing_lower: int      # lines whose lower level is unknown
    n_missing_upper: int
    n_outside_window: int     # outside [wl_min, wl_max]
    n_resolved: int           # endpoints matched by term + J instead of label


# ============================================================================
# ALKALI / ION (QDT) PATH
# ============================================================================

def j_selection_rule(j_upper, j_lower):
    """
    Return True if the J pair satisfies ΔJ = 0, ±1 and not (J=0 → J=0).
    If either J is None (QDT model level), returns True (rule not enforced).
    """
    if j_upper is None or j_lower is None:
        return True   # can't check — allow
    dJ = abs(j_upper - j_lower)
    if dJ > 1.001:
        return False
    if j_upper < 0.001 and j_lower < 0.001:
        return False  # J=0 → J=0 forbidden
    return True


def j_values(j_string):
    """
    Every J a level's J string carries: '5/2' -> [2.5]; '5/2,7/2' -> [2.5, 3.5] (NIST lists the
    f and g Rydberg levels from n = 12 as one merged level); None when there is no usable J.
    """
    if j_string is None:
        return None
    values = [parse_j(part) for part in str(j_string).split(',') if part.strip()]
    values = [v for v in values if v is not None]
    return values or None


def j_sets_allowed(js_upper, js_lower):
    """
    True when some J of the upper level and some J of the lower level satisfy the E1 rule.
    A level with no J (QDT model level) is not checked.  A merged level (several J in one string)
    is tested with every J it contains, not only the first: f(5/2,7/2) -> g9/2 is allowed through f7/2.
    """
    if not js_upper or not js_lower:
        return True
    return any(j_selection_rule(a, b) for a in js_upper for b in js_lower)


def ground_level(rydberg_data: Mapping, IE: float) -> tuple[Level, bool]:
    """(ground level, True when it was inferred from an old n_start-only JSON).

    The 'ground_state' block written by rydberg.py describes any element; old
    alkali JSONs only have 'n_start' (or nothing): assume l='s', J='1/2'.
    """
    gs_meta = rydberg_data.get('ground_state')
    if gs_meta:
        gs_n     = gs_meta['n']
        gs_l     = gs_meta['l']
        gs_j     = gs_meta.get('J') or ''
        gs_label = gs_meta.get('label') or f"{gs_n}{gs_l}"
    else:
        gs_n = rydberg_data.get('n_start', None)
        if gs_n is None:
            _s = [(ex['n'], ex['energy_eV']) for ex in rydberg_data.get('excitations', [])
                  if ex.get('l') == 's' and ex.get('n') and ex.get('energy_eV') is not None]
            gs_n = min(_s, key=lambda x: x[1])[0] if _s else 2
        gs_l     = 's'
        gs_j     = '1/2'
        gs_label = f"{gs_n}s"
    gs_term = (gs_meta or {}).get('term') or ''
    level = Level(
        n=gs_n,
        l=gs_l,
        l_int=L_CHAR.get(gs_l, 0),
        J=gs_j or None,
        J_float=parse_j(gs_j) if gs_j else None,
        term=gs_term or None,
        label=gs_label,
        energy=-IE,
        source='NIST',
    )
    return level, not gs_meta


def qdt_levels(ground: Level, excitations: Iterable[Mapping], n_max=None) -> list[Level]:
    """The ground level, then every distinct (n, l, J, term) of *excitations* up to n_max."""
    levels = [ground]
    seen = {(ground.n, ground.l, ground.J or '', ground.term or ground.label)}
    for ex in excitations:
        n      = ex.get('n')
        l_char = ex.get('l')
        j_str  = ex.get('J')          # may be None for QDT model levels
        label  = ex.get('label', '')
        E_abs  = ex.get('energy_eV')

        if n is None or l_char is None or E_abs is None:
            continue
        if n_max and n > n_max:
            continue

        # Normalise J key: use empty string when absent so distinct from '1/2' etc.
        # Include term (or label) so 3P and 1P with the same J are not dropped.
        j_key = str(j_str).strip() if j_str is not None else ''
        term  = ex.get('term') or ''
        key   = (n, l_char, j_key, term or label)
        if key in seen:
            continue
        seen.add(key)

        levels.append(Level(
            n=n,
            l=l_char,
            l_int=L_CHAR.get(l_char, -1),
            J=j_str,
            J_float=parse_j(j_str),
            term=term or None,
            label=label,
            energy=E_abs,
            source=ex.get('source', 'Rydberg'),
        ))
    return levels


def e1_transitions(levels: Sequence[Level], wl_min=0.0, wl_max=float('inf')) -> list[Transition]:
    """Every allowed E1 pair (Δl = ±1, ΔJ rule) with wl_min <= λ <= wl_max."""
    by_energy = sorted(levels, key=lambda s: s.energy)
    transitions = []
    for i in range(len(by_energy)):
        for j in range(i + 1, len(by_energy)):
            s_lower = by_energy[i]
            s_upper = by_energy[j]

            # Δl = ±1
            if abs(s_upper.l_int - s_lower.l_int) != 1:
                continue

            # ΔJ rule
            if not j_sets_allowed(j_values(s_upper.J), j_values(s_lower.J)):
                continue

            delta_E = s_upper.energy - s_lower.energy
            if delta_E <= 0:
                continue

            wavelength = HC_EV_NM_ROUNDED / delta_E

            if wavelength < wl_min or wavelength > wl_max:
                continue

            transitions.append(Transition(
                upper_label=s_upper.label,
                lower_label=s_lower.label,
                upper_n=s_upper.n,
                lower_n=s_lower.n,
                upper_l=s_upper.l,
                lower_l=s_lower.l,
                upper_J=s_upper.J,
                lower_J=s_lower.J,
                delta_n=s_upper.n - s_lower.n,
                upper_energy_eV=round(s_upper.energy, 6),
                lower_energy_eV=round(s_lower.energy, 6),
                delta_E_eV=round(delta_E, 6),
                wavelength_nm=round(wavelength, 4),
                region=spectral_region(wavelength),
                upper_source=s_upper.source,
                lower_source=s_lower.source,
            ))
    return transitions



# METAL PATH: transitions = ASD line list (no Δl invention)
# ============================================================================

def metal_levels(gs_meta: Mapping, excitations: Iterable[Mapping], IE: float) -> dict[str, MetalLevel]:
    """label -> level: the ground state, then the first level of every label."""
    gs_label = gs_meta['label']
    levels = {
        gs_label: MetalLevel(
            label=gs_label,
            label_pretty=unicode_digits_to_plain(
                gs_meta.get('label_pretty') or label_to_pretty(gs_label)),
            energy=-IE,
            n=gs_meta.get('n'),
            l=gs_meta.get('l'),
            J=gs_meta.get('J'),
            term=gs_meta.get('term') or '',
            source='NIST',
        )
    }
    for ex in excitations:
        lab = ex.get('label')
        E = ex.get('energy_eV')
        if not lab or E is None:
            continue
        if lab not in levels:
            levels[lab] = MetalLevel(
                label=lab,
                label_pretty=unicode_digits_to_plain(
                    ex.get('label_pretty') or label_to_pretty(lab)),
                energy=E,
                n=ex.get('n'),
                l=ex.get('l'),
                J=ex.get('J'),
                term=ex.get('term') or '',
                source=ex.get('source', 'NIST'),
            )
    return levels


def _term_j_key(term, j_str):
    _pref, _ls = split_asd_term(term or '')
    return normalize_term(_ls) or normalize_term(term or ''), str(j_str or '').strip()


def metal_transitions(lines: Iterable[Mapping], levels: Mapping[str, MetalLevel], IE: float,
                      wl_min=0.0, wl_max=float('inf')) -> MetalLines:
    """The ASD lines whose two endpoints are known levels, one per (upper, lower) pair."""
    # (normalized LS-term, J) -> [labels] for blank-config ASD lines
    by_term_j: dict[tuple[str, str], list[str]] = {}
    for lab, st in levels.items():
        _tkey, _jkey = _term_j_key(st.term, st.J)
        if _tkey and _jkey:
            by_term_j.setdefault((_tkey, _jkey), []).append(lab)

    def resolve(lab, term, j_str, e_from_gs):
        """Map line endpoint to a known level (label match or unique term+J)."""
        if lab and lab in levels:
            return lab, levels[lab]
        _tkey, _jkey = _term_j_key(term, j_str)
        cands = by_term_j.get((_tkey, _jkey), []) if _tkey and _jkey else []
        if len(cands) == 1:
            lab2 = cands[0]
            return lab2, levels[lab2]
        if len(cands) > 1 and e_from_gs is not None:
            # Pick closest excitation energy among term+J candidates
            best, best_d = None, 1e9
            for lab2 in cands:
                exc = levels[lab2].energy + IE
                d = abs(exc - float(e_from_gs))
                if d < best_d:
                    best_d, best = d, lab2
            if best is not None and best_d < 0.05:
                return best, levels[best]
        return None, None

    transitions = []
    n_miss_lo = n_miss_up = n_wl = n_resolved = 0
    seen_pair = set()
    for ln in lines:
        lo0 = ln.get('lower_label')
        up0 = ln.get('upper_label')
        lo, s_lo = resolve(lo0, ln.get('lower_term'), ln.get('lower_J'), ln.get('Ei_eV'))
        up, s_up = resolve(up0, ln.get('upper_term'), ln.get('upper_J'), ln.get('Ek_eV'))
        if lo0 != lo or up0 != up:
            if lo and up:
                n_resolved += 1
        if s_lo is None:
            n_miss_lo += 1
            continue
        if s_up is None:
            n_miss_up += 1
            continue
        # Prefer spectroscopic energies from levels JSON
        E_lo = s_lo.energy
        E_up = s_up.energy
        delta_E = E_up - E_lo
        if delta_E <= 0:
            continue
        wavelength = HC_EV_NM_ROUNDED / delta_E
        if ln.get('wavelength_nm'):
            # keep ASD wavelength when close; else levels-derived
            wl_asd = float(ln['wavelength_nm'])
            if abs(wl_asd - wavelength) / max(wavelength, 1e-9) < 0.05:
                wavelength = wl_asd
        if wavelength < wl_min or wavelength > wl_max:
            n_wl += 1
            continue
        pair = (up, lo)
        if pair in seen_pair:
            continue
        seen_pair.add(pair)
        nu, nl_ = s_up.n, s_lo.n
        transitions.append(MetalTransition(
            upper_label=up,
            lower_label=lo,
            upper_label_pretty=s_up.label_pretty or label_to_pretty(up),
            lower_label_pretty=s_lo.label_pretty or label_to_pretty(lo),
            upper_n=nu,
            lower_n=nl_,
            upper_l=s_up.l,
            lower_l=s_lo.l,
            upper_J=ln.get('upper_J') or s_up.J,
            lower_J=ln.get('lower_J') or s_lo.J,
            delta_n=(nu - nl_) if (nu is not None and nl_ is not None) else None,
            upper_energy_eV=round(E_up, 6),
            lower_energy_eV=round(E_lo, 6),
            delta_E_eV=round(delta_E, 6),
            wavelength_nm=round(wavelength, 4),
            region=spectral_region(wavelength),
            upper_source='NIST',
            lower_source='NIST',
            fik=ln.get('fik'),
            gA=ln.get('gA'),
            fosc=ln.get('fosc'),
            Acc=ln.get('Acc'),
        ))
    return MetalLines(transitions=transitions, n_missing_lower=n_miss_lo,
                      n_missing_upper=n_miss_up, n_outside_window=n_wl,
                      n_resolved=n_resolved)
