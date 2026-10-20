import re

import numpy as np

from linak.core.constants import (HARTREE_TO_EV, ATOMIC_MASS_AMU, ISOTOPE_MASS_AMU, default_isotope_A)

Ry = HARTREE_TO_EV / 2.0   # eV, Rydberg constant for infinite nuclear mass, R_inf (CODATA)
M_E_AMU = 5.48579909065e-4  # electron mass in u


def rydberg_constant_eV(element, isotope_A=None):
    """
    Reduced-mass Rydberg constant R_M = R_inf (1 - m_e / M_atom) in eV for a Rydberg electron
    orbiting the ion core of `element` (core mass = atom mass - m_e). Uses the mass of the
    chosen isotope (default: most abundant natural isotope), else the elemental weight, else
    R_inf. The correction is -2.4e-5 (relative) for Na, i.e. ~9 GHz at n = 13.
    """
    A = isotope_A if isotope_A is not None else default_isotope_A(element)
    M = ISOTOPE_MASS_AMU.get((element, A)) if A is not None else None
    if M is None:
        M = ATOMIC_MASS_AMU.get(element)
    if not M:
        return Ry
    return Ry * (1.0 - M_E_AMU / M)

MAX_OCC = {'s': 2, 'p': 6, 'd': 10, 'f': 14, 'g': 18, 'h': 22}


def parse_config(config_col):
    """
    Parse config string like '5p66s', '2p63p', '2p66h'.
    Uses max occupancy to split core from valence correctly.

    Open-shell atoms (halogens, noble gases, transition metals) embed a
    parenthesized parent term between core and valence, e.g. Cl
    '3s23p4(3P)4s', Ar '3s23p5(2P°3/2)4s', Fe '3d7(4F)4s'. These groups are
    skipped so the trailing valence shell is still captured.

    If the string ends in something unparseable (e.g. '3s23p4nd?'), returns
    [] so the caller skips the row instead of misassigning it to the last
    core subshell.

    Returns list of (n, l, occupancy) tuples; last entry is valence.
    """
    remaining = config_col.strip().strip('"')
    subshells = []
    pattern = re.compile(r'^(\d+)([spdfgh])(\d*)')
    while remaining:
        # ASD sometimes inserts spaces: '3d6.(5D).4s (6D).5s'
        remaining = remaining.lstrip()
        if not remaining:
            break
        # Full ASD exports separate subshells with dots: '2p6.3s'
        if remaining[0] == '.':
            remaining = remaining[1:]
            continue
        if remaining[0] == '(':
            close = remaining.find(')')
            if close < 0:
                break
            remaining = remaining[close + 1:]
            continue
        m = pattern.match(remaining)
        if not m:
            break
        n, l, occ_str = m.group(1), m.group(2), m.group(3)
        if occ_str:
            occ = int(occ_str)
            max_allowed = MAX_OCC.get(l, 99)
            if occ > max_allowed:
                real_occ = max_allowed
                leftover = str(occ)[len(str(real_occ)):]
                subshells.append((int(n), l, real_occ))
                remaining = leftover + remaining[m.end():]
                continue
            else:
                subshells.append((int(n), l, occ))
        else:
            subshells.append((int(n), l, 1))
        remaining = remaining[m.end():]
    # Leftover text (other than a trailing '?') means the valence shell could
    # not be identified: signal "unparseable" rather than return a core shell.
    if remaining.strip().strip('?'):
        return []
    return subshells


def parse_j(j_str):
    """
    Parse a J string to float. Handles:
      - Simple fractions: '1/2', '3/2'
      - Integers: '0', '1', '2'
      - Multiple J on one row: '1/2, 3/2'  -> takes first value
      - Non-J content (Term symbols etc.): returns None
    """
    if j_str is None:
        return None
    s = str(j_str).strip().strip('"')
    if not s:
        return None
    s = s.split(',')[0].strip()
    m = re.match(r'^(\d+)/(\d+)$', s)
    if m:
        try:
            return int(m.group(1)) / int(m.group(2))
        except ZeroDivisionError:
            return None
    try:
        v = float(s)
        if 0 <= v <= 20:
            return v
        return None
    except ValueError:
        return None


def j_label(l_char, j_float):
    """
    Return a human-readable J label suffix for a state label.

    Rules:
      - s states: always return None (J=1/2 is the only possibility, no suffix)
      - J = 0: return None: closed-shell singlet ground states (noble gases,
        alkaline earths) do not need a J suffix; it would be redundant and
        confusing (e.g. '3p' not '3p0' for the Ar ground state)
      - Half-integer J: return '1/2', '3/2', '5/2', ...
      - Integer J > 0: return '1', '2', ... (rare in alkali-like Rydberg)
    """
    if l_char == 's':
        return None          # s: J=1/2 always, no suffix
    if j_float is None:
        return None
    twice_j = round(2 * j_float)
    if twice_j == 0:
        return None          # J=0: closed shell, no suffix needed
    if twice_j % 2 == 1:
        return f"{twice_j}/2"
    else:
        return str(twice_j // 2)


def normalize_term(term):
    """Sanitize ASD term for labels/keys: '3P*' -> '3Po', '2[3/2]*' -> '2_3h2o'."""
    if not term:
        return ''
    t = str(term).strip()
    t = t.replace('°', 'o').replace('*', 'o')
    t = re.sub(r'\[(\d+)/(\d+)\]', r'_\1h\2', t)  # jK: 2[3/2] -> 2_3h2
    t = re.sub(r'[^0-9A-Za-z_]', '', t)
    return t


_SUP_DIGITS = str.maketrans('0123456789+-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻')
_SUB_DIGITS = str.maketrans('0123456789+-', '₀₁₂₃₄₅₆₇₈₉₊₋')
_FROM_SUP = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻', '0123456789+-')
_FROM_SUB = str.maketrans('₀₁₂₃₄₅₆₇₈₉₊₋', '0123456789+-')


def _to_sup(text):
    return str(text).translate(_SUP_DIGITS)


def _to_sub(text):
    return str(text).translate(_SUB_DIGITS)


def unicode_digits_to_plain(text):
    """Map Unicode sub/sup digits → normal ASCII (Plotly hover shrinks them)."""
    if not text:
        return text
    return str(text).translate(_FROM_SUB).translate(_FROM_SUP)


def j_to_plain(j_str):
    """J with normal-size digits: 2, 1/2, 3h2 → 3/2."""
    if j_str is None or j_str == '':
        return ''
    return str(j_str).strip().replace('h', '/')


def _jk_bracket_pretty(num, den):
    """jK bracket with normal-size digits: [3/2]."""
    return f"[{num}/{den}]"


def term_to_pretty(term):
    """
    ASD / normalized term → readable term symbol (full-size digits).
      '3P*' / '3Po' → '3P°'
      '2[3/2]*' / '2_3h2o' → '2[3/2]°'
    """
    if not term:
        return ''
    t = str(term).strip()
    # Normalized jK: 2_3h2o
    m_jk = re.match(r'^(\d+)_(\d+)h(\d+)(o?)$', t)
    if m_jk:
        mult, num, den, o = m_jk.groups()
        return f"{mult}{_jk_bracket_pretty(num, den)}{'°' if o else ''}"
    # Raw ASD jK: 2[3/2]*
    m_jk_raw = re.match(r'^(\d+)\[(\d+)/(\d+)\]([*°]?)$', t)
    if m_jk_raw:
        mult, num, den, mark = m_jk_raw.groups()
        return f"{mult}{_jk_bracket_pretty(num, den)}{'°' if mark else ''}"
    # LS: 3P* / 3Po / 1S / 3P
    m_ls = re.match(r'^(\d+)([A-Za-z])([*°o])?$', t)
    if m_ls:
        mult, letter, mark = m_ls.groups()
        return f"{mult}{letter.upper()}{'°' if mark else ''}"
    return t


# Larger-than-default HTML sub/sup for Plotly (browser default ~0.7em is tiny)
_HTML_SUP = '<sup style="font-size:0.92em;line-height:0">'
_HTML_SUB = '<sub style="font-size:0.92em;line-height:0">'


def is_alkali_doublet_term(term):
    """True for valence alkali-like terms 2S / 2P* / 2P° / 2D* … (short labels OK).

    NIST marks odd parity with ``*`` in CSV exports and ``°`` in text exports.
    """
    if not term:
        return False
    return bool(re.match(r'^2[SPDFGH][*°]?$', str(term).strip()))


def state_label(n, l_char, j_str, term=None, is_ground=False):
    """
    Build a unique ASCII state key (stable for JSON / matching).

    Alkali doublets keep legacy names (3s, 3p1/2). Multiplet ions include
    term + J so 2p5.3s 3P and 1P do not collide (3s_3Po_2 vs 3s_1Po_1).
    Closed-shell ground 1S (J=0) keeps the short nl label (e.g. 2p).
    Excited 1S must keep term+J (e.g. 3p_1S_0) so it does not collide.
    """
    j_float = parse_j(j_str)
    jl = j_label(l_char, j_float)
    base = f"{n}{l_char}{jl}" if jl else f"{n}{l_char}"
    if not term or is_alkali_doublet_term(term):
        return base
    # Only the true ground 1S may use the bare nl short name
    if is_ground and re.match(r'^1S\*?$', str(term).strip()) and (
            j_float is None or abs(j_float) < 1e-9):
        return base
    t_norm = normalize_term(term)
    if not t_norm:
        return base
    # Always attach J for non-alkali multiplets (s states otherwise lose J).
    # Underscore before J keeps keys readable: 3s_3Po_2
    if j_float is None:
        return f"{n}{l_char}_{t_norm}"
    if abs(j_float - round(j_float)) < 1e-9:
        return f"{n}{l_char}_{t_norm}_{int(round(j_float))}"
    j_txt = str(j_str).strip().replace('/', 'h')  # 1/2 -> 1h2
    return f"{n}{l_char}_{t_norm}_{j_txt}"


def state_label_pretty(n, l_char, j_str, term=None, is_ground=False):
    """
    Readable spectroscopic label for plots (full-size digits).
      alkali: 3p1/2   multiplet: 3s 3P° J=2   ground closed-shell: 2p
    """
    j_float = parse_j(j_str)
    jl = j_label(l_char, j_float)
    base = f"{n}{l_char}"
    if not term or is_alkali_doublet_term(term):
        return f"{base}{jl}" if jl else base
    if is_ground and re.match(r'^1S\*?$', str(term).strip()) and (
            j_float is None or abs(j_float) < 1e-9):
        return base
    t_pretty = term_to_pretty(term)
    if not t_pretty:
        return state_label(n, l_char, j_str, term=term, is_ground=is_ground)
    if j_float is None:
        return f"{base} {t_pretty}"
    return f"{base} {t_pretty} J={j_to_plain(j_str)}"


def label_to_pretty(label):
    """
    Convert a stored ASCII key to a readable display label.
    Accepts new keys (3s_3Po_2), legacy (3s_3Po2), metal TM keys, and
    old Unicode pretty. Always returns full-size digits (no Unicode sub/sup).
    """
    if not label:
        return label
    # Transition-metal ASD keys: config_prefix_term_J
    _metal = metal_ascii_to_pretty(label)
    if _metal:
        return unicode_digits_to_plain(_metal)
    # Strip old Unicode sub/sup if present → rebuild from ASCII when possible
    # Multiplet with underscore before J: 3s_3Po_2 / 3d_2_3h2o_3h2
    m = re.match(r'^(\d+)([spdfgh])_(.+)_(\d+h\d+|\d+)$', label)
    if m:
        n, l, t_norm, j_txt = m.groups()
        return unicode_digits_to_plain(
            f"{n}{l} {term_to_pretty(t_norm)} J={j_to_plain(j_txt)}")
    # Legacy multiplet without J underscore: 3s_3Po2
    m = re.match(r'^(\d+)([spdfgh])_([0-9A-Za-z]+?)(\d+h\d+|\d+)$', label)
    if m:
        n, l, t_norm, j_txt = m.groups()
        if t_norm:
            return unicode_digits_to_plain(
                f"{n}{l} {term_to_pretty(t_norm)} J={j_to_plain(j_txt)}")
    # Alkali fine structure: keep full-size ASCII (3p1/2)
    m = re.match(r'^(\d+)([spdfgh])(\d+/\d+)$', label)
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}"
    m = re.match(r'^(\d+)([spdfgh])(\d+)$', label)
    if m and m.group(2) != 's':
        return f"{m.group(1)}{m.group(2)}{m.group(3)}"
    # Already a pretty form, or old Unicode pretty (3p₁/₂) — normalize digits
    return unicode_digits_to_plain(label)


def j_to_unicode(j_str):
    """J as Unicode subscript: 2 → ₂, 1/2 → ₁/₂."""
    plain = j_to_plain(j_str)
    if not plain:
        return ''
    if '/' in plain:
        a, b = plain.split('/', 1)
        return f"{_to_sub(a)}/{_to_sub(b)}"
    return _to_sub(plain)


def _jk_bracket_unicode(num, den):
    """jK bracket with full-size digits: [3/2] (subscripts are too small in UI)."""
    return f"[{num}/{den}]"


def term_to_unicode(term):
    """
    ASD / normalized term → Unicode term symbol.
      '3P*' / '3Po' → '³P°'
      '2[3/2]*' / '2_3h2o' → '²[₃/₂]°'
    """
    if not term:
        return ''
    t = str(term).strip()
    m_jk = re.match(r'^(\d+)_(\d+)h(\d+)(o?)$', t)
    if m_jk:
        mult, num, den, o = m_jk.groups()
        return f"{_to_sup(mult)}{_jk_bracket_unicode(num, den)}{'°' if o else ''}"
    m_jk_raw = re.match(r'^(\d+)\[(\d+)/(\d+)\]([*°]?)$', t)
    if m_jk_raw:
        mult, num, den, mark = m_jk_raw.groups()
        return f"{_to_sup(mult)}{_jk_bracket_unicode(num, den)}{'°' if mark else ''}"
    m_ls = re.match(r'^(\d+)([A-Za-z])([*°o])?$', t)
    if m_ls:
        mult, letter, mark = m_ls.groups()
        return f"{_to_sup(mult)}{letter.upper()}{'°' if mark else ''}"
    return t


def label_to_unicode(label):
    """
    ASCII key → classic Unicode display (3D plots).
      3s_3Po_2 → 3s ³P°₂
      3p1/2    → 3p₁/₂
      metal: 3d6_4s2_a_5D_4 → a⁵D J=4
    """
    if not label:
        return label
    _metal = metal_ascii_to_pretty(label)
    if _metal:
        return _metal
    m = re.match(r'^(\d+)([spdfgh])_(.+)_(\d+h\d+|\d+)$', label)
    if m:
        n, l, t_norm, j_txt = m.groups()
        return f"{n}{l} {term_to_unicode(t_norm)}{j_to_unicode(j_txt)}"
    m = re.match(r'^(\d+)([spdfgh])_([0-9A-Za-z]+?)(\d+h\d+|\d+)$', label)
    if m:
        n, l, t_norm, j_txt = m.groups()
        if t_norm:
            return f"{n}{l} {term_to_unicode(t_norm)}{j_to_unicode(j_txt)}"
    m = re.match(r'^(\d+)([spdfgh])(\d+/\d+)$', label)
    if m:
        return f"{m.group(1)}{m.group(2)}{j_to_unicode(m.group(3))}"
    m = re.match(r'^(\d+)([spdfgh])(\d+)$', label)
    if m and m.group(2) != 's':
        return f"{m.group(1)}{m.group(2)}{j_to_unicode(m.group(3))}"
    # Already Unicode / plain pretty — leave as-is
    if any(c in label for c in '⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉') or ' J=' in label:
        return label
    return label


def unwrap_asd_field(val):
    """
    Normalize one NIST ASD export cell.

    Handles Excel CSV wrapping ('=""2p6.3s"' → '2p6.3s'), ordinary quotes,
    and bracketed uncertain values ('[5.09]' → '5.09').
    """
    if val is None:
        return ''
    s = str(val).strip()
    # Excel formula export: ="value"  (csv module already consumed outer quotes)
    if s.startswith('='):
        s = s[1:].strip()
    s = s.strip('"').strip()
    if s.startswith('='):
        s = s[1:].strip().strip('"').strip()
    return s.strip('"').strip()


def split_asd_term(term):
    """Split ASD term 'a 5D' / 'z 7F*' into (prefix, LS-term)."""
    t = (term or '').strip()
    m = re.match(r'^([a-z]{1,3})\s+(.+)$', t)
    if m:
        return m.group(1), m.group(2).strip()
    return '', t


def normalize_asd_config(config):
    """
    Stable config fingerprint for TM level/line matching.

    Levels and lines CSVs must produce the same key for the same ASD
    configuration. Strip whitespace; keep alphanumerics and dots; drop
    parentheses/punctuation (parent terms become embedded letters/digits).
    """
    s = unwrap_asd_field(config).strip()
    if not s:
        return ''
    s = re.sub(r'\s+', '', s)
    s = re.sub(r'[^0-9A-Za-z.]', '', s)
    return s.replace('.', '_')[:64]


def trailing_nl_from_config(config):
    """
    Last valence (n, l) from an ASD configuration string.

    Uses parse_config when possible; falls back to the last \\d+[spdfg]
    token so spaced Fe configs still get n/l for JSON/transitions.
    """
    s = unwrap_asd_field(config).strip()
    if not s:
        return None, None
    shells = parse_config(s)
    if shells:
        n, l, _ = shells[-1]
        return int(n), l
    hits = re.findall(r'(\d+)([spdfgh])\d*', s)
    if not hits:
        return None, None
    n_s, l = hits[-1]
    return int(n_s), l


def metal_state_label(config, term, j_str):
    """ASCII key for TM NIST levels: config_term_J (not alkali nl)."""
    prefix, ls = split_asd_term(term)
    conf_key = normalize_asd_config(config)
    parts = []
    if conf_key:
        parts.append(conf_key)
    if prefix:
        parts.append(prefix)
    t_norm = normalize_term(ls)
    if t_norm:
        parts.append(t_norm)
    j_plain = j_to_plain(j_str)
    if j_plain != '':
        parts.append(j_plain.replace('/', 'h'))
    return '_'.join(parts) if parts else 'level'


def metal_state_label_pretty(term, j_str, config=None):
    """Display label: 'a⁵D J=4' (optional short config prefix)."""
    prefix, ls = split_asd_term(term)
    m = re.match(r'^(\d+)([A-Za-z])([*°o])?$', (ls or '').strip())
    if m:
        mult, letter, mark = m.groups()
        pretty = f"{prefix}{_to_sup(mult)}{letter.upper()}{'°' if mark else ''}"
    else:
        ls_pretty = term_to_pretty(ls) if ls else ''
        pretty = f"{prefix}{ls_pretty}" if prefix else ls_pretty
        pretty = pretty or (term or '')
    j_plain = j_to_plain(j_str)
    if j_plain != '':
        pretty = f"{pretty} J={j_plain}".strip()
    return pretty or metal_state_label(config, term, j_str)


def metal_ascii_to_pretty(label):
    """
    ASCII metal key → display: '3d6_4s2_a_5D_4' → 'a⁵D J=4'.
    Returns None if the label is not a metal key.
    """
    if not label or not isinstance(label, str):
        return None
    m = re.match(
        r'^(.*)_([a-z]{1,3})_(\d+[A-Za-z]o?)_(\d+h\d+|\d+)$',
        label,
    )
    if not m:
        return None
    _conf, prefix, t_norm, j_txt = m.groups()
    # normalize_term stored oddness as trailing 'o'
    if t_norm.endswith('o') and len(t_norm) >= 2 and t_norm[-2].isalpha():
        ls = t_norm[:-1] + '*'
    else:
        ls = t_norm
    return metal_state_label_pretty(f"{prefix} {ls}", j_txt.replace('h', '/'))


def build_nist_term_excitations(ionization_energy, levels):
    """
    NIST term levels → plotinteractive-compatible excitations.

    energy_eV is ionization-relative (negative), same convention as QDT JSON.
    Ground (E≈0) is omitted; it lives in ground_state metadata.
    """
    states = []
    E_ground = -float(ionization_energy)
    state_num = 0
    for lv in sorted(levels, key=lambda x: (x['energy_eV'], x.get('J') or '')):
        e_from_gs = float(lv['energy_eV'])
        if e_from_gs <= 1e-12:
            continue
        E = -(ionization_energy - e_from_gs)
        if E >= 0:
            continue
        delta_E = E - E_ground
        if delta_E <= 0:
            continue
        conf = lv.get('config') or ''
        term = lv.get('term') or ''
        j_str = lv.get('J') or ''
        state_num += 1
        states.append({
            'state_number': state_num,
            'n': lv.get('n'),
            'l': lv.get('l'),
            'J': j_str or None,
            'term': term or None,
            'config': conf or None,
            'energy_eV': round(E, 8),
            'wavelength_nm': round(1239.84 / delta_E, 4),
            'label': metal_state_label(conf, term, j_str),
            'label_pretty': metal_state_label_pretty(term, j_str, config=conf),
            'source': 'NIST',
            'method': 'NIST ASD levels (no QDT)',
            'theoretical': bool(lv.get('theoretical')),
            'uncertainty_eV': lv.get('uncertainty_eV'),
        })
    return states


def compute_quantum_defects(ionization_energy, levels, ry_eff=None):
    """
    Compute delta = n - sqrt(Ry_eff / (IE - E)) for each level, average per orbital.

    ry_eff = Ry * Z^2 where Z is the residual ionic charge (1 for neutrals,
    2 for singly charged ions, …). Defaults to hydrogenic Ry.

    Theoretical ASD levels (Prefix/Suffix [ ]) are excluded from the fit.
    When Uncertainty (eV) is available, use inverse-variance weights
    w = 1 / max(σ, σ_floor)² so precisely known low-n levels dominate.
    """
    if ry_eff is None:
        ry_eff = Ry
    defects_by_orbital = {orb: [] for orb in 'spdfg'}  # list of (delta, weight)

    # Floor: 10% of median positive uncertainty, or 1e-9 eV
    pos_unc = [lv['uncertainty_eV'] for lv in levels
               if lv.get('uncertainty_eV') and lv['uncertainty_eV'] > 0
               and not lv.get('theoretical')]
    if pos_unc:
        unc_floor = max(0.1 * float(np.median(pos_unc)), 1e-12)
    else:
        unc_floor = 1e-9

    n_skip_theo = 0
    for lv in levels:
        if lv.get('theoretical'):
            n_skip_theo += 1
            continue
        n, l_char, E = lv['n'], lv['l'], lv['energy_eV']
        binding = ionization_energy - E
        if binding <= 0:
            continue
        n_eff = np.sqrt(ry_eff / binding)
        delta = n - n_eff
        if l_char not in defects_by_orbital:
            continue
        unc = lv.get('uncertainty_eV')
        if unc is None or unc <= 0:
            w = 1.0
        else:
            w = 1.0 / (max(unc, unc_floor) ** 2)
        defects_by_orbital[l_char].append((delta, w))

    if n_skip_theo:
        print(f"  (excluded {n_skip_theo} theoretical [ ] levels from QD fit)")

    averaged = {}
    for orbital in ['s', 'p', 'd', 'f', 'g']:
        vals = defects_by_orbital[orbital]
        if vals:
            deltas = np.array([d for d, _ in vals])
            weights = np.array([w for _, w in vals])
            wsum = float(weights.sum())
            mean = float(np.sum(deltas * weights) / wsum)
            # Weighted sample std (for print diagnostics)
            var = float(np.sum(weights * (deltas - mean)**2) / wsum)
            averaged[orbital] = mean
            print(f"  {orbital}: {len(vals):3d} levels,  "
                  f"delta_avg = {mean:.4f}  (wstd = {np.sqrt(var):.4f})")
        else:
            averaged[orbital] = 0.0
            print(f"  {orbital}:   0 levels,  delta_avg = 0.0000  (no data)")

    return averaged


def build_nist_lookup(ionization_energy, levels):
    """
    Build (n, l, J_str, term) -> {energy_eV, uncertainty_eV, theoretical, term}.

    energy_eV is ionization-relative (negative). Term is part of the key so
    e.g. Na II 2p5.3s 3P* J=1 and 1P* J=1 are not collapsed.
    """
    lookup = {}
    for lv in levels:
        E_abs = -(ionization_energy - lv['energy_eV'])
        term = lv.get('term') or ''
        key = (lv['n'], lv['l'], lv['J'], term)
        entry = dict(
            energy_eV=float(E_abs),
            uncertainty_eV=lv.get('uncertainty_eV'),
            theoretical=bool(lv.get('theoretical')),
            term=term,
        )
        if key not in lookup or E_abs < lookup[key]['energy_eV']:
            lookup[key] = entry
    return lookup


def rydberg_energy(n, l_char, delta, ry_eff=None):
    """Model energy using quantum defect (ionization-relative)."""
    if ry_eff is None:
        ry_eff = Ry
    n_eff = n - delta
    if n_eff <= 0:
        return None
    return -ry_eff / (n_eff ** 2)


RITZ_MIN_POINTS = 6        # distinct measured n required per orbital
RITZ_MAX_RMS = 2e-3        # reject the fit if the residual in delta exceeds this


def fit_ritz_defects(ionization_energy, levels, ry_eff=None):
    """
    Energy-dependent quantum defect  delta(n) = d0 + d2 / (n - d0)^2  (Ritz, 2 terms)
    fitted per orbital to the MEASURED (non-theoretical) levels.

    A single averaged defect (compute_quantum_defects) is dominated by the
    precisely known low-n levels (Na p: 0.8650 vs the Rydberg limit 0.8549), which
    mis-places high-n levels by ~1.5 % of the n* gap between neighbouring series
    (np vs ns) and wrecks near-resonant quantities (Rydberg polarizabilities
    +20..30 %). Levels of one n are averaged over J with weights 2J+1. The lowest two
    n of a series (core penetration beyond a 2-term Ritz law) are left out.
    Returns {orbital: {delta0, delta2, n_min, n_max, n_points, rms}}; orbitals with
    too few points or a poor fit are omitted (constant defect is used for them).
    """
    if ry_eff is None:
        ry_eff = Ry
    acc = {}
    for lv in levels:
        if lv.get('theoretical'):
            continue
        binding = ionization_energy - lv['energy_eV']
        if binding <= 0 or lv['l'] not in 'spdfg':
            continue
        j = parse_j(lv['J'])
        g = 2 * j + 1 if j is not None else 1.0
        acc.setdefault(lv['l'], {}).setdefault(lv['n'], []).append(
            (lv['n'] - float(np.sqrt(ry_eff / binding)), g))
    out = {}
    for l_char, by_n in acc.items():
        ns_all = sorted(by_n)
        n_lo = ns_all[0] + 2
        ns = np.array([n for n in ns_all if n >= n_lo], dtype=float)
        if ns.size < RITZ_MIN_POINTS:
            continue
        ds = np.array([sum(d * g for d, g in by_n[int(n)]) / sum(g for _, g in by_n[int(n)])
                       for n in ns])
        d0 = float(ds[-1])
        c = None
        for _ in range(30):
            x = 1.0 / (ns - d0) ** 2
            M = np.vstack([np.ones_like(x), x]).T
            c = np.linalg.lstsq(M, ds, rcond=None)[0]
            d0 = float(c[0])
        rms = float(np.sqrt(np.mean((c[0] + c[1] / (ns - c[0]) ** 2 - ds) ** 2)))
        if np.isfinite(rms) and rms < RITZ_MAX_RMS:
            out[l_char] = dict(delta0=float(c[0]), delta2=float(c[1]), n_min=int(ns[0]),
                               n_max=int(ns[-1]), n_points=int(ns.size), rms=rms)
    return out


def ritz_delta(ritz, l_char, n, default):
    """delta(n) from fit_ritz_defects(), or `default` (constant defect) if unavailable."""
    p = (ritz or {}).get(l_char)
    if not p:
        return default
    return p['delta0'] + p['delta2'] / (n - p['delta0']) ** 2


def generate_rydberg_series(ionization_energy, quantum_defects, nist_lookup,
                             n_start, n_max, ry_eff=None, ritz=None):
    """
    Generate Rydberg series with J-resolved levels where NIST data exists.

    For levels in nist_lookup: emit one state per (n, l, J) sublevel with the
    exact NIST energy and a label like '3p1/2', '3p3/2'.

    For high-n levels not in NIST (QDT model): emit one state per (n, l) using
    the quantum-defect energy. No J splitting is modelled (SOC ~ n^-3 → tiny
    at high n). Label is plain '10p', '15d', etc.

    Wavelength is computed as the transition energy from ground state.
    """
    if ry_eff is None:
        ry_eff = Ry
    states = []
    state_num = 0
    l_to_int  = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4}
    E_ground  = -ionization_energy

    # Collect which (n, l) pairs are covered by NIST lookup
    nist_nl_covered = set()
    for key in nist_lookup:
        n, l = key[0], key[1]
        nist_nl_covered.add((n, l))

    # Lowest NIST-known n per orbital: the QDT model must not extrapolate
    # BELOW the measured series. For open-shell atoms the low members can be
    # core-filled and nonexistent (Cl 3s2 3p5: the s series starts at 4s, but
    # blind QDT extrapolation would fabricate a bound '3s' Rydberg state).
    min_nist_n = {}
    for (n, l) in nist_nl_covered:
        if l not in min_nist_n or n < min_nist_n[l]:
            min_nist_n[l] = n

    for n in range(n_start, n_max + 1):
        for l_char in ['s', 'p', 'd', 'f', 'g']:
            l = l_to_int.get(l_char, -1)
            if l < 0 or l >= n:
                continue

            if (n, l_char) in nist_nl_covered:
                # NIST path: one entry per (J, term) sublevel
                sublevels = sorted(
                    [(j_str, term, meta)
                     for (nn, ll, j_str, term), meta in nist_lookup.items()
                     if nn == n and ll == l_char],
                    key=lambda x: (parse_j(x[0]) or 99, x[1] or '')
                )
                for j_str, term, meta in sublevels:
                    E = meta['energy_eV'] if isinstance(meta, dict) else meta
                    if E is None or E >= 0:
                        continue
                    delta_E = E - E_ground
                    if delta_E <= 0:
                        continue

                    label = state_label(n, l_char, j_str, term=term)
                    label_pretty = state_label_pretty(n, l_char, j_str, term=term)
                    theo = bool(meta.get('theoretical')) if isinstance(meta, dict) else False
                    unc = meta.get('uncertainty_eV') if isinstance(meta, dict) else None

                    state_num += 1
                    states.append({
                        'state_number': state_num,
                        'n':            n,
                        'l':            l_char,
                        'J':            j_str,
                        'term':         term or None,
                        'energy_eV':    round(E, 6),
                        'wavelength_nm': round(1239.84 / delta_E, 4),
                        'label':        label,
                        'label_pretty': label_pretty,
                        'source':       'NIST',
                        'method':       'Quantum Defect Theory (NIST-derived)',
                        'theoretical':  theo,
                        'uncertainty_eV': unc,
                    })

            else:
                # QDT model path: single entry (no J splitting) 
                # Never model n below the lowest NIST-observed member of the
                # series: those states are core-filled, not Rydberg states.
                if l_char in min_nist_n and n < min_nist_n[l_char]:
                    continue
                delta = ritz_delta(ritz, l_char, n, quantum_defects.get(l_char, 0.0))
                E = rydberg_energy(n, l_char, delta, ry_eff=ry_eff)
                if E is None or E >= 0:
                    continue
                delta_E = E - E_ground
                if delta_E <= 0:
                    continue

                state_num += 1
                _lbl = f"{n}{l_char}"
                states.append({
                    'state_number': state_num,
                    'n':            n,
                    'l':            l_char,
                    'J':            None,
                    'energy_eV':    round(E, 6),
                    'wavelength_nm': round(1239.84 / delta_E, 4),
                    'label':        _lbl,
                    'label_pretty': _lbl,
                    'source':       'QDT model',
                    'method':       'Quantum Defect Theory (NIST-derived)',
                    'theoretical':  False,
                    'uncertainty_eV': None,
                })
    return states
