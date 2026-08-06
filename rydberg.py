import csv
import io
import json
import os
import re
import sys
import argparse
import numpy as np

Ry = 13.6057  # eV
MAX_OCC = {'s': 2, 'p': 6, 'd': 10, 'f': 14, 'g': 18, 'h': 22}
# Project-relative path so the script works on any machine / cwd
_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_ROOT, "NIST Levels")

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


def _to_sup(text):
    return str(text).translate(_SUP_DIGITS)


def _to_sub(text):
    return str(text).translate(_SUB_DIGITS)


def j_to_plain(j_str):
    """J with normal-size digits: 2, 1/2, 3h2 → 3/2."""
    if j_str is None or j_str == '':
        return ''
    return str(j_str).strip().replace('h', '/')


def j_to_pretty(j_str):
    """J for display — plain digits (Unicode subscripts are too small in UI)."""
    return j_to_plain(j_str)


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


def term_to_html(term):
    """HTML term symbol with readable-size sup/sub tags."""
    if not term:
        return ''
    t = str(term).strip()
    m_jk = re.match(r'^(\d+)_(\d+)h(\d+)(o?)$', t)
    if m_jk:
        mult, num, den, o = m_jk.groups()
        return (f"{_HTML_SUP}{mult}</sup>[{_HTML_SUB}{num}/{den}</sub>]"
                f"{'°' if o else ''}")
    m_jk_raw = re.match(r'^(\d+)\[(\d+)/(\d+)\]([*°]?)$', t)
    if m_jk_raw:
        mult, num, den, mark = m_jk_raw.groups()
        return (f"{_HTML_SUP}{mult}</sup>[{_HTML_SUB}{num}/{den}</sub>]"
                f"{'°' if mark else ''}")
    m_ls = re.match(r'^(\d+)([A-Za-z])([*°o])?$', t)
    if m_ls:
        mult, letter, mark = m_ls.groups()
        return f"{_HTML_SUP}{mult}</sup>{letter.upper()}{'°' if mark else ''}"
    return t

def is_alkali_doublet_term(term):
    """True for valence alkali-like terms 2S / 2P* / 2D* … (short labels OK)."""
    if not term:
        return False
    return bool(re.match(r'^2[SPDFGH]\*?$', str(term).strip()))


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


def state_label_html(n, l_char, j_str, term=None, is_ground=False):
    """HTML label; prefers full-size plain form (same as pretty)."""
    return state_label_pretty(n, l_char, j_str, term=term, is_ground=is_ground)


def label_to_pretty(label):
    """
    Convert a stored ASCII key to a readable display label.
    Accepts new keys (3s_3Po_2), legacy (3s_3Po2), metal TM keys, and
    old Unicode pretty.
    """
    if not label:
        return label
    # Transition-metal ASD keys: config_prefix_term_J
    _metal = metal_ascii_to_pretty(label)
    if _metal:
        return _metal
    # Strip old Unicode sub/sup if present → rebuild from ASCII when possible
    # Multiplet with underscore before J: 3s_3Po_2 / 3d_2_3h2o_3h2
    m = re.match(r'^(\d+)([spdfgh])_(.+)_(\d+h\d+|\d+)$', label)
    if m:
        n, l, t_norm, j_txt = m.groups()
        return f"{n}{l} {term_to_pretty(t_norm)} J={j_to_plain(j_txt)}"
    # Legacy multiplet without J underscore: 3s_3Po2
    m = re.match(r'^(\d+)([spdfgh])_([0-9A-Za-z]+?)(\d+h\d+|\d+)$', label)
    if m:
        n, l, t_norm, j_txt = m.groups()
        if t_norm:
            return f"{n}{l} {term_to_pretty(t_norm)} J={j_to_plain(j_txt)}"
    # Alkali fine structure: keep full-size ASCII (3p1/2)
    m = re.match(r'^(\d+)([spdfgh])(\d+/\d+)$', label)
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}"
    m = re.match(r'^(\d+)([spdfgh])(\d+)$', label)
    if m and m.group(2) != 's':
        return f"{m.group(1)}{m.group(2)}{m.group(3)}"
    # Already a plain pretty form (has ' J=')
    if ' J=' in label or '°' in label:
        return label
    return label


def label_to_html(label):
    """ASCII key → display string (plain full-size; HTML optional via callers)."""
    return label_to_pretty(label)


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


def state_label_unicode(n, l_char, j_str, term=None, is_ground=False):
    """
    Classic spectroscopic Unicode label (for 3D Grotrian / orbital viewer).
      alkali: 3p₁/₂   multiplet: 3s ³P°₂   jK: 3d ²[₁/₂]°₁
    """
    j_float = parse_j(j_str)
    jl = j_label(l_char, j_float)
    base = f"{n}{l_char}"
    if not term or is_alkali_doublet_term(term):
        return f"{base}{j_to_unicode(jl)}" if jl else base
    if is_ground and re.match(r'^1S\*?$', str(term).strip()) and (
            j_float is None or abs(j_float) < 1e-9):
        return base
    t_uni = term_to_unicode(term)
    if not t_uni:
        return state_label(n, l_char, j_str, term=term, is_ground=is_ground)
    if j_float is None:
        return f"{base} {t_uni}"
    return f"{base} {t_uni}{j_to_unicode(j_str)}"


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


def find_nist_levels_file(element, data_dir=DATA_DIR):
    """Prefer <element>.csv (full ASD export), fall back to .txt."""
    for ext in ('.csv', '.txt'):
        path = os.path.join(data_dir, f"{element}{ext}")
        if os.path.exists(path):
            return path
    return None


def parse_nist_file(filepath):
    """
    Parse a NIST energy-levels export (latin-1).

    Handles:
      - Legacy/curated .txt: no header, tab-separated, undotted configs ('2p63s')
      - Full ASD tab .txt: header row, quoted fields, dotted configs ('2p6.3s')
      - Full ASD Excel .csv: comma-separated, Excel '=""…""' cell wrapping,
        Prefix/Suffix columns, multiple series-limit rows

    Returns:
        ionization_energy (float, eV): the LOWEST 'Limit' row
        levels: list of dicts with keys
            n, l, J, term, energy_eV, uncertainty_eV (or None), theoretical (bool)
            Bound spdfg levels only (below IE).
            theoretical=True when ASD marks the level with [ ] Prefix/Suffix
            (commonly high-n / high-ℓ calculated values).
    """
    levels = []
    ionization_energy = None
    current_n = None
    current_l = None

    with open(filepath, 'r', encoding='latin-1') as f:
        raw = f.read()

    first = next((ln for ln in raw.splitlines() if ln.strip()), '')
    delim = ',' if first.count(',') > first.count('\t') else '\t'
    rows = list(csv.reader(io.StringIO(raw), delimiter=delim))
    if not rows:
        return None, []

    idx = {
        'config': 0, 'term': 1, 'J': 2, 'level': 3,
        'unc': None, 'prefix': None, 'suffix': None,
    }
    start = 0
    header0 = unwrap_asd_field(rows[0][0]).lower() if rows[0] else ''
    if header0 == 'configuration':
        for i, name in enumerate(rows[0]):
            key = unwrap_asd_field(name).lower()
            if key == 'configuration':
                idx['config'] = i
            elif key == 'term':
                idx['term'] = i
            elif key == 'j':
                idx['J'] = i
            elif key.startswith('level'):
                idx['level'] = i
            elif key.startswith('uncertainty'):
                idx['unc'] = i
            elif key == 'prefix':
                idx['prefix'] = i
            elif key == 'suffix':
                idx['suffix'] = i
        start = 1

    def col(cols, key, strip_extra=''):
        i = idx.get(key)
        if i is None or len(cols) <= i:
            return ''
        return unwrap_asd_field(cols[i]).strip(strip_extra)

    for cols in rows[start:]:
        if not cols or not any(unwrap_asd_field(c) for c in cols):
            continue

        config_col = col(cols, 'config')
        term_col   = col(cols, 'term')
        j_col      = col(cols, 'J', strip_extra='?')
        energy_col = col(cols, 'level', strip_extra='[]')
        unc_col    = col(cols, 'unc', strip_extra='[]')
        prefix     = col(cols, 'prefix')
        suffix     = col(cols, 'suffix')
        theoretical = ('[' in prefix) or (']' in suffix)

        if term_col.lower() == 'limit' or 'limit' in term_col.lower():
            try:
                lim = float(energy_col)
                if ionization_energy is None or lim < ionization_energy:
                    ionization_energy = lim
            except ValueError:
                pass
            continue

        if config_col:
            shells = parse_config(config_col)
            if shells:
                current_n, current_l, _ = shells[-1]
            else:
                current_n = None
                current_l = None
        elif term_col:
            current_n = None
            current_l = None

        if current_n is None or current_l is None:
            continue
        if current_l not in ('s', 'p', 'd', 'f', 'g'):
            continue

        try:
            energy = float(energy_col)
        except ValueError:
            continue

        unc = None
        if unc_col:
            try:
                unc = float(unc_col)
            except ValueError:
                unc = None

        levels.append(dict(
            n=current_n, l=current_l, J=j_col,
            term=term_col or '',
            energy_eV=energy,
            uncertainty_eV=unc,
            theoretical=theoretical,
        ))

    if ionization_energy is not None:
        levels = [lv for lv in levels if lv['energy_eV'] < ionization_energy]

    return ionization_energy, levels


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


def parse_nist_asd_term_levels(filepath):
    """
    Parse full ASD levels keeping Configuration / Term / J.

    Unlike parse_nist_file (alkali QDT path), this does not require a trailing
    valence nl and does not rewrite identity as (n,l). Used for transition
    metals and other open-shell species where QDT labels are wrong.

    Returns:
        ionization_energy (eV), levels: list of dicts with
        config, term, J, energy_eV (from ground), uncertainty_eV, theoretical,
        and optional n,l from the trailing shell when parseable.
    """
    levels = []
    ionization_energy = None

    with open(filepath, 'r', encoding='latin-1') as f:
        raw = f.read()

    first = next((ln for ln in raw.splitlines() if ln.strip()), '')
    delim = ',' if first.count(',') > first.count('\t') else '\t'
    rows = list(csv.reader(io.StringIO(raw), delimiter=delim))
    if not rows:
        return None, []

    idx = {
        'config': 0, 'term': 1, 'J': 2, 'level': 3,
        'unc': None, 'prefix': None, 'suffix': None,
    }
    start = 0
    header0 = unwrap_asd_field(rows[0][0]).lower() if rows[0] else ''
    if header0 == 'configuration':
        for i, name in enumerate(rows[0]):
            key = unwrap_asd_field(name).lower()
            if key == 'configuration':
                idx['config'] = i
            elif key == 'term':
                idx['term'] = i
            elif key == 'j':
                idx['J'] = i
            elif key.startswith('level'):
                idx['level'] = i
            elif key.startswith('uncertainty'):
                idx['unc'] = i
            elif key == 'prefix':
                idx['prefix'] = i
            elif key == 'suffix':
                idx['suffix'] = i
        start = 1

    def col(cols, key, strip_extra=''):
        i = idx.get(key)
        if i is None or len(cols) <= i:
            return ''
        return unwrap_asd_field(cols[i]).strip(strip_extra)

    # ASD blanks Configuration on continuation rows → inherit last config.
    last_config = ''

    for cols in rows[start:]:
        if not cols or not any(unwrap_asd_field(c) for c in cols):
            continue

        config_col = col(cols, 'config')
        term_col = col(cols, 'term')
        j_col = col(cols, 'J', strip_extra='?')
        energy_col = col(cols, 'level', strip_extra='[]')
        unc_col = col(cols, 'unc', strip_extra='[]')
        prefix = col(cols, 'prefix')
        suffix = col(cols, 'suffix')
        theoretical = ('[' in prefix) or (']' in suffix)

        if term_col.lower() == 'limit' or 'limit' in term_col.lower():
            try:
                lim = float(energy_col)
                if ionization_energy is None or lim < ionization_energy:
                    ionization_energy = lim
            except ValueError:
                pass
            continue

        if not energy_col:
            continue
        try:
            energy = float(energy_col)
        except ValueError:
            continue

        # Ambiguous multi-J ASD rows — skip rather than invent a single J
        if ',' in (j_col or ''):
            continue

        # Bare '*' term = unclassified ASD placeholder
        if (term_col or '').strip() in ('*', '°', ''):
            if config_col:
                last_config = config_col
            continue

        if config_col:
            last_config = config_col
        else:
            config_col = last_config

        n_val, l_val = trailing_nl_from_config(config_col) if config_col else (None, None)

        unc = None
        if unc_col:
            try:
                unc = float(unc_col)
            except ValueError:
                unc = None

        levels.append(dict(
            config=config_col,
            term=term_col or '',
            J=j_col,
            energy_eV=energy,
            uncertainty_eV=unc,
            theoretical=theoretical,
            n=n_val,
            l=l_val,
        ))

    if ionization_energy is not None:
        levels = [lv for lv in levels if lv['energy_eV'] < ionization_energy]

    return ionization_energy, levels


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


def generate_rydberg_series(ionization_energy, quantum_defects, nist_lookup,
                             n_start, n_max, ry_eff=None):
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
                delta = quantum_defects.get(l_char, 0.0)
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


def main():
    parser = argparse.ArgumentParser(
        description='Build Rydberg series from NIST levels + quantum defect theory')
    parser.add_argument(
        'element', nargs='?', default=None,
        help='Species id: symbol (Na) or ion stem (Mg_c1) when NIST file exists')
    parser.add_argument(
        '--n-max', type=int, default=None,
        help='Maximum principal quantum number (prompted if omitted; default prompt 30)')
    parser.add_argument(
        '--nist-dir', default=None,
        help=f'NIST levels directory (default: {DATA_DIR})')
    parser.add_argument(
        '--nist-only', action='store_true',
        help='Write NIST ASD term levels only (no QDT). Auto-used for 3d metals.')
    args = parser.parse_args()

    # Prefer UTF-8 stdout on Windows consoles
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    from species import resolve_species

    nist_dir = args.nist_dir or DATA_DIR

    if args.element:
        raw = args.element.strip()
    else:
        raw = input("Enter species id (e.g. Cs, Na, Mg_c1): ").strip()
    try:
        sp = resolve_species(raw)
    except ValueError as exc:
        print(f"Error: {exc}")
        return
    element = sp["species_id"]

    txt_file = find_nist_levels_file(element, data_dir=nist_dir)
    if txt_file is None:
        print(f"Error: no '{element}.csv' or '{element}.txt' in {nist_dir}.")
        if sp["kind"] == "ion":
            print(f"  Ion {element} ({sp['label']}): add a NIST ASD export named "
                  f"'{element}.csv' (or .txt) before building Rydberg data. "
                  f"Do not invent level/f-value tables.")
        return

    # 3d metals: term/config NIST overlay only — do not invent QDT series.
    from orca_templates import TRANSITION_METAL_3D, get_element_class, ElementClass
    nist_only = bool(args.nist_only) or (
        sp["symbol"] in TRANSITION_METAL_3D
        or get_element_class(sp["symbol"], sp["charge"], sp["multiplicity"])
        == ElementClass.TRANSITION_METAL
    )

    print(f"\nParsing {txt_file}...")
    if nist_only:
        ionization_energy, levels = parse_nist_asd_term_levels(txt_file)
    else:
        ionization_energy, levels = parse_nist_file(txt_file)

    if ionization_energy is None:
        print("Error: Could not find ionization energy (Limit line) in file.")
        return

    print(f"Ionization energy : {ionization_energy:.6f} eV")
    print(f"Levels parsed     : {len(levels)}"
          + (" (term/config, no QDT)" if nist_only else " (J-resolved)"))

    if nist_only:
        gs_level = min(levels, key=lambda x: abs(x['energy_eV']))
        gs_term = gs_level.get('term') or ''
        gs_j = gs_level.get('J')
        gs_conf = gs_level.get('config') or ''
        gs_label = metal_state_label(gs_conf, gs_term, gs_j)
        gs_label_pretty = metal_state_label_pretty(gs_term, gs_j, config=gs_conf)
        print(f"\nGround state detected: {gs_label_pretty}  [{gs_label}]  "
              f"(config={gs_conf or '?'}, term={gs_term or '?'}, J={gs_j or '?'})")

        states = build_nist_term_excitations(ionization_energy, levels)
        if not states:
            print("No NIST term levels generated.")
            return
        theo_count = sum(1 for s in states if s.get('theoretical'))
        print(f"\nGenerated {len(states)} NIST term levels "
              f"({theo_count} marked theoretical [ ])")
        print(f"Energy range: {min(s['energy_eV'] for s in states):.4f} to "
              f"{max(s['energy_eV'] for s in states):.4f} eV")

        # Low-lying sanity: a5D / ground-term FS for Fe-like exports
        low = [s for s in states if (ionization_energy + s['energy_eV']) < 0.2]
        if low:
            print("\n  Near-ground NIST FS (E < 0.2 eV):")
            for s in low[:12]:
                e_exc = ionization_energy + s['energy_eV']
                disp = s.get('label_pretty') or s['label']
                print(f"    {disp:20s}  {e_exc:10.6f} eV")

        os.makedirs('data_json', exist_ok=True)
        json_file = f'data_json/{element}_rydberg.json'
        data = {
            'species_id': element,
            'element': sp["symbol"],
            'label': sp["label"],
            'kind': sp["kind"],
            'charge': sp["charge"],
            'multiplicity': sp["multiplicity"],
            'method': 'NIST ASD levels (no QDT)',
            'ionization_energy_eV': ionization_energy,
            'quantum_defects': {},
            'qd_notes': (
                'No quantum-defect series for this species. Excitations are '
                'measured/ tabulated ASD term levels (config + term + J).'
            ),
            'ground_state': {
                'n': gs_level.get('n'),
                'l': gs_level.get('l'),
                'J': gs_j,
                'term': gs_term or None,
                'config': gs_conf or None,
                'label': gs_label,
                'label_pretty': gs_label_pretty,
            },
            'n_start': None,
            'n_max': None,
            'num_states': len(states),
            'num_theoretical': theo_count,
            'excitations': states,
        }
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved NIST term levels to: {json_file}")
        print(f"\nIonization energy     : {ionization_energy:.5f} eV")
        print(f"Ionization threshold  : 0.00000 eV")
        return

    # Residual ionic charge seen by a Rydberg electron (Na+ -> Na2+ => Z=2).
    z_core = abs(sp["charge"]) + 1
    ry_eff = Ry * (z_core ** 2)
    if z_core != 1:
        print(f"Using Z={z_core} Rydberg scale: Ry_eff = {ry_eff:.4f} eV")

    print(f"\nComputing quantum defects (averaged over all n,J per orbital):")
    quantum_defects = compute_quantum_defects(
        ionization_energy, levels, ry_eff=ry_eff)

    nist_lookup = build_nist_lookup(ionization_energy, levels)
    print(f"\nNIST lookup table : {len(nist_lookup)} unique (n, l, J, term) entries")

    # Show fine-structure splitting for first few p levels (good sanity check)
    p_levels = sorted(
        [(n, j, term, meta['energy_eV'])
         for (n, l, j, term), meta in nist_lookup.items() if l == 'p'],
        key=lambda x: (x[0], parse_j(x[1]) or 0, x[2] or '')
    )
    if p_levels:
        print("\n  p-level fine structure (NIST):")
        prev_n = None
        for n, j, term, E in p_levels[:20]:
            marker = '  ' if n == prev_n else f'\n  n={n}'
            tbit = f" {term}" if term else ''
            print(f"  {marker}  {n}p_{j:>3}{tbit}  {E:+.6f} eV")
            prev_n = n

    # Ground state: lowest energy level in the NIST file 
    gs_level = min(levels, key=lambda x: abs(x['energy_eV']))
    gs_n, gs_l, gs_j = gs_level['n'], gs_level['l'], gs_level['J']
    gs_term = gs_level.get('term') or ''

    # n_start: principal quantum number of the ground state valence shell.
    # For alkalis (ns¹) this equals gs_n. For closed-shell np⁶ (noble /
    # Na+-like) the valence shell is full, so Rydberg series start at gs_n+1.
    n_start = gs_n
    if gs_l == 'p' and (parse_j(gs_j) == 0.0 or gs_j in ('0', '0.0', 0)):
        n_start = gs_n + 1

    gs_label = state_label(gs_n, gs_l, gs_j, term=gs_term, is_ground=True)
    gs_label_pretty = state_label_pretty(
        gs_n, gs_l, gs_j, term=gs_term, is_ground=True)

    print(f"\nGround state detected: {gs_label_pretty}  [{gs_label}]  "
          f"(n={gs_n}, l={gs_l}, J={gs_j or '?'}"
          + (f", term={gs_term}" if gs_term else "") + ")")

    if args.n_max is not None:
        n_max = int(args.n_max)
    else:
        n_max = int(input(f"\nMaximum n value (e.g. 50): ") or "50")

    if n_max < n_start:
        print(f"Error: --n-max ({n_max}) must be >= ground-state n ({n_start}).")
        return

    print(f"\nGenerating Rydberg series for {element} (n={n_start}..{n_max})...")
    states = generate_rydberg_series(ionization_energy, quantum_defects,
                                     nist_lookup, n_start, n_max, ry_eff=ry_eff)

    if not states:
        print("No states generated.")
        return

    nist_count  = sum(1 for s in states if s['source'] == 'NIST')
    model_count = sum(1 for s in states if s['source'] == 'QDT model')
    theo_count  = sum(1 for s in states if s.get('theoretical'))
    print(f"\nGenerated {len(states)} Rydberg states")
    print(f"  {nist_count} from NIST (J-resolved),  {model_count} from QDT model"
          f"  ({theo_count} marked theoretical [ ])")
    print(f"Energy range: {min(s['energy_eV'] for s in states):.4f} to "
          f"{max(s['energy_eV'] for s in states):.4f} eV")

    os.makedirs('data_json', exist_ok=True)

    json_file = f'data_json/{element}_rydberg.json'
    data = {
        'species_id':           element,
        'element':              sp["symbol"],
        'label':                sp["label"],
        'kind':                 sp["kind"],
        'charge':               sp["charge"],
        'multiplicity':         sp["multiplicity"],
        'method':               'Quantum Defect Theory (NIST-derived)',
        'ionization_energy_eV': ionization_energy,
        'z_core':               z_core,
        'rydberg_constant_eV':  ry_eff,
        'quantum_defects':      quantum_defects,
        'qd_notes': (
            'Averaged with inverse-variance weights from ASD Uncertainty (eV); '
            'theoretical [ ] levels excluded from the QD fit but kept in excitations.'
        ),
        'ground_state': {
            'n':     gs_n,
            'l':     gs_l,
            'J':     gs_j,
            'term':  gs_term or None,
            'label': gs_label,
            'label_pretty': gs_label_pretty,
        },
        'n_start':              n_start,
        'n_max':                n_max,
        'num_states':           len(states),
        'num_theoretical':      theo_count,
        'excitations':          states
    }
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved Rydberg series to: {json_file}")

    print("\nFirst 15 states:")
    for s in states[:15]:
        j_str = f"J={s['J']}" if s.get('J') else 'QDT'
        disp = s.get('label_pretty') or s['label']
        print(f"  {disp:>12} (n={s['n']:2}): {s['energy_eV']:>9.5f} eV  "
              f"{s['wavelength_nm']:>9.3f} nm  {j_str:>8}  [{s['source']}]")

    print("\nLast 10 states:")
    for s in states[-10:]:
        disp = s.get('label_pretty') or s['label']
        print(f"  {disp:>12} (n={s['n']:2}): {s['energy_eV']:>9.5f} eV  "
              f"{s['wavelength_nm']:>9.3f} nm  [{s['source']}]")
    print(f"\nIonization energy     : {ionization_energy:.5f} eV")
    print(f"Ionization threshold  : 0.00000 eV")


if __name__ == "__main__":
    main()
