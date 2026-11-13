"""Rydberg-series pipeline: NIST level IO + CLI (see qali.physics.rydberg).

The label formatting and quantum-defect/term computation live in the pure
``qali.physics.rydberg`` engine; this module adds the NIST file readers, the
command line and the JSON output.
"""

import argparse
import csv
import io
import os


from qali.core.errors import InvalidSpeciesError, MissingDataError
from qali.core.fileio import write_json
from qali.core.paths import bundled_file, reference_path
from qali.core.workspace import Workspace
from qali.physics.rydberg import (
    build_nist_lookup,
    build_nist_term_excitations,
    compute_quantum_defects,
    fit_ritz_defects,
    generate_rydberg_series,
    screen_nist_levels,
    metal_state_label,
    metal_state_label_pretty,
    parse_config,
    parse_j,
    state_label,
    state_label_pretty,
    rydberg_constant_eV,
    trailing_nl_from_config,
    unwrap_asd_field,
)

def find_nist_levels_file(element, data_dir):
    """Prefer <element>.csv (full ASD export), fall back to .txt."""
    for ext in ('.csv', '.txt'):
        path = bundled_file("NIST Levels", data_dir, f"{element}{ext}")
        if os.path.exists(path):
            return path
    return None


def _read_nist_export(filepath):
    """Text of a NIST ASD export: UTF-8 (current exports), else legacy latin-1."""
    with open(filepath, 'rb') as f:
        data = f.read()
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        return data.decode('latin-1')


def parse_nist_file(filepath):
    """
    Parse a NIST energy-levels export (UTF-8, or legacy latin-1).

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

    raw = _read_nist_export(filepath)

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

    raw = _read_nist_export(filepath)

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qali --pipeline rydberg",
        description='Build Rydberg series from NIST levels + quantum defect theory')
    parser.add_argument(
        'element',
        help='Species id: symbol (Na) or ion stem (Mg_c1) when NIST file exists')
    parser.add_argument(
        '--n-max', type=int, default=50,
        help='Maximum principal quantum number for the Rydberg series (default: 50)')
    parser.add_argument(
        '--nist-dir', default=None,
        help='NIST levels directory (default: the bundled "NIST Levels")')
    parser.add_argument(
        '--nist-only', action='store_true',
        help='Write NIST ASD term levels only (no QDT). Auto-used for 3d metals.')
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    from qali.core.species import resolve_species

    nist_dir = args.nist_dir or str(reference_path("NIST Levels"))

    raw = args.element.strip()
    try:
        sp = resolve_species(raw)
    except ValueError as exc:
        parser.error(str(exc))
    element = sp["species_id"]

    txt_file = find_nist_levels_file(element, data_dir=nist_dir)
    if txt_file is None:
        hint = ""
        if sp["kind"] == "ion":
            hint = (f" Ion {element} ({sp['label']}): add a NIST ASD export named "
                    f"'{element}.csv' (or .txt) before building Rydberg data; "
                    f"do not invent level/f-value tables.")
        raise MissingDataError(f"no '{element}.csv' or '{element}.txt' in {nist_dir}.{hint}")

    # 3d metals: term/config NIST overlay only — do not invent QDT series.
    from qali.integrations.orca_templates import TRANSITION_METAL_3D, get_element_class, ElementClass
    from qali.core.elements import ALKALI_SYMBOLS
    other_neutral = sp["kind"] == "neutral" and sp["symbol"] not in ALKALI_SYMBOLS and sp["symbol"] != "H"
    nist_only = bool(args.nist_only) or other_neutral or (
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
        raise MissingDataError(f"no ionization energy (Limit line) in {txt_file}")

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
            raise MissingDataError(f"no NIST term levels could be built from {txt_file}")
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

        data_dir = str(Workspace.current().data_json_dir)
        os.makedirs(data_dir, exist_ok=True)
        json_file = f'{data_dir}/{element}_rydberg.json'
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
        write_json(json_file, data)
        print(f"Saved NIST term levels to: {json_file}")
        print(f"\nIonization energy     : {ionization_energy:.5f} eV")
        print("Ionization threshold  : 0.00000 eV")
        return 0

    # Residual ionic charge seen by a Rydberg electron (Na+ -> Na2+ => Z=2).
    # The electron sees charge (species charge + 1): neutral 1, singly charged ion 2, ...; an
    # anion has no Rydberg series (the outer electron is not bound by a Coulomb tail).
    z_core = int(sp["charge"]) + 1
    if z_core < 1:
        raise InvalidSpeciesError(
            f"{element} (charge {sp['charge']:+d}) has no Rydberg series "
            f"(residual ionic charge {z_core}). Use --nist-only for its tabulated levels.")
    ry_eff = rydberg_constant_eV(sp["symbol"]) * (z_core ** 2)
    print(f"Rydberg constant (reduced-mass, {sp['symbol']}): R_M = {rydberg_constant_eV(sp['symbol']):.6f} eV"
          f"{'' if z_core == 1 else f'; Z={z_core} scale: Ry_eff = {ry_eff:.4f} eV'}")

    replaced_levels = []
    if sp["kind"] == "neutral":
        levels, replaced_levels = screen_nist_levels(ionization_energy, levels, ry_eff=ry_eff)
        if replaced_levels:
            print(f"Replaced by the QDT model: {len(replaced_levels)} NIST levels "
                  "(see nist_levels_replaced in the JSON)")

    print("\nComputing quantum defects (averaged over all n,J per orbital):")
    quantum_defects = compute_quantum_defects(
        ionization_energy, levels, ry_eff=ry_eff)

    ritz = fit_ritz_defects(ionization_energy, levels, ry_eff=ry_eff)
    for _l, _p in ritz.items():
        print(f"  Ritz {_l}: delta(n) = {_p['delta0']:.5f} + {_p['delta2']:+.4f}/(n-delta)^2  "
              f"({_p['n_points']} measured n={_p['n_min']}..{_p['n_max']}, rms {_p['rms']:.1e})")

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

    n_max = int(args.n_max)

    if n_max < n_start:
        parser.error(f"--n-max ({n_max}) must be >= ground-state n ({n_start})")

    print(f"\nGenerating Rydberg series for {element} (n={n_start}..{n_max})...")
    states = generate_rydberg_series(ionization_energy, quantum_defects,
                                     nist_lookup, n_start, n_max, ry_eff=ry_eff, ritz=ritz)

    if not states:
        raise MissingDataError(f"no Rydberg states could be generated for {element}")

    nist_count  = sum(1 for s in states if s['source'] == 'NIST')
    model_count = sum(1 for s in states if s['source'] == 'QDT model')
    theo_count  = sum(1 for s in states if s.get('theoretical'))
    print(f"\nGenerated {len(states)} Rydberg states")
    print(f"  {nist_count} from NIST (J-resolved),  {model_count} from QDT model"
          f"  ({theo_count} marked theoretical [ ])")
    print(f"Energy range: {min(s['energy_eV'] for s in states):.4f} to "
          f"{max(s['energy_eV'] for s in states):.4f} eV")

    data_dir = str(Workspace.current().data_json_dir)
    os.makedirs(data_dir, exist_ok=True)

    json_file = f'{data_dir}/{element}_rydberg.json'
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
        'quantum_defects_ritz': ritz,
        'qd_notes': (
            'Averaged with inverse-variance weights from ASD Uncertainty (eV); '
            'theoretical [ ] levels excluded from the QD fit but kept in excitations.'
        ),
        'nist_levels_replaced': [
            dict(n=lv['n'], l=lv['l'], J=lv['J'], energy_eV=lv['energy_eV'],
                 delta=round(lv['delta'], 4), reason=lv['reason'])
            for lv in replaced_levels
        ],
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
    write_json(json_file, data)
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
    print("Ionization threshold  : 0.00000 eV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
