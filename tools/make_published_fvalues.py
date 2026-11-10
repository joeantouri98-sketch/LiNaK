# -*- coding: utf-8 -*-
"""
Write line-level PUBLISHED rows of f_values/<El>_literature.json from published matrix elements.

For an element ARC does not model (Fr), where tools/make_arc_fvalues.py cannot run. Each row of
f_values/<El>_published_anchors.json gives |D| (a.u.) for one fine-structure line; its absorption
f = (2/3) dE D^2 / (2 J_lower + 1), with dE from the element's Rydberg JSON. The lower level is the
one lower in energy. Rows whose (lower, upper) already have a NIST f are left out.

    python tools/make_published_fvalues.py Fr
"""
from __future__ import annotations

import argparse
import json
import re

from linak.core.paths import reference_path
from linak.physics.polarizability_core import load_nist_fvalues

HARTREE_EV = 27.211386245988


def level_split(label):
    """('8s', '1/2') for '8s', ('7p', '3/2') for '7p3/2'."""
    m = re.fullmatch(r'(\d+[spdfg])(\d/2)?', label)
    return m.group(1), m.group(2) or '1/2'


def energies(rydberg):
    table = {}
    for lv in rydberg['excitations']:
        table[(f"{lv['n']}{lv['l']}", lv['J'] or '1/2')] = lv['energy_eV']
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('element')
    a = ap.parse_args()
    el = a.element
    anchors = json.loads(reference_path('f_values', f'{el}_published_anchors.json').read_text('utf-8'))
    path = reference_path('f_values', f'{el}_literature.json')
    out = json.loads(path.read_text('utf-8'))
    E = energies(json.loads(reference_path('data_json', f'{el}_rydberg.json').read_text('utf-8')))
    nist = load_nist_fvalues(el, verbose=False)['fosc']
    kept = [t for t in out.get('transitions', []) if not str(t.get('ref', '')).startswith('PUBLISHED')]
    rows = []
    for t in anchors['transitions']:
        a_key, b_key = level_split(t['lower']), level_split(t['upper'])
        lo, up = (a_key, b_key) if E[a_key] < E[b_key] else (b_key, a_key)
        if (lo[0], up[0]) in nist or (up[0], lo[0]) in nist:
            continue
        num, den = lo[1].split('/')
        two_j = 2 * int(num) / int(den)
        f = 2.0 / 3.0 * (E[up] - E[lo]) / HARTREE_EV * t['D'] ** 2 / (two_j + 1)
        rows.append(dict(lower=lo[0], lower_J=lo[1], upper=up[0], upper_J=up[1],
                         f_abs=float(f'{f:.5g}'), acc='B', ref=f"PUBLISHED: {t['ref']}"))
    out['transitions'] = kept + rows
    out['note'] = (f"{el}: curated NIST fit exclusions, and PUBLISHED line-level rows from "
                   f"{el}_published_anchors.json, used only where NIST lists no f.")
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n', 'utf-8')
    print(f'{el}: {len(rows)} PUBLISHED rows')
    for r in rows:
        print(f"  {r['lower']}{r['lower_J']} -> {r['upper']}{r['upper_J']}: f = {r['f_abs']}")


if __name__ == '__main__':
    main()
