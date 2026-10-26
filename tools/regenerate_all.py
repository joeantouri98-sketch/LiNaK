#!/usr/bin/env python3
"""
Regenerate every pipeline output for the alkalis in the README order, one command, same arguments each time.

    python tools/regenerate_all.py                 # Li Na K Rb Cs Fr, all stages
    python tools/regenerate_all.py Li Na           # subset
    python tools/regenerate_all.py --dry-run       # print the commands only
    python tools/regenerate_all.py --skip-blackbody

Stages per element: rydberg (n-max 50) -> transitions -> lifetimes; then the batch stages
polarizability --all, blackbody --all --field 100, tweezer --all, hyperfine --all, feshbach --all,
compare_elements. ORCA overlays (runorca.py) are NOT run: existing data_json/<El>.json / *_soc.json are used.
Blackbody needs --field 100 (without it the SFI anchors are 0). Run from the repository root.
"""
import argparse
import os
import subprocess
import sys
import time

ELEMENTS = ['Li', 'Na', 'K', 'Rb', 'Cs', 'Fr']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('elements', nargs='*', default=ELEMENTS)
    ap.add_argument('--n-max', default='50')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--skip-blackbody', action='store_true')
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    py = sys.executable
    # child scripts print Greek letters (mu_I, alpha, ...): when stdout is redirected, Windows Python falls back to
    # cp1252 and crashes with UnicodeEncodeError, so force UTF-8 for them
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    cmds = []
    for el in a.elements:
        cmds += [[py, 'rydberg.py', el, '--n-max', a.n_max], [py, 'transitions.py', el], [py, 'lifetimes.py', el]]
    batch = [[py, 'polarizability.py', '--all']]
    if not a.skip_blackbody:
        batch.append([py, 'blackbody.py', '--all', '--field', '100'])
    batch += [[py, 'tweezer.py', '--all'], [py, 'hyperfine.py', '--all'], [py, 'feshbach.py', '--all'],
              [py, 'compare_elements.py']]
    failed = []
    for c in cmds + batch:
        print('>>', ' '.join(c[1:]), flush=True)
        if a.dry_run:
            continue
        t0 = time.time()
        r = subprocess.run(c, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                           encoding='utf-8', errors='replace', env=env)
        print(f'   {"ok" if r.returncode == 0 else "FAILED"} ({time.time() - t0:.0f} s)', flush=True)
        if r.returncode:
            failed.append(' '.join(c[1:]))
            print(r.stderr[-800:])
    print('\nFAILED: ' + ('; '.join(failed) if failed else 'none'))
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
