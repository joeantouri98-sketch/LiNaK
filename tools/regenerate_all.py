"""
Regenerate every pipeline output for the alkalis in the README order, one command, same arguments each time.

    python tools/regenerate_all.py                 # Li Na K Rb Cs Fr, all stages
    python tools/regenerate_all.py Li Na           # subset
    python tools/regenerate_all.py --dry-run       # print the commands only
    python tools/regenerate_all.py --skip-blackbody

Stages per element: rydberg (n-max 50) -> transitions -> lifetimes; then the batch stages
polarizability --all, blackbody --all --field 100, tweezer --all, hyperfine --all, feshbach --all,
compare_elements, each as `linak --pipeline <stage>` writing into the output folder (LINAK_PROJECT_ROOT, else
the per-user data folder). ORCA overlays (runorca) are NOT run: existing data/<El>.json / *_soc.json are used.
Blackbody needs --field 100 (without it the SFI anchors are 0).
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
    linak = [sys.executable, '-m', 'linak', '--pipeline']
    # child scripts print Greek letters (mu_I, alpha, ...): when stdout is redirected, Windows Python falls back to
    # cp1252 and crashes with UnicodeEncodeError, so force UTF-8 for them
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    cmds = []
    for el in a.elements:
        cmds += [[*linak, 'rydberg', el, '--n-max', a.n_max], [*linak, 'transitions', el],
                 [*linak, 'lifetimes', el]]
    batch = [[*linak, 'polarizability', '--all']]
    if not a.skip_blackbody:
        batch.append([*linak, 'blackbody', '--all', '--field', '100'])
    batch += [[*linak, 'tweezer', '--all'], [*linak, 'hyperfine', '--all'], [*linak, 'feshbach', '--all'],
              [*linak, 'compare_elements']]
    failed = []
    for c in cmds + batch:
        print('>>', ' '.join(c[4:]), flush=True)
        if a.dry_run:
            continue
        t0 = time.time()
        r = subprocess.run(c, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                           encoding='utf-8', errors='replace', env=env)
        print(f'   {"ok" if r.returncode == 0 else "FAILED"} ({time.time() - t0:.0f} s)', flush=True)
        if r.returncode:
            failed.append(' '.join(c[4:]))
            print(r.stderr[-800:])
    print('\nFAILED: ' + ('; '.join(failed) if failed else 'none'))
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
