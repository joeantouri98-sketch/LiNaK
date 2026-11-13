"""
Regenerate every pipeline output for the alkalis in the README order, one command, same arguments each time.

    python tools/regenerate_all.py                 # Li Na K Rb Cs Fr, all stages
    python tools/regenerate_all.py Li Na           # subset
    python tools/regenerate_all.py --dry-run       # print the commands only
    python tools/regenerate_all.py --skip-blackbody
    python tools/regenerate_all.py --refresh-bundled   # update the example data shipped in src/qali/resources

Stages per element: rydberg (n-max 50) -> transitions -> lifetimes; then the batch stages
polarizability --all, blackbody --all --field 100, tweezer --all, hyperfine --all, feshbach --all,
compare_elements, each as `qali --pipeline <stage>` writing into the output folder (QALI_PROJECT_ROOT, else
the per-user data folder). ORCA overlays (runorca) are NOT run: existing data/<El>.json / *_soc.json are used.
Blackbody needs --field 100 (without it the SFI anchors are 0).

--refresh-bundled is the safe way to update the bundled example data: it runs everything in a fresh
temporary folder seeded from the bundle (so neither old input copies in your own output folder nor your
Preferences edits leak in), then copies back only what the stages wrote. Input files (constants, Feshbach
parameters, NIST tables) are never copied back. Nothing is copied if any stage fails. Never copy your own
output folder over src/qali/resources/data_json by hand: it holds older copies of the input files.
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ELEMENTS = ['Li', 'Na', 'K', 'Rb', 'Cs', 'Fr']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('elements', nargs='*', default=ELEMENTS)
    ap.add_argument('--n-max', default='50')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--skip-blackbody', action='store_true')
    ap.add_argument('--refresh-bundled', action='store_true',
                    help='run in a fresh temporary folder and copy the outputs into the bundled resources')
    a = ap.parse_args()
    if a.refresh_bundled and a.dry_run:
        ap.error('--refresh-bundled and --dry-run do not go together')
    qali = [sys.executable, '-m', 'qali', '--pipeline']
    # child scripts print Greek letters (mu_I, alpha, ...): when stdout is redirected, Windows Python falls back to
    # cp1252 and crashes with UnicodeEncodeError, so force UTF-8 for them
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    scratch, before = None, {}
    if a.refresh_bundled:
        scratch, before = _fresh_folder()
        env['QALI_PROJECT_ROOT'] = str(scratch)
        print(f'Running in a fresh folder: {scratch}', flush=True)
    cmds = []
    for el in a.elements:
        cmds += [[*qali, 'rydberg', el, '--n-max', a.n_max], [*qali, 'transitions', el],
                 [*qali, 'lifetimes', el]]
    batch = [[*qali, 'polarizability', '--all']]
    if not a.skip_blackbody:
        batch.append([*qali, 'blackbody', '--all', '--field', '100'])
    batch += [[*qali, 'tweezer', '--all'], [*qali, 'hyperfine', '--all'], [*qali, 'feshbach', '--all'],
              [*qali, 'compare_elements']]
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
    if scratch is not None:
        if failed:
            print(f'Bundled data left unchanged because a stage failed; the run is in {scratch}')
        else:
            _copy_back(scratch, before)
            shutil.rmtree(scratch, ignore_errors=True)
    sys.exit(1 if failed else 0)


def _bundle():
    from qali.core.paths import resource_root
    return resource_root()


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fresh_folder():
    """A temporary output folder holding a copy of the bundled inputs, and their digests."""
    scratch = Path(tempfile.mkdtemp(prefix='qali_regen_'))
    data = scratch / 'data'
    shutil.copytree(_bundle() / 'data_json', data)
    return scratch, {p.relative_to(data): _digest(p) for p in data.rglob('*') if p.is_file()}


def _copy_back(scratch, before):
    """Copy what the stages wrote (new or changed files) into the bundled data_json and plots."""
    bundle, copied = _bundle(), 0
    data = scratch / 'data'
    for path in sorted(data.rglob('*')):
        rel = path.relative_to(data)
        if path.is_file() and before.get(rel) != _digest(path):
            target = bundle / 'data_json' / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied += 1
    plots = scratch / 'plots'
    for path in sorted(plots.rglob('*')) if plots.is_dir() else []:
        if path.is_file():
            target = bundle / 'plots' / path.relative_to(plots)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied += 1
    print(f'Copied {copied} output files into {bundle}. Input files were left as they are.')


if __name__ == '__main__':
    main()
