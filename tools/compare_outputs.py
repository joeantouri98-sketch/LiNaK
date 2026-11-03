# -*- coding: utf-8 -*-
"""Check that a change did not move any pipeline output.

Runs the same pipeline chain twice, once with the code of a git revision
(default ``HEAD``) and once with the working tree, each into its own fresh
output folder, then compares every file the runs wrote:

* JSON must be byte-identical (these are the physics results);
* HTML is compared with Plotly's random per-run div ids normalised;
* exit codes must match.

Usage (from the repository root, in the project environment)::

    python tools/compare_outputs.py                  # HEAD vs working tree
    python tools/compare_outputs.py --before main    # main vs working tree
    python tools/compare_outputs.py --only rydberg transitions
    python tools/compare_outputs.py --keep           # keep the output folders

Exit status 0 means no differences; 1 lists what differed (the output
folders and per-run logs are then kept for inspection). A change that is
meant to move a number shows up here, and the PR should say so.
"""
from __future__ import annotations

import argparse
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEP_TIMEOUT_S = 900

# The chain, in dependency order (light settings: about a minute per side).
CHAIN = [
    ["rydberg", "Na"],
    ["transitions", "Na"],
    ["lifetimes", "Na"],
    ["polarizability", "Na", "--n-grid", "400"],
    ["blackbody", "Na", "--no-rydberg-shifts", "--no-mix"],
    ["tweezer", "Na", "--n-grid", "300"],
    ["hyperfine", "Na"],
    ["feshbach", "Na"],
    ["compare_elements", "--elements", "Li", "Na", "K"],
    ["scattering_rate", "--element", "Na"],
    ["spectra", "Na"],
    ["orbital3d", "Na", "--n-max", "4", "--grid", "24"],
    ["plotinteractive", "Na"],
    ["rydberg", "Na_c1"],
]

_PLOTLY_ID = re.compile(rb"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def export_revision(revision: str, dest: Path) -> Path:
    """Extract the tree of *revision* into *dest* (older trees keep data outside src/)."""
    archive = subprocess.run(
        ["git", "archive", revision], cwd=ROOT, capture_output=True, check=True
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(dest, filter="data")
    return dest


def run_chain(tree: Path, out: Path, steps: list[list[str]]) -> list[int | str]:
    """Run *steps* with the code in *tree* into the output folder *out*.

    *out* is made checkout-shaped, so every revision's code writes into
    ``out/.linak`` (older revisions only honoured ``LINAK_PROJECT_ROOT`` for
    a checkout).
    """
    (out / "src" / "linak").mkdir(parents=True)
    (out / "pyproject.toml").write_text("# output folder for compare_outputs\n", encoding="utf-8")
    env = dict(os.environ, LINAK_PROJECT_ROOT=str(out), PYTHONPATH=str(tree / "src"),
               PYTHONHASHSEED="0")
    codes: list[int | str] = []
    with (out / "chain.log").open("w", encoding="utf-8") as log:
        for step in steps:
            log.write(f"\n### linak --pipeline {' '.join(step)}\n")
            log.flush()
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "linak", "--pipeline", *step],
                    cwd=out, env=env, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, timeout=STEP_TIMEOUT_S,
                )
                codes.append(result.returncode)
            except subprocess.TimeoutExpired:
                codes.append("timeout")
    return codes


def _outputs(folder: Path) -> dict[str, Path]:
    runtime = folder / ".linak"
    return {
        path.relative_to(runtime).as_posix(): path   # same names on every OS
        for sub in ("data", "plots")
        if (runtime / sub).is_dir()
        for path in sorted((runtime / sub).rglob("*"))
        if path.is_file() and path.suffix in (".json", ".html")
    }


def compare(before: Path, after: Path) -> list[str]:
    """Human-readable differences between two output folders."""
    old, new = _outputs(before), _outputs(after)
    problems = [f"missing after: {name}" for name in sorted(old.keys() - new.keys())]
    problems += [f"new after:     {name}" for name in sorted(new.keys() - old.keys())]
    for name in sorted(old.keys() & new.keys()):
        a, b = old[name].read_bytes(), new[name].read_bytes()
        if name.endswith(".html"):
            a, b = _PLOTLY_ID.sub(b"ID", a), _PLOTLY_ID.sub(b"ID", b)
        if a != b:
            problems.append(f"changed:       {name}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--before", default="HEAD", help="git revision to compare against")
    parser.add_argument("--only", nargs="+", metavar="PIPELINE",
                        help="run only these pipelines (their inputs come from the bundle)")
    parser.add_argument("--keep", action="store_true", help="keep the output folders")
    args = parser.parse_args(argv)

    steps = [step for step in CHAIN if not args.only or step[0] in args.only]
    work = Path(tempfile.mkdtemp(prefix="linak_compare_"))
    before_tree = export_revision(args.before, work / "before_tree")
    with ThreadPoolExecutor(2) as pool:
        old = pool.submit(run_chain, before_tree, work / "before", steps)
        new = pool.submit(run_chain, ROOT, work / "after", steps)
        old_codes, new_codes = old.result(), new.result()

    problems = [
        f"exit code:     linak --pipeline {' '.join(step)}: {a} -> {b}"
        for step, a, b in zip(steps, old_codes, new_codes) if a != b
    ]
    problems += compare(work / "before", work / "after")

    print(f"{len(steps)} runs, {len(_outputs(work / 'after'))} output files "
          f"({args.before} vs working tree)")
    for line in problems:
        print("  " + line)
    print(f"{len(problems)} difference(s)" if problems else "no differences")
    if args.keep or problems:
        print(f"outputs and chain.log files kept in {work}")
    else:
        shutil.rmtree(work, ignore_errors=True)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
