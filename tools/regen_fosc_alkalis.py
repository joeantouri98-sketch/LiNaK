# -*- coding: utf-8 -*-
"""Regenerate lifetimes -> polarizability -> blackbody for all alkalis."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ELEMENTS = ["Li", "Na", "K", "Rb", "Cs", "Fr"]


def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def main():
    py = sys.executable
    for el in ELEMENTS:
        run([py, "lifetimes.py", el])
        run([py, "polarizability.py", el])
        # Na keeps field=100 for mix/SFI continuity with anchors; others free-space PI
        if el == "Na":
            run([py, "blackbody.py", el, "--field", "100"])
        else:
            run([py, "blackbody.py", el])
        run([py, "tools/fosc_audit.py", el])
    print("\nAll alkalis regenerated.")


if __name__ == "__main__":
    main()
