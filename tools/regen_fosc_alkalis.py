# -*- coding: utf-8 -*-
"""Regenerate lifetimes -> polarizability -> blackbody for all alkalis.

Writes into the workspace in effect (``QALI_PROJECT_ROOT``, else the
per-user data folder).
"""
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


def main() -> int:
    qali = [sys.executable, "-m", "qali", "--pipeline"]
    for el in ELEMENTS:
        run([*qali, "lifetimes", el])
        run([*qali, "polarizability", el])
        # Na keeps field=100 for mix/SFI continuity with anchors; others free-space PI
        if el == "Na":
            run([*qali, "blackbody", el, "--field", "100"])
        else:
            run([*qali, "blackbody", el])
        run([sys.executable, "tools/fosc_audit.py", el])
    print("\nAll alkalis regenerated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
