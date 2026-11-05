#!/usr/bin/env python3
"""Remove the 3D Grotrian plot (<El>_3d_visualization.html) from LiNaK completely.

Run from the LiNaK root (the folder with pyproject.toml):

    python remove_3d_visualization.py --dry-run     # show what would change
    python remove_3d_visualization.py               # do it

What it removes
  * plotinteractive/grotrian3d.py and its three templates (plotinteractive_levels4.html/js/css)
  * every packaged <El>_3d_visualization.html
  * the plot_3d import/call and every "plot 4" mention in plotinteractive/__init__.py
    (--plots now runs 1..3; 3 is the new default and "all")
  * the "4 - All four, incl. 3D view" choice in the GUI stage list (default becomes 3)
  * the _disp3d helper in plotinteractive/common.py (only the 3D plot used it)
  * the grotrian3d lines in tests/test_fork_fixes.py
  * the 3D lines in "Pipeline Overview.html"

The separate Orbital 3D pipeline (orbital3d.py, *_orbital3d.html) is NOT touched.

Every edit is a search-and-replace on exact text. If your copy differs, that edit is reported as
"not found" and skipped, and the final sweep lists any leftover mention so you can fix it by hand.
The script is safe to run twice.
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

ROOT = Path.cwd()
DRY = False
changes: list[str] = []


def say(msg: str) -> None:
    print(msg)
    changes.append(msg)


def rm(path: Path) -> None:
    if path.exists():
        say(f"  delete   {path.relative_to(ROOT)}")
        if not DRY:
            path.unlink()


def edit(path: Path, subs: list[tuple[str, str, str]], *, regex: bool = False) -> None:
    """subs: (label, old, new). Reports each one as applied / already done / not found."""
    if not path.exists():
        print(f"  skip     {path.relative_to(ROOT)} (file not found)")
        return
    text = path.read_text(encoding="utf-8")
    orig = text
    for label, old, new in subs:
        if regex:
            text2, n = re.subn(old, new, text, flags=re.S)
        else:
            n = text.count(old)
            text2 = text.replace(old, new)
        if n:
            say(f"  edit     {path.relative_to(ROOT)}: {label}")
            text = text2
        else:
            print(f"  --       {path.relative_to(ROOT)}: {label} (nothing to change)")
    if text != orig and not DRY:
        path.write_text(text, encoding="utf-8")


def remove_function(path: Path, name: str) -> None:
    """Delete a top-level function if nothing else in src/ or tests/ refers to it."""
    if not path.exists():
        return
    others = [p for p in list((ROOT / "src").rglob("*.py")) + list((ROOT / "tests").rglob("*.py"))
              if p != path and re.search(rf"\b{name}\b", p.read_text(encoding="utf-8", errors="ignore"))
              and p.name != "grotrian3d.py"]
    if others:
        print(f"  keep     {name}: still used by {', '.join(str(o.relative_to(ROOT)) for o in others)}")
        return
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            start = (node.decorator_list[0].lineno if node.decorator_list else node.lineno) - 1
            end = node.end_lineno
            while end < len(lines) and lines[end].strip() == "":
                end += 1                                   # swallow the blank lines after it
            say(f"  edit     {path.relative_to(ROOT)}: remove {name}()")
            if not DRY:
                path.write_text("".join(lines[:start] + lines[end:]), encoding="utf-8")
            return


def main() -> int:
    global ROOT, DRY
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="only report what would change")
    ap.add_argument("--root", default=".", help="LiNaK root (default: current folder)")
    args = ap.parse_args()
    DRY = args.dry_run
    ROOT = Path(args.root).resolve()
    if not (ROOT / "pyproject.toml").exists() or not (ROOT / "src" / "linak").is_dir():
        print("Run this from the LiNaK root (pyproject.toml and src/linak must be there).")
        return 1
    print(("DRY RUN - nothing is written\n" if DRY else "") + f"LiNaK root: {ROOT}\n")

    pi = ROOT / "src/linak/pipelines/plotinteractive"
    print("Files")
    rm(pi / "grotrian3d.py")
    for pyc in (pi / "__pycache__").glob("grotrian3d*.pyc"):
        rm(pyc)
    for ext in ("html", "js", "css"):
        rm(ROOT / f"src/linak/report_templates/plotinteractive_levels4.{ext}")
    for f in sorted((ROOT / "src/linak/resources").rglob("*_3d_visualization.html")):
        rm(f)

    print("\nplotinteractive/__init__.py")
    edit(pi / "__init__.py", [
        ("import of plot_3d", "from .grotrian3d import plot_3d\n", ""),
        ("plot_3d call", "    if cli.plots >= 4:\n        plot_3d(inputs, output_dir)\n", ""),
        ("docstring: --plots", "3=+ee_grotrian, 4=+3d). Example", "3=+ee_grotrian). Example"),
        ("docstring: module list", ", ``grotrian3d``", ""),
        ("accepted 'all' spellings", "('all', '*', '1-4', '1,2,3,4')", "('all', '*', '1-3', '1,2,3')"),
        ("'all' returns 3", "        return 4\n    wanted", "        return 3\n    wanted"),
        ("range check", "any(n < 1 or n > 4 for n in wanted)", "any(n < 1 or n > 3 for n in wanted)"),
        ("messages 1..4", "1..4", "1..3"),
        ("help text", "3=+ee_grotrian, 4=+3d (default: all). ", "3=+ee_grotrian (default: all). "),
        ("skip notice", "    if cli.plots < 4:", "    if cli.plots < 3:"),
        ("skip notice range", "range(cli.plots + 1, 5)", "range(cli.plots + 1, 4)"),
        ("summary threshold", "    if plots_max <= 3:", "    if plots_max <= 2:"),
        ("summary skipped notes",
         '{1: "   (plots 2-4 skipped)", 2: "   (plots 3-4 skipped)",\n               3: "   (plot 4 skipped)"}[plots_max]',
         '{1: "   (plots 2-3 skipped)", 2: "   (plot 3 skipped)"}[plots_max]'),
        ("summary file list",
         '    print(f"   4. {element}_3d_visualization.html  - 3D view (ROTATE ME!)")\n', ""),
    ])

    edit(pi / "__init__.py", [
        ("docstring: 3D view line", r"\n[ \t]*- 3D view option[^\n]*", ""),
        ("summary: 3D view hint", r'\n[ \t]*print\("   - 3D VIEW[^\n]*', ""),
    ], regex=True)

    print("\nGUI stage list")
    edit(ROOT / "src/linak/gui/stages.py", [
        ("remove choice 4", r'\n\s*\("4", "All four, incl\. 3D view \(default\)"\),', ""),
        ("choice 3 becomes the default", r'(\("3", "Grotrian \+ Spectrum \+ Excited\\u2194excited Grotrian)"\)', r'\1 (default)")'),
        ("default 4 -> 3", r'(choices=\[.*?\],\s*default=)"4"', r'\1"3"'),
        ("help text", "of generating all four.", "of generating all three."),
    ], regex=True)

    print("\nplotinteractive/common.py")
    remove_function(pi / "common.py", "_disp3d")
    common = pi / "common.py"
    if common.exists():
        txt = common.read_text(encoding="utf-8")
        # label_to_unicode was only imported for _disp3d: drop it from the import when nothing else uses it
        if len(re.findall(r"\blabel_to_unicode\b", txt)) == 1 and re.search(r"import[^\n]*\blabel_to_unicode\b", txt):
            edit(common, [("unused import label_to_unicode", r",\s*label_to_unicode\b|\blabel_to_unicode,\s*", "")], regex=True)

    print("\nTests")
    edit(ROOT / "tests/test_fork_fixes.py", [
        ("import", "common, ee_grotrian, grotrian3d", "common, ee_grotrian"),
        ("P19 assert", r"\n[ \t]*assert grotrian3d\._REGION_COLORS4 is common\.REGION_COLORS[^\n]*", ""),
    ], regex=True)

    print("\nPipeline Overview.html")
    edit(ROOT / "src/linak/resources/HTMLs/Pipeline Overview.html", [
        ("list entry", r",\s*<code>\*_3d_visualization\.html</code>", ""),
        ("table row", r"\n[ \t]*<tr><td><code>\w+_3d_visualization\.html</code></td>.*?</tr>", ""),
        ("output list", r"\s*·\s*\w+_3d_visualization\.html", ""),
    ], regex=True)
    edit(ROOT / "src/linak/resources/HTMLs/Pipeline Overview.html", [
        ("suite list", "Grotrian / 3D / spectra / comparison HTML suite", "Grotrian / spectra / comparison HTML suite"),
        ("section title", "plotinteractive.py: Grotrian, spectrum, 3D</h3>", "plotinteractive.py: Grotrian, spectrum</h3>"),
        ("--plots text", "3=+ee_grotrian, 4=+3d; default all", "3=+ee_grotrian; default all"),
        ("diagram box", "Grotrian / 3D / compare", "Grotrian / compare"),
        ("--extra note", "Spectrum / ee Grotrian / 3D are unchanged", "Spectrum / ee Grotrian are unchanged"),
    ])

    # ---- sweep -----------------------------------------------------------------------------------
    print("\nLeftover sweep (anything listed here still mentions the 3D plot)")
    pat = re.compile(r"3d_visualization|grotrian3d|plot_3d|levels4|_disp3d|3D view|_REGION_COLORS4|Grotrian / 3D|spectrum, 3D|4=\+3d|ee Grotrian / 3D", re.I)
    left = []
    for base in ("src", "tests", "docs"):
        for f in (ROOT / base).rglob("*") if (ROOT / base).exists() else []:
            if ".egg-info" in str(f) or "__pycache__" in str(f):
                continue                       # build metadata, regenerated by uv sync / pip install
            if f.is_file() and not (DRY and (f.name == "grotrian3d.py" or "levels4" in f.name)) and f.suffix in {".py", ".html", ".js", ".css", ".md", ".json", ".toml", ".txt"}:
                try:
                    for i, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                        if pat.search(line) and "orbital3d" not in line.lower():
                            left.append(f"  {f.relative_to(ROOT)}:{i}: {line.strip()[:110]}")
                except OSError:
                    pass
    for f in ROOT.glob("*.md"):
        for i, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if pat.search(line):
                left.append(f"  {f.name}:{i}: {line.strip()[:110]}")
    if DRY:
        print("  (dry run: files are not changed yet, so mentions in files that would be edited are still listed)")
    print("\n".join(left) if left else "  none")
    print(f"\n{'Would make' if DRY else 'Made'} {len(changes)} change(s). Test with: uv run pytest tests/test_fork_fixes.py -q")
    return 0


if __name__ == "__main__":
    sys.exit(main())
