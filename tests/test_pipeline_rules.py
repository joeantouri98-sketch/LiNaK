# -*- coding: utf-8 -*-
"""Rules every registered pipeline follows, checked over the source.

Pipelines take every option through argparse (a stdin prompt hangs the GUI
and CI), keep their state inside ``main`` (so they can run twice in one
process), and report HTML/CSS/JS lives in ``linak/report_templates`` rather
than in Python string builders.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPELINES = ROOT / "src" / "linak" / "pipelines"


def test_no_stdin_prompts_in_pipelines():
    """No registered pipeline may call input(); options go through argparse/flags.

    Scans every implementation module (``runorca`` lives in integrations/).
    """
    import importlib.util

    from linak.pipelines.runner import IMPLEMENTATIONS

    offenders = []
    for module in IMPLEMENTATIONS.values():
        origin = Path(importlib.util.find_spec(module).origin)
        # A package pipeline (plotinteractive) is every module in its folder.
        sources = origin.parent.rglob("*.py") if origin.name == "__init__.py" else [origin]
        for source in sources:
            text = source.read_text(encoding="utf-8", errors="replace")
            if re.search(r"(?<!\.)\binput\(", text):
                offenders.append(f"{module}:{source.name}")
    assert offenders == [], f"pipelines still prompting over stdin: {offenders}"


def test_no_inline_style_script_literals_in_pipelines():
    """Inline HTML builders move into templates; pipelines keep computation."""
    offenders = []
    for source in PIPELINES.rglob("*.py"):
        text = source.read_text(encoding="utf-8", errors="replace")
        if "'<style>'" in text or '"<style>"' in text or "<script>" in text:
            offenders.append(source.name)
    assert offenders == [], f"pipelines still embedding styles/scripts: {offenders}"


def test_pipelines_keep_their_state_out_of_module_globals():
    """No ``global`` statement: a run's state lives in its functions, so the GUI
    worker or a test can run a pipeline twice in one process without a reload."""
    import ast

    offenders = [source.name for source in sorted(PIPELINES.rglob("*.py"))
                 if any(isinstance(node, ast.Global)
                        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))))]
    assert offenders == [], f"pipelines with module-level run state: {offenders}"
