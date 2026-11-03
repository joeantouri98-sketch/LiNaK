# Contributing to LiNaK

Thanks for your interest in improving LiNaK.

## Reporting bugs / requesting features

Please open a [GitHub issue](https://github.com/joeantouri98-sketch/LiNaK/issues)
with:
- The element/species and command you ran.
- The full error output or a description of the unexpected result.
- Whether the issue is in a NIST-derived quantity, an ORCA-derived
  quantity, or a plotting/visualization bug; this determines which part
  of the pipeline needs to change.

## Contributing code

1. Fork the repository and create a branch from `main`.
2. Keep changes scoped to one module where possible; the pipeline is
   intentionally modular (`linak --pipeline rydberg -> transitions ->
   lifetimes`, etc., one module each under `src/linak/pipelines/`), and
   cross-cutting changes are harder to review. `ARCHITECTURE.md` is the map
   of the code; the rules below apply to every change.
3. Do not hand-edit generated JSON in the output folder's `data/`;
   regenerate it with the relevant pipeline instead.
4. If you add or change a physical formula, cite the source (paper, NIST
   page, etc.) in a comment and, if applicable, in `REFERENCES.bib`.
5. Run the linter, the test suite and the output check before opening a pull
   request:
   ```bash
   uv sync --dev
   uv run ruff check .
   uv run pytest
   uv run python tools/compare_outputs.py   # every changed output file vs HEAD
   ```
   A refactor must report "no differences"; a physics change lists exactly the
   files it meant to change, and the PR says which numbers moved and why.
6. Open a pull request describing what changed and why, and reference any
   related issue.

## Coding rules

- **Pipelines** expose `build_parser()` (with `prog="linak --pipeline NAME"`) and
  `main(argv) -> int`, are registered in `pipelines/runner.py::IMPLEMENTATIONS`, take
  every option through argparse (never `input()`), and do nothing at import time.
- **Errors:** bad arguments → `parser.error(...)`; missing or bad data →
  `raise MissingDataError(...)` / `PipelineError(...)`. Only `cli.py` turns exceptions
  into exit codes; library code never calls `sys.exit`.
- **Paths:** outputs through `Workspace.current()`, bundled data through
  `core.paths.reference_path()`; never paths relative to the current directory.
- **Files:** write with `core.fileio.write_json` / `atomic_write_text`; read with
  `encoding="utf-8"`; check user input used as a file name with `is_safe_name`.
- **Physics:** keep computations in pure functions (no files, prints or plots). A
  refactor must not move any number (`tools/compare_outputs.py` reports no
  differences); known defects are recorded in `PHYSICS_BUGS.md`, not patched in passing.
- **Reports:** HTML/CSS/JS live in `src/linak/report_templates/`, filled with `render()`.
- **One place for each thing:** import shared constants and tables from `core`; no dead
  or commented-out code; don't add rules to the ruff `ignore` list.
- **Tests ship with the change**, and write only to a temporary folder or `tests/results/`.

## Adding support for a new element or species

- Neutral alkalis follow the existing NIST + quantum-defect-theory (QDT)
  path; adding one requires NIST ASD level/line exports under
  `src/linak/resources/NIST Levels/` and `src/linak/resources/f_values/`.
- Ions and 3d transition metals currently use a reduced track (NIST
  plotting + optional ORCA spin-orbit structure only). Extending the full
  QDT/polarizability/BBR/tweezer/hyperfine/Feshbach chain to a new species
  class is a larger contribution. Please open an issue first to discuss
  scope before submitting a large pull request.

## Code of conduct

Be respectful and constructive. Physics disagreements should be resolved
by citing sources or providing a reproducible counter-example, not by
assertion.
