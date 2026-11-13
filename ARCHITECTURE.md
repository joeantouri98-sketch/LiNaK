# How Qali is put together

A map for people changing the code. Users want the [README](README.md); the coding rules
and workflow are in [CONTRIBUTING.md](CONTRIBUTING.md); known physics defects are in
[PHYSICS_BUGS.md](PHYSICS_BUGS.md).

## The idea in one paragraph

Qali is a chain of **pipelines**. Each one is a command (`qali --pipeline <name> ...`)
that reads JSON written by earlier pipelines (plus bundled reference data), computes
something, and writes JSON and interactive HTML reports into the **output folder**. The
desktop GUI is a front end that runs those same commands as subprocesses and displays
what they wrote. Nothing is shared between pipelines except files.

## Where things live

| Path | What it is | Rule of thumb |
|---|---|---|
| `src/qali/core/` | constants, elements, species ids, spectroscopy tables, paths, the workspace, safe file writes, errors | no plotting, no argparse |
| `src/qali/physics/` | physics engines (`polarizability_core`, `qdt_radial`, `rydberg`, `blackbody_core`, `feshbach`, `hyperfine_core`, `laser_cooling`, `lifetimes_core`, `orbital_radial`, `spectrum_core`, `transitions_core`, `tweezer_core`) | data in, results out: no files, no CLI |
| `src/qali/pipelines/` | one module per pipeline: CLI + file IO + plots, `main(argv) -> int`; `plotinteractive/` is a package (inputs, view, one module per plot) | registered in `pipelines/runner.py::IMPLEMENTATIONS` |
| `src/qali/integrations/` | ORCA (input templates, runner, output parser), ARC bridge | `runorca` lives here |
| `src/qali/report_templates/` | the HTML/CSS/JS of every report + a tiny `$!token` renderer | presentation only |
| `src/qali/gui/` | the PySide6 app (see below) | never computes physics |
| `src/qali/resources/` | bundled read-only data: NIST levels, f-values, nuclear moments, basis sets, example results, plotly.js | read through `core.paths.reference_path()` |
| `tests/` | pytest suite (anchors, engines, pipelines end to end, GUI offscreen) | |
| `tools/` | maintainer scripts, incl. `compare_outputs.py` | not imported by `src/` |

> Some physics still lives inside `pipelines/` (polarizability, compare_elements,
> …), and `polarizability_core` still reads files. Moving it into
> `physics/` engines, one pipeline at a time, is the main open structural task;
> `tweezer`, `feshbach`, `hyperfine`, `blackbody`, `lifetimes`, `transitions` and
> `plotinteractive` are done, and `tweezer` is the pattern (see "Split a pipeline into
> engine + story" below).

## Data flow

```text
 bundled NIST levels ─► rydberg ─► <El>_rydberg.json ─┬─► transitions ─► <El>_transitions.json
                                                      │        │
 bundled f-values ────────────────────────────────────┼────────┴─► lifetimes ─► <El>_lifetimes.json
                                                      │                              │
                                                      ├─► polarizability ─► <El>_polarizability.json
                                                      │        │                     │
                                                      │        ├─► blackbody ◄───────┤
                                                      │        ├─► tweezer ◄─────────┘
                                                      │        └─► feshbach (C₆) ◄── <El>_feshbach.json
                                                      ├─► hyperfine ◄── <El>_hf_constants.json, nuclear_data.json
                                                      ├─► polarizability_page (render only: reads <El>_polarizability.json, <El>_lifetimes.json)
                                                      ├─► (the blackbody stage also writes plots/<El>/<El>_blackbody.html, via pipelines/blackbody_report.py)
                                                      ├─► spectra (render only: <El>_lifetimes.json, <El>_hf_constants.json ─► plots/<El>/<El>_spectrum.html)
                                                      └─► orbital3d

 ORCA ─► runorca / orca_to_json ─► <El>.json, <El>_soc*.json ─► plotinteractive (Grotrian diagrams)

 NIST levels (optional) ─┬─► dhf ───────► <El>_dhf.json ───────┐
                         └─► brueckner ─► <El>_brueckner.json ─┴─► lifetimes, polarizability (--model-f):
                                          f-values where NIST and literature have none

 several elements' lifetimes + polarizability ─► compare_elements ─► comparison_table.json + plots/comparison.html
```

All of these files live in `<output folder>/data/`; reports go to `plots/<El>/`; raw ORCA
runs to `orca_outputs/<El>/<run_tag>/`.

## Key concepts

- **Output folder (workspace).** `core/workspace.py`. One rule for GUI and CLI: the folder
  the user chose (GUI menu or `QALI_PROJECT_ROOT`), else the per-user data folder. A
  source checkout used as the folder keeps its outputs in `.qali/`. Resolve paths with
  `Workspace.current()` at use time, never at import.
- **Seeding.** `seed_bundled_data()` copies the bundled inputs into the folder before a
  run (and, from the GUI, the example plots/ORCA runs). It never overwrites.
- **Bundled data.** Always `reference_path("data_json", ...)` etc. It resolves inside the
  `qali` package in every build — checkout, wheel and the Nuitka bundle alike.
- **User tables.** `core/user_tables.py`. Constant tables in the code (masses, core
  polarizabilities, known lifetimes, D-line values, ...) can be edited, and rows added, from
  the GUI; the changes live in `<output folder>/tables/` (`overrides.json`,
  `additions.json`, edited copies under `files/`). Each module calls
  `user_tables.apply("<id>", TABLE)` right after defining its dict, so a stage subprocess
  sees the changes; this read at import is the one exception to "nothing happens at import".
  A packaged row is never replaced by an added one. `core/species_data.py` lists what a
  species with no bundled data needs and imports its NIST files.
- **Pipeline contract.** `main(argv) -> int`; parse your own `argv`; no `input()`;
  nothing happens at import; a run's state lives in its functions, never in module
  globals, so a pipeline can run twice in one process. Bad arguments → `parser.error(...)` (exit 2); missing data →
  `raise MissingDataError(...)` (exit 1). Only `cli.py` turns exceptions into exit codes.
- **Writing files.** `core.fileio.write_json` / `atomic_write_text` (atomic, UTF-8,
  `allow_nan=False`); reports via `report_templates.write_offline_report`. File names
  built from user input must pass `is_safe_name`.
- **Reports.** Templates under `report_templates/`, filled by `render(name, **values)`;
  every token must be supplied and used, so layout drift fails loudly.

## The GUI

`gui/app.py` starts it. `main_window.py` holds the window (menus, layout, slots);
`stages.py` is the stage table (flags, defaults, outputs; pure data, checked against
each pipeline's parser by `tests/test_gui_stage_table.py`); `stage_process.py` builds
and kills the subprocess (`qali --pipeline …`, or the bundle exe when frozen);
`overview.py` is the Overview tab; `plot_cache.py` makes reports render offline in the
web view; `dataset_editor.py` edits input JSON/CSV. Preferences > Tables and variables is
`tables_page.py`; the same page limited to one element is `properties_dialog.py`, and
`new_species_dialog.py` opens when a species with no data is set. `find_bar.py` is Ctrl+F;
`tests_dialog.py` and `test_runner.py` are Run > Tests.

## Common tasks

**Add a pipeline.** Create `pipelines/<name>.py` with `build_parser()` (with
`prog="qali --pipeline <name>"`) and `main(argv) -> int`; register it in
`IMPLEMENTATIONS`; add a smoke test that runs it into a temp folder. To expose it in the
GUI, add a stage to `gui/stages.py` — the stage-table test will tell you if a flag name,
type or default disagrees with your parser.

**Split a pipeline into engine + story.** Follow `tweezer`: the engine
(`physics/tweezer_core.py`) takes its inputs as arguments and returns frozen dataclasses,
with no files, prints or plots; one `compute(...)` call does the whole calculation (a
pipeline made of independent blocks, like `blackbody`, has one `compute_*` per block). The
pipeline (`pipelines/tweezer.py`) loads the inputs, calls `compute`, prints a summary,
writes the JSON (`asdict` keeps the field order, so declare fields in the JSON's key order)
and plots. Add engine tests against published reference values
(`tests/test_tweezer_core.py`), and prove the split with `compare_outputs`.

**Change code without changing results.** Run

```bash
uv run ruff check .
uv run pytest
uv run python tools/compare_outputs.py          # HEAD vs your working tree
```

`compare_outputs` runs the pipeline chain on both versions and reports every output file
that changed (JSON byte-for-byte; HTML ignoring Plotly's random ids). For a refactor it
must say "no differences".

**Change physics on purpose.** Check `PHYSICS_BUGS.md` first, cite the source in a comment
and `REFERENCES.bib`, update or add an anchor in `tests/anchors/na_regression.json`, and
say in the PR which numbers moved (`compare_outputs` lists them).

**Package the app.** `packaging/build_pyside6.ps1` (Windows) or `.sh` (macOS/Linux) with
`pysidedeploy.spec`; then `QALI_TEST_FROZEN=1 uv run pytest tests/test_offline_plotly.py`.
