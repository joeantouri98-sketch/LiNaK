# LiNaK

**LiNaK**: an alkali-atom AMO pipeline (Li–Fr): NIST / QDT levels → transitions → lifetimes → polarizability → BBR → optical tweezers → hyperfine → Feshbach → interactive plots. Optional ORCA TD-DFT / SOC overlays and an optional ARC levels bridge.

Physics derivations and full CLI: [`src/linak/resources/HTMLs/Pipeline Overview.html`](src/linak/resources/HTMLs/Pipeline%20Overview.html) (§16 Na walkthrough, §17 flags).
Bibliography: [`REFERENCES.md`](REFERENCES.md) · BibTeX: [`REFERENCES.bib`](REFERENCES.bib).
Working on the code? Start with [`ARCHITECTURE.md`](ARCHITECTURE.md), then [`CONTRIBUTING.md`](CONTRIBUTING.md).

> **Naming:** this toolkit is **LiNaK**. **ORCA** in the docs always means the Neese quantum-chemistry package (`runorca` pipeline), not this pipeline.

---

## What this repo is

| Track | Status |
|-------|--------|
| Neutral alkalis Li–Fr | Full AMO chain (**tested**) |
| Ions (e.g. `Na_c1`) / 3d metals (Fe, Cr) | Limited: NIST plotting + ORCA SOC (**tested** for Fe, Cr) |
| ARC (`linak.integrations.arc`) | Optional **levels only** → `data/arc/` (never overwrites NIST) |

The pipeline has been exercised end-to-end for the alkalis and for Fe and Cr. **Shipped outputs** in this tree are mainly **Na** and **Na⁺ (`Na_c1`)** (plus curated Feshbach / hf tables). Fe, Cr, and other alkali run products are not bundled; if you need them, feel free to [contact](#contact).

---

## Requirements

- Python 3.9+ recommended  
- Run commands through the packaged entry points; generated data goes to the output folder (see [Where outputs go](#where-outputs-go)) and the read-only reference data ships as `linak` package data under `src/linak/resources/`.
- ORCA (optional; only for the ORCA pipeline stages)

```bash
uv sync
```

The project is also installable with standard Python packaging tools, without
uv:

```bash
python -m pip install .
linak-gui
linak --pipeline rydberg Na --n-max 50
```

To install the commands globally with uv instead of into the current
environment:

```bash
uv tool install .
linak-gui
linak --pipeline rydberg Na --n-max 50
```

Remove the uv-managed installation with:

```bash
uv tool uninstall linak
```

For checkout development, uv provides the equivalent isolated-environment
workflow. Use these stable entry points:

```bash
uv run linak-gui       # desktop application
uv run linak            # same application (GUI is the default)
uv run python -m linak  # module form
```

### Build the application

The GUI is deployed with the Qt-supported `pyside6-deploy` tool. It
uses Nuitka and the checked-in `pysidedeploy.spec` to include the Python/Qt
runtime and the read-only reference data tree (`src/linak/resources`, declared as
`linak` package data in `pyproject.toml`):

```bash
uv sync --dev
uv run pyside6-deploy -c pysidedeploy.spec
open dist/pyside6/LiNaK.app
```

The repeatable build wrappers are `./packaging/build_pyside6.sh --clean` on
macOS/Linux and `.\packaging\build_pyside6.ps1 -Clean` in PowerShell on Windows.
The wrappers use `--mode onefile` by default. On Windows the result is a
self-contained `dist/pyside6/LiNaK.exe`, on Linux `dist/pyside6/LiNaK.bin`,
and on macOS `dist/pyside6/LiNaK.app` (macOS always produces an application
bundle as Nuitka's one-file mode is not supported for macOS).

Package versions are derived from Git tags. Push a tag such as `v1.4.2` to
run `.github/workflows/build-release.yml`; GitHub Actions builds downloadable
Windows and macOS archives from that tagged source.

The checked-in spec packages the application shell, Python/Qt runtime, and
the read-only reference datasets (as `linak` package data) plus the vendored
Plotly JavaScript under `src/linak/resources/vendor/`. Plotly reports are fully offline: every HTML references
a sibling `plotly.min.js` copied next to it, and the frozen build ships the
`plotly` Python package as bytecode with only the data files it actually
reads at runtime (see `tests/test_offline_plotly.py`). The bundled data sits
at `linak/resources` and `linak/report_templates` inside the bundle, exactly
where the installed package keeps it, so one lookup rule serves every build.
ORCA remains an external installation: `orca` on `PATH`, or the `ORCA_EXE`
environment variable.

This creates (or updates) the project-local `.venv` and installs the locked
runtime dependencies. Run Python and project commands through `uv run` so they
always use that environment. The installed command itself remains
`linak --pipeline ...`:

```bash
uv run linak --pipeline rydberg Na --n-max 50
uv run pytest tests/
```

Optional integrations can be installed with extras:

```bash
uv sync --extra arc       # ARC levels integration
uv sync --extra basis     # heavy-element ORCA bases (e.g. Fr)
uv sync --extra web       # inline Qt WebEngine/HTML rendering
```

Development tools are in the `dev` dependency group:

```bash
uv sync --dev
```

---

## Desktop GUI

LiNaK includes a native PySide6 desktop control panel exposed as `linak-gui`.

The GUI provides:

- Species selection
- Pipeline-stage controls
- Command previews
- Console output from running stages
- HTML plot browsing
- JSON data browsing
- A species overview dashboard
- Pipeline completion status
- Rydberg, transition, lifetime, optical, blackbody, hyperfine, and Feshbach summaries
- Feshbach resonance and channel tables
- Hyperfine clock frequency and wavelength information

### GUI requirements

The GUI requires:

- Python 3.9 or newer
- PySide6
- The normal LiNaK dependencies installed by the package

Install the package from the project root:

```bash
python -m pip install .
```

For inline HTML and Plotly rendering inside the GUI, install:

```bash
python -m pip install '.[web]'
```

If `PySide6-Addons` is unavailable, the GUI still starts. HTML plots can
be opened with the system browser instead.

### Launching the GUI

Run the installed GUI command:

```bash
linak-gui
```

For checkout development, the equivalent uv command is:

```bash
uv run linak-gui
```

### Where outputs go

The GUI and the CLI share one rule. Outputs (`data/`, `plots/`,
`orca_outputs/`) go to the **output folder**:

1. the folder chosen with **File > Open output folder...** (remembered across
   sessions; **File > Use default output folder** forgets it), or the folder
   named by the `LINAK_PROJECT_ROOT` environment variable;
2. otherwise LiNaK's per-user data folder: `%LOCALAPPDATA%\LiNaK` on Windows,
   `~/Library/Application Support/LiNaK` on macOS, `~/.local/share/LiNaK`
   (or `$XDG_DATA_HOME/LiNaK`) on Linux.

The launch directory never matters. If the chosen folder is a LiNaK source
checkout, outputs go to its git-ignored `.linak/` subfolder so they never
touch the source tree:

```bash
LINAK_PROJECT_ROOT=. uv run linak --pipeline rydberg Na   # -> ./.linak/data/
```

The bundled reference data is copied into the output folder (existing files
are never overwritten): pipeline inputs on every run, and in the GUI also the
example `plots/` and `orca_outputs/` for Li, Na and Na⁺, so a fresh folder
opens with populated species.

### GUI workflow

1. Launch the GUI with `linak-gui`.
2. Select or enter a species, such as `Na`, `K`, or `Na_c1`.
3. Select a pipeline stage.
4. Configure the stage options.
5. Click **Run selected stage**.
6. Monitor output in the **Console** tab.
7. Review calculated results in the **Overview** tab.
8. Use the HTML and JSON tabs for detailed output files.

The Overview tab reads existing files from the output folder's `data/`. It does not
recalculate the physics independently. Run the relevant pipeline stages
first if a section reports that data is unavailable.

Check PySide6:

```bash
uv run python -c "from PySide6.QtWidgets import QApplication; print('PySide6 OK')"
```

Check optional Qt WebEngine support:

```bash
uv run python -c "from PySide6.QtWebEngineWidgets import QWebEngineView; print('Qt WebEngine OK')"
```


## Inputs you need on disk

| Path | Purpose |
|------|---------|
| `src/linak/resources/NIST Levels/<El>.txt` or `.csv` | ASD levels for `rydberg` |
| `src/linak/resources/f_values/<El>_lines.txt` or `.csv` | ASD lines / \(f\) for `lifetimes` |
| `src/linak/resources/data_json/<El>_feshbach.json` | Literature FR tables (Li, Na, K, Rb, Cs; **not Fr**) |
| `src/linak/resources/data_json/*_hf_constants.json` | Hyperfine constants (or rebuild nuclear data) |
| `src/linak/resources/Moments/*.csv` | Optional IAEA CSVs for `hyperfine --build-nuclear-data` |

Pre-built JSON under `src/linak/resources/data_json/` (copied into the output folder's `data/` on first run) lets you plot without re-running the whole chain. Prefer regenerating physics outputs with the scripts rather than hand-editing them.

---

## Quick start (Na, NIST path only)

Minimal path if ORCA is not installed:

```bash
uv run linak --pipeline rydberg Na --n-max 50
uv run linak --pipeline transitions Na
uv run linak --pipeline lifetimes Na

uv run linak --pipeline polarizability Na
uv run linak --pipeline blackbody Na --T 300
uv run linak --pipeline tweezer Na --wl 1064 --intensity 50 --waist 1
uv run linak --pipeline hyperfine Na
uv run linak --pipeline feshbach Na

uv run linak --pipeline plotinteractive Na --plots 1
uv run linak --pipeline compare_elements
uv run linak --pipeline scattering_rate --element Na --line D2
```

`linak --pipeline rydberg` no longer fits quantum defects through a stdin prompt, so all values come through argparse.  
`linak --pipeline plotinteractive` is fully flag-driven: the energy reference
(`--energy-shift 1|2|3` with `--absolute-ground` / `--custom-shift`), the
level/transition filters (`--filter-ie`, `--transition-display`,
`--orca-display` with `--*-wl-min/max`, `--plot3-choice`) and the optional
sub-launches (`--also-orbital3d` with `--orbital3d-n-max`/`--orbital3d-grid`,
`--also-spectra` with `--spectra-wl-min/max`) are plain options with defaults.

---

## Full alkali operating sequence

Recommended order for a fresh neutral alkali (example Na). Stages that support `--all` are noted; **`lifetimes` is one element per call**.

```bash
# Optional structure overlays (separate ORCA jobs - never one mixed job)
uv run linak --pipeline runorca Na                         # TD-DFT → data/Na.json
uv run linak --pipeline runorca Na --mode soc              # SOC → data/Na_soc*.json
# or reuse outs: uv run linak --pipeline orca_to_json Na [--run TAG] [--outname SUFFIX]

# Levels → network → rates
uv run linak --pipeline rydberg Na --n-max 50
uv run linak --pipeline transitions Na
uv run linak --pipeline lifetimes Na                       # no --all

# Trap / BBR / structure
uv run linak --pipeline polarizability Na                  # also: --all
uv run linak --pipeline blackbody Na --T 300               # also: --all
uv run linak --pipeline tweezer Na --wl 1064 --intensity 50
uv run linak --pipeline hyperfine Na                       # also: --all
uv run linak --pipeline feshbach Na                        # skip Fr (no FR JSON)

# Visualization
uv run linak --pipeline spectra Na
uv run linak --pipeline orbital3d Na
uv run linak --pipeline plotinteractive Na --plots 1       # 1=grotrian; higher N adds more
uv run linak --pipeline compare_elements                   # CompTable for Li-Fr
uv run linak --pipeline scattering_rate --element Na --line D2
```

Batch helpers (where implemented):

```bash
uv run linak --pipeline polarizability --all
uv run linak --pipeline blackbody --all
uv run linak --pipeline tweezer --all
uv run linak --pipeline hyperfine --all
uv run linak --pipeline feshbach --all          # every element that has *_feshbach.json
uv run linak --pipeline compare_elements        # default: all six alkalis at T=300 K
```

After any lifetimes / NIST refresh, rebuild CompTable:

```bash
uv run linak --pipeline lifetimes Na            # repeat for Li K Rb Cs Fr as needed
uv run linak --pipeline compare_elements
```

---

## Outputs

Paths are relative to the [output folder](#where-outputs-go).

| Output | Location |
|--------|----------|
| Levels, transitions, lifetimes, α, BBR, tweezer, hf, FR | `data/<El>_*.json` |
| ORCA parsed JSON | `data/<El>.json`, `data/<El>_soc*.json`, or `--outname` |
| ORCA run folders | `orca_outputs/<El>/<run_tag>/` |
| Element plots | `plots/<El>/` (Grotrian, spectrum, tweezer, BBR, Feshbach, …) |
| CompTable | regenerate: `uv run linak --pipeline compare_elements` → `plots/comparison_table.html` + `data/comparison_table.json` |
| ARC levels (optional) | `data/arc/<El>_rydberg.json` only |

Energy convention in pipeline rydberg JSON: **ionization-relative** (continuum = 0, ground = −IE). NIST ASD files on disk are ground-relative; `rydberg` converts.

---

## Optional: ORCA

TD-DFT and SOC are **separate** `runorca` invocations (`--mode` is exclusive).

```bash
uv run linak --pipeline runorca Na
uv run linak --pipeline runorca Na --mode soc --soc-recipe nevpt2
uv run linak --pipeline orca_to_json Na --run <TAG>          # re-parse without re-running
uv run linak --pipeline plotinteractive Na -e v13 --plots 1  # overlay a tagged JSON
uv run linak --pipeline plotinteractive Na --soc-only --plots 1
```

Useful flags: `--outname SUFFIX`, `--nroots`, `--functional-gs` / `--functional-ex`, `--scf-tightness`, `--basis-tightness` (heavy Z), `--basis-file`. Full list: Overview §17.

---

## Optional: ARC levels only

```bash
uv sync --extra arc
uv run python -m linak.integrations.arc Na --n-max 50
# → data/arc/Na_rydberg.json
```

- Exports **energies only** (no Stark / pair / blockade / \(f\)).  
- Never overwrites `data/<El>_rydberg.json`.  
- To try ARC levels: copy manually over the NIST rydberg JSON, run `transitions` / `lifetimes`, then restore NIST. Lifetimes still need NIST or ORCA \(f\).  
- Elements: Li, Na, K, Rb, Cs, Sr, Ca (Fr not in ARC). Sr/Ca are not a full alkali AMO path here.  
- Cite ARC CPC papers in `REFERENCES.md` if used.

---

## Limited tracks (ions / 3d metals)

Not the full AMO chain (no QDT polarizability / BBR / tweezers / Feshbach).  
Fe and Cr (and the alkalis) have been tested on this limited / full path as appropriate. **In-tree ion outputs:** `Na_c1` only. Fe/Cr JSON and ORCA folders are not shipped; ask under [Contact](#contact) if you need them.

**Ion example** (`Na_c1` = Na⁺; ids from `linak.core.species`; outputs shipped):

```bash
uv run linak --pipeline runorca Na_c1 --mode soc
uv run linak --pipeline rydberg Na_c1 --n-max 30
uv run linak --pipeline transitions Na_c1
uv run linak --pipeline lifetimes Na_c1
uv run linak --pipeline plotinteractive Na_c1 --plots 1
```

**3d metal example** (Fe / Cr; tested; regenerate locally or [ask](#contact) for outputs):

```bash
uv run linak --pipeline rydberg Fe                 # NIST terms; --nist-only auto for 3d
uv run linak --pipeline transitions Fe
uv run linak --pipeline lifetimes Fe --nist-only
uv run linak --pipeline runorca Fe --mode soc --soc-recipe nevpt2
uv run linak --pipeline plotinteractive Fe --soc-only --plots 1
```

---

## Other common operations

```bash
# Near-resonant MOT scattering widget (not tweezer R_sc)
uv run linak --pipeline scattering_rate --element Na --line D2

# CompTable temperature / subset
uv run linak --pipeline compare_elements --T 500 --elements Li Na K

# Rebuild nuclear moments table for hyperfine (bundled IAEA CSVs by default;
# pass --magn-csv / --elec-csv for your own files)
uv run linak --pipeline hyperfine --build-nuclear-data

# Lint, tests, and the output check (see CONTRIBUTING.md)
uv run ruff check .
uv run pytest
uv run python tools/compare_outputs.py
```

Convention notes used by CompTable / lifetimes:

- Pipeline peak cross-section: \(\sigma_{\rm nat}=(\lambda^2/4)(g_u/g_l)\) (isotropic).  
- Steck isotropic Lorentzian (~9.4 mW/cm² for Na D2) and cycling (~6.26) are different conventions - see Overview §07.

---

## Pipeline map

```text
src/linak/resources/NIST Levels/ + src/linak/resources/f_values/
        │
        ▼
   rydberg ──► transitions ──► lifetimes
        │                          │
        │                          ├─► polarizability ──► blackbody
        │                          │          │
        │                          │          └─► tweezer
        │                          ├─► hyperfine
        │                          └─► feshbach   (not Fr)
        │
        └─► plotinteractive / spectra / orbital3d / compare_elements / scattering_rate

Optional side inputs:
  runorca / orca_to_json  →  data/<El>.json (+ _soc* / --outname)
  linak.integrations.arc  →  data/arc/<El>_rydberg.json  (manual swap only)
```

---

## Repository layout

| Path | Role |
|------|------|
| `src/linak/` | Python package: `core/` (constants, elements, species, paths), `physics/` (models), `pipelines/` (CLI + IO + plots), `integrations/` (ORCA / ARC / BSE), `gui/` (PySide6 app) |
| `.linak/` | Outputs when the checkout itself is the output folder (git-ignored) |
| `src/linak/resources/NIST Levels/` | ASD level references (package data) |
| `src/linak/resources/f_values/` | ASD line / \(f\) references |
| `src/linak/resources/Moments/` | Nuclear-moment CSVs for hyperfine rebuild |
| `src/linak/resources/HTMLs/Pipeline Overview.html` | Physics + CLI reference |
| `tools/` | Maintainer helpers (regen / audit); not required for daily runs |
| `tests/` | Pytest checks |
| `REFERENCES.md` / `REFERENCES.bib` | Citations |

---

## Data provenance

- **Levels / lines:** NIST ASD (and Steck / literature overlays where noted).  
- **Feshbach \(B_0,\Delta,a_{\rm bg}\):** curated in `src/linak/resources/data_json/<El>_feshbach.json` with per-resonance refs (Chin RMP; Zürn, Knoop, D'Errico, Marte, Berninger, …). Default K table is **³⁹K**.  
- **Hyperfine:** Steck (Na/Rb/Cs), Allegrini22, Gehm (Li-6), Tiecke + Falke (K), Sansonetti (Fr) as listed in `*_hf_constants.json`; Stone/IAEA for `nuclear_data.json`.  
- **ORCA:** user-run; methods tagged in JSON metadata.  
- **ARC:** optional dependency; cite the ARC CPC papers in `REFERENCES.md` if used.

See [`REFERENCES.md`](REFERENCES.md) for the full list.

---

## License and third-party code

- **This repository:** [MIT](LICENSE) (Copyright 2026 Youssef Antoury).  
- **ARC:** BSD-3-Clause (optional dependency).  
- **ORCA:** separate license from the ORCA developers.

---

## Citation

If you use this pipeline in a paper or thesis, please cite:

1. **This software:** Youssef Antoury, *LiNaK: an alkali-atom AMO pipeline* (2026).  
   DOI: [10.5281/zenodo.21821149](https://doi.org/10.5281/zenodo.21821149) · Repository: [github.com/joeantouri98-sketch/LiNaK](https://github.com/joeantouri98-sketch/LiNaK)  
   BibTeX: `Antoury_LiNaK` in [`REFERENCES.bib`](REFERENCES.bib). Contact: [Joeantouri98@gmail.com](mailto:Joeantouri98@gmail.com).  
2. The literature sources for any numbers you quote - see the **Must cite** table in [`REFERENCES.md`](REFERENCES.md) and Overview §19. In particular:  
   - **NIST ASD** for levels / lines  
   - **Steck** (steck.us/alkalidata) for Na/Rb/Cs hyperfine and D-line checks  
   - **Allegrini et al.**, *J. Phys. Chem. Ref. Data* **51**, 043102 (2022) for recommended hyperfine  
   - **Gehm** (*Properties of $^6$Li*) / **Tiecke** (*Properties of Potassium*) when using those Li/K constants  
   - **Stone / IAEA** for nuclear moments  
   - Element-specific Feshbach papers + **Chin et al.** RMP (2010)  
   - **Grimm / Itano / Beterov / …** for trap and BBR methods as used  
3. ARC CPC papers if you used the ARC bridge (`linak.integrations.arc`).  
4. ORCA if you report ORCA results.

BibTeX: [`REFERENCES.bib`](REFERENCES.bib).

---

## Contributing / status notes

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for bug reports, pull requests, and how to add new elements.

- Prefer regenerating JSON via scripts over hand-editing physics outputs.  
- Do not invent NIST tables; drop ASD exports into `NIST Levels/` / `f_values/`.  
- Ions and 3d metals are **limited support** (Overview §17).  
- CompTable: `uv run linak --pipeline compare_elements` → `plots/comparison_table.html`.
- Keep Overview and README CLI examples in sync when flags change.

---

## Contact

Maintainers: Youssef Antoury / Joeantouri98@gmail.com.

Shipped demos: Na + Na⁺ (`Na_c1`). If you need outputs for other alkalis, Fe, or Cr (JSON, ORCA folders, CompTable, plots), feel free to contact me.
