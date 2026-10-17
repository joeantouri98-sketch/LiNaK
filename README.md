# LiNaK

**LiNaK**: an alkali-atom AMO pipeline (Li–Fr): NIST / QDT levels → transitions → lifetimes → polarizability → BBR → optical tweezers → hyperfine → Feshbach → interactive plots. Optional ORCA TD-DFT / SOC overlays and an optional ARC levels bridge.

Physics derivations and full CLI: [`resources/HTMLs/Pipeline Overview.html`](resources/HTMLs/Pipeline%20Overview.html) (§16 Na walkthrough, §17 flags).
Bibliography: [`REFERENCES.md`](REFERENCES.md) · BibTeX: [`REFERENCES.bib`](REFERENCES.bib).

> **Naming:** this toolkit is **LiNaK**. **ORCA** in the docs always means the Neese quantum-chemistry package (`runorca.py`), not this pipeline.

---

## What this repo is

| Track | Status |
|-------|--------|
| Neutral alkalis Li–Fr | Full AMO chain (**tested**) |
| Ions (e.g. `Na_c1`) / 3d metals (Fe, Cr) | Limited: NIST plotting + ORCA SOC (**tested** for Fe, Cr) |
| ARC (`arc_bridge.py`) | Optional **levels only** → `data_json/arc/` (never overwrites NIST) |

The pipeline has been exercised end-to-end for the alkalis and for Fe and Cr. **Shipped outputs** in this tree are mainly **Na** and **Na⁺ (`Na_c1`)** (plus curated Feshbach / hf tables). Fe, Cr, and other alkali run products are not bundled; if you need them, feel free to [contact](#contact).

---

## Requirements

- Python 3.9+ recommended  
- Run commands through the packaged entry points; generated data goes to `.linak/` and reference data lives under `resources/`.
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
runtime and the read-only `resources/` tree:

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

The checked-in spec packages the application shell and Python/Qt runtime plus
the read-only reference datasets and local Plotly JavaScript. The GUI can
operate on an external LiNaK project directory; set
`LINAK_PROJECT_ROOT` when launching the bundle from another directory. ORCA
remains an external installation and must be available on `PATH`.

Generated application data remains project-owned. Set `LINAK_PROJECT_ROOT`
when the GUI should operate on a particular checkout:

```bash
LINAK_PROJECT_ROOT=/path/to/LiNaK uv run linak-gui
```

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

The project root should contain:

```text
pyproject.toml
resources/                 # read-only reference data
.linak/                    # generated data, plots, and ORCA outputs
```

If the GUI is started from another directory, use:

```text
File > Open LiNaK project folder...
```

and select the folder containing `constants.py` and `rydberg.py`.

### GUI workflow

1. Launch `linak_gui.py`.
2. Select or enter a species, such as `Na`, `K`, or `Na_c1`.
3. Select a pipeline stage.
4. Configure the stage options.
5. Click **Run selected stage**.
6. Monitor output in the **Console** tab.
7. Review calculated results in the **Overview** tab.
8. Use the HTML and JSON tabs for detailed output files.

The Overview tab reads existing files from `data_json/`. It does not
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
| `NIST Levels/<El>.txt` or `.csv` | ASD levels for `rydberg.py` |
| `f_values/<El>_lines.txt` or `.csv` | ASD lines / \(f\) for `lifetimes.py` |
| `data_json/<El>_feshbach.json` | Literature FR tables (Li, Na, K, Rb, Cs; **not Fr**) |
| `data_json/*_hf_constants.json` | Hyperfine constants (or rebuild nuclear data) |
| `Moments/*.csv` | Optional IAEA CSVs for `hyperfine.py --build-nuclear-data` |

Pre-built JSON under `data_json/` lets you plot without re-running the whole chain. Prefer regenerating physics outputs with the scripts rather than hand-editing them.

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

`rydberg.py` prompts for `n-max` if you omit `--n-max` (default prompt 30).  
`plotinteractive.py` may still prompt for energy-zero / continuum choices even when flags are passed.

---

## Full alkali operating sequence

Recommended order for a fresh neutral alkali (example Na). Scripts that support `--all` are noted; **`lifetimes.py` is one element per call**.

```bash
# Optional structure overlays (separate ORCA jobs - never one mixed job)
uv run linak --pipeline runorca Na                         # TD-DFT → data_json/Na.json
uv run linak --pipeline runorca Na --mode soc              # SOC → data_json/Na_soc*.json
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

| Output | Location |
|--------|----------|
| Levels, transitions, lifetimes, α, BBR, tweezer, hf, FR | `data_json/<El>_*.json` |
| ORCA parsed JSON | `data_json/<El>.json`, `data_json/<El>_soc*.json`, or `--outname` |
| ORCA run folders | `orca_outputs/<El>/<run_tag>/` |
| Element plots | `resources/plots/<El>/` (Grotrian, spectrum, tweezer, BBR, Feshbach, …). Shipped samples: `resources/plots/Na/`, `resources/plots/Na_c1/` |
| CompTable | regenerate: `python compare_elements.py` → `resources/plots/comparison_table.html` + `data_json/comparison_table.json` |
| ARC levels (optional) | `data_json/arc/<El>_rydberg.json` only |

Energy convention in pipeline rydberg JSON: **ionization-relative** (continuum = 0, ground = −IE). NIST ASD files on disk are ground-relative; `rydberg.py` converts.

---

## Optional: ORCA

TD-DFT and SOC are **separate** `runorca.py` invocations (`--mode` is exclusive).

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
uv run python arc_bridge.py Na --n-max 50
# → data_json/arc/Na_rydberg.json
```

- Exports **energies only** (no Stark / pair / blockade / \(f\)).  
- Never overwrites `data_json/<El>_rydberg.json`.  
- To try ARC levels: copy manually over the NIST rydberg JSON, run `transitions.py` / `lifetimes.py`, then restore NIST. Lifetimes still need NIST or ORCA \(f\).  
- Elements: Li, Na, K, Rb, Cs, Sr, Ca (Fr not in ARC). Sr/Ca are not a full alkali AMO path here.  
- Cite ARC CPC papers in `REFERENCES.md` if used.

---

## Limited tracks (ions / 3d metals)

Not the full AMO chain (no QDT polarizability / BBR / tweezers / Feshbach).  
Fe and Cr (and the alkalis) have been tested on this limited / full path as appropriate. **In-tree ion outputs:** `Na_c1` only. Fe/Cr JSON and ORCA folders are not shipped; ask under [Contact](#contact) if you need them.

**Ion example** (`Na_c1` = Na⁺; ids from `species.py`; outputs shipped):

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

# Rebuild nuclear moments table for hyperfine
uv run linak --pipeline hyperfine --build-nuclear-data \
  --magn-csv Moments/magn_mom_recomm.csv \
  --elec-csv Moments/elec_mom_recomm.csv

# Tests
pytest tests/
```

Convention notes used by CompTable / lifetimes:

- Pipeline peak cross-section: \(\sigma_{\rm nat}=(\lambda^2/4)(g_u/g_l)\) (isotropic).  
- Steck isotropic Lorentzian (~9.4 mW/cm² for Na D2) and cycling (~6.26) are different conventions - see Overview §07.

---

## Pipeline map

```text
NIST Levels/ + f_values/
        │
        ▼
   rydberg.py ──► transitions.py ──► lifetimes.py
        │                                  │
        │                                  ├─► polarizability.py ──► blackbody.py
        │                                  │            │
        │                                  │            └─► tweezer.py
        │                                  ├─► hyperfine.py
        │                                  └─► feshbach.py   (not Fr)
        │
        └─► plotinteractive / spectra / orbital3d / compare_elements / scattering_rate

Optional side inputs:
  runorca / orca_to_json  →  data_json/<El>.json (+ _soc* / --outname)
  arc_bridge              →  data_json/arc/<El>_rydberg.json  (manual swap only)
```

---

## Repository layout

| Path | Role |
|------|------|
| `*.py` | Pipeline scripts and shared libraries (`constants.py`, `species.py`, `alpha_core.py`, …) |
| `.linak/data/` | Generated and project-local JSON |
| `.linak/data/arc/` | Optional ARC level exports |
| `resources/NIST Levels/` | ASD level references |
| `resources/f_values/` | ASD line / \(f\) references |
| `resources/Moments/` | Nuclear-moment CSVs for hyperfine rebuild |
| `.linak/orca_outputs/` | ORCA run directories (when ORCA is used) |
| `.linak/plots/` | Generated interactive HTML outputs |
| `HTMLs/Pipeline Overview.html` | Physics + CLI reference |
| `tools/` | Maintainer helpers (regen / audit); not required for daily runs |
| `tests/` | Pytest checks |
| `REFERENCES.md` / `REFERENCES.bib` | Citations |

---

## Data provenance

- **Levels / lines:** NIST ASD (and Steck / literature overlays where noted).  
- **Feshbach \(B_0,\Delta,a_{\rm bg}\):** curated in `data_json/<El>_feshbach.json` with per-resonance refs (Chin RMP; Zürn, Knoop, D'Errico, Marte, Berninger, …). Default K table is **³⁹K**.  
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
3. ARC CPC papers if you used `arc_bridge.py`.  
4. ORCA if you report ORCA results.

BibTeX: [`REFERENCES.bib`](REFERENCES.bib).

---

## Contributing / status notes

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for bug reports, pull requests, and how to add new elements.

- Prefer regenerating JSON via scripts over hand-editing physics outputs.  
- Do not invent NIST tables; drop ASD exports into `NIST Levels/` / `f_values/`.  
- Ions and 3d metals are **limited support** (Overview §17).  
- CompTable: `python compare_elements.py` → `resources/plots/comparison_table.html`.
- Keep Overview and README CLI examples in sync when flags change.

---

## Contact

Maintainers: Youssef Antoury / Joeantouri98@gmail.com.

Shipped demos: Na + Na⁺ (`Na_c1`). If you need outputs for other alkalis, Fe, or Cr (JSON, ORCA folders, CompTable, plots), feel free to contact me.
