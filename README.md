# LiNaK

**LiNaK**: an alkali-atom AMO pipeline (Li–Fr): NIST / QDT levels → transitions → lifetimes → polarizability → BBR → optical tweezers → hyperfine → Feshbach → interactive plots. Optional ORCA TD-DFT / SOC overlays and an optional ARC levels bridge.

Physics derivations and full CLI: [`HTMLs/Pipeline Overview.html`](HTMLs/Pipeline%20Overview.html) (§16 Na walkthrough, §17 flags).  
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
- Run all commands from the **project root** (so `data_json/`, `NIST Levels/`, `f_values/` resolve)  
- ORCA (optional; only for `runorca.py` / `orca_to_json.py`)  

```bash
pip install -r requirements.txt
```

Core: `numpy`, `scipy`, `plotly`. Optional extras (commented in `requirements.txt`):

```text
pip install ARC-Alkali-Rydberg-Calculator   # arc_bridge.py only
pip install basis-set-exchange             # heavy-element ORCA bases (e.g. Fr)
pip install pytest                         # tests/
```

---

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
python rydberg.py Na --n-max 50
python transitions.py Na
python lifetimes.py Na

python polarizability.py Na
python blackbody.py Na --T 300
python tweezer.py Na --wl 1064 --intensity 50 --waist 1
python hyperfine.py Na
python feshbach.py Na

python plotinteractive.py Na --plots 1
python compare_elements.py
python scattering_rate.py --element Na --line D2
```

`rydberg.py` prompts for `n-max` if you omit `--n-max` (default prompt 30).  
`plotinteractive.py` may still prompt for energy-zero / continuum choices even when flags are passed.

---

## Full alkali operating sequence

Recommended order for a fresh neutral alkali (example Na). Scripts that support `--all` are noted; **`lifetimes.py` is one element per call**.

```bash
# Optional structure overlays (separate ORCA jobs - never one mixed job)
python runorca.py Na                         # TD-DFT → data_json/Na.json
python runorca.py Na --mode soc              # SOC → data_json/Na_soc*.json
# or reuse outs: python orca_to_json.py Na [--run TAG] [--outname SUFFIX]

# Levels → network → rates
python rydberg.py Na --n-max 50
python transitions.py Na
python lifetimes.py Na                       # no --all

# Trap / BBR / structure
python polarizability.py Na                  # also: --all
python blackbody.py Na --T 300               # also: --all
python tweezer.py Na --wl 1064 --intensity 50
python hyperfine.py Na                       # also: --all
python feshbach.py Na                        # skip Fr (no FR JSON)

# Visualization
python spectra.py Na
python orbital3d.py Na
python plotinteractive.py Na --plots 1       # 1=grotrian; higher N adds more
python compare_elements.py                   # CompTable for Li-Fr
python scattering_rate.py --element Na --line D2
```

Batch helpers (where implemented):

```bash
python polarizability.py --all
python blackbody.py --all
python tweezer.py --all
python hyperfine.py --all
python feshbach.py --all          # every element that has *_feshbach.json
python compare_elements.py        # default: all six alkalis at T=300 K
```

After any lifetimes / NIST refresh, rebuild CompTable:

```bash
python lifetimes.py Na            # repeat for Li K Rb Cs Fr as needed
python compare_elements.py
```

---

## Outputs (where things land)

| Output | Location |
|--------|----------|
| Levels, transitions, lifetimes, α, BBR, tweezer, hf, FR | `data_json/<El>_*.json` |
| ORCA parsed JSON | `data_json/<El>.json`, `data_json/<El>_soc*.json`, or `--outname` |
| ORCA run folders | `orca_outputs/<El>/<run_tag>/` |
| Element plots | `plots/<El>/` (Grotrian, spectrum, tweezer, BBR, Feshbach, …). Shipped samples: `plots/Na/`, `plots/Na_c1/` |
| CompTable | regenerate: `python compare_elements.py` → `plots/comparison_table.html` + `data_json/comparison_table.json` |
| ARC levels (optional) | `data_json/arc/<El>_rydberg.json` only |

Energy convention in pipeline rydberg JSON: **ionization-relative** (continuum = 0, ground = −IE). NIST ASD files on disk are ground-relative; `rydberg.py` converts.

---

## Optional: ORCA

TD-DFT and SOC are **separate** `runorca.py` invocations (`--mode` is exclusive).

```bash
python runorca.py Na
python runorca.py Na --mode soc --soc-recipe nevpt2
python orca_to_json.py Na --run <TAG>          # re-parse without re-running
python plotinteractive.py Na -e v13 --plots 1  # overlay a tagged JSON
python plotinteractive.py Na --soc-only --plots 1
```

Useful flags: `--outname SUFFIX`, `--nroots`, `--functional-gs` / `--functional-ex`, `--scf-tightness`, `--basis-tightness` (heavy Z), `--basis-file`. Full list: Overview §17.

---

## Optional: ARC levels only

```bash
pip install ARC-Alkali-Rydberg-Calculator
python arc_bridge.py Na --n-max 50
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
python runorca.py Na_c1 --mode soc
python rydberg.py Na_c1 --n-max 30
python transitions.py Na_c1
python lifetimes.py Na_c1
python plotinteractive.py Na_c1 --plots 1
```

**3d metal example** (Fe / Cr; tested; regenerate locally or [ask](#contact) for outputs):

```bash
python rydberg.py Fe                 # NIST terms; --nist-only auto for 3d
python transitions.py Fe
python lifetimes.py Fe --nist-only
python runorca.py Fe --mode soc --soc-recipe nevpt2
python plotinteractive.py Fe --soc-only --plots 1
```

---

## Other common operations

```bash
# Near-resonant MOT scattering widget (not tweezer R_sc)
python scattering_rate.py --element Na --line D2

# CompTable temperature / subset
python compare_elements.py --T 500 --elements Li Na K

# Rebuild nuclear moments table for hyperfine
python hyperfine.py --build-nuclear-data \
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
| `data_json/` | Generated / curated JSON |
| `data_json/arc/` | Optional ARC level exports |
| `NIST Levels/` | ASD level exports |
| `f_values/` | ASD line / \(f\) exports |
| `Moments/` | Nuclear-moment CSVs for hyperfine rebuild |
| `orca_outputs/` | ORCA run directories (when ORCA is used) |
| `plots/` | Interactive HTML outputs (Na / Na⁺ samples shipped; regenerate others) |
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
   Repository: [github.com/joeantouri98-sketch/LiNaK](https://github.com/joeantouri98-sketch/LiNaK) (cite the commit hash you used; BibTeX: `Antoury_LiNaK` in [`REFERENCES.bib`](REFERENCES.bib)).  
   Contact: [Joeantouri98@gmail.com](mailto:Joeantouri98@gmail.com).  
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

- Prefer regenerating JSON via scripts over hand-editing physics outputs.  
- Do not invent NIST tables; drop ASD exports into `NIST Levels/` / `f_values/`.  
- Ions and 3d metals are **limited support** (Overview §17).  
- CompTable: `python compare_elements.py` → `plots/comparison_table.html`.  
- Keep Overview and README CLI examples in sync when flags change.

---

## Contact

Maintainers: Youssef Antoury / Joeantouri98@gmail.com.

Shipped demos: Na + Na⁺ (`Na_c1`). If you need outputs for other alkalis, Fe, or Cr (JSON, ORCA folders, CompTable, plots), feel free to contact me.
