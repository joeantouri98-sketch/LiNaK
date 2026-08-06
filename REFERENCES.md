# References

Bibliographic sources used by **LiNaK** (alkali AMO pipeline). BibTeX entries live in [`REFERENCES.bib`](REFERENCES.bib). When citing Steck / Gehm / Tiecke sheets, include the revision date used.

---

## Must cite (by what you use)

| If you use… | Cite at least… |
|-------------|----------------|
| This pipeline / code / regenerated tables | **Antoury_LiNaK** / **LiNaK** ([github.com/joeantouri98-sketch/LiNaK](https://github.com/joeantouri98-sketch/LiNaK); cite commit) |
| NIST levels / lines / Acc / CompTable lifetimes from ASD | **NIST ASD** |
| Na / Rb / Cs hyperfine constants or Steck validation of D-line / \(I_{\rm sat}\) | **Steck** (steck.us/alkalidata; name the sheet + revision) |
| Li hyperfine (especially \(^6\)Li block) | **Allegrini22** and **Gehm** (*Properties of \(^6\)Li*) |
| K hyperfine | **Tiecke** (*Properties of Potassium*) and **Falke06**; Allegrini for recommended survey |
| Fr hyperfine / Fr spectroscopy | **Allegrini22** and **Sansonetti07**; Simsarian for Fr lifetimes |
| Nuclear spins / moments (`nuclear_data.json`) | **Stone / IAEA** nuclear moments tables |
| Feshbach \(a(B)\) numbers | **Chin10** plus the element-specific papers in the Feshbach section |
| Trap depth / tweezer scattering formulas | **Grimm00** |
| BBR shifts / PI | **Itano82**, **FarleyWing81**, **Beterov09**, **Beloy25** as used |
| `arc_bridge.py` | **Sibalic17** and **Robertson21** |
| ORCA results | **ORCA** (Neese *et al.*) |

---

## Data sources and standards

| Key | Citation | Used for |
|-----|----------|----------|
| NIST-ASD | NIST Atomic Spectra Database (levels, lines, Acc codes) - [https://physics.nist.gov/asd](https://physics.nist.gov/asd) | `rydberg`, `lifetimes`, `compare_elements`, `f_values/` |
| CODATA2018 | CODATA 2018 / exact SI constants | `constants.py` |
| Steck | D. A. Steck, Alkali D Line Data (Na, \(^{85}\)Rb, \(^{87}\)Rb, Cs); [steck.us/alkalidata](https://steck.us/alkalidata/) | `*_hf_constants.json` (Na/Rb/Cs); CompTable / \(I_{\rm sat}\) validation |
| Gehm | M. E. Gehm, *Properties of \(^6\)Li* (standalone from PhD appendix, 2003); e.g. [NCSU JET copy](https://jet.physics.ncsu.edu/techdocs/pdf/PropertiesOfLi.pdf) | Li-6 hyperfine block in `Li_hf_constants.json` |
| Tiecke | T. G. Tiecke, *Properties of Potassium* (v1.0, 2010; from PhD appendix); [tobiastiecke.nl archive](https://www.tobiastiecke.nl/archive/PotassiumProperties.pdf) | K hyperfine in `K_hf_constants.json` |
| Allegrini22 | M. Allegrini, E. Arimondo, F. Tomassetti, *J. Phys. Chem. Ref. Data* **51**, 043102 (2022) [doi:10.1063/5.0090334](https://doi.org/10.1063/5.0090334) | Recommended alkali hyperfine (Li, K survey, Fr, …) |
| Sansonetti07 | J. E. Sansonetti, *J. Phys. Chem. Ref. Data* **36**, 497 (2007) | Fr spectroscopy / hyperfine |
| StoneIAEA | N. J. Stone (IAEA nuclear moments compilations / updates) | `nuclear_data.json` via `hyperfine.py --build-nuclear-data` |

---

## Core methods

| Key | Citation | Used for |
|-----|----------|----------|
| Grimm00 | R. Grimm, M. Weidemüller, Yu. B. Ovchinnikov, *Adv. At. Mol. Opt. Phys.* **42**, 95 (2000) | AC Stark / ODT depth; tweezer scattering context |
| Chin10 | C. Chin, R. Grimm, P. Julienne, E. Tiesinga, *Rev. Mod. Phys.* **82**, 1225 (2010) | Feshbach \(a(B)\) form; \(a_{\rm bg}\) tables |
| Itano82 | W. M. Itano, L. L. Lewis, D. J. Wineland, *Phys. Rev. A* **25**, 1233 (1982) | Static BBR Stark field |
| FarleyWing81 | J. W. Farley, W. H. Wing, *Phys. Rev. A* **23**, 2397 (1981) | High-\(n\) asymptotic BBR shift |
| Beterov09 | I. I. Beterov *et al.*, *New J. Phys.* **11**, 013052 (2009) | BBR PI / SFI / mixing rates |
| Beloy25 | K. Beloy *et al.*, arXiv:2507.00948 | Dynamic BBR shift \(\propto\alpha(\omega)\) |
| BatesDamgaard49 | D. R. Bates, A. Damgaard, *Phil. Trans. R. Soc. A* **242**, 101 (1949) | Coulomb / Bates–Damgaard \(f\) fallback |
| Mitroy10 | J. Mitroy, M. S. Safronova, C. W. Clark, *J. Phys. B* **43**, 202001 (2010) | Polarizability / core \(\alpha\) context |
| Armstrong71 | L. Armstrong Jr., *Adv. At. Mol. Phys.* **7**, 149 (1971) | Hyperfine multipole formulas |
| GribakinFlambaum93 | G. F. Gribakin, V. V. Flambaum, *Phys. Rev. A* **48**, 546 (1993) | Mean scattering length \(\bar a\) |

---

## Lifetimes / D-line precision

| Key | Citation | Used for |
|-----|----------|----------|
| Simsarian98 | J. E. Simsarian *et al.*, *Phys. Rev. A* **57**, 2448 (1998) | Fr D-line lifetimes / precision \(f\) |
| Simsarian96 | J. E. Simsarian *et al.*, *Phys. Rev. Lett.* **76**, 3522 (1996) | Fr MOT / isotope context |
| Falke06 | S. Falke *et al.*, *Phys. Rev. A* **74**, 032503 (2006) | K D-line / hyperfine |
| Volz96 | U. Volz, H. Schmoranzer, *Phys. Scr.* **T65**, 48 (1996) | Optional Li/Na/Rb lifetime-\(f\) overlays |

---

## Feshbach resonance parameters (per element)

| Element | Key papers |
|---------|------------|
| \(^6\)Li | Zürn *et al.*, *Phys. Rev. Lett.* **110**, 135301 (2013) [arXiv:1211.1512] |
| \(^{23}\)Na | Inouye *et al.*, *Nature* **392**, 151 (1998); Stenger *et al.*, *Phys. Rev. Lett.* **82**, 2422 & 4569 (1999); Knoop *et al.*, *Phys. Rev. A* **83**, 042704 (2011) |
| \(^{39}\)K | D'Errico *et al.*, *New J. Phys.* **9**, 223 (2007) [arXiv:0705.3036] |
| \(^{87}\)Rb | Marte *et al.*, *Phys. Rev. Lett.* **89**, 283202 (2002) [arXiv:cond-mat/0210651]; Chin *et al.* RMP (2010) Table IV |
| \(^{133}\)Cs | Berninger *et al.*, *Phys. Rev. A* **87**, 032517 (2013) [arXiv:1212.5584]; Chin *et al.* RMP (2010) |

See `data_json/<El>_feshbach.json` notes for resonance-by-resonance refs.

---

## Optional ARC levels (`arc_bridge.py`)

| Key | Citation | Used for |
|-----|----------|----------|
| Sibalic17 | N. Šibalić *et al.*, *Comput. Phys. Commun.* **220**, 319 (2017) | ARC alkali Rydberg library |
| Robertson21 | E. J. Robertson *et al.*, *Comput. Phys. Commun.* **261**, 107814 (2021) | ARC 3.0 divalent / extensions |

License: BSD-3-Clause. Install separately: `pip install ARC-Alkali-Rydberg-Calculator`.

---

## Software / tools (non-paper)

- **ORCA** quantum chemistry package (Neese *et al.*) - TD-DFT / CASSCF–NEVPT2+SOC jobs via `runorca.py`
- **NIST ASD** exports under `NIST Levels/` and `f_values/`
- **Plotly** - interactive HTML plots
