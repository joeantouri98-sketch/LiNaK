<p align="center"><img src="LiNaK.png" alt="LiNaK" width="180"></p>

# LiNaK

LiNaK is a free desktop app and Python toolkit for calculating and exploring the properties of alkali atoms (lithium to francium).

Pick an atom, run the calculations you need, and browse the results as interactive plots. No programming is required.

## What you can calculate

- Energy levels and Rydberg series
- Transitions, oscillator strengths and lifetimes
- Polarizability, C6 coefficients and magic wavelengths
- Blackbody radiation shifts
- Optical tweezer properties
- Extrapolated ns→np oscillator strengths for high Rydberg levels (including Fr)
- Hyperfine structure and Zeeman diagrams
- Feshbach resonances
- Grotrian diagrams, absorption and emission spectra (vapour cells, hyperfine D lines, saturated absorption), 3D orbitals and element comparisons

Reference data from the NIST Atomic Spectra Database is included. You can also add results from the [ORCA](https://www.faccts.de/orca/) quantum chemistry program if you have it installed.

## What is supported

| Species | Support |
|---|---|
| Li, Na, K, Rb, Cs, Fr | Full set of calculations. Fr has no Feshbach data. |
| Ions such as Na⁺ (written `Na_c1`) | Levels, transitions, lifetimes, plots and ORCA |
| Fe and Cr | Levels, transitions, lifetimes, plots and ORCA |
| Other atoms such as Mg | Set the species and import its NIST levels and lines in the dialog that opens. The stages that then work are listed there. |

Example results for Li, Na, K, Rb, Cs, Fr and Na⁺ are included, so you can explore the app straight away.

## Install

You need [Python](https://www.python.org/downloads/) 3.12, 3.13 or 3.14, [Git](https://git-scm.com/downloads) and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/joeantouri98-sketch/LiNaK.git
cd LiNaK
uv sync
```

Then start the app from the `LiNaK` folder:

```bash
uv run linak-gui
```

The first start copies the example data into your output folder (about 340 MB), so the lists may take a moment to fill.

If you prefer not to use uv, you can run `python -m pip install .` inside a virtual environment and then start the app with `linak-gui`.

## Using the app

1. Click a species at the top left, or type one such as `K` or `Na_c1` and press Enter.
2. Choose a stage from the **Pipeline** list. A filled dot means that stage already has results.
3. Adjust its settings in the **Options** tab on the right.
4. Press **Run stage** or F5, and follow the progress in the **Console**.
5. Open your results from the **Plots** and **Data** tabs. They appear in the **Viewer**.
6. Open the **Overview** tab for a summary of everything calculated for that species.

Work through the Pipeline list from top to bottom, because later stages use the results of earlier ones. The ORCA stages at the top are optional. Most stages finish within a few minutes, but ORCA calculations can take hours.

### Handy features

- **Themes:** choose Dark, Light or Follow system under **View > Theme**.
- **More room for plots:** drag a side panel or the Console panel to the window edge to collapse it, or use the shortcuts below.
- **Move the Console panel:** drag its tab bar or press its ⇅ button to place it above or below the viewer.
- **Delete a species:** right-click it and choose whether to remove it from the list only or delete its data as well. Deleted files go to the Recycle Bin. Type the name again to bring the species back.
- **Edit data:** create or open JSON and CSV datasets from the **Edit** menu.
- **Search:** press Ctrl+F in the viewer or the data editor to find text.
- **Built-in tables:** **Edit > Preferences > Tables and variables** shows the constants and data files that ship with LiNaK, with units, and lets you change them and add rows (a new lifetime, a new element). Tick **Override** on a known lifetime to use it instead of the computed one. Your changes are saved in the `tables/` folder of your output folder, and Restore Defaults brings the originals back.
- **Species properties:** right-click a species and choose **Properties...** to see, edit or add the built-in values and data files for that element. **Set up missing data...** there opens the same dialog that appears when you set a species LiNaK has no data for.
- **Extrapolation:** **Edit > Preferences > Extrapolator** sets the model used for ns→np oscillator strengths.
- **Run the tests:** **Run > Tests...** runs all, selected or failed tests with progress.
- **Help:** the **Help** menu explains every stage and option.

| Shortcut | Action |
|---|---|
| F5 | Run the selected stage |
| Shift+F5 | Stop the running stage |
| Ctrl+B | Show or hide the Pipeline panel |
| Ctrl+\ | Show or hide the Options panel |
| Ctrl+\` | Show or hide the Console panel |
| Ctrl+Shift+\` | Move the Console panel above or below the viewer |
| Ctrl+F | Find in the viewer or data editor (F3 / Shift+F3 for next / previous) |

## Where your results are saved

By default, LiNaK saves everything in your user data folder:

| System | Folder |
|---|---|
| Windows | `%LOCALAPPDATA%\LiNaK` |
| macOS | `~/Library/Application Support/LiNaK` |
| Linux | `~/.local/share/LiNaK` |

To save somewhere else, use **File > Open output folder...** LiNaK remembers your choice. **File > Use default output folder** switches back.

Inside the output folder:

| Folder | Contents |
|---|---|
| `data/` | Results as JSON files |
| `plots/` | Interactive HTML plots, which also open in any web browser |
| `orca_outputs/` | Raw ORCA runs |
| `tables/` | Your edits to the built-in tables (only if you make any) |

If you choose the LiNaK source folder itself, results go into a hidden `.linak` folder inside it.

## Optional: ORCA

ORCA is a separate program that you install yourself. LiNaK works without it, but it is needed for the ORCA stages. It finds ORCA in either of these ways:

- the ORCA folder is on your system `PATH`
- the `ORCA_EXE` environment variable holds the full path to the ORCA program

Restart the app after changing either one. Heavy elements such as Fr also need an extra package:

```bash
uv sync --extra basis
```

## Running the tests

From the `LiNaK` folder:

```bash
uv sync --group dev
uv run pytest
```

Or use **Run > Tests...** in the app.

## Using the command line

Everything in the app is also available from the terminal. Start each command from the `LiNaK` folder:

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
uv run linak --pipeline compare_elements          # add --html for the HTML page too
uv run linak --pipeline scattering_rate --element Na --line D2
```

To see every option for a stage:

```bash
uv run linak --pipeline rydberg --help
```

All stage names:

`runorca`, `orca_to_json`, `dhf`, `brueckner`, `rydberg`, `transitions`, `lifetimes`, `polarizability`, `tweezer`, `blackbody`, `hyperfine`, `feshbach`, `plotinteractive`, `polarizability_page`, `spectra`, `orbital3d`, `compare_elements`, `scattering_rate`

ORCA examples:

```bash
uv run linak --pipeline runorca Na
uv run linak --pipeline runorca Na --mode soc --soc-recipe nevpt2
```

To set the output folder for one command on macOS or Linux, use `LINAK_PROJECT_ROOT`:

```bash
LINAK_PROJECT_ROOT=/path/to/folder uv run linak --pipeline rydberg Na
```

## Good to know

- Energies in the Rydberg results are measured from the ionization limit, so that limit is 0 and the ground state is negative.
- Do not edit generated JSON files by hand. Run the stage again instead.
- The optional [ARC](https://arc-alkali-rydberg-calculator.readthedocs.io/) bridge exports energy levels only and never overwrites your NIST results. Install it with `uv sync --extra arc`, then run `uv run python -m linak.integrations.arc Na --n-max 50`.

## Troubleshooting

| Problem | What to try |
|---|---|
| The ORCA stage says ORCA was not found | Check the [ORCA section](#optional-orca), then restart the app. |
| A species or result is missing | Check **Output folder** in the status bar. LiNaK reads only the current folder. |
| Overview says data is unavailable | Run the missing stage. Overview shows saved results and does not calculate anything. |
| A stage reports missing data | Run the earlier stages first. |

If that does not help, please [open an issue](https://github.com/joeantouri98-sketch/LiNaK/issues) with the species, what you ran and the full error message.

## Learn more

- [Pipeline Overview](src/linak/resources/HTMLs/Pipeline%20Overview.html): the physics, derivations and all command-line options
- [References](REFERENCES.md): data sources and which papers to cite
- [Architecture](ARCHITECTURE.md): how the code is organised
- [Contributing](CONTRIBUTING.md): reporting bugs, making changes and building the app

## Citing LiNaK

If you use LiNaK in a paper or thesis, please cite:

Youssef Antoury, *LiNaK: an alkali-atom AMO pipeline* (2026). DOI: [10.5281/zenodo.21821149](https://doi.org/10.5281/zenodo.21821149)

Please also cite the sources of any numbers you quote, such as NIST ASD. [REFERENCES.md](REFERENCES.md) lists what to cite, and [REFERENCES.bib](REFERENCES.bib) has the BibTeX entries. If you report ORCA or ARC results, cite those programs too.

## License

LiNaK is released under the [GNU General Public License v3.0 or later](LICENSE) (GPL-3.0-or-later). Plotly.js, bundled for the offline plots, is MIT-licensed. ARC (BSD-3-Clause) is an optional dependency. ORCA is separate software with its own license.

## Contact

Youssef Antoury: [Joeantouri98@gmail.com](mailto:Joeantouri98@gmail.com)

Results for Fe and Cr are not included. If you need them, feel free to get in touch.
