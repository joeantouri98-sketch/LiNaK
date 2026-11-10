# Changelog

Short notes on what changed in LiNaK. Newest first.

## Latest

### New
- **Rydberg shifts, series repair (Blackbody stage):** the Na g series no longer jumps. Three causes were fixed: level energies the quantum-defect model built are used at the formula's value instead of the 1e-6 eV rounding (the f-g gap at n = 33 is the size of that step); f-g pairs lost when two rounded energies tied (n = 34, 35, 41, 42, 44, 45) are restored; and an h (l = 5) series from the same model supplies the upward partners g states lacked (the g dynamic shift was -460 to +1400 Hz at n = 25 to 45 and now runs from 2342 to 2396 Hz, towards the 2417 Hz Farley-Wing limit; all g states now close the f-sum rule). `l = h` is added to the angular-momentum table. The data files are not changed; the repair is made when the shifts are computed. Rerun the Blackbody stage to get it.
- **Rydberg shifts, `static_valid`:** each state in `<El>_blackbody.json` now says whether the static Itano formula applies: false where the coupling to a near-degenerate partner exceeds 0.3 of its gap (second-order theory fails) or where over 10% of α(0) comes from partners below k_B T. Most Rydberg states fail, which is the reason the dynamic shift is the one to use there. The Blackbody page marks them and gives the reason; the numbers stay in the file.
- **Blackbody page** (Visualise > Blackbody page, `--pipeline blackbody_page`): the blackbody results on one tabbed page, written to `plots/<El>/<El>_blackbody_page.html`. Tabs: Shifts vs T (static and dynamic shift of the ground, excited and difference, with the correction as a percent of the static shift), Depopulation (Γ_BBR or τ_eff against n, with a Γ(T) curve for any state you click and a table of the channels behind each rate), Photoionization (direct PI, field ionization at a typed field, mixing PI and total ion rate at the run temperature; click a legend entry to hide that rate) and Rydberg shifts. Temperature, units (Hz or Δν/ν₀), n range and field change live; click the plot to move the cursor temperature. Tick which tabs to build; input options: blackbody, polarizability and lifetimes data; output option: output directory. The Blackbody stage keeps its own plots and its Generate plot tick. Tabs whose data are missing are left out with a message.
- **Polarizability page** (Visualise > Polarizability page, `--pipeline polarizability_page`): polarizability, magic wavelengths and optical tweezer on one tabbed page, written to `plots/<El>/<El>_response.html`. α(ω) is summed in the browser from the stored transitions, so units (a.u., Hz per W/cm², µK per kW/cm²), wavelength range, trap intensity and cursor wavelength change live; a searchable, paged table shows which lines make up α at the cursor, and magic wavelengths are marked by robustness (circle, diamond, square). The Polarizability and Optical tweezer stages now write data only (their separate plots and the Generate plot tick are gone from the GUI; the command line keeps `--no-plot`). Tick which tabs to build: Polarizability, Magic wavelengths (the differential polarizability Δα and the magic wavelengths), Optical tweezer. Input options: polarizability and lifetimes data; output option: output directory (and Save as).
- **License:** LiNaK is now released under GPL-3.0-or-later (was MIT). `LICENSE`, `pyproject.toml` and `CITATION.cff` are updated.
- **Grotrian diagram rebuilt:** one page with levels in columns by orbital l, J fine-structure bars, a filter rail (region, level class, f source, f grade, n, wavelength, minimum f, ground or excited lower level), line weight by f, click / shift-click / isolate, a matching-transitions table, right-click **View more** (all data for the transition, including the extrapolation model f with its 68% and 95% bands), ORCA overlay markers at their exact energies (no residual panel), light and dark theme, and the Plotly toolbar's camera button for an SVG export. The excited-to-excited Grotrian (Plot 3, `*_ee_grotrian.html`) and `--plot3-choice` are gone: the Excited → excited tick covers it. `--plots` is now 1 or 2.
- **Grotrian diagram:** an "Extra data to overlay" field takes several ORCA-format files (separated by ;, or Add files...) and draws each as its own overlay.
- **NIST f-value scale** table (Tables and variables > Lifetimes): Fr 7d3/2 and 7d5/2 start ticked. The NIST 7p-7d lines give 53.0 and 77.0 ns; with the scale the lifetimes come out at the measured 73.6 and 67.7 ns, and the lifetimes, polarizability and blackbody stages all use the scaled f-values. Untick the rows to get the NIST values back (rerun the stages). The two measured lifetimes are also in the known lifetimes, as a check.
- **Compare elements** now shows the table live in the viewer when it finishes: sortable columns grouped in sections (Atom, D1, D2, Natural, Doppler, Laser cooling), Remove selected, a temperature box that recomputes the Doppler columns, an Add box for more elements (they are added to the saved table, not replacing it), Export CSV, and Raw JSON. The HTML page is optional: tick **Generate HTML**.
- **Run > Tests...** opens a window to run all, selected or failed tests, with progress and details.
- **Edit > Preferences > Tables and variables** shows the built-in constants and data files, with units, and lets you edit them. Edits are saved in the `tables/` folder of the output folder, and Restore Defaults brings the originals back.
- **Add...** and **Remove** in Tables and variables for the open lists (known lifetimes, element masses, abundances, core polarizabilities, A_L and quantum-defect differences, precision f-values, and whole new elements in the D-line table). New rows are checked for typos and duplicates, can carry a source note, and never replace a built-in row.
- **Edit > Preferences > Extrapolator** page for the extrapolation settings.
- **Override** column for known lifetimes: tick a row (or Override all) to use that lifetime instead of the computed one in the output and plots. The computed value is kept in the JSON.
- **Hyperfine constants** can be entered in the set-up dialog (mass number, then A, B, C for each state).
- **Properties...** (right-click a species, or Edit > Species properties...) shows that element's built-in values and data files in one place, flags what is missing, and lets you add or edit it.
- **New species:** setting an atom with no bundled data (for example Mg) opens a dialog to import its NIST levels and lines and type its numbers, and shows which stages that enables.
- **Ctrl+F** find bar in the JSON viewer, the data editor and the tables page (F3 / Shift+F3 for next / previous).
- **Fr** now has extrapolated ns→np oscillator strengths, with its own derived parameters (`tools/fr_derivation/`).
- Extrapolation of ns→np oscillator strengths with a calibrated Bayesian ridge model, user anchors and an optional threshold anchor.

### Interface
- Stage options are grouped into Input, Output and Options sections, with path pickers and grey default hints.
- Compare, transitions and lifetimes inputs reworked; combo boxes open downward.
- Extrapolation dialog: wider plot by default, the example in Edit datasets is now a template you can copy, unreliable rows are shown unticked and disabled.
- Shorter hover texts everywhere.
- Fixes for Windows test failures (viewer stack, paths).

### Physics fixes
- Rydberg: neutral atoms that are not alkalis always use the NIST levels (no quantum-defect model).
- Delete species can also remove the NIST files you imported for it.
- Scattering rate: an element you added is no longer dropped from the list.
- Lifetimes: A coefficients were counted twice; fine-structure 6j factors; J-resolved NIST multiplets rebuilt; Ritz levels used; literature values for K, Rb and Cs anchors.
- Numerov: grid fix and a gap-dependent scale; Fr calibration fixed.
- Extrapolation: Fr NIST rows for 7s→np (n ≥ 8) are a plain 2:1 split and are excluded.
- Polarizability: uncertainty (sigma) now propagates correctly.
- Blackbody: guards against bad input.
- Hyperfine: Breit-Rabi and octupole terms corrected.
- Isotope default is chosen by natural abundance.
- Tweezer: scattering rate (Gamma) corrected.
- C6 (Tang) formula, Feshbach grouping, sigma Lorentzian, hybrid Coulomb f, clock transition, D-line table and nuclear data fixed.
- Plot fixes.

### Other
- README updated; tests added for the extrapolator, Fr, find bar, tables, test runner.
