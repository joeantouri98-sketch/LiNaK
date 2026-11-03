# LiNaK — Known physics issues

The **physics / physical-data** defects found while reviewing the code (2026-10-17 and
2026-10-20). None of them has been fixed: every item would move a computed or displayed
number, so each needs a deliberate change of its own (cite the source, add a
reference-value test, and list the moved numbers — see `CONTRIBUTING.md`).

> **Path note:** line-number citations are as of the review date and may have shifted.

**How to read the entries:** each item lists the code evidence, the impact, and a
*suggested* fix (for a future, deliberately-scoped change). "Verified" means the claim was
confirmed by reading the source and/or arithmetic; "reported" means it came from static
review and has not been reproduced end-to-end.

---

## A. Wrong or ambiguous physical results / conventions

### P1 — Mean scattering length `ā` is 2× too large  *(Verified)*
**Evidence:** `src/linak/physics/feshbach.py:197`
```python
def a_bar_a0(beta6):
    """Mean scattering length ā ≈ 0.956 β₆ (Gribakin–Flambaum)."""
    return A_BAR_FACTOR * beta6      # A_BAR_FACTOR = 0.955978, constants.py:87
```
but `beta6_a0()` (`feshbach.py:185`) defines **β₆ = (2μC₆/ℏ²)^¼**:
```python
beta_m = (2.0 * mu * C6_SI / HBAR_SI**2) ** 0.25
```
For that β₆ the Gribakin–Flambaum mean scattering length is **ā = 0.477989 β₆**. The
0.955978 factor applies to R_vdW = β₆/2, i.e. exactly a factor of 2.

**Cross-check (arithmetic):** β₆(Rb) = 165 a₀ (literature R_vdW = 82.9 a₀), so ā should
be ≈ 79 a₀; the code yields ≈ 158 a₀. Shipped
`src/linak/resources/data_json/Na_feshbach_out.json` has `beta6_a0: 89.8894` and
`a_bar_a0: 85.9323`; it should be ≈ 42.97 a₀.

**Impact:** `a_bar_a0` is written to the Feshbach JSON and drawn in the plot/annotations.
It does **not** feed `a(B)` (see `compute_element`, `feshbach.py:279-321`), so only the
reported/plotted `ā` is affected — not the resonance curves.
**Suggested fix:** set `A_BAR_FACTOR = 0.477989`, correct the comment at
`constants.py:87`, and regenerate the `*_feshbach_out.json` artifacts.

---

### P2 — Default isotope choice picks the *minority* isotope  *(Verified)*
**Evidence:** `pipelines/hyperfine.py` `get_isotope`
```python
# comment: "...K-39 (93.3%), Rb-87 (28%) or Rb-85 (72%)..."
# "Heuristic: pick the stable isotope with highest A (often most abundant for alkalis)"
return stable[-1]
```
`src/linak/resources/data_json/nuclear_data.json` marks K-39 **and** K-41 as stable, so
`stable[-1]` is **K-41 (6.7 %)**; for Rb it returns **Rb-87 (28 %)** instead of Rb-85
(72 %).

**Impact:** I, μ_I, Q (hence A, B, and the clock frequency) are all derived from the
selected isotope, so the default result is silently wrong for K and Rb. Overridable via
`--isotope`, but the default is wrong.
**Suggested fix:** use an explicit abundance table (`nuclear_data.json` has no abundance
field) or per-element preferred masses from `constants.ALKALI_MASS_AMU`.

---

### P3 — Cycling transition is never computed  *(Verified)*
**Evidence:** the old `identify_special_transitions` looked for F_max → F_max+1 in a
transition list it was only ever called with as `[]`, so `special['cycling']` was never
set. That unreachable branch was removed when the physics moved to
`physics/hyperfine_core.py` (outputs unchanged); only `microwave_clock` remains.

**Impact:** the cycling transition, once advertised in the pipeline docstring, is absent
from the printed summary and the output JSON.
**Suggested fix:** compute it in `hyperfine_core.compute` from the ground state's and the
first excited state's F levels, and add it to `<El>_hyperfine.json` under `special`.

---

### P4 — Breit–Rabi quadrupole is not exact and disagrees with `hyperfine_energy`  *(Verified)*
**Evidence:** `physics/hyperfine_core.py` `breit_rabi_energies` (docstring "Exact
Breit–Rabi diagonalisation for arbitrary B field"), but the quadrupole is added
diagonally only in the uncoupled `|m_I,m_J⟩` basis:
```python
H_hf[r, r] += B_MHz_quad * (3*u*u + 1.5*u - I*(I+1)*J*(J+1)) / denom
```
The true quadrupole operator is rank-2 with Δm_J = ±1, ±2 couplings (the inline comment
there acknowledges the approximation). The numerator also differs from the one used by
`hyperfine_energy`:
```python
E += B_MHz * (1.5*K*(K+1) - 2*I*(I+1)*J*(J+1)) / denom_IJ
```
**Impact:** for any state with J ≥ 1 and B ≠ 0 (e.g. Na 3p3/2, B = 2.724 MHz), the B = 0
eigenvalues from `breit_rabi_energies` will not reproduce the `E_hf` values used by the
main hyperfine plot and JSON — the two outputs are inconsistent.
**Suggested fix:** implement the full quadrupole operator, or use one shared formula and
document the diagonal-only approximation.

---

### P5 — α(iξ) omits the transition sign (latent)  *(Verified)*
**Evidence:** `src/linak/physics/polarizability_core.py:1341-1349`
```python
def alpha_imag_freq(xi_au, transitions_list):
    for tr in transitions_list:
        dE    = abs(tr['delta_E_au'])
        ...
        result += f / denom          # <-- no 'sign'
```
whereas `alpha_real` uses `result += sgn * f / denom` (`:1332`).

**Impact:** for a ground state all signs are +1, so `compute_C6` (which uses
`transitions_gs`) is currently fine. But `alpha_imag_freq` is presented as shared core
(module docstring `:4-5` advertises reuse by BBO/tweezer) and would add instead of
subtract for any state with downward transitions.
**Suggested fix:** multiply by `tr['sign']` to match `alpha_real`.

---

### P6 — α / C₆ docstring formula is wrong  *(Verified — documentation only)*
**Evidence:** `src/linak/physics/polarizability.py:14-15` and `:27-29`
```
α_n(ω) = (2/3) Σ_k f_nk · (E_k - E_n) / [(E_k - E_n)² - ω²]
α_n(iξ) = (2/3) Σ_k f_nk · (E_k - E_n) / [(E_k - E_n)² + ξ²]
```
The code is the standard f-sum form (no `(2/3)` and no `(E_k - E_n)` numerator):
`polarizability_core.py:1327` `result += sgn * f / (dE*dE)` and `:1332`
`result += sgn * f / denom`.
**Impact:** documentation only; readers may attempt a wrong formula.
**Suggested fix:** correct the docstring to `α_n(ω) = Σ_k sgn·f_nk / ((E_k-E_n)² - ω²)`.

---

### P7 — Ground-state clock interval omits B/C  *(Verified, generically wrong)*
**Evidence:** `physics/hyperfine_core.py` `microwave_clock`
```python
dE = (hyperfine_energy(F1, I, state.J, state.A_MHz)
      - hyperfine_energy(F2, I, state.J, state.A_MHz))
```
Only the magnetic-dipole constant A is passed (no `B_MHz`, `C_kHz`).
**Impact:** correct for J = 1/2 ground states (the alkalis here), wrong if a J > 1/2
ground state is ever supplied. The label "Zero-field ground ΔF=1 interval"
(`CLOCK_NOTE`) makes the limitation easy to miss.
**Suggested fix:** pass B/C through, or document the J = 1/2 assumption.

---

### P8 — f-source provenance can be mislabelled  *(Verified)*
**Evidence:** `pipelines/plotinteractive/spectrum.py` `plot_spectrum`
```python
_src = _SOURCE_GROUPS.get(_t.get('fosc_source', ''), 'Coulomb')
```
`_SOURCE_GROUPS` maps `'unknown' -> 'Coulomb'`, and the default for any unrecognised
source is `'Coulomb'`.
**Impact:** unlabelled/approximate channels are presented with a specific provenance in
plots, which can mislead about data quality.
**Suggested fix:** add an explicit `'unknown'` group instead of defaulting to `'Coulomb'`.

---

### P9 — Recoil-velocity unit mismatch  *(Verified; currently dead code)*
**Evidence:** `src/linak/pipelines/scattering_rate.py:107`
```python
'vrec': d.get('v_rec_ms', 1.0),      # metres/second
...
def recoil_vel(el, line):
    return ELEMENTS[el][line].get('vrec', 1.0)  # comment claims cm/s
```
The `FALLBACK` table stores cm/s values, while `comparison_table.json` writes
`v_rec_ms`; `compare_elements.py:376-377` also writes `v_rec_cms = v_rec_ms*100`.
**Impact:** a "v_rec (cm/s)" column would be 100× too small when sourced from the pipeline
table. Currently inert because `recoil_vel` / `F_rad_aN` / `accel_km_s2` are unused.
**Suggested fix:** read `v_rec_cms`, or drop the dead functions.

---

### P10 — Resonance sign convention is undocumented  *(Verified)*
**Evidence:** `src/linak/physics/feshbach.py:155` comment: "Δ > 0 ⇒ a → +∞ just above B₀".
`scattering_length` (`:202`) uses `a_bg(1 − Δ/(B−B₀))`, so the sign of each curated `Δ`
in `*_feshbach.json` is load-bearing and not validated per resonance.
**Impact:** a wrong-signed `Δ` in a hand-curated file flips the curve silently.
**Suggested fix:** validate `Δ > 0` on load (or document the schema).

---

## B. Data-quality handling that changes physical outputs

### P11 — NIST ACC uncertainties are computed then discarded  *(Verified)*
**Evidence:** `polarizability_core.py` computes `ACC_UNCERTAINTY`-based `nist_sigma`,
`nist_sigma_j`, `sigma_line` (e.g. `:354`, `:542`, `:559-560`, `:587-588`, `:601-640`,
`:776`) but `load_element` returns only `nist_fosc`, `nist_acc`, `nist_fosc_j`,
`nist_acc_j` (`:943-946`) — **no** `nist_sigma`. `build_transitions_for_state` stores only
`delta_E_au, fosc, sign, other_label, source, wl_nm` (`:1272-1279`).
**Impact:** α(ω) and C₆ carry no propagated uncertainty; the ACC rating is decorative.
For `_nist_Ju_fik` entries with `_A_avg <= 0`, `nist_sigma` is set to `0.0` (`:634`),
i.e. an artificially perfect f.
**Suggested fix:** thread σ through `get_fosc` / `build_transitions_for_state`, or remove
the unused machinery to avoid implying it is used.

---

### P12 — Silent f-value mutations change Σf  *(Verified)*
**Evidence:**
1. A multiplet transition is **deleted** when it cannot be rebuilt from lines and neither
   shell is `s` (`polarizability_core.py:763-767`):
   ```python
   elif _ml.group(1) != 's' and _mu.group(1) != 's':
       del nist_fosc[_key]
   ```
2. Two published `fik` for the same key are **averaged** (`:358`):
   ```python
   fosc_line[key] = 0.5 * (fosc_line[key] + _fik)
   ```
**Impact:** both silently alter Σf (hence α). Only a count is printed for case 1
(`n_mult_dropped`); case 2 is unreported.
**Suggested fix:** log the affected keys/values, or make the policy configurable.

---

## C. Numerical edge cases inside physics code

### P13 — Trap frequency uses the average atomic weight, not the isotope mass  *(Verified)*
**Evidence:** `src/linak/physics/polarizability.py:493`
```python
mass_amu = ATOMIC_MASS_AMU.get(element)   # IUPAC average weight
```
`constants.py` provides `ISOTOPE_MASS_AMU` and `ALKALI_MASS_AMU` (`:127-140`).
**Impact:** for Li/Rb this gives a percent-level error in trap frequencies (∝ m^(−1/2)).
**Suggested fix:** prefer the isotope mass when known.

---

### P14 — Default excited-state selection can pick a bogus state  *(Verified)*
**Evidence:** `src/linak/physics/polarizability.py:455-459`
```python
def sort_key(lbl):
    E = data['state_energy_au'].get(lbl, -999)
    return E
candidates.sort(key=sort_key)
excited_label = candidates[0]
```
Missing energies default to −999, so an unknown-energy candidate sorts to the front and
is chosen as the "lowest" excited state.
**Suggested fix:** skip candidates with no known energy (or use `math.inf`).

*Status as-of 2026-10-19:* the *encoding* changed, not the behaviour — the
pipeline sites no longer carry the inline `−999` literal (absent energies read as
`None`, and `polarizability_core.unknown_energy_sort_key` maps them into the named
−999 bucket `UNKNOWN_ENERGY_AU`), but the ordering and the resulting state choice
are reproduced exactly (pinned by
`tests/test_deferred_targets.py::test_unknown_energy_sort_reproduces_the_minus999_ordering`).
The defect itself remains unfixed.

---

### P15 — Division by zero at T = 0 in the BBR summary  *(Verified)*
**Evidence:** `pipelines/blackbody.py` `_print_shifts`
```python
f"(dynamic correction: {corr_gs_T:+.5f} Hz, {corr_gs_T/sh.stat_gs_hz*100:+.3f}%)"
```
The static shift is ∝ T⁴ (from the `T_REF` scaling), so `--T 0` raises `ZeroDivisionError`.
Similarly the Farley–Wing ratio in `_print_rydberg` divides by `fw`, which is 0 at T = 0.
**Suggested fix:** guard `shift_gs_T != 0` (and `fw != 0`) before formatting.

---

### P16 — Negative C₆ yields a complex number  *(Verified)*
**Evidence:** `src/linak/physics/feshbach.py:192-193`
```python
C6_SI = C6_au * AU_TO_SI_C6
beta_m = (2.0 * mu * C6_SI / HBAR_SI**2) ** 0.25
```
If `load_C6_au()` returns a negative value, `(negative)**0.25` is a Python `complex`, which
then breaks `round(result['beta6_a0'], 4)` (`:483`) and plot formatting.
**Suggested fix:** reject `C6_au <= 0` at load with a clear error.

---

### P17 — Unguarded division by `lifetime_s` and float sentinels  *(Verified)*
**Evidence:**
- `src/linak/pipelines/compare_elements.py:206` divides by `(2*sigma_m2*tau_s)`; a
  `lifetime_s == 0.0` (read at `:324`) raises `ZeroDivisionError`.
- `physics/hyperfine_core.py` `compute_state` uses `A_MHz == 0.0` as a "not in file"
  sentinel; a genuine A = 0 would mis-trigger the extrapolation path. Same pattern for
  `B_MHz == 0.0` and `C_kHz == 0.0` in `hyperfine_energy` and `breit_rabi_energies`.
**Suggested fix:** use explicit `None`/key-presence checks instead of `0.0`.

*Status as-of 2026-10-20:* a state missing from the constants file is now explicit
(`constants.get(label)` is None), and absent B/C read as 0.0 in `HfConstants.from_json`;
the `A_MHz == 0.0` extrapolation trigger is kept bug-compatible, so the mis-trigger for a
genuine A = 0 remains (deliberately unfixed). The `compare_elements.py:206` division by
`lifetime_s` is untouched.

---

### P18 — The two hardcoded D-line tables disagree  *(Verified)*
**Evidence:** the D-line literature fallback data is declared twice with different
shapes *and* different numbers:

- `spectra.py` `DLINES_FALLBACK` — per-element list of transitions
  `{wl, fosc, label, src: "lit"}` (spectral-bar viewer fallback).
- `compare_elements.py` `LITERATURE` — per-element parameter record
  (`D1_nm`/`D2_nm`, `tau_D1_ns`/`tau_D2_ns`, `fosc_D1`/`fosc_D2`, g factors,
  quantum defects, SOC, IE_eV) for the cross-element comparison table.

Both are presented as "literature" values, yet for the same transitions:

| Element | Quantity | `DLINES_FALLBACK` | `LITERATURE` |
|---|---|---|---|
| Li | D-line wavelength | 670.776 nm | 670.976 (D1) / 670.962 (D2) nm |
| Li | fosc | 0.4956 | 0.2490 (D1) / 0.4980 (D2) |
| K | fosc_D2 / fosc_D1 | 0.6761 / 0.3380 | 0.6660 / 0.3320 |
| Rb | fosc_D2 / fosc_D1 | 0.6944 / 0.3472 | 0.6958 / 0.3424 |
| Cs | fosc_D1 | 0.3571 | 0.3435 |
| Fr | D2 / D1 wavelength | 718.183 / 817.165 nm | 718.2 / 817.2 nm |
| Fr | fosc_D2 / fosc_D1 | 0.5000 / 0.3000 | 0.736 / 0.340 |

(Na D2/D1 fosc differs only in the 4th digit, 0.6408/0.3204 vs 0.6410/0.3200, and
Cs D2 0.7143 vs 0.7142 — rounding-level.)

**Impact:** the two reports can show different numbers for the same transition
(e.g. the Fr D2 oscillator strength differs by ~1.5×, Li's D-line wavelength by
~0.2 nm) with nothing but `src: "lit"` to distinguish the provenance.

**Status as-of 2026-10-19:** the tables are single-sourced per dataset into
`src/linak/core/spectroscopy.py` (`DLINES_FALLBACK` + `LITERATURE`) with values
preserved byte-identical. The *numerical*
disagreement above is deliberately unresolved — reconciling the two datasets is
a physics change and stays deferred.

**Suggested fix:** pick one authoritative source set (e.g. the Steck D-line data
sheets already cited by `LITERATURE`) and reconcile both consumers in a single
deliberately physics-scoped change, then regenerate the affected outputs.

---

## D. Display-consistency findings from the report-template migration

### P19 — The two region colour palettes disagree  *(Verified)*
**Evidence:** `pipelines/plotinteractive/common.py` and `grotrian3d.py` declare two palette
maps for the same spectral-region keys:
```python
REGION_COLORS = {          # arrows in plots 1/3 + the JS filter panels
    'Vacuum UV':  '#8B00FF',  'UV':        '#4B0082',
    'Visible':    '#FF6600',  'Near-IR':   '#CC0000',
    'Mid/Far-IR': '#888888',  'Microwave': '#AAAAAA',
}
_REGION_COLORS4 = {        # level spheres/edges in plot 4 (3-D diagram)
    'Vacuum UV':  '#7B00FF',  'UV':        '#3a0ca3',
    'Visible':    '#f77f00',  'Near-IR':   '#d62828',
    'Mid/Far-IR': '#9aa4b8',  'Microwave': '#c8d0e0',
}
```
Only X-ray (`#9400D3`) matches; the other six regions render in different colours
between the 2-D Grotrian arrows/report panels and the 3-D diagram.

**Impact:** the same spectral region is drawn in two different colours across the
plots, so a user cross-reading plot 1 and plot 4 may mis-map regions.

**Status as-of 2026-10-19:** the *display order* of the regions is now
single-sourced (`_REGION_ORDER` + `_region_order_js()` feeding the three panel
templates), but the two colour maps are kept separate — merging them would change
the plot-4 output bytes/colours. The defect remains unfixed.

**Suggested fix:** pick one palette in core, use it in both places, and regenerate
the four interactive reports.

---

### P20 — Embedded JS kB mirror disagrees with `KB_EV`  *(Verified)*
**Evidence:** the spectrum filter panel re-broadens lines in the browser and embeds
its own copy of the Boltzmann constant:
```js
// template plotinteractive_spectrum_panel.js
const kB = 8.617333e-5;   // eV/K
```
while the Python-side Doppler width (`physics/spectrum_core.py` `doppler_sigma_nm`) uses
`constants.KB_EV = 8.617333262e-5` (`constants.py:46`).

**Impact:** only the browser-recomputed "Filtered" curves are affected; the
pre-rendered overview traces match the Python exactly. The relative deviation is
≈ 5 ppm in the Doppler width — invisible in practice but inconsistent with the
declared single-source rule for embedded mirrors.

**Status as-of 2026-10-19:** the literal was moved verbatim into the
template (`plotinteractive_spectrum_panel.js`) and is **not** routed through
`js_constant("KB_EV")` — that would change the emitted bytes and the browser
recomputation. The mirror stays pinned at the historical (2019 CODATA) value.

**Suggested fix:** in a deliberately physics-scoped change, embed the `KB_EV`
serialization and regenerate the spectrum report.

---

## E. Findings from the 2026-10-20 review (recorded, not fixed)

Found while reviewing the code for non-physics bugs. Nothing below was changed: every
item moves a computed or displayed number. Line numbers are as of 2026-10-20.
"Verified" means the formula was checked against the cited source and, where stated,
the pipeline was run.

### P21 — Tweezer scattering rate uses a frequency-independent damping  *(Verified, run)*
**Evidence:** `pipelines/tweezer.py:196-226,253-258` builds Im α with damping Γ·ω and a
constant Γ. The Lorentz-oscillator result needs Γ_ω = (ω/ω₀)²Γ (Grimm, Weidemüller &
Ovchinnikov, *Adv. At. Mol. Opt. Phys.* 42, 95 (2000), Eqs. 2, 10).
**Impact:** Na, 1064 nm, 50 kW/cm²: R_sc = 0.105 s⁻¹ vs 0.0322 s⁻¹ from Grimm Eq. 10 (D1+D2),
a factor (ω₀/ω)² ≈ 3.3; heating rate and τ_heat inherit it.
**Suggested fix:** damping Γω³/ω₀² in the imaginary part.

### P22 — Peak resonant cross-section is π/2 too large  *(Verified, arithmetic)*
**Evidence:** `physics/lifetimes_core.py` `lifetime_of`: `σ = (λ²/8π)(g_u/g_l)·A/Δν_nat`. A normalized
Lorentzian peaks at 2/(πΔν), not 1/Δν (Foot, *Atomic Physics*, ch. 7); with Δν = A/2π
the code gives (g_u/g_l)λ²/4 instead of (g_u/g_l)λ²/2π.
**Impact:** every `sigma_peak_*` is ×π/2 and every derived `I_sat` ×2/π (Na 3p3/2:
5.98 vs 9.39 mW/cm²); the Doppler-broadened values scale from it.

### P23 — Heteronuclear C₆ combining rule has the wrong dimensions  *(Verified, run)*
**Evidence:** `pipelines/polarizability.py:199-209` returns
2α_Aα_B C₆^A C₆^B / (α_B C₆^A + α_A C₆^B), which has units of α·C₆. Tang, *Phys. Rev.* 177,
108 (1969): C₆^AB = 2α_Aα_B C₆^A C₆^B / (α_B² C₆^A + α_A² C₆^B).
**Impact:** the `--heteronuclear` C₆ matrix; for A = B the code returns α·C₆
(Na–Na 2.53×10⁵ au instead of 1556).

### P24 — "Combined" Feshbach a(B) multiplies resonances from different channels  *(Verified)*
**Evidence:** `physics/feshbach.py:83-106`, `pipelines/feshbach.py:222-224` multiply every
resonance in the file with `resonances[0]`'s a_bg. The product form (Lange *et al.*,
*PRA* 79, 013622 (2009)) holds per s-wave entrance channel only. With the bundled data
Na's |1,−1⟩ 1195 G pole lands in the |1,1⟩ curve; K mixes three channels; Li-6 mixes
|12⟩, |13⟩, |23⟩; Cs folds in d- and g-wave entries.
**Impact:** `a_combined_a0`, the main JSON output and plot.

### P25 — Elastic cross-section is 2× low for identical bosons  *(Verified)*
**Evidence:** `physics/feshbach.py:75-80` σ = 4πa²/(1+k²a²) (distinguishable particles).
Identical bosons in one internal state (all bundled Na/K/Rb/Cs channels) need 8πa²
(Pethick & Smith, *Bose–Einstein Condensation in Dilute Gases*, §5.1).

### P26 — `coulomb_fosc` fails the hydrogen limit; two different copies  *(Verified, run)*
**Evidence:** `physics/polarizability_core.py:128` and `physics/lifetimes_core.py` (the two
functions share a name but are not identical). At δ = 0 a Coulomb approximation must be
exact: 1s–2p 0.593 (exact 0.4162), 2p–3d 0.237 (0.6958), 3d–4f 0.183 (1.0175)
(Bethe & Salpeter, Table 13).
**Impact:** last-resort f-values, and every QDT-tail partner up to n = 200 in
`physics/blackbody_core.py` `discrete_tail_partners`.

### P27 — Fr precision D-line f-values are bypassed in LS mode  *(Verified, run)*
**Evidence:** `physics/polarizability_core.py:1197-1240`: `_ls_state_term` never reads
`PRECISION_FOSC_J` (`:50-65`, used at `:1086-1090` otherwise); LS mode is on for every
bundled alkali. Fr 7s: f(7p1/2) = 0.322 (NIST) instead of 0.340; α₀ = 310.0 vs 315.0 au.
`lifetimes` does use the precision values, so the two pipelines disagree.

### P28 — Non-LS path omits g_l/g_u on downward channels  *(Verified)*
**Evidence:** `physics/polarizability_core.py:1297-1305`. Emission f_nk = −(g_k/g_n) f_kn
(Mitroy, Safronova & Clark, *J. Phys. B* 43, 202001 (2010)); the LS path has the factor.
**Impact:** ions, species without a lines file, and metals (forcing LS off, Na 3p3/2→3s
gives 0.641 instead of 0.3205).

### P29 — Quantum defects: one n-independent δ_l and R∞ instead of R_M  *(Verified)*
**Evidence:** `physics/rydberg.py:590-654`: an inverse-variance mean δ_l dominated by low
n, with R∞ = 13.6057 eV rather than the reduced-mass R_M (−2.4×10⁻⁵ relative for Na).
Rydberg–Ritz δ₀ + δ₂/(n−δ₀)² with R_M is standard (Gallagher, *Rydberg Atoms*, ch. 2).
**Impact:** Na δ_s = 1.35052 vs 1.34838 at 12s; QDT energies ~9 GHz off at 13s.

### P30 — Magnetic-octupole hyperfine term is wrong  *(Verified, latent)*
**Evidence:** the octupole term of `physics/hyperfine_core.py` `hyperfine_energy` is not
proportional to Steck's Eq. 16 / Armstrong for I = J = 3/2. Latent: every bundled C = 0.
The engine's module docstring shows half of the B-term numerator (the code is right
there).

### P31 — Anions get a Z = 2 Rydberg scale  *(Verified)*
**Evidence:** `pipelines/rydberg.py` `z_core = abs(charge) + 1` gives anions Z = 2 (4 Ry);
it should be charge + 1 = 0, i.e. no Rydberg series.

### P32 — Wrong ground-state multiplicities for La, Ce, Ac  *(Verified)*
**Evidence:** `core/elements.py`: La 4 (should be 2, ²D₃/₂), Ce 3 (1, ¹G°₄), Ac 4 (2,
²D₃/₂) per NIST ASD. These feed the ORCA multiplicity.

### P33 — Rounded physical constants next to the exact ones  *(Verified)*
**Evidence:** hc = 1239.84 eV·nm in `physics/rydberg.py`, `physics/transitions_core.py`
(`HC_EV_NM_ROUNDED`), `pipelines/plotinteractive/` (several sites) and
`integrations/arc.py:57` (`HC_EV_NM = 1239.84` shadows `core.constants.HC_EV_NM =
1239.841984`); c = 3e8 in `physics/hyperfine_core.py` (`microwave_clock`) and
`physics/spectrum_core.py` `gamma_nm_from_A` (with π = 3.14159265).
**Impact:** 1.6 ppm in hc (visible in 4-decimal `wavelength_nm`), 0.07% in c.

### P34 — Doppler width: the FWHM is used as σ  *(Verified)*
**Evidence:** `physics/spectrum_core.py` `doppler_sigma_nm` returns
E·√(8kT ln2/mc²) — the Doppler FWHM — but feeds `exp(-0.5(x/σ)²)`; the docstring says σ.
The browser panel copies it on purpose (`plotinteractive_spectrum_panel.js:46-50`), and
the printed "Doppler σ" is the FWHM. `compare_elements.py:110-114` uses the same
expression correctly as a FWHM.
**Impact:** simulated Doppler profiles 2√(2 ln 2) ≈ 2.355× too broad.

### P35 — Natural width differs between the Python and the JS spectrum  *(Reported)*
**Evidence:** Python uses ħ/τ_total (`physics/spectrum_core.py` `natural_gamma_nm`); the JS
input `gamma_nm_from_A` uses a single line's A (with c = 3e8). Filtered curves in the browser come
out narrower than the pre-rendered ones when the upper state has several decay channels.

### P36 — Three source-colour maps disagree on `Coulomb(approx)`  *(Verified)*
**Evidence:** `pipelines/plotinteractive/ee_grotrian.py` vs `spectra.py:70-77`: `#52be80` vs
`#8e44ad`. Same pattern as P19.

### P37 — Inconsistent missing-ionization-energy fallbacks  *(Verified)*
**Evidence:** `pipelines/plotinteractive/`: 0.0 (`grotrian.py` panel), 5.139
(`ee_grotrian.py`) and 5.0 (`grotrian3d.py`) eV
when the Rydberg data has no `ionization_energy_eV`.

### P38 — `NIST-extrap` is not in `_SOURCE_GROUPS` (P8 triggered in practice)  *(Verified, run)*
**Evidence:** `pipelines/plotinteractive/spectrum.py` `_SOURCE_GROUPS`. Na has 154 `NIST-extrap` transitions:
the 34 ground-state ones are shown as "Coulomb" in plot 2, and all 154 match no category
in plot 3, so they never appear there.

### P39 — Transition metals get a meaningless α  *(Verified, run)*
**Evidence:** `polarizability_core.py:979-995,1243-1326`: Fe passes
`require_neutral_species`; `build_transitions_for_state` ignores the per-line NIST fosc
and invents two "Numerov" channels from the configuration (Fe α₀ = 23.0 au from
f = 0.6375 each; the 58 NIST ground-state lines, Σf = 1.49, are ignored).
**Suggested fix:** exclude metal/no-QDT species as ions are excluded.

### P40 — Hyperfine "ground state" is the first state processed  *(Verified, run)*
**Evidence:** `physics/hyperfine_core.py` `compute` takes the clock of `states[0]`.
`--states 3p3/2 3s1/2` writes
`microwave_clock` for 3p3/2 (F = 3↔2, 55.6 MHz).
**Suggested fix:** take the lowest-energy state or the rydberg JSON `ground_state`.

### P41 — Bundled `nuclear_data.json` picks Fr-219 (22.5 ms) as francium's default  *(Verified, run)*
**Evidence:** the IAEA parser that built it read "ms" half-lives as minutes, lost
flattened exponents ("5.0 1010y" → 5 s) and dropped spins written "3/2(-)", "(3)+"
(Fr-223, Fr-224 among ~100 rows). The **parser** was fixed in this review
(`hyperfine._halflife_s`/`_parse_spin`, tests in `tests/test_hyperfine_nuclear_parse.py`),
but the bundled data file was not regenerated: `get_isotope(..., "Fr")` still returns
Fr-219 (I = 9/2) from it; a rebuilt file returns Fr-223 (I = 3/2, 22 min). Li/Na/K/Rb/Cs
defaults are unchanged by a rebuild.
**Suggested fix:** `linak --pipeline hyperfine --build-nuclear-data` and commit the
regenerated `src/linak/resources/data_json/nuclear_data.json` (a data change).

### P42 — The GUI always sends `--nroots 20`, also in SOC mode  *(Verified)*
**Evidence:** `gui/stages.py` runorca stage default "20"; `orca_runner.default_nroots`
then returns 20 in soc mode, where the CLI uses the per-class SOC default (4–6,
`_DEFAULT_SOC_NROOTS`). The GUI's own help text ("or the SOC class default in soc mode")
is therefore false. Blanking the default changes GUI SOC results and run time, so it was
left (and is allow-listed in `tests/test_gui_stage_table.py`).

### P43 — Basis-set converters drop or mislabel SP shells  *(Verified for GAMESS)*
**Evidence:** `integrations/orca_templates.py:370-371` skips GAMESS `L` (SP) shells (no
error: an S/L/D block yields only S and D); `:287-290` takes `angular_momentum[0]` of a
BSE combined-SP shell (written as two S contractions) and maps l ≥ 6 to S.
**Impact:** `--basis-file` with a Pople-family GAMESS file, or the `--pick-basis` path,
silently produces a different basis.

### P44 — `transitions` writes 0 transitions after `rydberg --nist-only`  *(Verified, run)*
**Evidence:** `pipelines/transitions.py` `main`: any no-QDT rydberg JSON takes the metal branch,
whose loader (`load_nist_fvalues`) returns no lines for non-metal species; `rydberg Na
--nist-only` followed by `transitions Na` saves 0 transitions and exits 0.
**Suggested fix:** use `load_nist_metal_lines` in that branch (output-changing).

### P45 — Breit–Rabi levels of a state with A = 0 get impossible (F, m_F) labels  *(Verified, run)*
**Evidence:** `physics/hyperfine_core.py` `breit_rabi_energies` labels each B = 0
eigenvector with the F whose energy is closest; when A = 0 (a `--states` label with no
constants and nothing to extrapolate from, e.g. `hyperfine Na --states 3s1/2 4d5/2`) all
F levels are degenerate, so every level is called F = F_min while m_F runs to ±(I+J).
The Zeeman legend then shows labels such as "F=1, m_F=+4". (The colour code used to crash
on them; `mF_color` now clamps.)
**Suggested fix:** label from the eigenvectors' F² expectation value, or skip the
Breit–Rabi panel for states without constants.

### P46 — BBR photoionization reads the bound-bound rates after JSON rounding  *(Verified)*
**Evidence:** `physics/blackbody_core.py` `compute_continuum` takes Γ_rad and Γ_BBR rounded
to 10⁻⁶ s⁻¹, as the `rates` block of `<El>_blackbody.json` stores them (the code used to
read them back from that block). The PI-inclusive τ_eff, Γ_quench and the eq. (35)
mixing rates are computed from the rounded values.
**Impact:** relative changes of order 10⁻¹⁰ or less for the bundled data (every reported
state has Γ ≳ 10³ s⁻¹), but removing the rounding moves last digits of the output.
**Suggested fix:** pass the unrounded rates, and accept the last-digit changes.

---

## Summary of the physics list

| # | Item | File | Impact |
|---|------|------|--------|
| P1 | `ā` 2× too large | `feshbach.py:197` | wrong reported/plotted `ā` |
| P2 | default isotope = minority isotope | `pipelines/hyperfine.py` `get_isotope` | wrong A/B/clock for K, Rb |
| P3 | cycling transition never computed | `hyperfine_core.py` | missing result |
| P4 | quadrupole not exact / inconsistent | `hyperfine_core.py` `breit_rabi_energies` | two inconsistent outputs |
| P5 | α(iξ) missing sign | `polarizability_core.py:1341` | latent wrong physics |
| P6 | α docstring formula wrong | `polarizability.py:14,27` | documentation |
| P7 | clock interval omits B/C | `hyperfine_core.py` `microwave_clock` | wrong for J>1/2 ground states |
| P8 | provenance default `'Coulomb'` | `plotinteractive/spectrum.py` | misleading data labels |
| P9 | v_rec cm/s vs m/s | `scattering_rate.py:107,158` | 100× unit error (dead code) |
| P10 | resonance-sign convention undocumented | `feshbach.py:155` | silent sign errors |
| P11 | uncertainty never propagated | `polarizability_core.py:943` | α/C₆ have no error bars |
| P12 | silent f deletions/averaging | `polarizability_core.py:358,763` | Σf changes silently |
| P13 | trap mass = average weight | `polarizability.py:493` | % level trap-frequency error |
| P14 | excited-state −999 fallback | `polarizability.py:456` | wrong state selected |
| P15 | ÷0 at T=0 | `pipelines/blackbody.py` `_print_shifts`, `_print_rydberg` | crash |
| P16 | negative C₆ → complex | `feshbach.py:192` | crash |
| P17 | float sentinels / ÷τ | `compare_elements.py:206`, `hyperfine_core.py` `compute_state` | crash / mis-trigger |
| P18 | the two D-line tables disagree | `spectra.py` `DLINES_FALLBACK` vs `compare_elements.py` `LITERATURE` | inconsistent displayed "literature" values |
| P19 | region colour palettes disagree | `plotinteractive/` `REGION_COLORS` vs `_REGION_COLORS4` | same region → two colours across plots |
| P20 | embedded JS kB ≠ `KB_EV` | template `plotinteractive_spectrum_panel.js` vs `constants.py:46` | browser re-broadening uses 2019 kB |
| P21 | tweezer Im α uses constant Γ | `tweezer.py:196` | R_sc, heating ~3× too high at 1064 nm |
| P22 | peak cross-section uses 1/Δν | `lifetimes_core.py` `lifetime_of` | σ ×π/2, I_sat ×2/π |
| P23 | C₆ combining rule wrong dimensions | `polarizability.py:199` | heteronuclear C₆ matrix |
| P24 | combined a(B) mixes channels | `physics/feshbach.py:83` | main Feshbach output |
| P25 | σ_el missing boson factor 2 | `physics/feshbach.py:75` | σ(B) 2× low |
| P26 | `coulomb_fosc` fails hydrogen limit | `polarizability_core.py:128`, `lifetimes_core.py` | fallback/tail f-values |
| P27 | Fr precision f bypassed in LS mode | `polarizability_core.py:1197` | Fr α₀, pipelines disagree |
| P28 | non-LS emission f lacks g ratio | `polarizability_core.py:1297` | ions/metals α |
| P29 | single δ_l, R∞ not R_M | `physics/rydberg.py:590` | GHz-level QDT energies |
| P30 | octupole term wrong | `hyperfine_core.py` `hyperfine_energy` | latent (C = 0) |
| P31 | anions Z = 2 | `pipelines/rydberg.py` | anion Rydberg series |
| P32 | La/Ce/Ac multiplicities | `core/elements.py` | ORCA multiplicity |
| P33 | hc = 1239.84, c = 3e8 | several | ppm-level |
| P34 | Doppler FWHM used as σ | `spectrum_core.py` `doppler_sigma_nm` | spectra 2.355× too broad |
| P35 | natural width Python ≠ JS | `spectrum_core.py` | browser vs pre-rendered |
| P36 | `Coulomb(approx)` colour maps disagree | `plotinteractive/ee_grotrian.py`, `spectra.py` | display |
| P37 | IE fallback 0.0/5.139/5.0 | `plotinteractive/` | display |
| P38 | `NIST-extrap` not grouped | `plotinteractive/spectrum.py` | 154 Na lines mislabeled/missing |
| P39 | metals get invented α channels | `polarizability_core.py:979` | Fe α meaningless |
| P40 | hyperfine clock = first state | `hyperfine_core.py` `compute` | wrong clock with custom `--states` |
| P41 | bundled nuclear data picks Fr-219 | `data_json/nuclear_data.json` | Fr default isotope |
| P42 | GUI always sends `--nroots 20` | `gui/stages.py` | GUI SOC runs |
| P43 | SP shells dropped/mislabelled | `orca_templates.py:287,370` | silent basis change |
| P44 | transitions after `--nist-only` empty | `transitions.py` | 0 transitions, exit 0 |
| P45 | Breit–Rabi labels for A = 0 states | `hyperfine_core.py` `breit_rabi_energies` | wrong Zeeman legend |
| P46 | BBR PI uses rounded Γ_rad, Γ_BBR | `blackbody_core.py` `compute_continuum` | last digits |

*No changes were made to any physics code (P41: only the parser was fixed; the data file it produced was left as is).*
