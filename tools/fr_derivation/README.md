# Francium extrapolation model: derivation and checks

ARC has no Fr, so the Fr model for the extrapolator (`model_params.json` "Fr", `exact.MARINESCU["Fr"]`, `frlevels.py`) was derived here.
It is semi-empirical, in three layers, each tuned against something known.

## 1. Potential (fitted to NIST energies)
Marinescu model potential, effective charge `Z_l(r) = 1 + (Z-1)e^{-a1 r} - r(a3 + a4 r)e^{-a2 r}` (Z = 87) plus the core-polarization tail
`-alphaC/(2r^4) (1 - e^{-(r/rc)^6})`, alphaC = 20.41 a.u. (Fr+ RRPA, arXiv:physics/9812028; not fitted).
a1-a4 and rc of the s and p series were fitted by bounded least squares to the NIST 7s-14s and 7p-14p energies (soft prior near the Cs values).
d and f are Cs values and unused by ns -> np. Levels above the NIST list come from a Ritz quantum-defect extension (`frlevels.py`).
The np cutoff rc[1] is retuned to 7p (J average) at run time, as for the other species.

## 2. Spin-orbit (one calibration)
Screened `(1/r) dV/dr L.S` term with one scale per species: `so_scale = 1.2965`, root-found so that the 7p splitting is 1686.6 cm-1.
Only 7p is fitted. The splittings at 8p and 9p are 538.7 / 245.0 cm-1 against NIST 545.3 / 248.0 (1.2% low): a check, not a fit.
Side scan (not shipped): about 1.31 is best for 8p-10p (all within 1 cm-1) but makes 7p 18 cm-1 too wide and worsens the 7p and 8p3/2 levels, and the dipole
elements move by under 0.2%. The model's n-dependence of the splitting is off by about 1% between 7p and 8p, so no single scale fits both; 7p was kept
because 7s -> 7p and 7s -> 8p dominate the table.

## 3. Dipole operator (one fitted parameter)
`d(r) = r [1 - lam s(w) alpha_D / r^3 sqrt(1 - e^{-(r/r~c)^6})]`, lam = 1, no core pole, alpha_D = 20.41, **r~c = 4.2** fitted to the four
`<7s||D||7p1/2, 7p3/2, 8p1/2, 8p3/2>` all-order values of Roberts, Fairhall & Ginges, PRA 107, 052812 (2023), Table X final (chi2 = 0.06).
The core term lowers 7s-7p1/2 from 4.69 to 4.29 a.u. Unfitted checks: 8s-8p and 7p-8s agree to 0.2% or better.

## Caveats
- r~c is degenerate with an optional core pole: (4.2, none), (4.4, 16 eV), (4.4, 12 eV), (4.6, 9 eV) all fit the four anchors. At n = 20-40 they differ by up to 10%
  (p3/2) and a factor 2-3 (p1/2), hence the `extra_sys` widening for Fr (sigma 0.5 p1/2, 0.1 p3/2 at nu >= 9).
- 8p1/2 is 9.9 cm-1 off in energy; the other p levels are within about 3 cm-1.
- NIST T6898 7s -> np rows (n >= 8) split each doublet exactly 2:1 and disagree with the all-order values by 2x or more; they are excluded
  from the extrapolation fit (`unreliable_tp_tags`) and replaced by the four reference values (`precision_in_stage`). Lifetimes and polarizability still read them.
- The 6d series and the 7p -> 6d / 6d -> 8p anchors are not validated (d unfitted).

## Files
- `check_fr_model.py`: prints the five checks from the shipped parameters (about 7 s): `uv run python tools/fr_derivation/check_fr_model.py`.
- `tests/test_fr_derivation.py`: the same checks as assertions.
- `fit/`: the scripts that produced the parameters (`fit1.py` per-l level fit, `val2.py` spin-orbit scale and rc[1], `val3.py` chi2 scan,
  `val4.py` dipole sensitivity, `val5.py` NIST T6898 comparison; `frmodel.py`, `frD.py`, `frlev.py` helpers; `mp_sp.json`, `tuned_sp.json` fitted values).
  They need `linak` importable and write JSON next to themselves; run from the repo root.
