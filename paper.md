---
title: 'LiNaK: A Python pipeline for alkali-atom atomic, molecular, and optical (AMO) physics'
tags:
  - Python
  - atomic physics
  - Rydberg atoms
  - quantum defect theory
  - laser cooling
  - optical tweezers
  - spectroscopy
authors:
  - name: Youssef Antoury
    orcid: 0009-0009-4665-0029
    affiliation: 1
affiliations:
  - name: "Independent Researcher"
    index: 1
date: 12 August 2026
bibliography: paper.bib
---

**Youssef Antoury**  
Independent Researcher  
[ORCID: 0009-0009-4665-0029](https://orcid.org/0009-0009-4665-0029)

# Summary

`LiNaK` is a Python pipeline that turns public spectroscopic data for the
alkali metals (Li, Na, K, Rb, Cs, Fr) into a complete, cross-checked set of
atomic, molecular, and optical (AMO) physics quantities, together with an
interactive visualization suite. Starting from NIST Atomic Spectra Database
(ASD) level and line lists [@NIST_ASD], the pipeline builds a quantum-defect
(QDT) description of each atom's Rydberg series, applies electric-dipole
selection rules to construct the full allowed-transition network, and
computes Einstein A coefficients and radiative lifetimes with
`Bates & Damgaard`-scaled or full Numerov radial oscillator strengths where
experimental values are unavailable [@BatesDamgaard1949]. From this
network the pipeline derives dynamic polarizabilities, van der Waals
$C_6$ coefficients and magic wavelengths, static and dynamic blackbody
radiation (BBR) shifts and depopulation/photoionization rates
[@Itano1982; @Beloy2025; @Beterov2009], optical-tweezer scattering and
recoil-heating rates [@Grimm2000], hyperfine structure and Breit-Rabi/Zeeman
diagrams, and magnetic Feshbach resonance curves from curated literature
parameters [@Chin2010]. An optional bridge to ORCA adds
TD-DFT and CASSCF/NEVPT2+spin-orbit-coupling calculations for cross-checking
fine structure [@Neese2020], and an optional bridge to ARC [@Sibalic2017] allows
Rydberg-level cross-validation. Every physical quantity is written to
machine-readable JSON and rendered as a fully interactive Plotly HTML page
(Grotrian diagrams, 3D level/orbital viewers, spectra, polarizability and
Feshbach curves), so results can be explored without re-running Python.

# Statement of need

Alkali atoms are the workhorse species of modern cold-atom and quantum-
simulation experiments, and calculations of their level structure,
lifetimes, polarizabilities, and BBR shifts are a routine but
labor-intensive part of experiment design. In practice this work is
usually redone by hand for each new experiment: NIST energy levels are
copied into a spreadsheet, quantum defects are fit separately, oscillator
strengths are looked up or estimated with the Coulomb approximation, and
downstream quantities (lifetimes, $C_6$, magic wavelengths, BBR shifts,
scattering rates) are computed in disconnected, one-off scripts that are
rarely validated against each other or shared. Existing open tools such as
ARC [@Sibalic2017; @Robertson2021] provide excellent, validated Rydberg
energy-level and dipole-matrix-element calculations for alkali and
alkaline-earth atoms, but do not connect that data to NIST-sourced
oscillator strengths, radiative lifetimes, blackbody shifts, hyperfine
structure, Feshbach resonances, or an interactive visualization layer in
one pipeline.

`LiNaK` addresses this gap by unifying the full chain, from raw ASD
exports through to publication-ready interactive figures, behind a
single, consistent set of physical conventions (energy zero, oscillator
strength sources and their priority order, degeneracy factors) that are
documented explicitly in an accompanying derivation reference so that every
number in the output JSON can be traced back to a specific formula and
data source. Numerical building blocks (e.g. the dynamic-BBR Planck-spectrum
integral, the Casimir-Polder $C_6$ integral, and the Numerov radial
oscillator strength scale) are independently validated in the code against
closed-form limits (the Itano static field and Farley-Wing high-$n$
asymptote for BBR, the Farley-Wing formula, and NIST-overlap medians for
Numerov scaling) before being reported, and the sodium results reproduce
literature lifetimes, static polarizability, and $C_6$ to better than a few
percent. The pipeline is intended for AMO experimentalists and students who
need trustworthy, cross-checked numbers for a specific alkali transition or
Rydberg state without building the QDT-to-observable chain themselves, and
for instructors who want a worked, fully derived reference (from the
central-potential Schrödinger equation up through the optical Bloch
equations) tied to runnable code. Ions and 3d transition metals are
supported on a reduced track (NIST plotting and optional ORCA spin-orbit
structure only), reflecting that the quantum-defect and Numerov machinery
at the pipeline's core is specific to single-valence-electron, alkali-like
systems.

# Functionality

The pipeline is organized as a sequence of independent, composable command-line
scripts (`rydberg.py` -> `transitions.py` -> `lifetimes.py`, followed by
`polarizability.py`, `blackbody.py`, `tweezer.py`, `hyperfine.py`, and
`feshbach.py`), each of which reads and writes plain JSON so any stage can be
rerun, inspected, or replaced independently. A separate visualization layer
(`plotinteractive.py`, `orbital3d.py`, `spectra.py`, `compare_elements.py`,
`scattering_rate.py`) consumes that JSON to produce self-contained,
filterable Plotly HTML pages. A `pytest` suite and a set
of maintainer audit tools (`tools/`) cross-check oscillator-strength sourcing
and physical conventions across elements.

# Acknowledgements

We acknowledge the NIST Atomic Spectra Database, the ARC and ORCA
development teams for the software this pipeline optionally bridges to, and
the authors of the literature sources cited in `REFERENCES.bib`, from which
Feshbach, hyperfine, and lifetime benchmark values were drawn.

# References
