# Contributing to LiNaK

Thanks for your interest in improving LiNaK.

## Reporting bugs / requesting features

Please open a [GitHub issue](https://github.com/joeantouri98-sketch/LiNaK/issues)
with:
- The element/species and command you ran.
- The full error output or a description of the unexpected result.
- Whether the issue is in a NIST-derived quantity, an ORCA-derived
  quantity, or a plotting/visualization bug; this determines which part
  of the pipeline needs to change.

## Contributing code

1. Fork the repository and create a branch from `main`.
2. Keep changes scoped to one script/module where possible; the pipeline
   is intentionally modular (`rydberg.py -> transitions.py -> lifetimes.py`,
   etc.), and cross-cutting changes are harder to review.
3. Do not hand-edit generated JSON under `data_json/`; regenerate it via
   the relevant script instead.
4. If you add or change a physical formula, cite the source (paper, NIST
   page, etc.) in a comment and, if applicable, in `REFERENCES.bib`.
5. Run the test suite before opening a pull request:
   ```bash
   pip install -r requirements.txt
   pip install pytest
   pytest tests/
   ```
6. Open a pull request describing what changed and why, and reference any
   related issue.

## Adding support for a new element or species

- Neutral alkalis follow the existing NIST + quantum-defect-theory (QDT)
  path; adding one requires NIST ASD level/line exports under
  `NIST Levels/` and `f_values/`.
- Ions and 3d transition metals currently use a reduced track (NIST
  plotting + optional ORCA spin-orbit structure only). Extending the full
  QDT/polarizability/BBR/tweezer/hyperfine/Feshbach chain to a new species
  class is a larger contribution. Please open an issue first to discuss
  scope before submitting a large pull request.

## Code of conduct

Be respectful and constructive. Physics disagreements should be resolved
by citing sources or providing a reproducible counter-example, not by
assertion.
