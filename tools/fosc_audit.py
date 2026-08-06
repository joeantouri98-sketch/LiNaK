# -*- coding: utf-8 -*-
"""
fosc_audit.py — Oscillator-strength source scorecard

Reports, per element, how lifetime channels and gs polarizability networks
are sourced (NIST / literature / Numerov / Coulomb / ORCA / …).

Usage (from repo root):
  python tools/fosc_audit.py Na
  python tools/fosc_audit.py Na --json tests/results/fosc_audit_Na.json

Writes JSON under tests/results/ by default (keeps artifacts out of data_json/).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "tests" / "results"
DATA = ROOT / "data_json"


def _load(path: Path):
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def audit_lifetimes(element: str, data_dir: Path) -> dict:
    lt = _load(data_dir / f"{element}_lifetimes.json")
    if not lt:
        return {"error": f"missing {element}_lifetimes.json"}

    src_total = Counter()
    by_n = defaultdict(Counter)
    by_series = defaultdict(Counter)  # upper l
    A_weighted = Counter()

    for row in lt.get("lifetimes", []):
        n = row.get("n")
        l = row.get("l") or "?"
        for ch in row.get("channels", []):
            src = ch.get("fosc_source") or "?"
            src_total[src] += 1
            if n is not None:
                by_n[int(n)][src] += 1
            by_series[l][src] += 1
            A_weighted[src] += float(ch.get("A_s") or 0.0)

    n_channels = sum(src_total.values()) or 1
    coulombish = src_total.get("Coulomb", 0) + src_total.get("Coulomb(approx)", 0)
    return {
        "n_states": lt.get("n_states"),
        "n_channels": n_channels,
        "by_source": dict(src_total),
        "fraction_coulomb": coulombish / n_channels,
        "A_weighted_by_source": {k: round(v, 6) for k, v in A_weighted.items()},
        "by_n": {str(n): dict(by_n[n]) for n in sorted(by_n)},
        "by_upper_l": {k: dict(v) for k, v in sorted(by_series.items())},
        "n_max_mostly_nist": _n_max_mostly_trusted(by_n),
    }


def _n_max_mostly_trusted(by_n: dict) -> int | None:
    """Largest n where NIST+precision+literature+ORCA still outnumber Coulomb."""
    best = None
    for n in sorted(by_n):
        c = by_n[n]
        good = sum(c[s] for s in c if s in (
            "NIST", "precision", "literature", "ORCA", "EOM-CCSD", "Numerov"
        ))
        bad = c.get("Coulomb", 0) + c.get("Coulomb(approx)", 0)
        if good >= bad and good > 0:
            best = n
        elif n > 12:
            break
    return best


def audit_polarizability(element: str, data_dir: Path) -> dict:
    pol = _load(data_dir / f"{element}_polarizability.json")
    if not pol:
        return {"error": f"missing {element}_polarizability.json"}
    gs = pol.get("transitions_gs") or []
    src = Counter(t.get("source") or "?" for t in gs)
    return {
        "gs_label": pol.get("gs_label"),
        "alpha0_au": pol.get("alpha0_au"),
        "n_transitions_gs": len(gs),
        "by_source": dict(src),
        "fraction_coulomb": (src.get("Coulomb", 0) + src.get("Coulomb(approx)", 0))
        / max(len(gs), 1),
    }


def build_scorecard(element: str, data_dir: Path) -> dict:
    card = {
        "element": element,
        "lifetimes": audit_lifetimes(element, data_dir),
        "polarizability_gs": audit_polarizability(element, data_dir),
    }
    # Runtime Numerov scale from this element's NIST + QD (no hardcoded factors)
    ryd = _load(data_dir / f"{element}_rydberg.json")
    try:
        from alpha_core import load_nist_fvalues
        from qdt_radial import fit_numerov_scale_from_nist
        nist = load_nist_fvalues(element, verbose=False)
        qd = (ryd or {}).get("quantum_defects", {})
        card["numerov_calibration"] = fit_numerov_scale_from_nist(qd, nist["fosc"])
    except Exception as exc:
        card["numerov_calibration"] = {"error": str(exc)}
    return card


def main(argv=None):
    p = argparse.ArgumentParser(description="f-source scorecard")
    p.add_argument("element", nargs="?", default="Na")
    p.add_argument("--data-dir", default=str(DATA))
    p.add_argument("--json", default=None,
                   help="Output path (default: tests/results/fosc_audit_<el>.json)")
    args = p.parse_args(argv)

    data_dir = Path(args.data_dir)
    card = build_scorecard(args.element, data_dir)

    out = Path(args.json) if args.json else RESULTS / f"fosc_audit_{args.element}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)
        f.write("\n")

    lt = card["lifetimes"]
    print(f"=== fosc audit: {args.element} ===")
    if "error" in lt:
        print(" lifetimes:", lt["error"])
    else:
        print(f" channels: {lt['n_channels']}")
        print(f" by source: {lt['by_source']}")
        print(f" fraction Coulomb*: {lt['fraction_coulomb']:.3f}")
        print(f" n_max mostly trusted: {lt.get('n_max_mostly_nist')}")
    pol = card["polarizability_gs"]
    if "error" not in pol:
        print(f" alpha gs sources: {pol['by_source']}  alpha0={pol.get('alpha0_au')}")
    print(f" wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
