# -*- coding: utf-8 -*-
"""Cross-check tweezer/spectra/feshbach/compare_elements after f-upgrade regen."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PI = math.pi

# Accepted experimental D2 anchors (Steck Na/Rb/Cs; Sansonetti K; NIST Li; Simsarian Fr)
LIT_D2 = {
    "Li": {"wl_nm": 670.977, "tau_ns": 27.102, "tol_wl_nm": 0.05, "tol_tau_rel": 0.05},
    "Na": {"wl_nm": 589.158326, "tau_ns": 16.249, "tol_wl_nm": 0.01, "tol_tau_rel": 0.02},
    "K": {"wl_nm": 766.701, "tau_ns": 26.37, "tol_wl_nm": 0.05, "tol_tau_rel": 0.03},
    "Rb": {"wl_nm": 780.241209, "tau_ns": 26.2348, "tol_wl_nm": 0.01, "tol_tau_rel": 0.01},
    "Cs": {"wl_nm": 852.3472758, "tau_ns": 30.405, "tol_wl_nm": 0.01, "tol_tau_rel": 0.02},
    "Fr": {"wl_nm": 718.0, "tau_ns": 21.02, "tol_wl_nm": 0.5, "tol_tau_rel": 0.05},
}

ok, warn, fail = [], [], []


def check(cond: bool, msg: str, soft: bool = False) -> None:
    if cond:
        ok.append(msg)
    elif soft:
        warn.append(msg)
    else:
        fail.append(msg)


def _find(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = _find(v, key)
            if r is not None:
                return r
    return None


def main() -> int:
    comp_path = ROOT / "plots" / "comparison_table.json"
    comp = json.loads(comp_path.read_text(encoding="utf-8"))
    check(comp["T_K"] == 300.0, "compare T_K=300")
    rows = {r["element"]: r for r in comp["elements"]}
    check(set(rows) == {"Li", "Na", "K", "Rb", "Cs", "Fr"}, "all 6 alkalis in compare table")

    print("=== compare_elements vs lifetimes / rydberg / literature ===")
    for el, row in rows.items():
        check(row["source"] == "pipeline", f"{el}: source=pipeline (not literature fallback)")
        lt = json.loads((ROOT / "data_json" / f"{el}_lifetimes.json").read_text(encoding="utf-8"))
        ry = json.loads((ROOT / "data_json" / f"{el}_rydberg.json").read_text(encoding="utf-8"))
        by = {x["state"]: x for x in lt["lifetimes"]}

        for line in ("D1", "D2"):
            d = row[line]
            st = by.get(d["label"])
            check(st is not None, f"{el} {line} label {d['label']} present in lifetimes")
            if st is None:
                continue
            wl_json = float(st["dominant_wl_nm"])
            tau_json = float(st["lifetime_s"]) * 1e9
            A_json = float(st["A_total_s"])
            check(abs(d["wl_nm"] - wl_json) / wl_json < 1e-6, f"{el} {line} wl matches lifetimes")
            check(abs(d["tau_ns"] - tau_json) / tau_json < 1e-5, f"{el} {line} tau matches lifetimes")
            check(abs(d["A_s"] - A_json) / A_json < 1e-5, f"{el} {line} A matches lifetimes")
            dnu_calc = A_json / (2 * PI) / 1e6
            check(
                abs(d["dnu_nat_MHz"] - dnu_calc) / dnu_calc < 1e-3,
                f"{el} {line} dnu_nat = A/(2 pi)",
            )
            # compare_elements stores R_scatt_max in units of 1e6 / s
            r_calc = (A_json / 2) / 1e6
            check(
                abs(d["R_scatt_max"] - r_calc) / r_calc < 1e-3,
                f"{el} {line} R_scatt_max = (A/2)/1e6",
            )
            check(abs(d["tau_ns"] * 1e-9 * A_json - 1.0) < 1e-5, f"{el} {line} tau*A = 1")

        ratio = row["D2"]["fosc"] / row["D1"]["fosc"]
        check(1.7 < ratio < 2.3, f"{el}: f_D2/f_D1 = {ratio:.3f} ~ 2")

        qd = ry["quantum_defects"]
        check(abs(row["delta_s"] - float(qd["s"])) < 1e-6, f"{el} delta_s from rydberg")
        check(abs(row["delta_p"] - float(qd["p"])) < 1e-6, f"{el} delta_p from rydberg")
        check(
            abs(row["IE_eV"] - float(ry["ionization_energy_eV"])) < 1e-6,
            f"{el} IE from rydberg",
        )

        L = LIT_D2[el]
        dwl = abs(row["D2"]["wl_nm"] - L["wl_nm"])
        rtau = abs(row["D2"]["tau_ns"] - L["tau_ns"]) / L["tau_ns"]
        check(
            dwl <= L["tol_wl_nm"],
            f"{el} D2 wl vs lit: {row['D2']['wl_nm']:.4f} vs {L['wl_nm']} (dwl={dwl:.4f})",
        )
        check(
            rtau <= L["tol_tau_rel"],
            f"{el} D2 tau vs lit: {row['D2']['tau_ns']:.3f} vs {L['tau_ns']} (rel={rtau:.4f})",
        )

    socs = [rows[e]["soc_meV"] for e in ["Li", "Na", "K", "Rb", "Cs", "Fr"]]
    check(all(socs[i] < socs[i + 1] for i in range(5)), f"SOC increases Li->Fr: {socs}")
    for el in ["Na", "K", "Rb", "Cs", "Fr"]:
        check(
            rows[el]["D1"]["wl_nm"] > rows[el]["D2"]["wl_nm"],
            f"{el}: D1 redder than D2",
        )

    print("=== tweezer ===")
    for el in ["Li", "Na", "K", "Rb", "Cs", "Fr"]:
        tw = json.loads((ROOT / "data_json" / f"{el}_tweezer.json").read_text(encoding="utf-8"))
        html = ROOT / "plots" / el / f"{el}_tweezer.html"
        check(
            html.exists() and html.stat().st_size > 1000,
            f"{el} tweezer.html size={html.stat().st_size if html.exists() else 0}",
        )
        re_a = _find(tw, "Re_alpha_au")
        if re_a is None:
            re_a = _find(tw, "alpha_re_au")
        check(re_a is not None and float(re_a) > 0, f"{el} tweezer Re[alpha]={re_a} > 0")
        rsc = _find(tw, "R_sc_s") or _find(tw, "R_sc")
        if rsc is not None:
            check(float(rsc) > 0, f"{el} tweezer R_sc={rsc} > 0")

    print("=== feshbach C6 ===")
    for el in ["Li", "Na", "K", "Rb", "Cs", "Fr"]:
        pol = json.loads(
            (ROOT / "data_json" / f"{el}_polarizability.json").read_text(encoding="utf-8")
        )
        fout = ROOT / "data_json" / f"{el}_feshbach_out.json"
        fin = ROOT / "data_json" / f"{el}_feshbach.json"
        html = ROOT / "plots" / el / f"{el}_feshbach.html"
        if not fout.exists():
            check(
                False,
                f"{el}: no feshbach_out.json (input JSON exists={fin.exists()})",
                soft=True,
            )
            continue
        fe = json.loads(fout.read_text(encoding="utf-8"))
        c6p = float(pol["C6_au"])
        c6f = _find(fe, "C6_au")
        if c6f is None:
            c6f = _find(fe, "C6")
        check(
            c6f is not None and abs(float(c6f) - c6p) / c6p < 1e-6,
            f"{el} feshbach C6={c6f} matches pol {c6p}",
        )
        check(html.exists(), f"{el} feshbach.html exists")

    print("=== spectra ===")
    for el in ["Li", "Na", "K", "Rb", "Cs", "Fr"]:
        p = ROOT / "plots" / el / f"{el}_spectra.html"
        check(
            p.exists() and p.stat().st_size > 1000,
            f"{el} spectra.html size={p.stat().st_size if p.exists() else 0}",
        )

    print("\n=== SUMMARY ===")
    print(f"OK={len(ok)}  WARN={len(warn)}  FAIL={len(fail)}")
    for w in warn:
        print("WARN:", w)
    for fmsg in fail:
        print("FAIL:", fmsg)
    if fail:
        return 1
    print("All hard checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
