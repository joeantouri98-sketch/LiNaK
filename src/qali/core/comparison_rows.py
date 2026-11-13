"""Flat rows for the cross-element comparison table (Qt-free).

``comparison_table.json`` holds one nested record per element. The viewer
table and its CSV export need one flat row per element at a chosen
temperature; the Doppler-dependent columns are recomputed here with the same
formulas as the Compare page (``report_templates/compare_page.html``).
"""

from __future__ import annotations

import math
from pathlib import Path

from qali.core.constants import AMU_KG, C_SI, EV_TO_J, H_SI, KB_EV

# (key, header shown in the viewer, CSV header, digits for display and CSV or None for text)
COLUMNS = (
    ("element", "Atom", "Element", None),
    ("Z", "Z", "Z", None),
    ("IE_eV", "IE (eV)", "IE(eV)", None),
    ("delta_s", "δs", "delta_s", None),
    ("delta_p", "δp", "delta_p", None),
    ("delta_d", "δd", "delta_d", None),
    ("D1_wl", "λ (nm)", "D1_lambda(nm)", None),
    ("D1_tau", "τ (ns)", "D1_tau(ns)", None),
    ("D1_A", "A (1e7/s)", "D1_A(1e7/s)", None),
    ("D1_fosc", "fosc", "D1_fosc", None),
    ("soc", "SOC (meV)", "SOC(meV)", None),
    ("D2_wl", "λ (nm)", "D2_lambda(nm)", None),
    ("D2_tau", "τ (ns)", "D2_tau(ns)", None),
    ("D2_A", "A (1e7/s)", "D2_A(1e7/s)", None),
    ("D2_fosc", "fosc", "D2_fosc", None),
    ("D2_acc", "Acc", "D2_Acc", None),
    ("dnu_nat", "Δν (MHz)", "dnu_nat(MHz)", 3),
    ("sig_nat", "σ peak (1e-9 cm²)", "sig_nat(1e-9cm2)", 3),
    ("Isat_nat", "I_sat (mW/cm²)", "Isat_nat(mW/cm2)", 3),
    ("dnu_D", "Δν_D (GHz)", "dnu_D(GHz)", 3),
    ("sig_D", "σ_D (1e-12 cm²)", "sig_D(1e-12cm2)", 3),
    ("Isat_D", "I_sat_D (mW/cm²)", "Isat_D(mW/cm2)", 1),
    ("R_scatt", "R_scatt (1e6/s)", "R_scatt(1e6/s)", 3),
    ("F_rad", "F_rad (aN)", "F_rad(aN)", 3),
    ("v_rec", "v_rec (cm/s)", "v_rec(cm/s)", 2),
    ("E_rec", "E_rec (nK)", "E_rec(nK)", 2),
)
CSV_FILENAME = "alkali_comparison.csv"

# Column sections as in the HTML table: (title, first key, last key). "{T}" is the temperature.
GROUPS = (
    ("Atom", "element", "delta_d"),
    ("D1 line (np 1/2)", "D1_wl", "soc"),
    ("D2 line (np 3/2)", "D2_wl", "D2_acc"),
    ("Natural linewidth (D2)", "dnu_nat", "Isat_nat"),
    ("Doppler, T = {T} K (D2)", "dnu_D", "Isat_D"),
    ("Laser cooling (D2)", "R_scatt", "E_rec"),
)


def doppler_fwhm_mhz(wl_nm: float, T_K: float, mass_amu: float) -> float:
    nu0 = C_SI / (wl_nm * 1e-9)
    mc2_eV = mass_amu * AMU_KG * C_SI * C_SI / EV_TO_J
    return nu0 * math.sqrt(8 * math.log(2) * KB_EV * T_K / mc2_eV) / 1e6


def _num(value):
    return value if isinstance(value, (int, float)) else None


def build_rows(elements: list[dict], T_K: float) -> list[dict]:
    """One flat row per element record; the Doppler columns follow *T_K*.
    Records without a D line are skipped, like the HTML table."""
    rows = []
    for rec in elements:
        d1 = rec.get("D1")
        if not d1:
            continue
        d2: dict = rec.get("D2") or d1
        sig_nat = d2.get("sig_nat_1e9cm2") or 0.0
        dnu_nat = d2.get("dnu_nat_MHz") or 0.0
        dnu_d = sig_d = isat_d = None
        if T_K > 0 and rec.get("mass_amu") and d2.get("wl_nm"):
            dnu_d_mhz = doppler_fwhm_mhz(d2["wl_nm"], T_K, rec["mass_amu"])
            dnu_d = dnu_d_mhz / 1e3
            sig_d_cm2 = (sig_nat * 1e-9 * (dnu_nat / dnu_d_mhz) * math.sqrt(math.log(2) * math.pi)
                         if sig_nat and dnu_nat and dnu_d_mhz else 0.0)
            sig_d = sig_d_cm2 * 1e12
            tau_s = (d2.get("tau_ns") or 0.0) * 1e-9
            if sig_d_cm2 > 0 and tau_s > 0:
                e_photon = H_SI * C_SI / (d2["wl_nm"] * 1e-9)
                isat_d = e_photon / (2 * sig_d_cm2 * 1e-4 * tau_s) * 0.1
        rows.append({
            "element": rec["element"], "Z": rec.get("Z"), "IE_eV": rec.get("IE_eV"),
            "delta_s": rec.get("delta_s"), "delta_p": rec.get("delta_p"),
            "delta_d": rec.get("delta_d"), "soc": rec.get("soc_meV"),
            "D1_wl": d1.get("wl_nm"), "D1_tau": d1.get("tau_ns"),
            "D1_A": (d1.get("A_s") or 0.0) / 1e7, "D1_fosc": d1.get("fosc"),
            "D2_wl": d2.get("wl_nm"), "D2_tau": d2.get("tau_ns"),
            "D2_A": (d2.get("A_s") or 0.0) / 1e7, "D2_fosc": d2.get("fosc"),
            "D2_acc": d2.get("acc"),
            "dnu_nat": d2.get("dnu_nat_MHz"), "sig_nat": d2.get("sig_nat_1e9cm2"),
            "Isat_nat": d2.get("Isat_nat_mWcm2"),
            "dnu_D": dnu_d, "sig_D": sig_d, "Isat_D": isat_d,
            "R_scatt": d2.get("R_scatt_max"), "F_rad": d2.get("F_rad_aN"),
            "v_rec": d2.get("v_rec_cms"), "E_rec": d2.get("E_rec_nK"),
            "source": rec.get("source", ""),
        })
    return rows


def cell_text(key: str, value, digits) -> str:
    """The text of one table cell or CSV field ('' for a missing value)."""
    if value is None:
        return ""
    if digits is not None and _num(value) is not None:
        return f"{value:.{digits}f}"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def csv_text(rows: list[dict]) -> str:
    """The table as CSV, with the HTML export's headers and rounding."""
    lines = [",".join(csv for _k, _h, csv, _d in COLUMNS)]
    for r in rows:
        lines.append(",".join(cell_text(k, r.get(k), d) for k, _h, _c, d in COLUMNS))
    return "\n".join(lines) + "\n"


def load_table(path: Path) -> tuple[float, list[dict]]:
    """(T_K, element records) from comparison_table.json."""
    import json
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return float(data.get("T_K") or 300.0), list(data.get("elements") or [])
