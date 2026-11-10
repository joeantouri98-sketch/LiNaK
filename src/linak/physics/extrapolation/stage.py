"""Stage-facing layer: config, anchors from the stage's own data, results with plain-text warnings.

A stage (lifetimes today; transitions, polarizability, plots later) calls :func:`run_for_species`
with its loaded NIST f-values and Rydberg levels. The result lists, per fine-structure channel,
measured anchors and extrapolated values with 68/95% bands, plus warnings for the dialog.
No Qt, no file writing except the explicit ``save_config``.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from linak.core.fileio import write_json

from .api import extrapolate, flag_low_weight
from . import fit_settings, model_settings
from .params import UnsupportedSpeciesError, anchor_config, fit_overrides, model_overrides, species_params
from .tables import get_model, table_status

MODES = ("off", "show", "use")
RY_FALLBACK_EV = 13.605693122994
J_VALUES = {"1/2": 0.5, "3/2": 1.5}
_NP = re.compile(r"^(\d+)p$")


@dataclass
class ExtrapolationConfig:
    """What the user chose in the dialog; round-trips through JSON."""

    mode: str = "off"                 # off | show (add alongside) | use (fill gaps)
    n_max: int = 40
    threshold: bool = True            # threshold pseudo-anchor, where the species has one
    threshold_aware: bool = False     # opt-in; unvalidated at high n
    excluded: list[str] = field(default_factory=list)       # anchor ids, e.g. "12p3/2"
    user_anchors: list[dict] = field(default_factory=list)  # {n, J:'1/2', f, unc}
    user_threshold: dict | None = None  # {sigma_Mb, unc_Mb, src}: your own threshold cross-section, replaces the packaged one
    fit: dict = field(default_factory=dict)   # fit settings (see fit_settings); read from fit_settings.json, not saved here
    model: dict = field(default_factory=dict)  # model parameters per species (see model_settings); read from model_settings.json, not saved here
    tables_dir: str = ""                      # rebuilt tables and the exact-calculation cache; set when the config is loaded

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"extrapolation mode must be one of {MODES}, got {self.mode!r}")
        if not 3 <= int(self.n_max) <= 200:
            raise ValueError("n_max must be between 3 and 200")

    @classmethod
    def from_dict(cls, d: dict) -> ExtrapolationConfig:
        known = {k: d[k] for k in ("mode", "n_max", "threshold", "threshold_aware", "excluded", "user_anchors", "user_threshold") if k in d}
        cfg = cls(**known)
        cfg.validate()
        return cfg


def parse_user_threshold(raw) -> tuple[dict | None, str]:
    """(threshold, problem) from the 'threshold' entry of a user anchors file; None/empty is fine."""
    if not raw:
        return None, ""
    try:
        sigma, unc = float(raw["sigma_Mb"]), float(raw["unc_Mb"])
        if not (sigma > 0 and unc > 0):
            raise ValueError("sigma_Mb and unc_Mb must be positive")
        return {"sigma_Mb": sigma, "unc_Mb": unc, "src": str(raw.get("src") or "your value")}, ""
    except (KeyError, TypeError, ValueError) as exc:
        return None, f"Your threshold value was ignored ({exc}); it needs sigma_Mb and unc_Mb as positive numbers."


def threshold_for(element: str, cfg: ExtrapolationConfig) -> dict | None:
    """The threshold reference in use: yours if you gave one, else the packaged one, else None."""
    mine, _problem = parse_user_threshold(cfg.user_threshold)
    return mine or anchor_config()["threshold_anchors"].get(element)


def load_config(path: str | Path) -> ExtrapolationConfig:
    """The config file plus the fit settings that sit next to it (``fit_settings.json``)."""
    with Path(path).open(encoding="utf-8") as fh:
        cfg = ExtrapolationConfig.from_dict(json.load(fh))
    cfg.fit = fit_settings.load(fit_settings.settings_path(path))
    cfg.model = model_settings.load(model_settings.settings_path(path))
    cfg.tables_dir = str(model_settings.tables_dir(path))
    return cfg


def save_config(cfg: ExtrapolationConfig, path: str | Path) -> None:
    cfg.validate()
    write_json(str(path), {k: v for k, v in asdict(cfg).items() if k not in ("fit", "model", "tables_dir")})   # these live in their own files


def _line_tags(element: str) -> dict[tuple[int, str], str]:
    """(n, J_upper) -> tp_ref for ground ns->np lines in the bundled NIST list (alkali .txt only)."""
    from linak.core.paths import reference_path

    path = reference_path("f_values", f"{element}_lines.txt")
    if not path.is_file():
        return {}
    out: dict[tuple[int, str], str] = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            clean = {k.strip(): (v or "").strip().strip('"') for k, v in row.items() if k}
            try:
                if float(clean["Ei(eV)"]) != 0.0:
                    continue
            except (KeyError, ValueError):
                continue
            m = re.search(r"(\d+)p$", clean.get("conf_k", ""))
            if m:
                out[(int(m.group(1)), clean.get("J_k", ""))] = clean.get("tp_ref", "")
    return out


def _nu_by_label(rydberg: dict) -> dict[str, float]:
    ry = float(rydberg.get("rydberg_constant_eV") or RY_FALLBACK_EV)
    out = {}
    for ex in rydberg.get("excitations", []):
        e = ex.get("energy_eV")
        if ex.get("l") == "p" and e:
            out[ex["label"]] = math.sqrt(ry / abs(float(e)))
    return out


def _anchor_sigma(a: dict) -> float:
    """The relative sigma the fit gives this anchor: its quoted uncertainty (or the floor / low-weight value) with the systematic."""
    cfg = anchor_config()
    unc = max(a.get("unc") or 0.0, cfg["low_weight_unc"] if a.get("low_weight") else cfg["floor_unc"])
    return float(math.hypot(unc, a.get("sys", cfg["ssys_by_type"]["nist"])))


def collect_anchors(element: str, nist: dict, rydberg: dict) -> list[dict]:
    """ns->np_J rows from the stage's NIST f-values, ready for the fit and for the dialog table."""
    nl = species_params(element)["nl"]
    gs = f"{nl}s"
    nus = _nu_by_label(rydberg)
    tags = _line_tags(element)
    cfg = anchor_config()
    rows = []
    for (lo, up, jstr), f in nist.get("fosc_j", {}).items():
        m = _NP.match(up)
        if lo != gs or not m or jstr not in J_VALUES or not f or f <= 0:
            continue
        n = int(m.group(1))
        nu = nus.get(f"{n}p{jstr}")
        if nu is None:
            continue
        sig = nist.get("sigma_j", {}).get((lo, up, jstr))
        acc = nist.get("acc_j", {}).get((lo, up, jstr), "")
        precision = acc == "precision"
        rows.append({
            "id": f"{n}p{jstr}", "n": n, "J": J_VALUES[jstr], "jstr": jstr, "nu": nu, "f": float(f),
            "unc": float(sig) / float(f) if sig else 0.0, "acc": acc,
            "source": "precision" if precision else "NIST", "tp_ref": tags.get((n, jstr), ""),
            "sys": cfg["ssys_by_type"]["precision" if precision else "nist"],
        })
    bad = cfg.get("unreliable_tp_tags", {})
    for r in rows:
        flag_low_weight([r], element, r["J"])
        # only above the ground-state series: the resonance line (n = nl) is fine and lies below the table anyway
        r["unreliable"] = r["n"] > species_params(element)["nl"] and any(t in str(r["tp_ref"]) for t in bad)
    rows = _with_precision_rows(element, rows, nus, rydberg)
    return sorted(rows, key=lambda r: (r["J"], r["n"]))


def _with_precision_rows(element: str, rows: list[dict], nus: dict[str, float], rydberg: dict) -> list[dict]:
    """Add the packaged precision matrix elements as anchors, for species whose NIST series cannot stand alone.

    A precision row replaces a NIST row of the same level only when that row is marked unreliable.
    """
    cfg = anchor_config()
    if element not in cfg.get("precision_in_stage", []):
        return rows
    nl = species_params(element)["nl"]
    ry = float(rydberg.get("rydberg_constant_eV") or RY_FALLBACK_EV)
    inv_gs = float(rydberg["ionization_energy_eV"]) / ry          # 1 / nu_gs^2
    by_id = {r["id"]: r for r in rows}
    for jstr, J in J_VALUES.items():
        for n_s, (D, u) in cfg["precision_anchors"].get(element, {}).get(str(J), {}).items():
            n = int(n_s)
            nu = nus.get(f"{n}p{jstr}")
            old = by_id.get(f"{n}p{jstr}")
            if n <= nl or nu is None or (old is not None and not old.get("unreliable")):
                continue
            omega = 0.5 * (inv_gs - 1.0 / nu**2)
            new = {"id": f"{n}p{jstr}", "n": n, "J": J, "jstr": jstr, "nu": nu, "f": omega * D**2 / 3, "unc": 2 * u,
                   "acc": "precision", "source": "precision", "tp_ref": "", "sys": cfg["ssys_by_type"]["precision"],
                   "unreliable": False}
            rows = [r for r in rows if r is not old] + [new]
    return rows


def _num(v, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _user_rows(element: str, cfg: ExtrapolationConfig, nus: dict[str, float], problems: list[str]) -> list[dict]:
    out = []
    for u in cfg.user_anchors:
        jstr = u.get("J")
        try:
            n_user, f_user = int(u["n"]), float(u["f"])
        except (KeyError, TypeError, ValueError):
            problems.append(f"A user anchor was ignored because n or f is missing or not a number: {u}.")
            continue
        nu = nus.get(f"{n_user}p{jstr}") or float(u.get("nu", 0)) or None
        if jstr not in J_VALUES or f_user <= 0 or nu is None:
            why = "J must be 1/2 or 3/2" if jstr not in J_VALUES else "f must be positive" if f_user <= 0 else "that level is not in the Rydberg list, so add its effective quantum number as \"nu\""
            problems.append(f"User anchor n={n_user}, J={jstr} was ignored: {why}.")
            continue
        out.append({"id": f"{int(u['n'])}p{jstr}*", "n": int(u["n"]), "J": J_VALUES[jstr], "jstr": jstr, "nu": nu,
                    "f": float(u["f"]), "unc": _num(u.get("unc"), 0.05), "acc": "user", "source": "user",
                    "tp_ref": "", "sys": anchor_config()["ssys_by_type"]["precision"]})
    return out


def run_for_species(element: str, cfg: ExtrapolationConfig, nist: dict, rydberg: dict) -> dict:
    """Extrapolated ns->np f-values for both J channels, with bands and warnings."""
    cfg.validate()
    fit, problems = fit_settings.validate(cfg.fit)
    model, model_problems = model_settings.validate(cfg.model)
    with fit_overrides(fit_settings.to_overrides(fit)), model_overrides(
        model_settings.to_overrides(model), Path(cfg.tables_dir) if cfg.tables_dir else None
    ):
        out = _run_for_species(element, cfg, nist, rydberg)
    out["fit_settings"] = fit
    out["model_settings"] = model.get(element, {})
    out["warnings"] = list(out["warnings"]) + problems + model_problems
    if fit:
        out["warnings"].append("Custom fit settings in use: " + fit_settings.describe(fit) + ".")
    if model.get(element):
        out["warnings"].append("Custom model parameters in use: " + model_settings.describe(model, element) + ".")
    return out


def _run_for_species(element: str, cfg: ExtrapolationConfig, nist: dict, rydberg: dict) -> dict:
    sp = species_params(element)
    nl = sp["nl"]
    acfg = anchor_config()
    tab = get_model(element)
    exact = not hasattr(tab, "splines")
    nus = _nu_by_label(rydberg)
    warnings: list[str] = []
    anchors = collect_anchors(element, nist, rydberg) + _user_rows(element, cfg, nus, warnings)
    excluded_ids = set(cfg.excluded)
    unreliable = [a for a in anchors if a.get("unreliable")]
    if unreliable:                     # a source known to be wrong: never fitted, whatever the user ticked
        excluded_ids |= {a["id"] for a in unreliable}
        why = next(w for t, w in acfg["unreliable_tp_tags"].items() if t in str(unreliable[0]["tp_ref"]))
        warnings.append(f"{len(unreliable)} NIST anchor(s) were left out of the fit: {why}.")
    if exact:
        warnings.append(
            "The shipped tables do not match the current model parameters, so the exact model calculation was used "
            "(slower). Rebuild the tables in Edit > Preferences > Extrapolator to make it fast again."
        )
    if element == "Li":
        warnings.append("Lithium: some model parameters are not yet sourced, so results may vary between species.")
    _mine, problem = parse_user_threshold(cfg.user_threshold)
    if problem:
        warnings.append(problem)
    thr = threshold_for(element, cfg) if cfg.threshold else None
    if cfg.threshold_aware and not thr:
        warnings.append("Threshold-aware mode needs a threshold reference; none exists for this species, so it was ignored.")
    if cfg.threshold_aware and thr:
        warnings.append(f"{element}: threshold-aware fit is experimental. Its extra term acts mainly above n of about 7, "
                        f"where it has not been validated, so bands there are less reliable than they look.")

    channels = {}
    for jstr, J in J_VALUES.items():
        used = [a for a in anchors if a["J"] == J and a["id"] not in excluded_ids]
        skipped = [a["id"] for a in anchors if a["J"] == J and a["id"] in excluded_ids]
        measured = {a["n"]: a for a in anchors if a["J"] == J}
        targets = []
        for n in range(nl + 1, int(cfg.n_max) + 1):
            nu = nus.get(f"{n}p{jstr}")
            if nu is None:
                # fall back to the measured-series quantum defect of the highest anchor
                ref = max(used or anchors, key=lambda a: a["n"], default=None)
                if ref is None:
                    continue
                nu = n - (ref["n"] - ref["nu"])
            targets.append({"n": n, "nu": nu})
        try:
            o = extrapolate(element, J, used, targets, threshold=thr,
                            threshold_aware=cfg.threshold_aware)
        except ValueError as exc:
            warnings.append(f"p{jstr}: {exc}")
            continue
        rows = []
        for i, t in enumerate(targets):
            if math.isnan(o["value"][i]):
                continue
            m = measured.get(t["n"])
            rows.append({"n": t["n"], "nu": t["nu"], "f": o["value"][i], "f_model": tab.f(t["nu"], J), "lo68": o["lo68"][i], "hi68": o["hi68"][i],
                         "lo95": o["lo95"][i], "hi95": o["hi95"][i],
                         "f_measured": m["f"] if m else None, "measured": m is not None})
        if o["out_of_range_targets"]:
            warnings.append(f"p{jstr}: {len(o['out_of_range_targets'])} level(s) lie outside the model range (ν {tab.nu_min:.1f}-{tab.nu_max:.0f}) and were skipped.")
        if o["excluded_series_nu"]:
            names = [a["id"].rstrip("*") for a in used if not tab.contains(a["nu"])]
            warnings.append(f"p{jstr}: {len(names)} anchor(s) outside the model range (ν {tab.nu_min:.1f}-{tab.nu_max:.0f}; e.g. the resonance line) are not used in the fit ({', '.join(names)}).")
        if o["conflict"]:
            warnings.append(f"p{jstr}: the model needs a large overall correction (exp({o['c0']:+.2f}) = {100 * (math.exp(o['c0']) - 1):+.0f}%); check the anchors.")
        if thr:
            ratio = o["sigma_th_Mb"] / thr["sigma_Mb"]
            if not 0.77 <= ratio <= 1.3:
                warnings.append(f"p{jstr}: fitted threshold cross-section {o['sigma_th_Mb']:.3g} Mb differs from the reference {thr['sigma_Mb']:.3g} Mb ({thr['src']}) by {100 * (ratio - 1):+.0f}%.")
        lw = [a["id"] for a in used if a.get("low_weight")]
        if lw:
            warnings.append(f"p{jstr}: {len(lw)} anchor(s) from spliced/extrapolated NIST sources were down-weighted ({', '.join(lw[:4])}{'...' if len(lw) > 4 else ''}).")
        if len(used) < 3:
            warnings.append(f"p{jstr}: only {len(used)} anchor(s) in use; the correction is weakly constrained.")
        for rule in acfg["extra_sys"]:
            if rule["species"] == element and rule["J"] == J:
                hit = [r["n"] for r in rows if r["nu"] >= rule["nu_min"]]
                if hit:
                    span = f"{hit[0]}p{jstr}" if len(hit) == 1 else f"{hit[0]}p{jstr} to {hit[-1]}p{jstr}"
                    warnings.append(f"p{jstr}: uncertainty widened for {span} (n >= {hit[0]}): {rule['why']}.")
        channels[jstr] = {"J": J, "c0": o["c0"], "c1": o["c1"], "sigma_th_Mb": o["sigma_th_Mb"],
                          "n_anchors_used": len(used), "excluded": skipped, "rows": rows}
    return {"element": element, "mode": cfg.mode, "n_max": cfg.n_max, "backend": "exact" if exact else "table",
            "sigma_th_model_Mb": tab.sigma_th_Mb(), "threshold_reference": thr,
            "table": table_status(element), "channels": channels, "warnings": warnings,
            "anchors": [{k: v for k, v in a.items() if k != "sys"} | {"included": a["id"] not in excluded_ids, "sigma": _anchor_sigma(a)}
                        for a in anchors]}


def fill_entries(result: dict, nist: dict, literature: dict) -> list[dict]:
    """Extrapolated rows for levels with no measured, precision, NIST or published value (what 'Use' may fill).

    Literature rows that are themselves model-potential values (ARC) do not count as covering a level: the
    anchor-calibrated extrapolation ranks above them and replaces them (the stage records what it replaced).
    """
    nl = species_params(result["element"])["nl"]
    gs = f"{nl}s"
    model_j = set(literature.get("model_j", set()))
    model = {tuple(k) for k in literature.get("model", set())}
    have = set(nist.get("fosc_j", {})) | (set(literature.get("fosc_j", {})) - model_j)
    # multiplet totals (NIST or published) cover both J components of that np level
    multiplet = {tuple(k) for k in nist.get("fosc", {})} | ({tuple(k) for k in literature.get("fosc", {})} - model)
    out = []
    for jstr, ch in result["channels"].items():
        for r in ch["rows"]:
            if (gs, f"{r['n']}p") in multiplet or (f"{r['n']}p", gs) in multiplet:
                continue
            if (gs, f"{r['n']}p", jstr) not in have:
                out.append({"key": (gs, f"{r['n']}p", jstr), **r})
    return out


def published_disagreements(result: dict, literature: dict, *, n_show: int = 6) -> list[str]:
    """Warnings for published (non-model) literature f-values that lie outside the extrapolated 95% band."""
    gs = f"{species_params(result['element'])['nl']}s"
    model_j = set(literature.get("model_j", set()))
    model = {tuple(k) for k in literature.get("model", set())}
    rows = {(j, r["n"]): r for j, ch in result["channels"].items() for r in ch["rows"]}
    off: list[tuple[str, float]] = []
    for (lo, up, jstr), f in literature.get("fosc_j", {}).items():
        if lo != gs or (lo, up, jstr) in model_j or not up.endswith("p"):
            continue
        r = rows.get((jstr, int(up[:-1])))
        if r and not (r["lo95"] <= f <= r["hi95"]):
            off.append((f"{up}{jstr}", f / r["f"] - 1.0))
    for (lo, up), f in literature.get("fosc", {}).items():
        if lo != gs or (lo, up) in model or not up.endswith("p"):
            continue
        parts = [rows.get((j, int(up[:-1]))) for j in ("1/2", "3/2")]
        if all(parts):
            tot = sum(p["f"] for p in parts)
            lo95, hi95 = sum(p["lo95"] for p in parts), sum(p["hi95"] for p in parts)
            if not (lo95 <= f <= hi95):
                off.append((up, f / tot - 1.0))
    if not off:
        return []
    shown = ", ".join(f"{lab} ({d:+.0%})" for lab, d in off[:n_show]) + (", ..." if len(off) > n_show else "")
    return [f"{len(off)} published literature value(s) lie outside the extrapolated 95% band and are kept "
            f"(published values rank above the extrapolation): {shown}."]


def overlay_literature(literature: dict, entries: list[dict]) -> dict:
    """Copy of the stage's literature mapping with the gap-filling values added, tagged 'extrapolated'."""
    lit = dict(literature)
    fj = dict(lit.get("fosc_j", {}))
    aj = dict(lit.get("acc_j", {}))
    for e in entries:
        fj[e["key"]] = e["f"]
        aj[e["key"]] = "extrapolated"
    lit["fosc_j"] = fj
    lit["acc_j"] = aj
    return lit


def output_block(result: dict, entries: list[dict] | None = None) -> dict:
    """JSON-ready 'extrapolation' section: everything tagged as extrapolated, nothing mixed with measured data."""
    return {
        "mode": result["mode"], "backend": result["backend"], "table": result["table"], "warnings": result["warnings"],
        "tag": "extrapolated", "fit_settings": result.get("fit_settings", {}),
        "channels": {j: {k: v for k, v in ch.items() if k != "rows"} | {"rows": [r for r in ch["rows"]]}
                     for j, ch in result["channels"].items()},
        "filled": [{"lower": e["key"][0], "upper": e["key"][1], "J": e["key"][2], "f": e["f"], "lo68": e["lo68"], "hi68": e["hi68"]}
                   for e in (entries or [])],
    }


def apply_to_stage(element: str, config_path, nist: dict, literature: dict, rydberg: dict, *, log=print):
    """What a stage does with ``--extrapolation``: ``(result, entries, literature)``.

    ``result`` is None when extrapolation is off, not requested, or the species is unsupported (e.g. Ra).
    In 'use' mode ``literature`` comes back with the gap-filling values added (tagged 'extrapolated');
    otherwise it is returned unchanged. Any stage that reads f-values can call this.
    """
    if not config_path:
        return None, [], literature
    cfg = load_config(config_path)
    if cfg.mode == "off":
        return None, [], literature
    try:
        result = run_for_species(element, cfg, nist, rydberg)
    except UnsupportedSpeciesError as exc:
        log(f"  Extrapolation skipped: {exc}")
        return None, [], literature
    entries = fill_entries(result, nist, literature) if cfg.mode == "use" else []
    result["warnings"] = list(result["warnings"]) + published_disagreements(result, literature)
    n_rows = sum(len(c["rows"]) for c in result["channels"].values())
    log(f"  Extrapolation ({cfg.mode}): {n_rows} ns->np values" + (f", {len(entries)} fill gaps" if cfg.mode == "use" else ""))
    for w in result["warnings"]:
        log(f"    ! {w}")
    if cfg.mode == "use":
        literature = overlay_literature(literature, entries)
    return result, entries, literature


def rebuild_tables(element: str, config_path: str | Path, model: dict | None = None, progress=None, cancelled=None) -> dict:
    """Build the tables of ``element`` for the model parameters in ``model`` (default: the saved ones) into ``extrapolation_tables/``."""
    from .exact import build_table

    path = Path(config_path)
    clean, _ = model_settings.validate(model if model is not None else model_settings.load(model_settings.settings_path(path)))
    out_dir = model_settings.tables_dir(path)
    with model_overrides(model_settings.to_overrides(clean), out_dir):
        return build_table(element, out_dir, progress=progress, cancelled=cancelled)


def table_state(element: str, config_path: str | Path, model: dict | None = None) -> dict:
    """Which tables the given model parameters would use: 'shipped', 'rebuilt', or 'exact' (no table matches)."""
    from .tables import packaged_table, rebuilt_table

    path = Path(config_path)
    clean, _ = model_settings.validate(model if model is not None else model_settings.load(model_settings.settings_path(path)))
    with model_overrides(model_settings.to_overrides(clean), model_settings.tables_dir(path)):
        if rebuilt_table(element) is not None:
            return {"state": "rebuilt"}
        return {"state": "shipped" if packaged_table(element).fresh else "exact"}
