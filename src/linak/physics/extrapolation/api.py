"""Public entry points: anchor preparation and :func:`extrapolate`."""

from __future__ import annotations

from .params import anchor_config, species_params
from .ridge import fit_and_predict
from .tables import load_table


def flag_low_weight(rows: list[dict], element: str | None = None, J: float | None = None) -> list[dict]:
    """Mark NIST rows from spliced/extrapolated sources (by tp_ref tag or species rule) as low weight."""
    cfg = anchor_config()
    nmin = None
    for rule in cfg["low_weight_rules"]:
        if rule["species"] == element and rule["J"] == J:
            nmin = rule["n_min"]
    for r in rows:
        if r.get("source", "NIST") != "NIST":
            continue
        tagged = any(t in str(r.get("tp_ref", "")) for t in cfg["low_weight_tp_tags"])
        if tagged or (nmin is not None and r.get("n", 0) >= nmin):
            r["low_weight"] = True
    return rows


def precision_rows(element: str, J: float, omega_au_by_n: dict, nu_by_n: dict, first_n: int | None = None) -> list[dict]:
    """f-rows from the packaged precision anchors (f = omega D^2 / 3, omega in Hartree).

    The resonance line lies below the table range, so rows start at ``nl+1`` by default.
    """
    first_n = species_params(element)["nl"] + 1 if first_n is None else first_n
    table = anchor_config()["precision_anchors"].get(element, {}).get(str(J), {})
    out = []
    for n_s, (D, u) in table.items():
        n = int(n_s)
        if n >= first_n:
            out.append(
                {"n": n, "nu": nu_by_n[n], "f": omega_au_by_n[n] * D**2 / 3, "unc": 2 * u, "sys": 0.08, "source": "precision"}
            )
    return out


def extrapolate(
    element: str,
    J: float,
    series: list[dict],
    targets: list[dict],
    *,
    threshold: str | dict | None = "default",
    threshold_aware: bool = False,
    anchor_type: str = "nist",
) -> dict:
    """Extrapolate ns->np_J f-values to ``targets`` (dicts with ``n`` and ``nu``).

    ``threshold='default'`` uses the packaged pseudo-anchor for the species (if any); pass
    ``None`` to switch it off or a dict ``{sigma_Mb, unc_Mb}`` to supply your own.
    Raises ``UnsupportedSpeciesError`` for species without a validated model (e.g. Fr).
    """
    species_params(element)
    cfg = anchor_config()
    thr = cfg["threshold_anchors"].get(element) if threshold == "default" else threshold
    if isinstance(thr, str):
        raise TypeError("threshold must be 'default', None or a dict with sigma_Mb and unc_Mb")
    out = fit_and_predict(
        element,
        J,
        series,
        [t["nu"] for t in targets],
        anchor_type=anchor_type,
        threshold=thr,
        threshold_aware=threshold_aware,
        config=cfg,
    )
    out["n"] = [t["n"] for t in targets]
    out["backend"] = "table"
    return out


def model_range(element: str) -> tuple[float, float]:
    t = load_table(element)
    return t.nu_min, t.nu_max
