"""Bayesian ridge correction of the model to anchor data, with predictive bands.

ln(f_data / f_model) = c0 + c1*s, s = 1/nu^2, Gaussian priors (tau0, tau1) on the coefficients,
per-anchor sigma = hypot(max(unc, floor), sys). Optional localised third term (threshold-aware
mode) is exp(-s/0.01), acting only at very high nu.
"""

from __future__ import annotations

import numpy as np

from .params import anchor_config, species_params
from .tables import Table, load_table

_NAN6 = ("value", "lo68", "hi68", "lo95", "hi95")


def _extra_sys(element: str, J: float, nu: float, cfg: dict) -> float:
    tot = 0.0
    for r in cfg["extra_sys"]:
        if r["species"] == element and r["J"] == J and nu >= r["nu_min"]:
            tot += r["sigma"] ** 2
    return tot


def fit_and_predict(
    element: str,
    J: float,
    series: list[dict],
    targets_nu: list[float],
    *,
    anchor_type: str = "nist",
    ssys: float | None = None,
    threshold: dict | None = None,
    threshold_aware: bool = False,
    table: Table | None = None,
    config: dict | None = None,
) -> dict:
    """series: dicts with ``nu``, ``f``, ``unc`` (relative) and optional ``sys``, ``low_weight``.

    Returns values and 68/95% bands for ``targets_nu`` (NaN where outside the table), the
    fitted coefficients and the diagnostics the dialog shows as warnings.
    """
    cfg = config or anchor_config()
    species_params(element)
    tab = table or load_table(element)
    rg = cfg["ridge"]
    tau0, tau1, tau2 = rg["tau0"], rg["tau1"], rg["tau2_threshold_aware"]
    ssys = cfg["ssys_by_type"][anchor_type] if ssys is None else ssys

    excluded = [r["nu"] for r in series if not tab.contains(r["nu"])]
    tr = sorted((r for r in series if tab.contains(r["nu"])), key=lambda r: r["nu"])
    if len(tr) < 1 and not threshold:
        raise ValueError(f"{element} J={J}: no anchors inside the model range [{tab.nu_min:.2f}, {tab.nu_max:.2f}]")

    y = np.array([np.log(r["f"] / tab.f(r["nu"], J)) for r in tr])
    s_ = np.array([1 / r["nu"] ** 2 for r in tr])
    lw = cfg["low_weight_unc"]
    sg = np.array(
        [np.hypot(max(r["unc"], lw if r.get("low_weight") else cfg["floor_unc"]), r.get("sys", ssys)) for r in tr]
    )
    if threshold:  # pseudo-anchor at s=0: weighted, never an override
        y0 = np.log(threshold["sigma_Mb"] / tab.sigma_th_Mb())
        sg0 = np.hypot(threshold["unc_Mb"] / threshold["sigma_Mb"], ssys)
        y, s_, sg = np.append(y, y0), np.append(s_, 0.0), np.append(sg, sg0)

    ta = bool(threshold and threshold_aware)
    scale = rg["threshold_aware_scale"]

    def basis(s):
        cols = [np.ones_like(s), s] + ([np.exp(-s / scale)] if ta else [])
        return np.array(cols).T

    X = basis(np.asarray(s_, float))
    W = np.diag(1 / sg**2)
    prior = np.diag([1 / tau0**2, 1 / tau1**2] + ([1 / tau2**2] if ta else []))
    C = np.linalg.inv(X.T @ W @ X + prior)
    b = C @ X.T @ W @ y

    out: dict = {k: [] for k in ("nu", *_NAN6)}
    out_of_range = []
    for nu in targets_nu:
        if not tab.contains(nu):
            out_of_range.append(nu)
            out["nu"].append(float(nu))
            for k in _NAN6:
                out[k].append(float("nan"))
            continue
        xt = basis(np.array([1 / nu**2]))[0]
        mu = float(xt @ b)
        sd = float(np.sqrt(xt @ C @ xt + ssys**2 + _extra_sys(element, J, nu, cfg)))
        fm = tab.f(nu, J)
        out["nu"].append(float(nu))
        for k, v in (
            ("value", fm * np.exp(mu)),
            ("lo68", fm * np.exp(mu - sd)),
            ("hi68", fm * np.exp(mu + sd)),
            ("lo95", fm * np.exp(mu - 2 * sd)),
            ("hi95", fm * np.exp(mu + 2 * sd)),
        ):
            out[k].append(float(v))
    out.update(
        c0=float(b[0]),
        c1=float(b[1]),
        conflict=bool(abs(b[0]) > 2 * tau0),
        excluded_series_nu=excluded,
        out_of_range_targets=out_of_range,
        threshold_used=bool(threshold),
        threshold_aware=ta,
        sigma_th_Mb=float(tab.sigma_th_Mb() * np.exp(b[0] + (b[2] if ta else 0.0))),
        table_hash=tab.table_hash,
        table_fresh=tab.fresh,
    )
    return out
