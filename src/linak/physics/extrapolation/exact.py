"""The exact model calculation behind the tables: radial wavefunctions in the Marinescu potential.

ns->np_J oscillator strengths come from the l-dependent model potential (parameters as in ARC) with a screened
spin-orbit term, an np cutoff tuned to the lowest np level, and the dipole operator
r * [1 - lam * s(w) * alpha_D * sqrt(1 - exp(-(r/rc)^6)) / r^3] (Singor alpha_D and rc, s(w) the dynamic core factor).
Slow (seconds per level) and independent of any table, so it is used whenever the tables do not match the model
settings, and to build new tables. Energies of the real levels come from ARC, an optional dependency.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import ClassVar

import numpy as np
from scipy.optimize import brentq

from .params import model_params, params_hash, species_params, user_dir

ALPHA = 1.0 / 137.035999084
HA_EV = 27.211386245988
RCUT = 0.01          # innermost radius seen by the spin-orbit term
H_STEP = 0.002       # log-grid step
R_MIN = 1e-5
SIGMA_TH_FACTOR_MB = 4.0334
N_GRID_TAIL = (30, 35, 40, 50, 60)
N_GRID_TOP = 25


class ExactBackendUnavailable(RuntimeError):
    """The exact calculation needs ARC (optional dependency) for the energies of the real levels."""


class Cancelled(RuntimeError):
    """A rebuild was cancelled by the user."""


# Marinescu-Sadeghpour-Dalgarno parameters, the values ARC uses (read-only: not editable in the GUI).
MARINESCU = {
    "Li": dict(Z=3, alphaC=0.1923,
               a1=[2.47718079, 3.45414648, 2.51909839, 2.51909839], a2=[1.84150932, 2.5515108, 2.4371245, 2.4371245],
               a3=[-0.02169712, -0.21646561, 0.32505524, 0.32505524], a4=[-0.11988362, -0.06990078, 0.1060243, 0.1060243],
               rc=[0.61340824, 0.61566441, 2.34126273, 2.34126273]),
    "Na": dict(Z=11, alphaC=0.9448,
               a1=[4.82223117, 5.08382502, 3.53324124, 1.11056646], a2=[2.45449865, 2.18226881, 2.48697936, 1.05458759],
               a3=[-1.12255048, -1.19534623, -0.75688448, 1.73203428], a4=[-1.42631393, -1.03142861, -1.27852357, -0.09265696],
               rc=[0.45489422, 0.45798739, 0.71875312, 28.6735059]),
    "K": dict(Z=19, alphaC=5.331,
              a1=[3.56079437, 3.65670429, 4.12713694, 1.42310446], a2=[1.83909642, 1.67520788, 1.79837462, 1.27861156],
              a3=[-1.74701102, -2.07416615, -1.69935174, 4.77441476], a4=[-1.03237313, -0.89030421, -0.98913582, -0.94829262],
              rc=[0.83167545, 0.85235381, 0.83216907, 6.50294371]),
    "Rb": dict(Z=37, alphaC=9.0760,
               a1=[3.69628474, 4.44088978, 3.78717363, 2.39848933], a2=[1.64915255, 1.92828831, 1.57027864, 1.76810544],
               a3=[-9.86069196, -16.79597770, -11.65588970, -12.07106780], a4=[0.19579987, -0.8163314, 0.52942835, 0.77256589],
               rc=[1.66242117, 1.50195124, 4.86851938, 4.79831327]),
    "Cs": dict(Z=55, alphaC=15.6440,
               a1=[3.49546309, 4.69366096, 4.32466196, 3.01048361], a2=[1.47533800, 1.71398344, 1.61365288, 1.40000001],
               a3=[-9.72143084, -24.65624280, -6.70128850, -3.20036138], a4=[0.02629242, -0.09543125, -0.74095193, 0.00034538],
               rc=[1.92046930, 2.13383095, 0.93007296, 1.99969677]),
}


def _arc_class(element: str):
    try:
        import arc
    except ImportError as exc:
        raise ExactBackendUnavailable(
            "The exact model calculation needs the ARC package for level energies "
            "(install it with: uv sync --extra arc, or pip install arc-alkali-rydberg-calculator)."
        ) from exc
    return {"Li": arc.Lithium7, "Na": arc.Sodium, "K": arc.Potassium39, "Rb": arc.Rubidium85, "Cs": arc.Caesium}[element]


def polarization_term(form: str, r, alpha_d: float, rc: float, user: tuple[str, list] | None = None):
    """The core polarization correction g(r) in r * [1 - lam * g(r)] for the chosen form (prefix of the form name)."""
    kind = form[:1]
    if kind == "U":
        from .userform import evaluate
        return evaluate(user[0], user[1], r, alpha_d, rc)
    if kind == "N":
        return alpha_d / r**3 * (1 - np.exp(-((r / rc) ** 3)))
    if kind == "G":
        return alpha_d / r**3 * (1 - np.exp(-((r / rc) ** 6)))
    if kind == "P":
        return alpha_d / (r * r + rc * rc) ** 1.5
    return alpha_d / r**3 * np.sqrt(1 - np.exp(-((r / rc) ** 6)))        # "E": the shipped form


def _numerov(Q: np.ndarray, h: float, u0: float, u1: float) -> np.ndarray:
    """Outward solution of u'' = Q u on a uniform grid."""
    f = (1.0 - h * h / 12.0 * Q).tolist()
    u = [0.0] * len(f)
    u[0], u[1] = u0, u1
    for i in range(1, len(f) - 1):
        u[i + 1] = ((12.0 - 10.0 * f[i]) * u[i] - f[i - 1] * u[i - 1]) / f[i + 1]
    return np.array(u)


class ExactModel:
    """Model f-values for one species and one set of model settings."""

    def __init__(self, element: str, params: dict | None = None, cache_dir=None):
        self.element = element
        self.params = dict(params or species_params(element))
        self.hash = params_hash(element)
        self.mp = {k: (list(v) if isinstance(v, list) else v) for k, v in MARINESCU[element].items()}
        self.nl = self.params["nl"]
        self._atom = None
        self._bs: dict = {}
        self._f: dict = {}
        self._sigma = None
        self._tuned = False
        self._cache_path = None
        d = cache_dir if cache_dir is not None else user_dir()
        if d is not None:
            from pathlib import Path
            self._cache_path = Path(d) / f"exact_cache_{element}_{self.hash}.json"
            self._load_cache()

    # ---- the real levels (ARC) ---------------------------------------------------------------
    @property
    def atom(self):
        if self._atom is None:
            self._atom = _arc_class(self.element)()
        return self._atom

    def energy_ha(self, n: int, l: int, j: float) -> float:
        return float(self.atom.getEnergy(n, l, j)) / HA_EV

    def nu(self, n: int, J: float) -> float:
        return 1.0 / math.sqrt(2 * abs(self.energy_ha(n, 1, J)))

    # ---- model potential and bound states ----------------------------------------------------
    def _vc(self, l: int, r):
        p = self.mp
        zl = 1.0 + (p["Z"] - 1.0) * np.exp(-p["a1"][l] * r) - r * (p["a3"][l] + p["a4"][l] * r) * np.exp(-p["a2"][l] * r)
        return -zl / r - p["alphaC"] / (2 * r**4) * (1 - np.exp(-((r / p["rc"][l]) ** 6)))

    def potential(self, l: int, j: float, r):
        v = self._vc(l, r)
        ls = (j * (j + 1) - l * (l + 1) - 0.75) / 2.0
        re = np.maximum(r, RCUT)
        h = 1e-5 * re
        dv = (self._vc(l, re + h) - self._vc(l, re - h)) / (2 * h)
        scale = self.params["so_scale"] if l == 1 else 1.0
        return v + ALPHA**2 / 2 * ls * dv / re * scale

    def _radial_u(self, l, j, E, x, r, h):
        Q = 2.0 * r * r * (self.potential(l, j, r) - E) + (l + 0.5) ** 2
        return _numerov(Q, h, math.exp((l + 0.5) * x[0]), math.exp((l + 0.5) * x[1]))

    def bound_state(self, n: int, l: int, j: float, nu_hint: float | None = None):
        """(E, r, P, x) for the state with n-l-1 nodes. ``nu_hint`` narrows the energy scan to that level."""
        rmax = 2.0 * n * n + 60.0 * n + 40
        x = np.arange(math.log(R_MIN), math.log(rmax), H_STEP)
        r = np.exp(x)
        target = n - l - 1

        def nodes_and_tail(E):
            u = self._radial_u(l, j, E, x, r, H_STEP)
            sg = np.sign(u)
            return int(np.sum(sg[1:] * sg[:-1] < 0)), u[-1]

        scans = []
        if nu_hint:
            scans.append(np.linspace(nu_hint - 0.4, nu_hint + 0.4, 41))
        scans.append(np.linspace(max(0.9, n - 6.0), n + 1.5, 500))
        for nus in scans:
            prev_nodes, e_prev = None, None
            for E in -0.5 / nus**2:
                nd, _ = nodes_and_tail(E)
                if prev_nodes == target and nd == target + 1:
                    eb = brentq(lambda ee: nodes_and_tail(ee)[1], e_prev, E, xtol=1e-15)
                    u = self._radial_u(l, j, eb, x, r, H_STEP)
                    P = u * np.sqrt(r)
                    k0 = int(np.searchsorted(r, 1.0 / abs(eb)))
                    cut = max(k0 + int(np.argmin(np.abs(P)[k0:])), k0 + 1)
                    P, rr, xx = P[:cut], r[:cut], x[:cut]
                    return eb, rr, P / np.sqrt(np.trapezoid(P * P * rr, xx)), xx
                prev_nodes, e_prev = nd, E
        raise RuntimeError(f"no bound state found for {self.element} n={n} l={l} j={j}")

    def _bound(self, n, l, j, nu_hint=None):
        key = (n, l, j)
        if key not in self._bs:
            self._bs[key] = self.bound_state(n, l, j, nu_hint)
        return self._bs[key]

    def _tune(self):
        """np cutoff: the lowest np level (J average) must sit at its real energy."""
        if self._tuned:
            return
        nl = self.nl
        nus = {j: self.nu(nl, j) for j in (0.5, 1.5)}   # ns level is s: use the np level at n=nl
        r0 = self.mp["rc"][1]

        def miss(rc):
            self.mp["rc"][1] = rc
            e = [self.bound_state(nl, 1, j, nus[j])[0] - self.energy_ha(nl, 1, j) for j in (0.5, 1.5)]
            return 0.5 * (e[0] + e[1])

        self.mp["rc"][1] = brentq(miss, r0 * 0.5, r0 * 1.6, xtol=1e-6)
        self._bs.clear()
        self._tuned = True

    # ---- oscillator strengths ----------------------------------------------------------------
    def f(self, n: int, J: float) -> float:
        """Absorption oscillator strength ns -> n p_J."""
        key = f"{n}_{J}"
        if key in self._f:
            return self._f[key]
        self._tune()
        nl = self.nl
        b1 = self._bound(nl, 0, 0.5)
        b2 = self._bound(n, 1, J, self.nu(n, J))
        r1, P1 = b1[1], b1[2]
        r2, P2 = b2[1], b2[2]
        m = min(len(r1), len(r2))
        r = r1[:m]
        x = b1[3][:m]
        base = P1[:m] * np.interp(r, r2, P2) * r
        al, rc, lam = self.params["singor_alpha_D"], self.params["singor_rc"], self.params["lam"]
        i0 = np.trapezoid(base * r, x)
        i1 = np.trapezoid(base * r * polarization_term(self.params.get("form") or model_params()["form"], r, al, rc, self._user()), x)
        om = self.energy_ha(n, 1, J) - self.energy_ha(nl, 0, 0.5)
        wcore = self.params.get("wcore_eV")
        sd = 1.0 / (1.0 - (om / (wcore / HA_EV)) ** 2) if wcore else 1.0
        value = float(om * (2 * J + 1) / 9.0 * (i0 - lam * sd * i1) ** 2)   # f = 2 w R^2 (2J+1)/9 with the 6j of ns->np
        self._f[key] = value
        self._save_cache()
        return value

    def _user(self):
        return (self.params.get("user_formula", ""), list(self.params.get("user_params") or []))

    def sigma_th_Mb(self) -> float:
        """Model threshold cross-section from F = f nu^3 at n = 30, 40, 60 extrapolated (quadratic in 1/nu^2) to threshold."""
        if self._sigma is None:
            total = 0.0
            for J in (0.5, 1.5):
                s, F = [], []
                for n in (30, 40, 60):
                    nu = self.nu(n, J)
                    s.append(1 / nu**2)
                    F.append(self.f(n, J) * nu**3)
                total += float(np.polyfit(s, F, 2)[-1])
            self._sigma = SIGMA_TH_FACTOR_MB * total
        return self._sigma

    # ---- cache of finished levels (same settings only: the file name carries the hash) -------
    def _load_cache(self):
        try:
            self._f.update(json.loads(self._cache_path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass

    def _save_cache(self):
        if self._cache_path is None:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(self._f), encoding="utf-8")
        except OSError:
            pass


def n_grid(nl: int) -> list[int]:
    return [*range(nl + 1, N_GRID_TOP + 1), *N_GRID_TAIL]


def build_table(element: str, out_dir, progress: Callable[[int, int, str], None] | None = None,
                cancelled: Callable[[], bool] | None = None) -> dict:
    """Compute F = f nu^3 on the standard grid for the current settings and write ``phys_table_<El>.json`` to ``out_dir``."""
    from pathlib import Path
    model = ExactModel(element, cache_dir=None)
    grid = n_grid(model.nl)
    total = 2 * len(grid)
    rows = []
    done = 0
    for n in grid:
        for J in (0.5, 1.5):
            if cancelled and cancelled():
                raise Cancelled("Rebuild cancelled.")
            nu = model.nu(n, J)
            f = model.f(n, J)
            rows.append({"n": n, "J": J, "nu": nu, "f": f, "F": f * nu**3})
            done += 1
            if progress:
                progress(done, total, f"{element} n={n} J={J}")
    sp = model.params
    table = {"el": element, "hash": model.hash, "rows": rows,
             "params": {"lam": sp["lam"], "wcore_eV": sp.get("wcore_eV")}}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"phys_table_{element}.json").write_text(json.dumps(table), encoding="utf-8")
    return table


class ExactModelView:
    """The :class:`~linak.physics.extrapolation.tables.Table` interface on top of the exact calculation.

    The fit asks for f at an effective quantum number; each is matched to the real level of that nu, so only real
    levels inside the table range can be evaluated (the same range as the tables).
    """

    fresh = True
    _models: ClassVar[dict] = {}

    def __init__(self, element: str):
        from .tables import OutOfTableRange, packaged_table
        self._oor = OutOfTableRange
        self.element = element
        key = (element, params_hash(element), str(user_dir()))
        if key not in self._models:
            type(self)._models[key] = ExactModel(element)
        self.model = self._models[key]
        self.table_hash = self.model.hash
        ref = packaged_table(element)               # same range as the tables
        self.nu_min, self.nu_max = ref.nu_min, ref.nu_max
        self._levels: dict = {}

    def _level(self, nu: float, J: float) -> int:
        if J not in self._levels:
            nl = self.model.nl
            self._levels[J] = [(n, self.model.nu(n, J)) for n in range(nl + 1, N_GRID_TAIL[-1] + 1)]
        n, nun = min(self._levels[J], key=lambda t: abs(t[1] - nu))
        if abs(nun - nu) > 0.25:
            raise self._oor(f"{self.element} nu={nu:.3f} matches no p level of J={J}")
        return n

    def contains(self, nu: float) -> bool:
        return self.nu_min * 0.99 <= nu <= self.nu_max * 1.001

    def f(self, nu: float, J: float) -> float:
        if not self.contains(nu):
            raise self._oor(f"{self.element} nu={nu:.3f} outside the range [{self.nu_min:.3f}, {self.nu_max:.3f}]")
        return self.model.f(self._level(nu, J), J)

    def F(self, nu: float, J: float) -> float:
        return self.f(nu, J) * nu**3

    def sigma_th_Mb(self) -> float:
        return self.model.sigma_th_Mb()
