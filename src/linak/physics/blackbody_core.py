"""Blackbody radiation (BBR) shifts, depopulation and photoionization: the pure computation.

Four calculations, one ``compute_*`` entry point each:

1. ``compute_shifts`` — the static (Itano) shift −½ α(0) ⟨E²⟩_T of the ground and one
   excited state, and the full static + dynamic shift from the Planck spectrum
   integrated against α(ω), at T and across 10–1000 K.
2. ``compute_rates`` — bound-bound BBR depopulation Γ_BBR from the Einstein A of the
   lifetimes network and the thermal photon number n̄(ω, T).
3. ``compute_continuum`` — direct BBR photoionization Γ_PI (Beterov et al., New J. Phys.
   11, 013052 (2009), eq. 27), optional field ionization after BBR transfer (eq. 30),
   and ionization through neighbouring states (eq. 35).
4. ``compute_rydberg_shifts`` — static shifts of high-n states from α(0) networks, and
   dynamic shifts (each channel a principal value against the Planck spectrum) on a
   network completed by a QDT tail, TRK closure and a Kramers continuum, so high n
   approaches the Farley–Wing limit
   π (k_B T)² / (3 c³) = +2416.666 Hz at 300 K.

The continuum reads the bound-bound rates as stored in the JSON, with 12
significant digits (P46: no longer rounded to 10⁻⁶ s⁻¹).

No files, prints or plots here: the ``blackbody`` pipeline loads the element data,
the α networks and the lifetimes JSON and passes them in.
"""

from __future__ import annotations

from linak.core import user_tables as _user_tables
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline

from linak.core.constants import (
    A0_SI,
    ATOMIC_FREQ_RAD_S,
    AU_TO_SI_ALPHA,
    C_SI,
    E_FIELD_REF_T_K,
    E_FIELD_REF_V_M,
    EATOMIC_V_M,
    EPS0_SI,
    H_SI,
    HARTREE_TO_EV,
    HARTREE_TO_HZ,
    HBAR_SI,
    KB_SI,
)
from linak.physics.qdt_radial import level_defect
from linak.physics.polarizability_core import (
    L_CHAR,
    alpha0_static,
    alpha_real,
    alpha_real_vectorized,
    build_transitions_for_state,
    coulomb_fosc,
)

# Headroom: alpha(0) for target n needs upward partners above n; n near
# series n_max is truncated and unreliable.
N_PAD_DEFAULT = 5

# Discrete Rydberg tail above series n_max + Kramers continuum, with TRK
# closure so signed oscillator strength sums to 1 (one valence electron).
# Coulomb same-n f values otherwise overshoot TRK (~3) and inflate dyn to
# ~3x Farley-Wing; completion restores the high-n asymptote.
N_TAIL_DEFAULT = 200
# Table of the thermal channel integral F(y), y = ħω/k_BT: dense enough for cubic
# interpolation to about 1e-7; beyond it the exact limits take over.
_PV_Y_MIN = 1e-4
_PV_Y_MAX = 1e3
_PV_POINTS = 800
_PV_THERMAL_CUTOFF = 80.0
N_CONT_BINS_DEFAULT = 80
EPS_MAX_AU_DEFAULT = 0.05


# ============================================================================
# RESULT RECORDS
# ============================================================================

@dataclass(frozen=True)
class Shifts:
    """Static and static + dynamic BBR shifts of the ground and one excited state."""

    T_K: float
    alpha0_gs_au: float
    alpha0_ex_au: float | None          # None without an excited state
    stat_gs_hz: float                   # static (Itano) at T_K
    stat_ex_hz: float | None
    dyn_gs_hz: float                    # static + dynamic (Planck integral) at T_K
    dyn_ex_hz: float | None
    T_stat_K: np.ndarray                # 10..1000 K, 400 points
    stat_gs_curve_hz: np.ndarray
    stat_ex_curve_hz: np.ndarray | None
    T_dyn_K: np.ndarray                 # 10..1000 K, 60 points
    dyn_gs_curve_hz: np.ndarray
    dyn_ex_curve_hz: np.ndarray | None

    @property
    def stat_diff_hz(self) -> float | None:
        """Shift of the ground → excited transition (excited − ground)."""
        return None if self.stat_ex_hz is None else self.stat_ex_hz - self.stat_gs_hz

    @property
    def dyn_diff_hz(self) -> float | None:
        return None if self.dyn_ex_hz is None else self.dyn_ex_hz - self.dyn_gs_hz

    @property
    def stat_diff_curve_hz(self) -> np.ndarray | None:
        if self.stat_ex_curve_hz is None:
            return None
        return self.stat_ex_curve_hz - self.stat_gs_curve_hz

    @property
    def dyn_diff_curve_hz(self) -> np.ndarray | None:
        if self.dyn_ex_curve_hz is None:
            return None
        return self.dyn_ex_curve_hz - self.dyn_gs_curve_hz


@dataclass(frozen=True)
class LifetimeGraph:
    """The bidirectional E1 network of a ``<El>_lifetimes.json``.

    states: {label: {n, l, A_total_s, lifetime_s}}
    edges_from: {label: [(partner, A_ul, g_u, g_l, wl_nm, kind), ...]}
      kind = 'stim' (upper->lower stimulated emission) or 'abs' (lower->upper)
    """

    states: dict
    edges_from: dict
    n_start: int | None


@dataclass(frozen=True)
class BBRChannel:
    """One BBR-driven transfer out of a state."""

    partner: str
    kind: str               # 'stim' (to a lower state) or 'abs' (to an upper state)
    wavelength_nm: float
    A_s: float
    nbar: float
    rate_s: float


@dataclass(frozen=True)
class Depopulation:
    """Bound-bound BBR depopulation of one state."""

    state: str
    n: int
    l: str
    Gamma_rad_s: float
    Gamma_BBR_s: float
    tau_rad_s: float | None
    tau_eff_s: float | None
    channels: list[BBRChannel]          # strongest first

    @property
    def BBR_over_rad(self) -> float | None:
        return self.Gamma_BBR_s / self.Gamma_rad_s if self.Gamma_rad_s > 0 else None


@dataclass(frozen=True)
class Rates:
    T_K: float
    n_min: int
    n_start: int
    states: list[Depopulation]          # n >= n_min, sorted (n, l, state)
    sanity: list[Depopulation]          # the n = n_start states
    T_sweep_K: np.ndarray               # 50..500 K, 40 points
    sweep_Gamma_BBR_s: dict[str, list[float]]   # a few s/p/d states across T_sweep_K


@dataclass(frozen=True)
class ContinuumRow:
    """Photoionization / field ionization / quenching of one state."""

    state: str
    n: int
    l: str
    n_star: float
    L: int
    A_L: float
    phase_weight: float
    E_bind_eV: float
    Gamma_PI_s: float
    Gamma_SFI_s: float
    field_ionized: bool       # the state itself is above the SFI threshold
    Gamma_mix_PI_s: float
    Gamma_mix_SFI_s: float
    Gamma_ion_s: float
    Gamma_BBR_s: float
    Gamma_rad_s: float
    Gamma_quench_s: float
    tau_eff_s: float | None
    n_mix_partners: int


@dataclass(frozen=True)
class Continuum:
    T_K: float
    n_min: int
    field_V_cm: float | None
    n_c: float | None                   # SFI threshold n*, None without a field
    mixed: bool                         # eq. (35) mixing ran
    t1_s: float
    t2_s: float
    mix_dn_max: int
    rows: list[ContinuumRow]            # incl. rate states below n_min (mixing partners)
    probes: list[ContinuumRow]          # ground, lowest p, 10s, 20s, 25s when present

    @property
    def table(self) -> list[ContinuumRow]:
        """The reported states: n >= n_min."""
        return [r for r in self.rows if r.n >= self.n_min]


@dataclass(frozen=True)
class Completion:
    """TRK / tail / continuum completion of one dynamic network (rounded to 1e-6)."""

    trk_scale: float
    f_cont: float
    n_tail: int
    n_cont: int
    n_tail_max: int
    trk_sum: float | None


@dataclass(frozen=True)
class RydbergShift:
    state: str
    n: int | None
    l: str | None
    alpha0_au: float
    shift_stat_hz: float
    shift_dyn_hz: float | None          # None when the dynamic integral is skipped
    n_transitions: int
    min_dE_au: float | None
    fosc_sources: dict[str, int]
    completion: Completion | None       # None when the dynamic integral is skipped
    static_valid: bool = True           # False: the quadratic static formula does not apply (see static_validity)
    static_note: str = ''               # why, in a few words
    max_coupling: float | None = None   # largest |d.E_rms| / |dE| over the valence partners

    @property
    def dyn_reliable(self) -> bool:
        """False when the valence network cannot close the TRK sum.

        The completion puts missing strength into the continuum, so a negative signed f
        sum, which needs f_cont > 1, means a nearby partner is absent from the network:
        the shift is then wrong however exactly it is integrated.
        """
        return self.completion is None or self.completion.f_cont <= 1.0


@dataclass(frozen=True)
class RydbergShifts:
    T_K: float
    table: bool                         # False: only the --state labels
    n_min: int | None                   # the table window, None without a table
    n_max: int | None
    n_pad: int
    series_n_max: int
    do_dyn: bool
    do_cont: bool
    n_tail_max: int
    farley_wing_hz: float
    states: list[RydbergShift]
    skipped: list[str]                  # labels without an energy or a network
    probes: list[RydbergShift]          # ground, lowest p, 10s … 40s when present



# BLACKBODY RADIATION SHIFT

def mean_E_squared_au(T_K):
    """
    Mean-square thermal electric field <E^2>_T, in atomic units.
        <E^2>_T = (831.9 V/m)^2 (T / 300 K)^4     (Itano, Lewis & Wineland 1982)
    """
    E_rms_V_m = E_FIELD_REF_V_M * (T_K / E_FIELD_REF_T_K) ** 2
    E_rms_au = E_rms_V_m / EATOMIC_V_M
    return E_rms_au ** 2


def bbr_static_shift_hz(alpha0_au, T_K):
    """
    Static (leading-order) blackbody radiation shift, in Hz.
        ΔE = -(1/2) α(0) <E^2>_T                          [Hartree]
        Δν = ΔE * (Eh/h)                                    [Hz]
    Dominant contribution to the BBR shift (>98% typically) for alkali-like
    atoms at T ~ 300 K; bbr_total_shift_hz adds the dynamic correction.
    """
    E2_au = mean_E_squared_au(T_K)
    dE_au = -0.5 * alpha0_au * E2_au
    return dE_au * HARTREE_TO_HZ + 0.0      # + 0.0 turns IEEE -0.0 (T = 0) into 0.0


def _u_planck_arr(omega_rad_s, T_K):
    """Planck spectral energy density u(omega,T), J / (m^3 * rad/s)."""
    x = np.clip(HBAR_SI * omega_rad_s / (KB_SI * T_K), 1e-300, 700.0)
    return (HBAR_SI * omega_rad_s**3) / (math.pi**2 * C_SI**3) / np.expm1(x)


def bbr_total_shift_hz(transitions_list, T_K, n_pts=4000, omega_max_factor=40):
    """
    Full (static + dynamic) blackbody shift, in Hz, via direct numerical
    integration of the Planck spectrum against the real dynamic
    polarizability alpha(omega):

        Delta nu (T) = -(1/(2 eps0 h)) integral_0^omega_max u(omega,T) alpha_SI(omega) d omega

    omega_max_factor * (kB T / hbar) sets the upper integration cutoff --
    the Planck factor suppresses the integrand exponentially well before
    this for any alkali D-line-scale resonance at T <~ 1000 K. n_pts=4000
    was checked to be converged to 6 significant figures against
    n_pts=20000 on real Na data.
    """
    if not transitions_list:
        return 0.0
    kT_over_hbar = KB_SI * T_K / HBAR_SI
    omega_max = omega_max_factor * kT_over_hbar
    w_grid = np.linspace(1e-3 * kT_over_hbar, omega_max, n_pts)
    w_au = w_grid / ATOMIC_FREQ_RAD_S
    alpha_SI = alpha_real_vectorized(w_au, transitions_list) * AU_TO_SI_ALPHA
    integrand = _u_planck_arr(w_grid, T_K) * alpha_SI
    val_J = -0.5 / EPS0_SI * np.trapezoid(integrand, w_grid)
    return val_J / H_SI


def farley_wing_shift_hz(T_K):
    """
    Farley-Wing high-n asymptotic BBR Stark shift (PRA 23, 2397 (1981)):
        delta_E = pi (k_B T)^2 / (3 c^3)     [Hartree]
        delta_nu = delta_E * (E_h / h)       [Hz]
    At T = 300 K this is +2416.666 Hz (independent of n, l).
    """
    kb_au = KB_SI / (H_SI * HARTREE_TO_HZ)
    c_au = C_SI / (A0_SI * ATOMIC_FREQ_RAD_S)
    dE_au = math.pi * (kb_au * float(T_K)) ** 2 / (3.0 * c_au ** 3)
    return dE_au * HARTREE_TO_HZ


@lru_cache(maxsize=1)
def _planck_pv_spline() -> CubicSpline:
    """F(y) tabulated on a log grid; see :func:`planck_pv_integral`."""
    def thermal(x):
        return x * x * (x / math.expm1(x) if x > 0.0 else 1.0) if x < 700.0 else 0.0

    def value(y):
        if y > _PV_THERMAL_CUTOFF:   # the pole lies where the spectrum has vanished
            return quad(lambda x: thermal(x) / ((y - x) * (y + x)), 0.0, _PV_THERMAL_CUTOFF,
                        limit=400, epsabs=0.0, epsrel=1e-11)[0]
        # F = -PV∫ g(x)/(x - y) dx with g = x³/(eˣ-1)/(x + y), the pole isolated on [0, 2y].
        near = quad(lambda x: thermal(x) / (x + y), 0.0, 2.0 * y,
                    weight='cauchy', wvar=y, limit=200, epsabs=0.0, epsrel=1e-11)[0]
        far = quad(lambda x: thermal(x) / ((y - x) * (y + x)), 2.0 * y, np.inf,
                   limit=400, epsabs=1e-13, epsrel=1e-11)[0]
        return far - near

    log_y = np.linspace(math.log(_PV_Y_MIN), math.log(_PV_Y_MAX), _PV_POINTS)
    return CubicSpline(log_y, [value(math.exp(v)) for v in log_y])


def planck_pv_integral(y):
    """
    The universal thermal integral of one E1 channel, for y = ħω_k / k_B T ≥ 0:

        F(y) = PV ∫₀^∞ x³ / (eˣ - 1) · dx / (y² - x²)

    Its limits are exact: F → -π²/6 for y → 0 (a partner much closer than k_B T, which
    is what gives the Farley-Wing shift) and F → π⁴ / (15 y²) for y → ∞ (a partner far
    above k_B T, which gives the static Itano shift).
    """
    y = np.asarray(y, dtype=float)
    spline = _planck_pv_spline()
    inside = spline(np.log(np.clip(y, _PV_Y_MIN, _PV_Y_MAX)))
    with np.errstate(divide='ignore', invalid='ignore'):
        above = math.pi ** 4 / (15.0 * y ** 2) + 8.0 * math.pi ** 6 / (63.0 * y ** 4)
    return np.where(y > _PV_Y_MAX, above, inside)


def bbr_total_shift_pv_hz(transitions_list, T_K):
    """
    Full BBR shift (Hz) of a state whose partners may sit inside the thermal spectrum.

    Each channel of α(ω) = Σ sign f / (ω_k² - ω²) is integrated against the Planck
    spectrum as a principal value. With x = ħω/k_BT and θ = k_BT/ħ,

        Δν = -(1 / 2ε₀h) (ħθ⁴ / π²c³) Σ sign f F(y_k) / θ²   (θ in a.u. in the last factor)

    A grid over ω cannot do this: the resonances are many orders of magnitude narrower
    than any practical step, so the sum samples them at random and the result depends
    on the grid. Their widths need no treatment here, because Γ ≪ k_BT/ħ for every
    channel: the integrated effect of a narrow line depends only on its position.
    """
    if not transitions_list:
        return 0.0
    theta = KB_SI * T_K / HBAR_SI
    theta_au = theta / ATOMIC_FREQ_RAD_S
    f = np.array([t['fosc'] for t in transitions_list], dtype=float)
    sign = np.array([t['sign'] for t in transitions_list], dtype=float)
    y = np.abs(np.array([t['delta_E_au'] for t in transitions_list], dtype=float)) / theta_au
    channels = float(np.sum(sign * f * planck_pv_integral(y)))
    planck = HBAR_SI * theta ** 4 / (math.pi ** 2 * C_SI ** 3)
    return -0.5 / EPS0_SI * AU_TO_SI_ALPHA * planck * channels / theta_au ** 2 / H_SI


def compute_shifts(trans_gs: Sequence[dict], trans_ex: Sequence[dict] | None,
                   T_K: float) -> Shifts:
    """Ground (and excited, when *trans_ex* is non-empty) BBR shifts at T_K and vs T."""
    alpha0_gs = alpha_real(0.0, trans_gs)
    alpha0_ex = alpha_real(0.0, trans_ex) if trans_ex else None

    # T-sweep (10 K to 1000 K): static term (cheap, analytic)
    T_arr = np.linspace(10, 1000, 400)
    shift_gs_arr = np.array([bbr_static_shift_hz(alpha0_gs, T) for T in T_arr])
    shift_ex_arr = (np.array([bbr_static_shift_hz(alpha0_ex, T) for T in T_arr])
                    if alpha0_ex is not None else None)

    # T-sweep: full dynamic-corrected term (numerical integral; coarser
    # grid than the static sweep since each point costs a full integration,
    # but n_pts=4000 per point converges to 6 s.f. so 60 T-points is ample)
    T_dyn_arr = np.linspace(10, 1000, 60)
    dyn_gs_arr = np.array([bbr_total_shift_hz(trans_gs, T) for T in T_dyn_arr])
    dyn_ex_arr = (np.array([bbr_total_shift_hz(trans_ex, T) for T in T_dyn_arr])
                  if trans_ex else None)

    return Shifts(
        T_K=T_K,
        alpha0_gs_au=alpha0_gs,
        alpha0_ex_au=alpha0_ex,
        stat_gs_hz=bbr_static_shift_hz(alpha0_gs, T_K),
        stat_ex_hz=bbr_static_shift_hz(alpha0_ex, T_K) if alpha0_ex is not None else None,
        dyn_gs_hz=bbr_total_shift_hz(trans_gs, T_K),
        dyn_ex_hz=bbr_total_shift_hz(trans_ex, T_K) if trans_ex else None,
        T_stat_K=T_arr,
        stat_gs_curve_hz=shift_gs_arr,
        stat_ex_curve_hz=shift_ex_arr,
        T_dyn_K=T_dyn_arr,
        dyn_gs_curve_hz=dyn_gs_arr,
        dyn_ex_curve_hz=dyn_ex_arr,
    )


# ============================================================================
# NETWORK COMPLETION FOR THE DYNAMIC RYDBERG SHIFTS
# ============================================================================

def _qdt_energy_au(n, l_char, qd, mass_factor=1.0):
    """E = -(R_M/R_inf)/(2 n*^2) in Hartree; mass_factor = R_M/(Z^2 R_inf) (1.0 = infinite mass)."""
    n_eff = int(n) - level_defect(qd, n, l_char)
    if n_eff <= 0:
        return None
    return -0.5 * mass_factor / (n_eff * n_eff)


def _rydberg_mass_factor(data):
    """R_M / (Z^2 R_inf) from the Rydberg JSON's reduced-mass constant (1.0 if absent)."""
    ryd = data.get('ryd') or {}
    ry = ryd.get('rydberg_constant_eV')
    if not ry:
        return 1.0
    z = float(ryd.get('z_core') or 1.0)
    return float(ry) / (z * z * (HARTREE_TO_EV / 2.0))


def discrete_tail_partners(label, data, n_from, n_to):
    """
    Synthetic E1 partners n_from..n_to via QDT energies + Coulomb fosc.
    Upward only (higher Rydberg / toward continuum).
    """
    n0, l0 = parse_nl_label(label)
    if n0 is None or l0 is None:
        return []
    E0 = data['state_energy_au'].get(label)
    if E0 is None:
        return []
    L0 = L_CHAR.get(l0)
    if L0 is None:
        return []
    qd = data.get('qd') or {}
    _mass_factor = _rydberg_mass_factor(data)
    out = []
    n_from, n_to = int(n_from), int(n_to)
    d0 = level_defect(qd, n0, l0)
    # Exact radial integrals (hydrogen-validated Coulomb/Numerov hybrid in polarizability_core) for n' up to
    # n_dense, then a PCHIP interpolation of g(n') = f n*'^3 / dE in 1/n*' on a geometric anchor set.
    # g is smooth for n' >> n0 (threshold law f ~ dE / n*'^3); it is NOT smooth near the diagonal
    # (n' ~ n0, dE -> 0), hence the dense exact region n_dense ~ 1.5 n0. Interpolation error < 2 %.
    n_dense = min(n_to, max(n_from + 9, int(math.ceil(1.5 * n0))))
    for l_char, L in L_CHAR.items():
        if abs(L - L0) != 1:
            continue
        du = {n: level_defect(qd, n, l_char) for n in range(n_from, n_to + 1)}
        exact_ns = list(range(n_from, n_dense + 1))
        f_of = {n: coulomb_fosc(n, L, n0, L0, du[n], d0) for n in exact_ns}
        if n_to > n_dense:
            anchors = sorted(set([n_dense] + [int(round(n_dense * (n_to / n_dense) ** (k / 8.0)))
                                              for k in range(1, 9)] + [n_to]))
            xs, ys = [], []
            for na in anchors:
                nsu, nsl = na - du[na], n0 - d0
                dE = 0.5 / nsl ** 2 - 0.5 / nsu ** 2
                fa = f_of[na] if na in f_of else coulomb_fosc(na, L, n0, L0, du[na], d0)
                if fa > 0 and dE > 0:
                    xs.append(1.0 / nsu)
                    ys.append(fa * nsu ** 3 / dE)
            if len(xs) >= 3:
                from scipy.interpolate import PchipInterpolator
                P = PchipInterpolator(xs[::-1], ys[::-1])
                for n in range(n_dense + 1, n_to + 1):
                    nsu, nsl = n - du[n], n0 - d0
                    dE = 0.5 / nsl ** 2 - 0.5 / nsu ** 2
                    f_of[n] = max(float(P(1.0 / nsu)), 0.0) * dE / nsu ** 3
        for n in range(n_from, n_to + 1):
            E = _qdt_energy_au(n, l_char, qd, _mass_factor)
            if E is None or E <= E0:
                continue
            f = f_of.get(n, 0.0)
            if f <= 0:
                continue
            out.append(dict(
                delta_E_au=E - E0,
                fosc=f,
                sign=+1,
                other_label=f'{n}{l_char}',
                source='QDT-tail',
                wl_nm=0.0,
            ))
    return out


def kramers_continuum_partners(E_bind_au, f_cont, n_bins=N_CONT_BINS_DEFAULT,
                               eps_max_au=EPS_MAX_AU_DEFAULT):
    """
    Distribute residual TRK strength f_cont over continuum bins with
    Kramers weight (I / (I+eps))^3 above the ionization threshold.
    """
    if f_cont <= 0 or E_bind_au <= 0:
        return []
    I = float(E_bind_au)
    n_bins = max(int(n_bins), 8)
    eps_max = float(eps_max_au)
    de = eps_max / n_bins
    weights = []
    for i in range(n_bins):
        eps = (i + 0.5) * de
        weights.append((I / (I + eps)) ** 3)
    W = sum(weights) * de
    if W <= 0:
        return []
    out = []
    for i, w in enumerate(weights):
        eps = (i + 0.5) * de
        f = f_cont * (w * de) / W
        out.append(dict(
            delta_E_au=I + eps,
            fosc=f,
            sign=+1,
            other_label=f'cont:{eps:.5g}',
            source='continuum',
            wl_nm=0.0,
        ))
    return out


def complete_transitions_for_dyn(label, data, valence, n_tail_max=N_TAIL_DEFAULT,
                                 n_cont_bins=N_CONT_BINS_DEFAULT,
                                 eps_max_au=EPS_MAX_AU_DEFAULT):
    """
    Bound valence + QDT discrete tail + TRK closure + Kramers continuum.

    Returns (channels, completion).
    Static alpha(0) should keep using the uncompleted valence network.
    """
    series_n_max = int(data.get('ryd', {}).get('n_max') or 0)
    channels = [dict(t) for t in valence]
    n_tail = 0
    if n_tail_max and series_n_max and n_tail_max > series_n_max:
        tail = discrete_tail_partners(
            label, data, series_n_max + 1, n_tail_max)
        channels.extend(tail)
        n_tail = len(tail)

    S = sum(t['sign'] * t['fosc'] for t in channels)
    if S > 1.0 + 1e-12:
        scale = 1.0 / S
        for t in channels:
            t['fosc'] = t['fosc'] * scale
            src = str(t.get('source') or '')
            if 'TRK' not in src:
                t['source'] = src + '-TRK' if src else 'TRK'
    else:
        scale = 1.0

    S2 = sum(t['sign'] * t['fosc'] for t in channels)
    f_cont = max(0.0, 1.0 - S2)
    E0 = data['state_energy_au'].get(label)
    E_bind = -float(E0) if E0 is not None and E0 < 0 else 0.0
    cont = kramers_continuum_partners(
        E_bind, f_cont, n_bins=n_cont_bins, eps_max_au=eps_max_au)
    channels.extend(cont)

    completion = Completion(
        trk_scale=round(scale, 6),
        f_cont=round(f_cont, 6),
        n_tail=n_tail,
        n_cont=len(cont),
        n_tail_max=n_tail_max,
        trk_sum=round(sum(t['sign'] * t['fosc'] for t in channels), 6),
    )
    return channels, completion



# BLACKBODY DEPOPULATION RATES (bound-bound)

def nbar_photon(omega_rad_s, T_K):
    """
    Mean thermal photon number nbar(omega, T) = 1 / (exp(hbar omega / kT) - 1).
    """
    if T_K <= 0 or omega_rad_s <= 0:
        return 0.0
    x = HBAR_SI * float(omega_rad_s) / (KB_SI * float(T_K))
    if x > 700.0:
        return 0.0
    if x < 1e-12:
        return 1.0 / max(x, 1e-300)  # Rayleigh-Jeans
    return 1.0 / math.expm1(x)


def omega_from_nm(wl_nm):
    """Angular frequency (rad/s) from vacuum wavelength in nm."""
    return 2.0 * math.pi * C_SI / (float(wl_nm) * 1e-9)


def label_degeneracy(label):
    """
    Statistical weight g = 2J+1 from a state label.
    Examples: '3p1/2' -> 2, '3p3/2' -> 4, '4s' -> 2 (alkali ns = 2S1/2).
    Unresolved non-s: 2*(2l+1).
    """
    m = re.match(r'^(\d+)([spdfgh])(.*)$', str(label).strip())
    if not m:
        return 2
    l_char = m.group(2)
    suffix = m.group(3).strip()
    if suffix:
        if '/' in suffix:
            try:
                num, den = (int(x) for x in suffix.split('/', 1))
                return 2 * num // den + 1
            except (ValueError, ZeroDivisionError):
                pass
        else:
            try:
                return 2 * int(suffix) + 1
            except ValueError:
                pass
    if l_char == 's':
        return 2
    Lmap = {'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5}
    L = Lmap.get(l_char, 0)
    return 2 * (2 * L + 1)


def build_lifetime_graph(life_data: Mapping) -> LifetimeGraph:
    """Bidirectional E1 graph from a lifetimes JSON."""
    states = {}
    edges_from = {}

    def ensure_state(label, n=None, l=None, A_total=0.0, lifetime_s=None):
        if label not in states:
            if n is None or l is None:
                m = re.match(r'^(\d+)([spdfgh])', label)
                n = int(m.group(1)) if m else 0
                l = m.group(2) if m else '?'
            states[label] = dict(
                n=n, l=l,
                A_total_s=float(A_total or 0.0),
                lifetime_s=lifetime_s,
            )
            edges_from[label] = []
        elif A_total:
            states[label]['A_total_s'] = float(A_total)
            if lifetime_s is not None:
                states[label]['lifetime_s'] = lifetime_s

    for entry in life_data.get('lifetimes', []):
        upper = entry['state']
        ensure_state(
            upper, n=entry.get('n'), l=entry.get('l'),
            A_total=entry.get('A_total_s', 0.0),
            lifetime_s=entry.get('lifetime_s'),
        )
        g_u = label_degeneracy(upper)
        for ch in entry.get('channels', []):
            lower = ch['lower_label']
            A = float(ch.get('A_s') or 0.0)
            wl = float(ch.get('wavelength_nm') or 0.0)
            if A <= 0 or wl <= 0:
                continue
            ensure_state(lower)
            g_l = label_degeneracy(lower)
            # From upper: stimulated emission toward lower
            edges_from[upper].append((lower, A, g_u, g_l, wl, 'stim'))
            # From lower: absorption toward upper
            edges_from[lower].append((upper, A, g_u, g_l, wl, 'abs'))

    return LifetimeGraph(states=states, edges_from=edges_from,
                         n_start=life_data.get('n_start'))


def bbr_depopulation(state, graph: LifetimeGraph, T_K) -> Depopulation:
    """
    Bound-bound BBR depopulation for one state of *graph*.

    stim: Gamma = A * nbar
    abs:  Gamma = A * (g_u / g_l) * nbar
    """
    info = graph.states[state]
    Gamma_rad = float(info.get('A_total_s') or 0.0)
    channels = []
    Gamma_BBR = 0.0
    for partner, A, g_u, g_l, wl, kind in graph.edges_from.get(state, []):
        omega = omega_from_nm(wl)
        nb = nbar_photon(omega, T_K)
        if kind == 'stim':
            rate = A * nb
        else:
            rate = A * (g_u / max(g_l, 1e-30)) * nb
        if rate <= 0:
            continue
        Gamma_BBR += rate
        channels.append(BBRChannel(
            partner=partner, kind=kind, wavelength_nm=wl,
            A_s=A, nbar=nb, rate_s=rate,
        ))
    channels.sort(key=lambda c: -c.rate_s)
    Gamma_tot = Gamma_rad + Gamma_BBR
    return Depopulation(
        state=state,
        n=info['n'],
        l=info['l'],
        Gamma_rad_s=Gamma_rad,
        Gamma_BBR_s=Gamma_BBR,
        tau_rad_s=(1.0 / Gamma_rad) if Gamma_rad > 0 else None,
        tau_eff_s=(1.0 / Gamma_tot) if Gamma_tot > 0 else None,
        channels=channels,
    )


def compute_rates(graph: LifetimeGraph, T_K: float, n_min: int | None = None) -> Rates:
    """
    Gamma_BBR / tau_eff for all states with n >= n_min.
    Default n_min = n_start + 2 (skip lowest optical manifold).
    """
    n_start = graph.n_start
    if n_start is None:
        n_start = min((s['n'] for s in graph.states.values() if s['n']), default=1)
    if n_min is None:
        n_min = int(n_start) + 2

    results = [bbr_depopulation(label, graph, T_K)
               for label, info in graph.states.items() if info['n'] >= n_min]
    results.sort(key=lambda r: (r.n, r.l, r.state))

    # Sanity: ground-like absorption-only states (n == n_start, often not in table)
    # and lowest p if present.
    sanity = [bbr_depopulation(label, graph, T_K)
              for label, info in graph.states.items() if info['n'] == int(n_start)]

    # Representative T-sweep for a few s/p states
    by_series: dict[str, list[Depopulation]] = {'s': [], 'p': [], 'd': []}
    for r in results:
        if r.l in by_series:
            by_series[r.l].append(r)
    reps = []
    for rows in by_series.values():
        if not rows:
            continue
        rows_n = sorted(rows, key=lambda r: r.n)
        picks = {rows_n[0].state, rows_n[len(rows_n)//2].state, rows_n[-1].state}
        reps.extend(sorted(picks))
    reps = list(dict.fromkeys(reps))  # unique, preserve order

    T_sweep = np.linspace(50, 500, 40)
    sweep = {lab: [bbr_depopulation(lab, graph, float(T)).Gamma_BBR_s for T in T_sweep]
             for lab in reps}

    return Rates(T_K=T_K, n_min=n_min, n_start=int(n_start), states=results,
                 sanity=sanity, T_sweep_K=T_sweep, sweep_Gamma_BBR_s=sweep)



# HIGH-n RYDBERG alpha(0) / STATIC BBR SHIFTS

# SERIES REPAIR AND VALIDITY OF THE STATIC FORMULA

QDT_ROUND_AU = 2.5e-8           # the Rydberg JSON rounds energies to 1e-6 eV: half a step is 1.84e-8 a.u.
STATIC_COUPLING_MAX = 0.3       # |d E_rms| / |dE| above this: second-order theory is not valid
STATIC_LOWFREQ_MAX = 0.1        # share of alpha(0) from partners with hbar w < k_B T above which "static" fails
H_SERIES_N_MIN = 6              # lowest h state (l = 5 needs n >= 6)


def refine_series_data(data):
    """
    A copy of *data* whose model Rydberg series are consistent for near-degenerate pairs.

    1. Levels the Rydberg stage built from the quantum-defect formula ("QDT model") are taken at
       the formula's own energy instead of its 1e-6 eV rounding. The f-g gap at n = 33 is 3.3e-8 a.u.,
       the size of that rounding step; the stored gap was 0 or one step.
    2. An E1 pair between such levels (n differing by at most 2) that the transitions list
       lacks, because the rounded energies tied, is added with the same f-value code the other pairs use.
    3. An l = 5 (h) series, absent from the data, is added from the quantum-defect formula
       (hydrogenic when the h defect is unknown) so g states have their upward partners.
       Without it a g state has f partners only, the signed f sum comes out near -1 and the dynamic
       shift is wrong (-460 Hz at 25g against the 2417 Hz Farley-Wing limit).
    Energies that differ from the formula by more than the rounding are left alone.
    """
    if data.get('_series_refined'):
        return data
    ryd = data.get('ryd') or {}
    qd = data.get('qd') or {}
    mass_factor = _rydberg_mass_factor(data)
    energies = dict(data['state_energy_au'])
    state_nl = dict(data.get('state_nl') or {})
    model = {}
    for ex in ryd.get('excitations') or []:
        lab, n, l_char = ex.get('label'), ex.get('n'), ex.get('l')
        if (ex.get('source') != 'QDT model' or ex.get('J') is not None or lab not in energies
                or not n or l_char not in L_CHAR):
            continue
        E = _qdt_energy_au(n, l_char, qd, mass_factor)
        if E is None or abs(E - energies[lab]) > QDT_ROUND_AU:
            continue
        energies[lab] = E
        model[lab] = (int(n), l_char)
    n_energy = len(model)

    n_h = 0
    synthetic = []
    series_n_max = int(ryd.get('n_max') or 0)
    if any(l == 'g' for _, l in model.values()) and not any(l == 'h' for _, l in state_nl.values()):
        for n in range(H_SERIES_N_MIN, series_n_max + 1):
            E = _qdt_energy_au(n, 'h', qd, mass_factor)
            if E is None or E >= 0:
                continue
            lab = f'{n}h'
            energies[lab] = E
            state_nl[lab] = (n, 'h')
            model[lab] = (n, 'h')
            synthetic.append(lab)
            n_h += 1

    have = {frozenset((t['upper_label'], t['lower_label'])) for t in data['transitions']}
    added = []

    def _pair(a, na, la, ja, b, nb, lb, jb):
        dE = energies[a] - energies[b]
        if dE == 0:
            return
        if dE < 0:
            a, na, la, ja, b, nb, lb, jb = b, nb, lb, jb, a, na, la, ja
        if lb == 'h' and ja:
            # Fine-structure g level above a collapsed h level: build_transitions_for_state takes the
            # emission factor g_lo / g_up from the J degeneracies when both J are given and from the
            # full nl degeneracies when either is missing.  Giving h the multiplet degeneracy
            # 2 (2l + 1) = 22 (J = 21/2 stands for it) keeps the factor that of the g level.
            jb = '21/2'
        added.append(dict(
            upper_label=a, lower_label=b, upper_n=na, lower_n=nb, upper_l=la, lower_l=lb,
            upper_J=None if la == 'h' else ja, lower_J=None if lb == 'h' and not ja else jb,
            delta_n=abs(na - nb), wavelength_nm=45.56335 / abs(dE),
            upper_source='QDT model', lower_source='QDT model'))

    labels = sorted(model)
    for i, a in enumerate(labels):
        na, la = model[a]
        if la == 'h':
            continue
        for b in labels[i + 1:]:
            nb, lb = model[b]
            if lb == 'h' or abs(L_CHAR[la] - L_CHAR[lb]) != 1 or frozenset((a, b)) in have:
                continue
            if abs(na - nb) > 2:
                continue
            _pair(a, na, la, None, b, nb, lb, None)
    jmap = {ex.get('label'): ex.get('J') for ex in ryd.get('excitations') or []}
    n_merged = _add_merged_j_pairs(state_nl, jmap, energies, have, _pair)
    if n_h:
        h_labels = [lab for lab in synthetic]
        for g_lab, (ng, lg) in state_nl.items():
            if lg != 'g' or g_lab in synthetic or ng is None or g_lab not in energies:
                continue
            for h_lab in h_labels:
                _pair(g_lab, ng, 'g', jmap.get(g_lab), h_lab, state_nl[h_lab][0], 'h', None)
    out = dict(data)
    out['state_energy_au'] = energies
    out['state_nl'] = state_nl
    out['transitions'] = list(data['transitions']) + added
    out['_series_refined'] = dict(n_energy=n_energy, n_h=n_h, n_pairs=len(added), n_merged=n_merged, synthetic=frozenset(synthetic))
    return out


def _j_values(j_string):
    """The J values of a level's J string ('5/2', or '5/2,7/2' for a merged doublet); None when unknown."""
    if not j_string:
        return None
    try:
        return [int(a) / int(b) if '/' in x else float(x) for x in j_string.split(',')
                for a, b in [(x.split('/') + ['1'])[:2]]]
    except ValueError:
        return None


def _add_merged_j_pairs(state_nl, jmap, energies, have, add_pair):
    """
    Pairs the transitions list drops for a merged-J level.

    NIST gives f and g levels from n = 12 on as one level '5/2,7/2' (f) or '7/2,9/2' (g).  The list was
    built with the first J only, so f(5/2,7/2) -> 18g9/2 failed the dJ rule although f7/2 -> g9/2 is allowed,
    and the n = 17 f state kept only 8/18 of its strength to the 18g multiplet (17f dynamic shift
    490 Hz against about 2300).  Every pair between a merged level and a J-resolved level that some
    member of the doublet allows (|dJ| <= 1) is added; the f-value code then weights it by g_J / g_total.
    """
    count = 0
    for x, jx_string in jmap.items():
        jx = _j_values(jx_string)
        if jx is None or ',' not in (jx_string or '') or x not in state_nl or x not in energies:
            continue
        nx, lx = state_nl[x]
        for y, (ny, ly) in state_nl.items():
            if y == x or y not in energies or ly not in L_CHAR or lx not in L_CHAR:
                continue
            if abs(L_CHAR[lx] - L_CHAR[ly]) != 1 or frozenset((x, y)) in have:
                continue
            jy = _j_values(jmap.get(y))
            if jy is None or not any(abs(a - b) <= 1.0 + 1e-9 for a in jx for b in jy):
                continue
            add_pair(x, nx, lx, jx_string, y, ny, ly, jmap.get(y))
            count += 1
    return count


def _kT_au(T_K):
    return KB_SI * T_K / (HARTREE_TO_HZ * H_SI)


def static_validity(valence, T_K, L0=None):
    """
    (valid, note, max_coupling) for the static Itano shift -1/2 alpha(0) <E^2> of one state.

    The formula needs (a) second-order perturbation theory: the coupling d.E_rms to every partner is much
    smaller than its gap, here |d E_rms|/|dE| <= 0.3 with d_z^2 = f/(2 dE) from the absorption f;
    and (b) transition energies far above k_B T, so the thermal field is quasi-static for that partner:
    here at most 10% of alpha(0) may come from partners with dE < k_B T (Itano, Lewis and Wineland 1982).
    Rydberg states usually fail (b); the dynamic Planck integral is then the shift to use.
    """
    E_rms = math.sqrt(mean_E_squared_au(T_K))
    kT = _kT_au(T_K)
    worst = 0.0
    total = low = 0.0
    for t in valence:
        dE = abs(t['delta_E_au'])
        f = t['fosc']
        if dE <= 0 or f <= 0:
            continue
        f_abs = f
        if t['sign'] < 0:       # stored emission f = f_abs g_lo / g_up with nl degeneracies
            n_o, l_o = parse_nl_label(t.get('other_label'))
            if L0 is not None and l_o in L_CHAR:
                f_abs = f * (2 * L0 + 1) / (2 * L_CHAR[l_o] + 1)
        worst = max(worst, math.sqrt(f_abs / (2.0 * dE)) * E_rms / dE)
        w = f / dE ** 2
        total += w
        if dE < kT:
            low += w
    if worst > STATIC_COUPLING_MAX:
        return False, f'coupling {worst:.2g} x gap: perturbation theory fails', worst
    if total > 0 and low / total > STATIC_LOWFREQ_MAX:
        return False, (f'{100 * low / total:.0f}% of alpha(0) from partners below k_BT: '
                       f'not static, use the dynamic shift'), worst
    return True, '', worst



def parse_nl_label(label):
    """Return (n, l_char) from a state label, or (None, None)."""
    m = re.match(r'^(\d+)([spdfgh])', str(label).strip())
    if not m:
        return None, None
    return int(m.group(1)), m.group(2)


def rydberg_shift_for_state(label, data, T_K, do_dyn=True,
                            do_cont=True, n_tail_max=N_TAIL_DEFAULT) -> RydbergShift | None:
    """
    Build E1 alpha network for label; return static Itano + optional dynamic Planck
    BBR shift. Dynamic uses (by default) TRK + QDT-tail + Kramers continuum completion
    so high-n approaches the Farley-Wing asymptote.
    """
    data = refine_series_data(data)
    all_transitions = data['transitions']
    if label not in data['state_energy_au']:
        return None
    trans = build_transitions_for_state(label, data, all_transitions)
    valence = [t for t in trans if t.get('other_label') != 'core']
    alpha0 = alpha0_static(trans)
    if not math.isfinite(alpha0):
        return None
    n, l_char = parse_nl_label(label)
    if n is None:
        n_l = data.get('state_nl', {}).get(label)
        if n_l:
            n, l_char = n_l[0], n_l[1]
    min_dE = None
    if valence:
        min_dE = min(abs(t['delta_E_au']) for t in valence)
    n_sources: dict[str, int] = {}
    for t in valence:
        src = t.get('source') or 'unknown'
        n_sources[src] = n_sources.get(src, 0) + 1
    shift_hz = bbr_static_shift_hz(alpha0, T_K)
    s_valid, s_note, s_coupling = static_validity(valence, T_K, L_CHAR.get(l_char) if l_char else None)
    shift_dyn = None
    completion = None
    if do_dyn:
        if do_cont:
            dyn_channels, completion = complete_transitions_for_dyn(
                label, data, valence, n_tail_max=n_tail_max)
        else:
            dyn_channels = [dict(t) for t in valence]
            completion = Completion(trk_scale=1.0, f_cont=0.0, n_tail=0, n_cont=0,
                                    n_tail_max=0, trk_sum=None)
        shift_dyn = bbr_total_shift_pv_hz(dyn_channels, T_K)
    return RydbergShift(
        state=label,
        n=n,
        l=l_char,
        alpha0_au=alpha0,
        shift_stat_hz=shift_hz,
        shift_dyn_hz=shift_dyn,
        n_transitions=len(valence),
        min_dE_au=min_dE,
        fosc_sources=n_sources,
        completion=completion,
        static_valid=s_valid,
        static_note=s_note,
        max_coupling=s_coupling,
    )


def compute_rydberg_shifts(data, T_K, *,
                           n_min=None, n_max=None, extra_states: Iterable[str] = (),
                           n_pad=N_PAD_DEFAULT, table=True, do_dyn=True, do_cont=True,
                           n_tail_max=N_TAIL_DEFAULT) -> RydbergShifts | None:
    """
    Static + optional dynamic BBR shifts from on-the-fly alpha networks,
    or None when no state is selected.

    Default n window: [gs_n+2, series_n_max - n_pad]. Near the series end,
    upward partners are missing and alpha(0) is truncated; raise n_max in
    rydberg.py / transitions.py before trusting those states.
    Set table=False to compute only extra_states (CLI --state with
    --no-rydberg-shifts).
    """
    data = refine_series_data(data)
    gs_n = int(data.get('gs_n') or 1)
    series_n_max = int(data.get('ryd', {}).get('n_max') or gs_n)
    if n_min is None:
        n_min = gs_n + 2
    if n_max is None:
        n_max = max(n_min, series_n_max - int(n_pad))

    labels = set()
    synthetic = (data.get('_series_refined') or {}).get('synthetic', ())
    if table:
        for lab, (n, _l) in data.get('state_nl', {}).items():
            if n is None or lab in synthetic:
                continue
            if n_min <= int(n) <= n_max:
                labels.add(lab)
    for lab in extra_states:
        lab = str(lab).strip()
        if lab:
            labels.add(lab)

    if not labels:
        return None

    def shift_for(label):
        return rydberg_shift_for_state(
            label, data, T_K, do_dyn=do_dyn,
            do_cont=do_cont, n_tail_max=n_tail_max)

    rows = []
    missing = []
    for lab in sorted(labels, key=lambda s: (parse_nl_label(s)[0] or 0, s)):
        row = shift_for(lab)
        if row is None:
            missing.append(lab)
            continue
        rows.append(row)
    rows.sort(key=lambda r: (r.n or 0, r.l or '', r.state))

    probes = []
    for probe in (f'{gs_n}s', f'{gs_n}p1/2', f'{gs_n}p', '10s', '15s', '20s', '25s', '40s'):
        hit = next((r for r in rows if r.state == probe), None)
        if hit is None and probe in data['state_energy_au']:
            hit = shift_for(probe)
        if hit is not None:
            probes.append(hit)

    return RydbergShifts(
        T_K=T_K,
        table=table,
        n_min=n_min if table else None,
        n_max=n_max if table else None,
        n_pad=n_pad,
        series_n_max=series_n_max,
        do_dyn=do_dyn,
        do_cont=do_cont,
        n_tail_max=n_tail_max,
        farley_wing_hz=farley_wing_shift_hz(T_K),
        states=rows,
        skipped=missing,
        probes=probes,
    )



# CONTINUUM BBR PHOTOIONIZATION / QUENCHING

# Beterov et al., New J. Phys. 11, 013052 (2009):
#   eq. (27) direct PI with quantum-defect phases + A_L (Table 2)
#   eq. (30) BBR-induced SFI into n' >= n_c
#   eq. (9)  E_c ≈ 3.2e8 / n_eff^4  (V/cm)
BETEROV_PREF = 11500.0
BETEROV_E = 157890.0
E_C_PREF_V_CM = 3.2e8
L_QUANTUM = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5}

# Table 1: mu_L - mu_{L+1} near n~20 (continuum-adjacent)
QD_DIFF = {
    'Li': {'s': 0.352417, 'p': 0.0451664, 'd': 0.00162407},
    'Na': {'s': 0.493519, 'p': 0.840023,  'd': 0.0148029},
    'K':  {'s': 0.466733, 'p': 1.43762,   'd': 0.264237},
    'Rb': {'s': 0.490134, 'p': 1.29456,   'd': 1.34636},
    'Cs': {'s': 0.458701, 'p': 1.12661,   'd': 2.43295},
    'Fr': {'s': 0.46,     'p': 1.13,      'd': 2.4},  # Cs-like fallback
}
_user_tables.apply("blackbody.qd_diff", QD_DIFF)

# Table 2: A_L scaling in eq. (27)/(30)
A_L_TABLE = {
    'Li': {'s': 1.0,  'p': 1.0,  'd': 0.9},
    'Na': {'s': 1.0,  'p': 1.0,  'd': 1.1},
    'K':  {'s': 0.9,  'p': 0.45, 'd': 1.3},
    'Rb': {'s': 1.0,  'p': 1.0,  'd': 0.6},
    'Cs': {'s': 0.85, 'p': 1.1,  'd': 0.35},
    'Fr': {'s': 0.85, 'p': 1.1,  'd': 0.35},
}
_user_tables.apply("blackbody.a_l", A_L_TABLE)


def quantum_defect_for_l(qd_table, l_char):
    """Lookup delta_l from rydberg quantum_defects dict."""
    if not qd_table or not l_char:
        return 0.0
    return float(qd_table.get(l_char, 0.0) or 0.0)


def binding_energy_eV(label, data):
    """
    Binding energy below continuum (eV).
    Rydberg JSON stores absolute energies with continuum = 0 (negative E).
    Ground uses -IE from load_element.
    """
    E_au = data['state_energy_au'].get(label)
    if E_au is None:
        return None
    E_bind = -float(E_au) * HARTREE_TO_EV
    if E_bind <= 0:
        return None
    return E_bind


def beterov_A_L(element, l_char):
    tab = A_L_TABLE.get(element) or {}
    return float(tab.get(l_char, 1.0))


def beterov_phase_weight(element, l_char):
    """
    cos^2 phase factor(s) from eq. (27).
    nS: only L' = L+1 term.  nP/nD: both L'+1 and L'-1.
    """
    diffs = QD_DIFF.get(element) or QD_DIFF['Na']
    pi6 = math.pi / 6.0

    def dmu(lc):
        return float(diffs.get(lc, 0.0))

    if l_char == 's':
        d_plus = math.pi * dmu('s')  # mu_S - mu_P
        return math.cos(d_plus + pi6) ** 2
    if l_char == 'p':
        d_plus = math.pi * dmu('p')   # mu_P - mu_D
        d_minus = math.pi * dmu('s')  # mu_S - mu_P
        return math.cos(d_plus + pi6) ** 2 + math.cos(d_minus - pi6) ** 2
    if l_char == 'd':
        d_plus = math.pi * dmu('d')   # mu_D - mu_F
        d_minus = math.pi * dmu('p')  # mu_P - mu_D
        return math.cos(d_plus + pi6) ** 2 + math.cos(d_minus - pi6) ** 2
    # Higher L: fall back to both-channel form with small diffs -> ~2
    return 2.0


def beterov_ln_term(x):
    """-ln(1 - e^{-x}) with stable limits."""
    if x > 40.0:
        return math.exp(-x)
    emx = math.exp(-x)
    return -math.log1p(-emx)


def gamma_pi_beterov(element, n_star, l_char, T_K):
    """
    Direct BBR photoionization (1/s), Beterov eq. (27):

        W = A_L * 11500 T / n*^(7/3) * [phase cos^2 weights]
            * ln(1 / (1 - exp(-157890 /(T n*^2))))
    """
    if n_star is None or n_star <= 0.5 or T_K <= 0:
        return 0.0
    n_star = float(n_star)
    A_L = beterov_A_L(element, l_char)
    phase = beterov_phase_weight(element, l_char)
    x = BETEROV_E / (float(T_K) * n_star * n_star)
    gamma = (A_L * BETEROV_PREF * float(T_K) / (n_star ** (7.0 / 3.0))
             * phase * beterov_ln_term(x))
    if not math.isfinite(gamma) or gamma < 0:
        return 0.0
    return gamma


def n_c_from_field(E_V_cm):
    """Critical n_eff for SFI: E_c ≈ 3.2e8 / n_eff^4  (V/cm)."""
    if E_V_cm is None or E_V_cm <= 0:
        return None
    return (E_C_PREF_V_CM / float(E_V_cm)) ** 0.25


def gamma_sfi_beterov(element, n_star, l_char, T_K, n_c):
    """
    BBR-induced transfer to n' >= n_c then field-ionized (1/s), eq. (30).
    Returns (Gamma_SFI, field_ionized_initial) where the flag means the
    initial state itself sits above the SFI threshold.
    """
    if n_c is None or n_star is None or n_star <= 0.5 or T_K <= 0:
        return 0.0, False
    n_star = float(n_star)
    n_c = float(n_c)
    if n_star >= n_c:
        return 0.0, True
    A_L = beterov_A_L(element, l_char)
    phase = beterov_phase_weight(element, l_char)
    # ln(1/(1-exp(E/(T nc^2) - E/(T n^2)))) - ln(1/(1-exp(-E/(T n^2))))
    x_n = BETEROV_E / (float(T_K) * n_star * n_star)
    # term1: -ln(1 - exp(x_nc - x_n)) with x_nc = E/(T nc^2), x_n = E/(T n^2)
    x_nc = BETEROV_E / (float(T_K) * n_c * n_c)
    # exp(x_nc - x_n) = exp(-(x_n - x_nc)) since x_nc < x_n
    delta = x_n - x_nc  # positive
    if delta > 40.0:
        # 1 - exp(x_nc - x_n) = 1 - exp(-delta) ≈ 1; -ln(...) ≈ 0? No:
        # ln(1/(1-exp(x_nc-x_n))) = -ln(1 - e^{-delta}) ≈ e^{-delta} for large delta
        ln1 = beterov_ln_term(delta)
    else:
        ln1 = -math.log1p(-math.exp(-delta))
    ln2 = beterov_ln_term(x_n)
    bracket = ln1 - ln2
    if bracket < 0:
        bracket = 0.0
    gamma = (A_L * BETEROV_PREF * float(T_K) / (n_star ** (7.0 / 3.0))
             * phase * bracket)
    if not math.isfinite(gamma) or gamma < 0:
        return 0.0, False
    return gamma, False


def continuum_row_for_state(label, data, T_K, rates_by_state=None,
                            element=None, n_c=None) -> ContinuumRow | None:
    """One continuum / quenching row for a state label (direct PI/SFI only).

    rates_by_state: {label: (Gamma_rad_s, Gamma_BBR_s)} from the bound-bound rates.
    """
    element = element or data.get('element')
    n, l_char = parse_nl_label(label)
    if n is None:
        n_l = data.get('state_nl', {}).get(label)
        if n_l:
            n, l_char = n_l[0], n_l[1]
    if n is None or not l_char:
        return None
    E_bind = binding_energy_eV(label, data)
    if E_bind is None:
        return None
    qd = data.get('qd') or data.get('ryd', {}).get('quantum_defects') or {}
    delta = quantum_defect_for_l(qd, l_char)
    n_star = float(n) - delta
    if n_star <= 0.5:
        return None
    Gamma_PI = gamma_pi_beterov(element, n_star, l_char, T_K)
    Gamma_SFI = 0.0
    field_ionized = False
    if n_c is not None:
        Gamma_SFI, field_ionized = gamma_sfi_beterov(
            element, n_star, l_char, T_K, n_c)
    Gamma_rad = 0.0
    Gamma_BBR = 0.0
    if rates_by_state and label in rates_by_state:
        Gamma_rad, Gamma_BBR = rates_by_state[label]
    # Free-space quench: BBR + direct PI (SFI / mix are experiment-side)
    Gamma_quench = Gamma_BBR + Gamma_PI
    Gamma_tot = Gamma_rad + Gamma_quench
    return ContinuumRow(
        state=label,
        n=n,
        l=l_char,
        n_star=n_star,
        L=L_QUANTUM.get(l_char, 0),
        A_L=beterov_A_L(element, l_char),
        phase_weight=beterov_phase_weight(element, l_char),
        E_bind_eV=E_bind,
        Gamma_PI_s=Gamma_PI,
        Gamma_SFI_s=Gamma_SFI,
        field_ionized=field_ionized,
        Gamma_mix_PI_s=0.0,
        Gamma_mix_SFI_s=0.0,
        Gamma_ion_s=Gamma_PI + Gamma_SFI,
        Gamma_BBR_s=Gamma_BBR,
        Gamma_rad_s=Gamma_rad,
        Gamma_quench_s=Gamma_quench,
        tau_eff_s=(1.0 / Gamma_tot) if Gamma_tot > 0 else None,
        n_mix_partners=0,
    )


def t_eff_gate(tau_s, t1_s, t2_s):
    """
    Effective interaction time for gated ion collection:
        t_eff = tau * (exp(-t1/tau) - exp(-t2/tau))
    """
    if tau_s is None or tau_s <= 0 or t2_s <= t1_s:
        return 0.0
    return float(tau_s) * (math.exp(-t1_s / tau_s) - math.exp(-t2_s / tau_s))


def _l_delta_ok(l_a, l_b):
    La = L_QUANTUM.get(l_a)
    Lb = L_QUANTUM.get(l_b)
    if La is None or Lb is None:
        return False
    return abs(La - Lb) == 1


def transfer_channels_from_state(label, graph: LifetimeGraph, T_K, dn_max=3):
    """
    Spontaneous + BBR transfer rates from label to E1 partners (Delta L = +/-1).
    Returns list of (partner, Gamma_transfer_s, n_p, l_p).
    """
    info = graph.states.get(label)
    if not info:
        return []
    n0 = info['n']
    out = []
    for partner, A, g_u, g_l, wl, kind in graph.edges_from.get(label, []):
        n_p, l_p = parse_nl_label(partner)
        if n_p is None:
            pinfo = graph.states.get(partner)
            if not pinfo:
                continue
            n_p, l_p = pinfo['n'], pinfo['l']
        if not _l_delta_ok(info['l'], l_p):
            continue
        if dn_max is not None and abs(int(n_p) - int(n0)) > int(dn_max):
            continue
        if wl <= 0 or A <= 0:
            continue
        omega = omega_from_nm(wl)
        nb = nbar_photon(omega, T_K)
        if kind == 'stim':
            # label is upper: spontaneous A + stimulated emission
            G_tr = A * (1.0 + nb)
        else:
            # label is lower: absorption only
            G_tr = A * (g_u / max(g_l, 1e-30)) * nb
        if G_tr > 0:
            out.append((partner, G_tr, int(n_p), l_p))
    return out


def mix_ionization_rate(Gamma_transfer, W_ion_p, tau_s, tau_p, t1_s, t2_s):
    """
    Effective ionization rate of the *initial* state via population of a
    neighbor that then ionizes (rate-equation solution).

        W_mix = Gamma_tr * W_ion_p / (gamma_p - gamma_s)
                * (1 - t_eff_p / t_eff_s)

    with gamma = 1/tau, t_eff = tau*(e^{-t1/tau}-e^{-t2/tau}).
    Always returns a non-negative rate.
    """
    if (Gamma_transfer <= 0 or W_ion_p <= 0 or tau_s is None or tau_p is None
            or tau_s <= 0 or tau_p <= 0):
        return 0.0
    te_s = t_eff_gate(tau_s, t1_s, t2_s)
    te_p = t_eff_gate(tau_p, t1_s, t2_s)
    if te_s <= 0:
        return 0.0
    gamma_s = 1.0 / tau_s
    gamma_p = 1.0 / tau_p
    dg = gamma_p - gamma_s
    if abs(dg) < 1e-12 * max(gamma_s, gamma_p, 1.0):
        # Equal-lifetime limit: N_p(t) = Gamma_tr * N0 * t * e^{-gamma t}
        # ions = W_ion * Gamma_tr * N0 * I, W_mix = W_ion*Gamma_tr*I/te_s
        # I = int_{t1}^{t2} t e^{-g t} dt
        g = gamma_s
        def I_part(t):
            # int t e^{-g t} dt = -e^{-gt}(t/g + 1/g^2)
            return -math.exp(-g * t) * (t / g + 1.0 / (g * g))
        I = I_part(t2_s) - I_part(t1_s)
        if I < 0:
            I = 0.0
        return Gamma_transfer * W_ion_p * I / te_s
    # Derived from integrating N_p(t); (1 - te_p/te_s)/(gamma_p - gamma_s)
    # is positive when the rate-equation solution is physical.
    factor = (1.0 - te_p / te_s) / dg
    if factor < 0:
        # Swap-equivalent form if gate/lifetime ordering flips the sign
        factor = (1.0 - te_s / te_p) / (gamma_s - gamma_p) if te_p > 0 else 0.0
    if factor < 0:
        return 0.0
    return Gamma_transfer * W_ion_p * factor


def apply_beterov_mixing(rows: Sequence[ContinuumRow], graph: LifetimeGraph, T_K,
                         t1_s, t2_s, dn_max=3, do_sfi_mix=False) -> list[ContinuumRow]:
    """
    The rows with Gamma_mix_PI / Gamma_mix_SFI filled in (Beterov eq. 35).
    Main weight from Delta n ~ 1 partners with Delta L = +/-1.
    """
    by_label = {r.state: r for r in rows}
    mixed = []
    for r in rows:
        transfers = transfer_channels_from_state(
            r.state, graph, T_K, dn_max=dn_max)
        mix_pi = 0.0
        mix_sfi = 0.0
        n_used = 0
        for partner, G_tr, _n_p, _l_p in transfers:
            pr = by_label.get(partner)
            if pr is None:
                continue
            # Need tau for partner; if missing rates, skip
            if r.tau_eff_s is None or pr.tau_eff_s is None:
                continue
            contrib = mix_ionization_rate(
                G_tr, pr.Gamma_PI_s,
                r.tau_eff_s, pr.tau_eff_s, t1_s, t2_s)
            mix_pi += contrib
            # Mix-SFI only via neighbors that are not already field-ionized
            # (direct SFI covers n' >= n_c).
            if do_sfi_mix and not pr.field_ionized:
                mix_sfi += mix_ionization_rate(
                    G_tr, pr.Gamma_SFI_s,
                    r.tau_eff_s, pr.tau_eff_s, t1_s, t2_s)
            n_used += 1
        mixed.append(replace(
            r,
            Gamma_mix_PI_s=mix_pi,
            Gamma_mix_SFI_s=mix_sfi,
            n_mix_partners=n_used,
            Gamma_ion_s=r.Gamma_PI_s + r.Gamma_SFI_s + mix_pi + mix_sfi,
        ))
    return mixed


def compute_continuum(element, data, T_K, *, n_min=None, rates: Rates | None = None,
                      extra_states: Iterable[str] = (), field_V_cm=None,
                      graph: LifetimeGraph | None = None,
                      t1_s=0.3e-6, t2_s=2.1e-6, mix_dn_max=3) -> Continuum:
    """
    Continuum BBR photoionization (Beterov eq.27) + optional SFI (eq.30)
    + multi-step mixing (eq.35) when the lifetime *graph* is given.
    Free-space quench / tau_eff use Gamma_BBR + Gamma_PI only.
    """
    gs_n = int(data.get('gs_n') or 1)
    if n_min is None:
        n_min = gs_n + 2
    n_c = n_c_from_field(field_V_cm)

    # The bound-bound rates as written to the JSON: 12 significant digits (P46).
    rates_by_state = {}
    if rates and rates.states:
        rates_by_state = {r.state: (float(f"{r.Gamma_rad_s:.12g}"), float(f"{r.Gamma_BBR_s:.12g}"))
                          for r in rates.states}

    labels = set()
    for lab, (n, _l) in data.get('state_nl', {}).items():
        if n is not None and int(n) >= n_min:
            labels.add(lab)
    for lab in extra_states:
        lab = str(lab).strip()
        if lab:
            labels.add(lab)

    # Also include rate states below n_min so mix partners have PI/tau
    for lab in rates_by_state:
        labels.add(lab)

    rows = []
    for lab in sorted(labels, key=lambda s: (parse_nl_label(s)[0] or 0, s)):
        row = continuum_row_for_state(
            lab, data, T_K, rates_by_state, element=element, n_c=n_c)
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda r: (r.n, r.l, r.state))

    if graph is not None:
        rows = apply_beterov_mixing(
            rows, graph, T_K, t1_s, t2_s,
            dn_max=mix_dn_max, do_sfi_mix=(n_c is not None),
        )

    probes = []
    for probe in (data.get('gs_label'), f'{gs_n}p1/2', f'{gs_n}p', '10s', '20s', '25s'):
        if not probe:
            continue
        hit = next((r for r in rows if r.state == probe), None)
        if hit is not None:
            probes.append(hit)

    return Continuum(T_K=T_K, n_min=n_min, field_V_cm=field_V_cm, n_c=n_c,
                     mixed=graph is not None, t1_s=t1_s, t2_s=t2_s,
                     mix_dn_max=mix_dn_max, rows=rows, probes=probes)
