"""Magnetic Feshbach resonances: the pure computation.

Van der Waals scales (β₆, ā) from C₆ and the mass, the isolated-resonance
scattering length a(B) = a_bg (1 − Δ/(B − B₀)) (Chin, Grimm, Julienne &
Tiesinga, Rev. Mod. Phys. 82, 1225 (2010)), the product of the s-wave
resonances of each entrance channel, and the elastic cross section. No files,
prints or plots: the ``feshbach`` pipeline loads the literature table, C₆ and
mass and passes them in.

Sign convention (Chin et al.): the zero crossing is at B₀ + Δ. For a_bg·Δ > 0,
a → +∞ just BELOW B₀ and → −∞ just above it; Δ and a_bg may each be negative,
so the sign of Δ is physical and is not validated (only Δ ≠ 0 and finite values
are required).

    β₆ = (2 μ C₆ / ℏ²)^{1/4}            van der Waals length
    ā  = 0.477989 β₆ = 0.955978 R_vdW   mean scattering length (Gribakin–Flambaum),
                                        R_vdW = β₆/2 (so ā = 0.956 R_vdW, not 0.956 β₆)
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from qali.core.constants import (
    A0_M,
    A_BAR_FACTOR,
    AMU_KG,
    AU_TO_SI_C6,
    HBAR_SI,
    H_SI,
    K_B,
)


# ============================================================================
# INPUT VALIDATION
# ============================================================================

def validate_feshbach_params(params, source='feshbach json'):
    """Schema check for the curated resonance list (the sign of Delta is physical and not checked)."""
    res = params.get('resonances') or []
    for i, r in enumerate(res):
        for key in ('B0_G', 'Delta_G'):
            if key not in r:
                raise ValueError(f'{source}: resonance #{i} ({r.get("label", "?")}) lacks {key}')
            v = float(r[key])
            if not math.isfinite(v):
                raise ValueError(f'{source}: resonance #{i} {key} is not finite')
        if float(r['Delta_G']) == 0.0:
            raise ValueError(f'{source}: resonance #{i} ({r.get("label", "?")}) has Delta_G = 0 (no resonance)')
        if 'a_bg_a0' in r and not math.isfinite(float(r['a_bg_a0'])):
            raise ValueError(f'{source}: resonance #{i} a_bg_a0 is not finite')
    return params


def _checked_C6(value, source):
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f'C6 from {source} is {value!r}; a positive, finite C6 is required '
                         f'(beta6 = (2 mu C6/hbar^2)^(1/4) is undefined otherwise)')
    return value


# ============================================================================
# PHYSICS
# ============================================================================

def reduced_mass_kg(mass_amu):
    """Identical-atom reduced mass μ = m/2."""
    m = mass_amu * AMU_KG
    return 0.5 * m


def beta6_a0(C6_au, mass_amu):
    """
    van der Waals length β₆ = (2 μ C₆ / ℏ²)^{1/4} in units of a₀.

    Working in SI then converting to a₀ keeps units transparent.
    """
    C6_au = _checked_C6(C6_au, 'beta6_a0 argument')
    mu = reduced_mass_kg(mass_amu)
    C6_SI = C6_au * AU_TO_SI_C6
    beta_m = (2.0 * mu * C6_SI / HBAR_SI**2) ** 0.25
    return beta_m / A0_M


def a_bar_a0(beta6):
    """Mean scattering length ā = [2π/Γ(1/4)²] β₆ = 0.477989 β₆ (Gribakin–Flambaum),
    for β₆ = (2μC₆/ℏ²)^{1/4}. Equivalently ā = 0.955978 R_vdW with R_vdW = β₆/2."""
    return A_BAR_FACTOR * beta6


def scattering_length(B, a_bg, B0, Delta, gamma=None):
    """
    a(B) = a_bg (1 − Δ/(B − B₀)).

    Returns NaN exactly at B = B₀ (pole). With an inelastic decay width *gamma* (field units) the
    resonance is a(B) = a_bg [1 − Δ/(B − B₀ − iγ/2)] and this returns its real part, which stays
    finite: |Re a − a_bg| ≤ |a_bg Δ|/γ.
    """
    B = np.asarray(B, dtype=float)
    if gamma:
        return np.real(a_bg * (1.0 - Delta / (B - B0 - 0.5j * gamma)))
    denom = B - B0
    with np.errstate(divide='ignore', invalid='ignore'):
        a = a_bg * (1.0 - Delta / denom)
    a = np.where(np.abs(denom) < 1e-12, np.nan, a)
    return a


def wave_number_a0(kT_uK, mass_amu):
    """
    Thermal wave number k = √(2 μ k_B T) / ℏ, returned in a₀⁻¹.

    Uses E = k_B T (not 3/2 kT) — convention for quoting a single energy scale.
    """
    mu = reduced_mass_kg(mass_amu)
    E = K_B * (kT_uK * 1e-6)
    k_SI = math.sqrt(2.0 * mu * E) / HBAR_SI
    return k_SI * A0_M   # 1/m · m/a₀⁻¹ → a₀⁻¹


_Z_ALKALI = {'Li': 3, 'Na': 11, 'K': 19, 'Rb': 37, 'Cs': 55, 'Fr': 87}


def is_boson(element, A):
    """Quantum statistics of the neutral atom: it is a boson when its number of fermions
    (Z electrons + Z protons + (A-Z) neutrons = A + Z) is even. None if Z is unknown."""
    Z = _Z_ALKALI.get(element)
    if Z is None or A is None:
        return None
    return (int(A) + Z) % 2 == 0


def channel_kets(channel):
    """Internal states of an entrance channel string like '|1,+1> + |1,+1>' -> ['|1,+1>', '|1,+1>']."""
    import re
    return [k.replace(' ', '') for k in re.findall(r'\|[^>]*>', channel or '')]


def identical_particles(channel):
    """True when both atoms are in the same internal state (needs a parsable '|..> + |..>')."""
    kets = channel_kets(channel)
    return len(kets) == 2 and kets[0] == kets[1]


def sigma_el_a0(a_a0, k_a0inv, identical_bosons=False):
    """
    s-wave elastic cross section, in a0^2:
        sigma = 4 pi a^2 / (1 + k^2 a^2)   distinguishable atoms (different internal states)
        sigma = 8 pi a^2 / (1 + k^2 a^2)   two identical bosons (same internal state)
    (Pethick & Smith, Bose-Einstein Condensation in Dilute Gases, Sec. 5.1). Identical fermions
    have no s-wave scattering; callers must not ask for it.
    """
    a = np.asarray(a_a0, dtype=float)
    ka2 = (k_a0inv * a) ** 2
    pref = 8.0 * np.pi if identical_bosons else 4.0 * np.pi
    with np.errstate(invalid='ignore'):
        return pref * a**2 / (1.0 + ka2)


def couples_to_s_wave(partial_wave) -> bool:
    """True when a resonance is a pole of the s-wave scattering length.

    *partial_wave* is the ℓ of the closed-channel molecular state (the "d-wave" of a d-wave
    resonance). Spin-dipole and second-order spin-orbit coupling change ℓ by 0 or 2, and each
    step keeps the parity, so an even-ℓ state (s, d, g, i) couples to an s-wave entrance channel
    and makes a pole of a(B): the narrow Cs d- and g-wave lines are such poles (Chin et al.,
    RMP 82, 1225 (2010), Sec. V.B; Jachymski & Julienne, PRA 88, 052701 (2013)). An odd-ℓ
    state (p, f) needs a p-wave entrance channel and leaves a(B) alone.
    """
    return str(partial_wave).strip().lower()[:1] in ('s', 'd', 'g', 'i')


def group_s_wave_by_channel(resonances: Sequence[Resonance]) -> dict[str, list[Resonance]]:
    """
    {channel: [resonances]} for the resonances that are poles of the s-wave a(B) (even ℓ,
    ``couples_to_s_wave``), in file order. The product form a(B) = a_bg prod_i (1 - Delta_i/(B - B0_i))
    holds for the poles of ONE entrance channel (Lange et al., PRA 79, 013622 (2009)); resonances of
    other channels belong to other a(B) curves.
    """
    groups: dict[str, list[Resonance]] = {}
    for r in resonances:
        if not couples_to_s_wave(r.partial_wave):
            continue
        groups.setdefault(r.channel or '', []).append(r)
    return groups


@dataclass(frozen=True)
class Resonance:
    """One literature resonance (a row of ``<El>_feshbach.json``)."""

    label: str
    channel: str
    partial_wave: str
    B0_G: float
    Delta_G: float
    a_bg_a0: float | None   # None: use the table's default background
    ref: str
    dmu_muB: float | None = None   # δμ, closed- minus open-channel magnetic moment (μ_B)
    s_res: float | None = None     # published resonance strength (used when δμ is not given)
    gamma_G: float | None = None   # inelastic decay width γ (field units): a = a_bg[1 − Δ/(B − B₀ − iγ/2)]

    @classmethod
    def from_json(cls, row: Mapping) -> Resonance:
        B0 = float(row['B0_G'])
        a_bg = row.get('a_bg_a0')
        return cls(
            label=row.get('label', f"B0={B0}G"),
            channel=row.get('channel', ''),
            partial_wave=row.get('partial_wave', 's'),
            B0_G=B0,
            Delta_G=float(row['Delta_G']),
            a_bg_a0=None if a_bg is None else float(a_bg),
            ref=row.get('ref', ''),
            dmu_muB=None if row.get('dmu_muB') is None else float(row['dmu_muB']),
            s_res=None if row.get('s_res') is None else float(row['s_res']),
            gamma_G=float(row['gamma_G']) if row.get('gamma_G') else None,
        )

    def background(self, a_bg_default: float) -> float:
        return self.a_bg_a0 if self.a_bg_a0 is not None else a_bg_default

    @property
    def is_s_wave(self) -> bool:
        """A pole of the s-wave a(B) (even closed-channel ℓ, see ``couples_to_s_wave``)."""
        return couples_to_s_wave(self.partial_wave)

    @property
    def is_pole(self) -> bool:
        """A true pole of Re a(B): s-wave coupled and without an inelastic decay width."""
        return self.is_s_wave and not self.gamma_G


@dataclass(frozen=True)
class ResonanceCurve:
    """a(B) and σ(B) of one resonance on its own (isolated single pole).

    ``None`` for an odd-ℓ (p, f) resonance: it is not a pole of the s-wave a(B).
    """

    resonance: Resonance
    a_bg_a0: float
    a_a0: np.ndarray | None
    sigma_a02: np.ndarray | None


@dataclass(frozen=True)
class ChannelCurve:
    """a(B) and σ(B) of one s-wave entrance channel.

    *source*: 'resonances' (product over the channel's poles), 'table' or 'fit' (published curve,
    ``TableCurve`` / ``FitCurve``). *bridged*: True where no published table or fit covers B.
    """

    channel: str
    labels: list[str]
    a_bg_a0: float
    identical_bosons: bool
    a_a0: np.ndarray
    sigma_a02: np.ndarray
    source: str = 'resonances'
    ref: str = ''
    bridged: np.ndarray | None = None
    poles_G: tuple[float, ...] = ()
    zeros_G: tuple[float, ...] = ()


@dataclass(frozen=True)
class FeshbachResult:
    C6_au: float
    mass_amu: float
    beta6_a0: float
    a_bar_a0: float
    a_bg_default_a0: float
    kT_uK: float
    k_a0inv: float
    B_G: np.ndarray
    a_combined_a0: np.ndarray        # the default channel's a(B) and σ(B)
    sigma_combined_a02: np.ndarray
    default_channel: str
    channels: list[ChannelCurve]
    is_boson: bool | None
    curves: list[ResonanceCurve]


def multi_resonance_a(B, resonances: Sequence[Resonance], a_bg_default):
    """
    Product of isolated-resonance factors (valid when resonances are well
    separated compared to their widths):

        a(B)/a_bg = Π_i (1 − Δ_i / (B − B₀,i − iγ_i/2))

    γ_i is the inelastic decay width of a resonance in a channel that can relax (zero for most);
    the result is Re a. The overall a_bg is the first resonance's own, else *a_bg_default*: pass
    the resonances of ONE entrance channel (``group_s_wave_by_channel``).
    """
    B = np.asarray(B, dtype=float)
    a_bg = resonances[0].background(a_bg_default)
    factor = np.ones_like(B, dtype=complex)
    for res in resonances:
        B0 = res.B0_G
        Delta = res.Delta_G
        denom = B - B0 - 0.5j * (res.gamma_G or 0.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            f = 1.0 - Delta / denom
        f = np.where(np.abs(denom) < 1e-12, np.nan, f)
        factor = factor * f
    return a_bg * np.real(factor)


def default_B_max(resonances: Sequence[Resonance]) -> float:
    """15 % past the highest resonance, at least 500 G."""
    return max(max(r.B0_G for r in resonances) * 1.15, 500.0)


def _product_poles_and_zeros(B, a, poles):
    """Pole fields (the B₀ themselves) and zero crossings of a sampled product curve (sign changes
    that are not poles), inside the sampled range."""
    B = np.asarray(B)
    lo, hi = float(B[0]), float(B[-1])
    pz = sorted(p for p in poles if lo <= p <= hi)
    zeros = []
    a = np.asarray(a, dtype=float)
    ok = np.isfinite(a)
    idx = np.flatnonzero(ok)
    for k0, k1 in itertools.pairwise(idx):
        a0, a1 = a[k0], a[k1]
        if a0 == 0.0 and (not zeros or zeros[-1] != float(B[k0])):
            zeros.append(float(B[k0]))                     # sampled exactly at the zero
        elif a0 * a1 < 0 and not any(B[k0] <= p <= B[k1] for p in pz):
            zeros.append(float(B[k0] - a0 * (B[k1] - B[k0]) / (a1 - a0)))
    return tuple(pz), tuple(zeros)


def compute(
    resonances: Sequence[Resonance], a_bg_default_a0: float, C6_au: float, mass_amu: float,
    *, kT_uK: float = 1.0, B_max: float | None = None, B_points: int = 2000,
    boson: bool | None = None, channel: str | None = None,
    published: Mapping[str, object] | None = None, refine: bool = False,
) -> FeshbachResult:
    """Van der Waals scales, per-channel and per-resonance a(B) and σ(B) on 0..B_max.

    *boson* is the atom's quantum statistics (``is_boson``; None if unknown) and sets
    the identical-particle factor of σ. The combined curve is entrance *channel*
    (default: the channel with the most s-wave resonances). *published* maps a channel to its
    ``TableCurve`` / ``FitCurve``; that channel's a(B) then comes from the published curve.
    With *refine*, the grid adds points around every pole so narrow resonances are resolved.
    """
    published = dict(published or {})
    beta6 = beta6_a0(C6_au, mass_amu)
    k_th = wave_number_a0(kT_uK, mass_amu)
    if B_max is None:
        B_max = default_B_max(resonances)
        tables = [cur for cur in published.values() if isinstance(cur, TableCurve)]
        if tables:     # a table reaches far past the poles: show the broad resonances' BCS side too
            B_max = min(max(B_max, 1.4 * max(r.B0_G for r in resonances)), min(t.B_range[1] for t in tables))
    groups = group_s_wave_by_channel(resonances)
    for ch in published:
        groups.setdefault(ch, [])
    if not groups:
        raise ValueError('the resonance table has no s-wave resonances')
    if channel is not None and channel not in groups:
        raise ValueError(f"channel {channel!r} not in {list(groups)}")
    default_channel = channel if channel is not None else max(groups, key=lambda c: len(groups[c]))
    if refine:
        pole_list = [(r.B0_G, r.Delta_G) for rs in groups.values() for r in rs]
        for cur in published.values():
            if isinstance(cur, FitCurve):
                pole_list += [p for sgm in cur.segments for p in sgm.poles] + list(cur.extra_poles)
            elif isinstance(cur, TableCurve):
                narrow = [n[0] for n in cur.narrow]
                pole_list += [(p, 1.0) for p in cur.poles_and_zeros()[0] if p not in narrow] + list(cur.narrow)
        B = refined_field_grid(B_max, pole_list, n_uniform=B_points)
    else:
        B = np.linspace(0.0, B_max, B_points)

    # a(B) per entrance channel (poles of ONE channel multiply; channels never mix)
    channels = []
    for ch, rs in groups.items():
        ident = identical_particles(ch)
        cur = published.get(ch)
        bridged = None
        if isinstance(cur, TableCurve):
            a_ch = cur.a(B)
            bridged = ~np.isfinite(a_ch) | (B > cur.B_range[1])
            source, ref = 'table', cur.ref
            pz = cur.poles_and_zeros()
            poles = tuple(p for p in pz[0] if p <= B_max)
            zeros = tuple(z for z in pz[1] if z <= B_max)
            a_bg = rs[0].background(a_bg_default_a0) if rs else float('nan')
        elif isinstance(cur, FitCurve):
            a_ch, bridged = cur.a_and_flag(B)
            source, ref = 'fit', cur.ref
            fit_poles = [p[0] for sgm in cur.segments for p in sgm.poles] + [p[0] for p in cur.extra_poles]
            poles, zeros = _product_poles_and_zeros(B, a_ch, fit_poles)
            a_bg = rs[0].background(a_bg_default_a0) if rs else cur.global_a_bg_a0
        else:
            a_ch = multi_resonance_a(B, rs, a_bg_default_a0)
            source, ref = 'resonances', ''
            poles, zeros = _product_poles_and_zeros(B, a_ch, [r.B0_G for r in rs if r.is_pole])
            a_bg = rs[0].background(a_bg_default_a0)
        if ident and boson is False:
            sig_ch = np.zeros_like(a_ch)       # identical fermions: no s-wave collisions
        else:
            sig_ch = sigma_el_a0(a_ch, k_th, identical_bosons=bool(ident and boson))
        channels.append(ChannelCurve(
            channel=ch,
            labels=[r.label for r in rs],
            a_bg_a0=a_bg,
            identical_bosons=bool(ident and boson),
            a_a0=a_ch,
            sigma_a02=sig_ch,
            source=source,
            ref=ref,
            bridged=bridged,
            poles_G=poles,
            zeros_G=zeros,
        ))
    sel = next(c for c in channels if c.channel == default_channel)

    # Per-resonance curves (isolated single-pole form; s-wave poles only)
    curves = []
    for res in resonances:
        a_bg = res.background(a_bg_default_a0)
        if not res.is_s_wave:
            curves.append(ResonanceCurve(res, a_bg, None, None))
            continue
        a = scattering_length(B, a_bg, res.B0_G, res.Delta_G, res.gamma_G)
        sigma = sigma_el_a0(a, k_th, identical_bosons=bool(identical_particles(res.channel) and boson))
        curves.append(ResonanceCurve(res, a_bg, a, sigma))
    return FeshbachResult(
        C6_au=C6_au,
        mass_amu=mass_amu,
        beta6_a0=beta6,
        a_bar_a0=a_bar_a0(beta6),
        a_bg_default_a0=a_bg_default_a0,
        kT_uK=kT_uK,
        k_a0inv=k_th,
        B_G=B,
        a_combined_a0=sel.a_a0,
        sigma_combined_a02=sel.sigma_a02,
        default_channel=default_channel,
        channels=channels,
        is_boson=boson,
        curves=curves,
    )


# ============================================================================
# PUBLISHED a(B): COUPLED-CHANNEL TABLES AND FIELD-DEPENDENT FITS
# ============================================================================
#
# The isolated-pole product above is the literature's local description of each resonance. Where
# the literature gives more (a coupled-channel table of a(B), or fits with a field-dependent
# background that are valid over a stated field range), the channel's a(B) comes from that instead.
# ``<El>_feshbach.json`` names them under "curves": {channel: spec}.

_THETA_SCALE_A0 = 1000.0     # a = L tan θ: θ is smooth through poles (θ = π/2 mod π) and zeros


def _phase(a, scale=_THETA_SCALE_A0):
    """θ = arctan(a/L), unwrapped modulo π along the field axis, so it runs smoothly through poles."""
    th = np.arctan(np.asarray(a, dtype=float) / scale)
    return np.unwrap(2.0 * th) / 2.0


@dataclass(frozen=True)
class TableCurve:
    """a(B) tabulated on a field grid (a coupled-channel calculation), interpolated in θ = arctan(a/L).

    Interpolating θ rather than a keeps a pole a pole: a = L tan θ diverges where θ crosses π/2,
    with the 1/(B − B₀) shape the tabulation implies, instead of a straight line through ±10⁴ a₀.
    """

    B_G: np.ndarray
    a_a0: np.ndarray
    ref: str
    narrow: tuple[tuple[float, float], ...] = ()

    @classmethod
    def from_rows(cls, rows: Sequence[Sequence[float]], column: int, ref: str = '',
                  narrow: Sequence[tuple[float, float]] = ()) -> TableCurve:
        """*narrow*: (B₀, Δ) of resonances narrower than the table step. Their pole is divided out
        of the rows, the rows within 1.5 G of B₀ (which sample it too coarsely) are dropped, and
        the pole is multiplied back in analytically: a = a_table,bg(B) (1 − Δ/(B − B₀))."""
        arr = np.asarray(rows, dtype=float)
        B, a = arr[:, 0], arr[:, column]
        order = np.argsort(B)
        B, a = B[order], a[order]
        narrow = tuple((float(b0), float(d)) for b0, d in narrow)
        for B0, D in narrow:
            keep = np.abs(B - B0) >= 1.5
            B, a = B[keep], a[keep] / (1.0 - D / (B[keep] - B0))
        if B[0] > 0.0:          # the tables start at 1 G; continue the first step linearly to B = 0
            a0 = a[0] - (a[1] - a[0]) / (B[1] - B[0]) * B[0]
            B, a = np.concatenate([[0.0], B]), np.concatenate([[a0], a])
        return cls(B_G=B, a_a0=a, ref=ref, narrow=narrow)

    @property
    def B_range(self) -> tuple[float, float]:
        return float(self.B_G[0]), float(self.B_G[-1])

    def a(self, B):
        """a(B) inside the table; NaN outside it."""
        from scipy.interpolate import PchipInterpolator
        B = np.asarray(B, dtype=float)
        th = PchipInterpolator(self.B_G, _phase(self.a_a0))(np.clip(B, *self.B_range))
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            out = _THETA_SCALE_A0 * np.tan(th)
            for B0, D in self.narrow:
                out = out * (1.0 - D / (B - B0))
        return np.where((B >= self.B_range[0]) & (B <= self.B_range[1]), out, np.nan)

    def poles_and_zeros(self) -> tuple[list[float], list[float]]:
        """Fields where the tabulated θ crosses π/2 (poles) and 0 (zeros), mod π, linearly interpolated."""
        th = _phase(self.a_a0)
        poles: list[float] = []
        zeros: list[float] = []
        for k in range(len(th) - 1):
            for target, out in ((0.5, poles), (0.0, zeros)):
                lo, hi = sorted((th[k], th[k + 1]))
                for m in range(int(math.floor(lo / math.pi - target)), int(math.ceil(hi / math.pi - target)) + 1):
                    v = (m + target) * math.pi
                    if lo < v <= hi and th[k + 1] != th[k]:
                        out.append(float(self.B_G[k] + (v - th[k]) / (th[k + 1] - th[k]) * (self.B_G[k + 1] - self.B_G[k])))
        for B0, D in self.narrow:
            poles.append(B0)
            zeros.append(B0 + D)
        return sorted(poles), sorted(zeros)


@dataclass(frozen=True)
class FitSegment:
    """One published fit, valid on [B_min, B_max]:

        a(B) = (a_bg + s (B − B_ref)) Π_i (1 − Δ_i/(B − B₀,i))

    with a linear background. Two published conventions map onto it: Kraemer et al. give
    a_bg + s·B directly; Jachymski & Julienne give a_bg (1 + α (B − B_ref)) with a_bg in units of
    R₆, i.e. s = α a_bg.
    """

    B_min: float
    B_max: float
    a_bg_a0: float
    slope_a0_per_G: float
    B_ref_G: float
    poles: tuple[tuple[float, float], ...]
    ref: str

    @classmethod
    def from_json(cls, row: Mapping) -> FitSegment:
        if 'a_bg_R6' in row:
            a_bg = float(row['a_bg_R6']) * float(row['R6_a0'])
        else:
            a_bg = float(row['a_bg_a0'])
        if 'alpha_per_G' in row:
            slope = a_bg * float(row['alpha_per_G'])
        else:
            slope = float(row.get('slope_a0_per_G', 0.0))
        poles = tuple((float(p[0]), float(p[1])) for p in row.get('poles', []))
        for B0, D in poles:
            if not (math.isfinite(B0) and math.isfinite(D)) or D == 0.0:
                raise ValueError(f'fit segment pole ({B0}, {D}) needs finite B0 and nonzero Delta')
        return cls(B_min=float(row['B_min_G']), B_max=float(row['B_max_G']), a_bg_a0=a_bg,
                   slope_a0_per_G=slope, B_ref_G=float(row.get('B_ref_G', 0.0)), poles=poles,
                   ref=row.get('ref', ''))

    def a(self, B):
        B = np.asarray(B, dtype=float)
        out = self.a_bg_a0 + self.slope_a0_per_G * (B - self.B_ref_G)
        for B0, D in self.poles:
            with np.errstate(divide='ignore', invalid='ignore'):
                out = out * (1.0 - D / (B - B0))
        return out


def _near(B0, poles, tol=1.0):
    return any(abs(B0 - p) <= tol for p in poles)


@dataclass(frozen=True)
class FitCurve:
    """Piecewise published fits, joined across the fields no fit covers.

    Between two segments (and past the last one) a(B) is a global model G(B) — a linear background
    times every pole of every segment and the narrow lines of the resonance list — multiplied by a
    ratio a_fit/G taken at the neighbouring segment edges and interpolated linearly. Inside a
    segment, narrow resonances of the list that the fit leaves out are multiplied in (Kraemer's
    low-field fit excludes the narrow d- and g-wave lines). Points not covered by a fit are flagged.
    """

    segments: tuple[FitSegment, ...]
    global_a_bg_a0: float
    global_slope_a0_per_G: float
    extra_poles: tuple[tuple[float, float], ...]    # narrow list resonances no segment includes
    ref: str

    @classmethod
    def from_json(cls, spec: Mapping, resonances: Sequence[Resonance] = ()) -> FitCurve:
        segs = tuple(sorted((FitSegment.from_json(r) for r in spec['segments']), key=lambda s: s.B_min))
        fit_poles = [p[0] for s in segs for p in s.poles]
        extra = tuple((r.B0_G, r.Delta_G) for r in resonances
                      if r.is_s_wave and not _near(r.B0_G, fit_poles))
        g = spec.get('global', {})
        return cls(segments=segs, global_a_bg_a0=float(g.get('a_bg_a0', segs[0].a_bg_a0)),
                   global_slope_a0_per_G=float(g.get('slope_a0_per_G', 0.0)), extra_poles=extra,
                   ref=spec.get('ref', ''))

    def _extra(self, B):
        f = np.ones_like(B)
        for B0, D in self.extra_poles:
            with np.errstate(divide='ignore', invalid='ignore'):
                f = f * (1.0 - D / (B - B0))
        return f

    def global_model(self, B):
        B = np.asarray(B, dtype=float)
        out = self.global_a_bg_a0 + self.global_slope_a0_per_G * B
        for s in self.segments:
            for B0, D in s.poles:
                with np.errstate(divide='ignore', invalid='ignore'):
                    out = out * (1.0 - D / (B - B0))
        return out * self._extra(B)

    def _seg_value(self, s, B):
        return s.a(B) * self._extra(np.asarray(B, dtype=float))

    def a_and_flag(self, B):
        """(a(B), bridged) with bridged True where no published fit covers B."""
        B = np.atleast_1d(np.asarray(B, dtype=float))
        a = np.full(B.shape, np.nan)
        bridged = np.ones(B.shape, dtype=bool)
        for s in self.segments:
            m = (B >= s.B_min) & (B <= s.B_max)
            a[m] = self._seg_value(s, B[m])
            bridged[m] = False
        G = self.global_model(B)
        edges = []            # (field, ratio) at every segment edge, in field order
        for s in self.segments:
            for b in (s.B_min, s.B_max):
                edges.append((b, float(self._seg_value(s, b) / self.global_model(b))))
        eb = np.array([e[0] for e in edges])
        er = np.array([e[1] for e in edges])
        m = bridged
        if m.any():
            r = np.interp(B[m], eb, er)          # constant beyond the first/last edge
            a[m] = G[m] * r
        return a, bridged

    def a(self, B):
        return self.a_and_flag(B)[0]


def channel_curve_spec(spec: Mapping, resonances: Sequence[Resonance], table_rows=None):
    """The published a(B) of one channel from its "curves" entry: a TableCurve or a FitCurve."""
    kind = spec.get('kind')
    if kind == 'table':
        if table_rows is None:
            raise ValueError(f"channel table {spec.get('file')!r} was not loaded")
        narrow = [(float(n['B0_G']), float(n['Delta_G'])) for n in spec.get('narrow', [])]
        return TableCurve.from_rows(table_rows, int(spec.get('column', 1)), spec.get('ref', ''), narrow)
    if kind == 'fit':
        return FitCurve.from_json(spec, resonances)
    raise ValueError(f"unknown a(B) curve kind {kind!r} (expected 'table' or 'fit')")


def refined_field_grid(B_max, poles: Sequence[tuple[float, float]], n_uniform=3000, B_min=0.0):
    """A uniform grid plus log-spaced points on both sides of every pole, so a 1 mG-wide resonance is
    resolved on a 1000 G axis. *poles*: (B₀, Δ) pairs; points reach from 10⁻³|Δ| to 30|Δ| from B₀."""
    B = [np.linspace(B_min, B_max, n_uniform)]
    for B0, D in poles:
        w = max(abs(D), 1e-6)
        off = w * np.logspace(-3, math.log10(30.0), 60)
        for side in (-1.0, 1.0):
            pts = B0 + side * off
            B.append(pts[(pts > B_min) & (pts < B_max)])
        # the zero crossing B0 + Δ sits inside that span already
    g = np.unique(np.concatenate(B))
    return g


# ============================================================================
# DERIVED QUANTITIES (mirrored in feshbach_physics.js)
# ============================================================================

MU_B_J_T = 9.2740100783e-24


def mean_scattering_energy_J(a_bar, mass_amu):
    """Ē = ℏ²/(m ā²), the van der Waals energy scale of the resonance strength (m: atomic mass)."""
    m = mass_amu * AMU_KG
    return HBAR_SI ** 2 / (m * (a_bar * A0_M) ** 2)


def resonance_strength(a_bg, Delta_G, dmu_muB, a_bar, mass_amu):
    """s_res = (a_bg/ā)(δμ Δ/Ē) (Chin et al., RMP 82, 1225, Eq. 39); ≫ 1 open-channel dominated."""
    return abs(a_bg / a_bar) * abs(dmu_muB * MU_B_J_T * abs(Delta_G) * 1e-4) / mean_scattering_energy_J(a_bar, mass_amu)


def binding_energy_hz(a_a0, mass_amu, a_bar=None):
    """Feshbach-molecule binding energy E_b/h for a > 0.

    Universal: E_b = ℏ²/(m a²) (= ℏ²/(2μa²), μ = m/2). With *a_bar*, the Gribakin–Flambaum
    correction E_b = ℏ²/(m (a − ā)²), valid for a ≫ ā near open-channel-dominated resonances
    (Gribakin & Flambaum, PRA 48, 546 (1993); Chin et al. Eq. 30). NaN where there is no shallow
    dimer (a ≤ 0, or a ≤ ā with the correction).
    """
    a = np.asarray(a_a0, dtype=float)
    m = mass_amu * AMU_KG
    eff = a - (a_bar or 0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        E = HBAR_SI ** 2 / (m * (eff * A0_M) ** 2) / H_SI
    return np.where(eff > 0, E, np.nan)


def fermi_wave_number_a0(n_sigma_cm3):
    """k_F = (6π² n_σ)^{1/3} of one spin component at density n_σ (cm⁻³), in a₀⁻¹."""
    n = n_sigma_cm3 * 1e6
    return (6.0 * math.pi ** 2 * n) ** (1.0 / 3.0) * A0_M


def fermi_temperature_K(n_sigma_cm3, mass_amu):
    k = fermi_wave_number_a0(n_sigma_cm3) / A0_M
    return HBAR_SI ** 2 * k ** 2 / (2.0 * mass_amu * AMU_KG * K_B)


def thermal_factor(b):
    """f(b) = ∫₀^∞ u e^{−u}/(1 + b u) du, the Maxwell–Boltzmann average of 1/(1 + k²a²) weighted by
    the relative speed (b = k_T² a², k_T² = 2μk_BT/ℏ²); f(0) = 1, f → 1/b as b → ∞ (unitarity).
    Trapezoid rule in ln u on 10⁻¹⁰ … 60 (the e^{−u} tail beyond is < 10⁻²⁴)."""
    b = np.asarray(b, dtype=float)
    lu = np.linspace(math.log(1e-10), math.log(60.0), 700)
    u = np.exp(lu)
    w = u * u * np.exp(-u)                       # u e^{-u} · du/d(ln u)
    integ = w[None, :] / (1.0 + b.reshape(-1, 1) * u[None, :])
    out = np.trapezoid(integ, lu, axis=1) if hasattr(np, 'trapezoid') else np.trapz(integ, lu, axis=1)
    return out.reshape(b.shape)


def thermal_k_a0(T_uK, mass_amu):
    """k_T = √(2 μ k_B T)/ℏ in a₀⁻¹ (μ = m/2)."""
    mu = reduced_mass_kg(mass_amu)
    return math.sqrt(2.0 * mu * K_B * T_uK * 1e-6) / HBAR_SI * A0_M


def mean_relative_speed_m_s(T_uK, mass_amu):
    """⟨v_rel⟩ = √(8 k_B T/(π μ)), μ = m/2."""
    return math.sqrt(8.0 * K_B * T_uK * 1e-6 / (math.pi * reduced_mass_kg(mass_amu)))


def thermal_sigma_a0(a_a0, T_uK, mass_amu, identical_bosons=False):
    """⟨σ v_rel⟩/⟨v_rel⟩ of a thermal gas: C a² f(k_T² a²), C = 8π (identical bosons) or 4π.
    Saturates at the unitarity limit C/k_T² as |a| → ∞ (effective range neglected)."""
    a = np.asarray(a_a0, dtype=float)
    kT = thermal_k_a0(T_uK, mass_amu)
    C = 8.0 * math.pi if identical_bosons else 4.0 * math.pi
    return C * a ** 2 * thermal_factor(kT ** 2 * a ** 2)


def three_body_L3_cm6_s(a_a0, mass_amu, C=67.1, T_uK=None):
    """Three-body loss coefficient of a thermal Bose gas, dn/dt = −L₃ n³:
    L₃ = 3 C ℏ a⁴/m (Weber et al., PRL 91, 123201 (2003); Kraemer et al., Nature 440, 315 (2006)),
    C the Efimov factor (0 … 67.1 for a > 0 without deep-dimer loss). With *T_uK*, capped at the
    unitarity limit 36√3 π² ℏ⁵/(m³ (k_BT)²) (Rem et al., PRL 110, 163202 (2013))."""
    a = np.asarray(a_a0, dtype=float) * A0_M
    m = mass_amu * AMU_KG
    L = 3.0 * C * HBAR_SI * a ** 4 / m
    if T_uK:
        L = np.minimum(L, 36.0 * math.sqrt(3.0) * math.pi ** 2 * HBAR_SI ** 5 / (m ** 3 * (K_B * T_uK * 1e-6) ** 2))
    return L * 1e12
