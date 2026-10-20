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

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from linak.core.constants import (
    A0_M,
    A_BAR_FACTOR,
    AMU_KG,
    AU_TO_SI_C6,
    HBAR_SI,
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


def scattering_length(B, a_bg, B0, Delta):
    """
    a(B) = a_bg (1 − Δ/(B − B₀)).

    Returns NaN exactly at B = B₀ (pole).
    """
    B = np.asarray(B, dtype=float)
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


def group_s_wave_by_channel(resonances: Sequence[Resonance]) -> dict[str, list[Resonance]]:
    """
    {channel: [resonances]} for the s-wave resonances only, in file order.
    The product form a(B) = a_bg prod_i (1 - Delta_i/(B - B0_i)) holds for the poles of ONE
    s-wave entrance channel (Lange et al., PRA 79, 013622 (2009)); resonances of other channels
    belong to other a(B) curves, and d/g-wave poles are not poles of the s-wave scattering length.
    """
    groups: dict[str, list[Resonance]] = {}
    for r in resonances:
        if not r.is_s_wave:
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
        )

    def background(self, a_bg_default: float) -> float:
        return self.a_bg_a0 if self.a_bg_a0 is not None else a_bg_default

    @property
    def is_s_wave(self) -> bool:
        return str(self.partial_wave).lower() == 's'


@dataclass(frozen=True)
class ResonanceCurve:
    """a(B) and σ(B) of one resonance on its own (isolated single pole).

    ``None`` for a d/g-wave resonance: it is not a pole of the s-wave a(B).
    """

    resonance: Resonance
    a_bg_a0: float
    a_a0: np.ndarray | None
    sigma_a02: np.ndarray | None


@dataclass(frozen=True)
class ChannelCurve:
    """a(B) and σ(B) of one s-wave entrance channel (product over its poles only)."""

    channel: str
    labels: list[str]
    a_bg_a0: float
    identical_bosons: bool
    a_a0: np.ndarray
    sigma_a02: np.ndarray


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

        a(B)/a_bg = Π_i (1 − Δ_i / (B − B₀,i))

    The overall a_bg is the first resonance's own, else *a_bg_default*: pass
    the resonances of ONE entrance channel (``group_s_wave_by_channel``).
    """
    B = np.asarray(B, dtype=float)
    a_bg = resonances[0].background(a_bg_default)
    factor = np.ones_like(B, dtype=float)
    for res in resonances:
        B0 = res.B0_G
        Delta = res.Delta_G
        denom = B - B0
        with np.errstate(divide='ignore', invalid='ignore'):
            f = 1.0 - Delta / denom
        f = np.where(np.abs(denom) < 1e-12, np.nan, f)
        factor = factor * f
    return a_bg * factor


def default_B_max(resonances: Sequence[Resonance]) -> float:
    """15 % past the highest resonance, at least 500 G."""
    return max(max(r.B0_G for r in resonances) * 1.15, 500.0)


def compute(
    resonances: Sequence[Resonance], a_bg_default_a0: float, C6_au: float, mass_amu: float,
    *, kT_uK: float = 1.0, B_max: float | None = None, B_points: int = 2000,
    boson: bool | None = None, channel: str | None = None,
) -> FeshbachResult:
    """Van der Waals scales, per-channel and per-resonance a(B) and σ(B) on 0..B_max.

    *boson* is the atom's quantum statistics (``is_boson``; None if unknown) and sets
    the identical-particle factor of σ. The combined curve is entrance *channel*
    (default: the channel with the most s-wave resonances).
    """
    beta6 = beta6_a0(C6_au, mass_amu)
    k_th = wave_number_a0(kT_uK, mass_amu)
    B = np.linspace(0.0, default_B_max(resonances) if B_max is None else B_max, B_points)

    # a(B) per entrance channel (s-wave poles of ONE channel multiply; channels never mix)
    groups = group_s_wave_by_channel(resonances)
    if not groups:
        raise ValueError('the resonance table has no s-wave resonances')
    if channel is not None and channel not in groups:
        raise ValueError(f"channel {channel!r} not in {list(groups)}")
    default_channel = channel if channel is not None else max(groups, key=lambda c: len(groups[c]))
    channels = []
    for ch, rs in groups.items():
        ident = identical_particles(ch)
        a_ch = multi_resonance_a(B, rs, a_bg_default_a0)
        if ident and boson is False:
            sig_ch = np.zeros_like(a_ch)       # identical fermions: no s-wave collisions
        else:
            sig_ch = sigma_el_a0(a_ch, k_th, identical_bosons=bool(ident and boson))
        channels.append(ChannelCurve(
            channel=ch,
            labels=[r.label for r in rs],
            a_bg_a0=rs[0].background(a_bg_default_a0),
            identical_bosons=bool(ident and boson),
            a_a0=a_ch,
            sigma_a02=sig_ch,
        ))
    sel = next(c for c in channels if c.channel == default_channel)

    # Per-resonance curves (isolated single-pole form; s-wave only)
    curves = []
    for res in resonances:
        a_bg = res.background(a_bg_default_a0)
        if not res.is_s_wave:
            curves.append(ResonanceCurve(res, a_bg, None, None))
            continue
        a = scattering_length(B, a_bg, res.B0_G, res.Delta_G)
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
