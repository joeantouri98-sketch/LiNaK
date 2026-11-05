"""Einstein A coefficients, radiative lifetimes and line parameters: the pure computation.

For every dipole transition the absorption oscillator strength f_lu comes from the
first source that has it:

  1. NIST (term + J for ions, J-resolved lines with LS 6j completion, or the multiplet
     resolved into fine structure with LS 6j weights), 1b. the curated literature
     overlay, 2. ORCA TD-DFT for ground-state lines matched within 0.10 eV, 3. ORCA
     EOM-CCSD excited → excited moments, 4. Numerov radial integrals calibrated on
     NIST, 5. the Coulomb approximation.

Metals use the published NIST gA / f of each line only. Lines below the Numerov gate
(n* ≤ 2.5), and model (Numerov / Coulomb) lines beyond the last NIST point of a series,
take their f from extrapolating NIST series as f n*³ = a + b/n*². NIST rows are never
changed, only flagged: a NIST multiplet that disagrees with the calibrated Numerov value
by more than ×1.5, or a curated series caveat, widens its uncertainty.

  A_ul = (EINSTEIN_PREFACTOR / λ²) (g_l / g_u) f_lu,   τ = 1 / Σ A_ul

and, for each state's dominant channel, the Einstein B, natural and Doppler widths,
peak cross sections, saturation intensities, recoil, Doppler limit and capture velocity.

``coulomb_fosc`` is the single implementation in ``polarizability_core`` (P26).

No files or prints here: the ``lifetimes`` pipeline loads the transitions, the
NIST / literature / ORCA f-values and passes them in.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from fractions import Fraction

import numpy as _np

from linak.core.constants import AMU_KG, C_M_S, EINSTEIN_PREFACTOR, H_J_S, KB_J_K
from linak.physics.polarizability_core import (
    ACC_UNCERTAINTY,
    coulomb_fosc,
    parse_term_from_label,
)
from linak.physics.qdt_radial import (
    L_CHAR,
    L_INT,
    _parse_nl_token,
    allowed_J2,
    build_true_multiplet_f,
    fit_numerov_scales_by_type,
    ls_group_branching,
    ls_line_weight,
    numerov_fosc_scaled,
    level_defect,
    n_star,
    numerov_scale_for,
    parse_J2_set,
)

# Reference temperature for Doppler width (typical vapour cell)
T_DOPPLER_K = 300.0   # K

# NIST series extrapolation: y(n*) = f n*^3 = a + b/n*^2 (Seaton: f n*^3 = 2 Ry df/dE is a
# smooth function of energy; first two terms of an expansion in 1/n*^2 = -E/Ry), fitted per
# J-resolved series (lower level, upper l, upper J) on the last K NIST points. Accepted when
# >= K points exist, the in-sample rms < 5 % and y > 0. Uncertainty: NIST accuracy (+)
# in-sample rms (+) half the spread of the same fit on windows ending 0/2/4 points earlier.
_EXTRAP_K = 6
_EXTRAP_MAX_RMS = 0.05
_EXTRAP_MODEL_AGREE = 0.10   # extrap and model within 10 %: window spread is not used
_EXTRAP_MODEL_FLOOR = 0.10   # typical Numerov scale spread (s->p ~ 10-20 %), kept as an uncertainty floor
_EXTRAP_BACKOFF = (0, 2, 4)
_EXTRAP_MAX_DROP = 2         # at most this many outlying NIST points may be left out of a fit

# NIST vs model consistency flags (flag only: NIST f-values are never replaced). For every
# NIST multiplet pair inside the Numerov domain (n* > 2.5) the NIST f is compared with the
# NIST-calibrated Numerov value. A ratio outside [1/R, R] is flagged on every NIST row of
# that pair and the row's fractional uncertainty is widened by half the discrepancy (the
# usual half-difference of two independent estimates). Example (Na): the np -> 10s column,
# where 9p->10s (grade A) is 0.68x and 4p->10s 1.9x the model.
NIST_FLAG_RATIO = 1.5
NIST_FLAG_WIDEN_MAX = 3.0   # widen the NIST uncertainty only if the ratio is within [1/3, 3]


# ============================================================================
# RECORDS
# ============================================================================

@dataclass(frozen=True)
class FValueSources:
    """Everything the oscillator-strength priority list can draw on."""

    nist: Mapping                   # polarizability_core.load_nist_fvalues(...)
    literature: Mapping             # polarizability_core.load_literature_fvalues(...)
    # ORCA ground → excited f, summed per excitation energy (eV, 4 decimals)
    orca_fosc: Mapping[float, float] = field(default_factory=dict)
    # ORCA EOM-CCSD excited → excited f by (from, to) state number (1-based)
    ee_fosc: Mapping[tuple[int, int], float] = field(default_factory=dict)
    orca_excitations: Sequence[Mapping] = ()
    # f_values/<El>_literature.json["nist_series_caveats"]: NIST series known to deviate
    # from the smooth physical curve beyond some n (uncertainty only; values unchanged)
    nist_series_caveats: Sequence[Mapping] = ()
    # f_values/<El>_literature.json["nist_fit_exclusions"]: NIST points left out of the
    # extrapolation fits (the table value itself is never altered)
    nist_fit_exclusions: Sequence[Mapping] = ()


@dataclass(frozen=True)
class TransitionA:
    """One line with an Einstein A (a row of ``transitions`` in the lifetimes JSON)."""

    upper_label: str
    lower_label: str
    upper_n: int | None
    lower_n: int | None
    upper_l: str | None
    lower_l: str | None
    wavelength_nm: float
    delta_E_eV: float
    region: str
    fosc: float
    fosc_source: str
    fosc_acc: str | None     # NIST Acc string or None
    fosc_delta: float        # fractional 1-sigma on fosc
    g_upper: float
    g_lower: float
    A_s: float
    sigma_A_s: float
    # Present (not None) only on the rows they concern:
    fosc_model: float | None = None          # model f the NIST extrapolation replaced
    fosc_model_source: str | None = None     # 'Numerov' / 'Coulomb'
    nist_model_ratio: float | None = None    # NIST / calibrated-Numerov f, when outside [1/1.5, 1.5]
    model_unreliable: bool | None = None     # ratio beyond [1/3, 3]: the model, not NIST, is off
    nist_series_caveat: str | None = None    # reference of the curated caveat applied


@dataclass(frozen=True)
class Dropped:
    """A transition that got no A, and why."""

    transition: Mapping
    reason: str


@dataclass(frozen=True)
class SeriesFit:
    """A NIST series fit f n*³ = a + b/n*² used to extrapolate the series."""

    n_fit_max: int
    n_points: int
    a: float
    b: float
    rms: float
    excluded: tuple = ()   # NIST n left out of the fit by the robust retry


@dataclass(frozen=True)
class Lifetime:
    """One upper state: τ, its decay channels, and its dominant channel's line parameters."""

    state: str
    n: int | None
    l: str | None
    A_total_s: float
    sigma_A_s: float
    lifetime_s: float
    sigma_tau_s: float
    quality_acc: str | None    # accuracy of the least accurate channel above 5 % of A
    quality_delta: float
    channels: list[TransitionA]   # strongest first
    # The dominant (strongest) channel:
    B21_m3_J_s2: float
    B12_m3_J_s2: float
    delta_nu_nat_Hz: float
    sigma_peak_nat_m2: float
    I_sat_nat_W_m2: float
    mass_amu: float
    delta_nu_dop_Hz: float     # at T_DOPPLER_K
    sigma_peak_dop_m2: float
    I_sat_dop_W_m2: float
    v_rec_ms: float
    E_rec_J: float
    E_rec_nK: float
    T_dop_uK: float
    v_cap_ms: float

    @property
    def dominant(self) -> TransitionA:
        return self.channels[0]


@dataclass(frozen=True)
class LifetimesResult:
    numerov_calibration: dict | None      # None on the metal path
    transitions: list[TransitionA]        # input order, then the extrapolated lines
    dropped: list[Dropped]
    n_extrapolated: int                   # recovered (previously dropped) channels
    n_extrap_replaced: int                # model values replaced by the NIST extrapolation
    extrap_vs_model: dict[tuple[str, str | None], list[float]]   # (lower, upper l) -> extrap/model f
    series_fits: dict[str, SeriesFit]     # '<lower>-><l><J>'
    nist_flag_pairs: dict[frozenset, float]   # {(n, l), (n', l')} -> NIST / model f
    n_nist_flagged_rows: int
    series_caveats_applied: list[dict]
    lifetimes: list[Lifetime]


@dataclass(frozen=True)
class _Context:
    sources: FValueSources
    qd: dict[str, float]
    z_core: float
    IE: float
    gs_label: str
    is_ion: bool
    is_metal: bool
    orca_only: bool
    nist_only: bool
    nist_true: Mapping                  # multiplet f rebuilt from the NIST lines
    numerov_cal: dict | None


# ============================================================================
# OSCILLATOR-STRENGTH HELPERS
# ============================================================================



def resolve_multiplet_f(f_mult, lu, ll, J_up, J_lo, wl):
    """
    Multiplet f (nl -> n'l', lower-multiplet averaged) -> (fosc, g_upper, g_lower) for
    the channel upper-group -> lower-group, using LS 6j line strengths. A level label
    without J is an unresolved multiplet averaged statistically over its fine-structure
    levels (so no double counting); returns None if the J combination is forbidden.
        A(Ju -> Jl) = A_mult (2Lu+1)(2Jl+1){Lu Ju 1/2; Jl Ll 1}^2,
        A_mult = PREF/lambda^2 (2Ll+1)/(2Lu+1) f_mult
    """
    ratio, gU, gL = ls_group_branching(lu, parse_J2_set(J_up, lu), ll, parse_J2_set(J_lo, ll))
    if ratio <= 1e-12:
        return None
    A_mult = (EINSTEIN_PREFACTOR / wl**2) * (2*ll + 1) / (2*lu + 1) * f_mult
    return A_mult * ratio * wl**2 / EINSTEIN_PREFACTOR * gU / gL, gU, gL


def nist_line_f(lo_nl, up_nl, lu, J_up, ll, J_lo, fosc_jl):
    """
    f of one line (L_l, J_l) -> (L_u, J_u): the NIST line itself if listed, else the
    value implied by another listed lower-J line of the same upper level through the
    LS 6j ratio  f ~ w(Ju,Jl)/(2Jl+1).  Returns (f, acc, reconstructed) or None when
    not applicable (lower s level, merged J labels, ions, no line data).
    """
    if lu < 0 or ll < 1 or not J_up or not J_lo or ',' in str(J_up) or ',' in str(J_lo):
        return None
    ju = parse_J2_set(J_up, lu)
    jl = parse_J2_set(J_lo, ll)
    if len(ju) != 1 or len(jl) != 1:
        return None
    ju, jl = ju[0], jl[0]
    w_this = ls_line_weight(lu, ju, ll, jl)
    if w_this <= 1e-12:
        return (0.0, None, False)
    hit = fosc_jl.get((lo_nl, up_nl, J_up, J_lo))
    if hit:
        return (hit[0], hit[1], False)
    for j2 in allowed_J2(ll):
        if j2 == jl:
            continue
        w_k = ls_line_weight(lu, ju, ll, j2)
        if w_k <= 1e-12:
            continue
        Jk = f"{j2}/2"
        other = fosc_jl.get((lo_nl, up_nl, J_up, Jk))
        if other:
            f = other[0] * (w_this / (jl + 1)) / (w_k / (j2 + 1))
            return (f, other[1], True)
    return None


def degeneracy(l):
    """Degeneracy of orbital l ignoring spin-orbit: 2l+1."""
    return 2 * l + 1


def parse_J_label(label):
    """Extract J from '3p1/2' or ion multiplet key '3s_1Po_1'."""
    if not label:
        return None
    m = re.search(r'(\d+/\d+)$', label)
    if m:
        return m.group(1)
    m = re.search(r'_(\d+h\d+|\d+)$', str(label))
    return m.group(1) if m else None


def J_degeneracy(J_str):
    """Statistical weight g = 2J+1 from a J string like '1/2'."""
    if not J_str:
        return None
    try:
        if '/' in J_str:
            Jn, Jd = (int(x) for x in J_str.split('/'))
            return 2 * Jn // Jd + 1
        return 2 * int(J_str) + 1
    except (ValueError, ZeroDivisionError):
        return None


def default_lower_J(l_char):
    """Default J when the lower label has no J suffix (alkali ns ground → ²S₁/₂)."""
    return '1/2' if l_char == 's' else None


def parse_state_label(label):
    """Parse state label like '3p', '10d', '13s' into (n, l) for sorting."""
    digits = ''.join(filter(str.isdigit, label))
    letters = ''.join(filter(str.isalpha, label))
    return (int(digits) if digits else 0, letters)


def calibration_fvalues(nist_fosc, nist_true):
    """The NIST multiplet f the Numerov scale is fitted on.

    The loader's multiplet f is 1.45-1.48x too large whenever BOTH levels have two J
    (p->d, d->f, d->p, f->d) and exact for s->p and p->s, so it is kept only for pairs
    involving an s level; the rest comes from the multiplets rebuilt from the lines.
    """
    cal = dict(nist_true)
    for k, v in (nist_fosc or {}).items():
        if k in cal or not isinstance(k, tuple):
            continue
        a = re.match(r'^\d+([spdfg])$', k[0])
        b = re.match(r'^\d+([spdfg])$', k[1])
        if a and b and 's' in (a.group(1), b.group(1)):
            cal[k] = v
    return cal


# ============================================================================
# ONE TRANSITION
# ============================================================================

def _row(t, fosc, fosc_source, fosc_acc, fosc_delta, g_upper, g_lower, A, sigma_A):
    return TransitionA(
        upper_label=t['upper_label'], lower_label=t['lower_label'],
        upper_n=t['upper_n'], lower_n=t['lower_n'],
        upper_l=t['upper_l'], lower_l=t['lower_l'],
        wavelength_nm=t['wavelength_nm'], delta_E_eV=t['delta_E_eV'],
        region=t['region'],
        fosc=fosc, fosc_source=fosc_source, fosc_acc=fosc_acc, fosc_delta=fosc_delta,
        g_upper=g_upper, g_lower=g_lower,
        A_s=A, sigma_A_s=sigma_A,
    )


def _metal_row(t, c: _Context) -> TransitionA | None:
    """Metal path: the published NIST gA / f of the line only (no Numerov / Coulomb)."""
    nist = c.sources.nist
    nist_fosc_line = nist.get('fosc_line') or {}
    nist_acc_line = nist.get('acc_line') or {}
    nist_gA_line = nist.get('gA_line') or {}

    wl = t['wavelength_nm']
    lu = L_CHAR.get(t['upper_l'], -1) if t['upper_l'] else -1
    ll_int = L_CHAR.get(t['lower_l'], -1) if t['lower_l'] else -1
    g_upper = degeneracy(lu) if lu >= 0 else 1
    g_lower = degeneracy(ll_int) if ll_int >= 0 else 1
    _gJ_u = J_degeneracy(t.get('upper_J'))
    _gJ_l = J_degeneracy(t.get('lower_J'))
    if _gJ_u:
        g_upper = _gJ_u
    if _gJ_l:
        g_lower = _gJ_l
    _key = (t['lower_label'], t['upper_label'])
    _gA = t.get('gA')
    if _gA is None:
        _gA = nist_gA_line.get(_key)
    _fik = t.get('fik')
    _f_line = t.get('fosc') or nist_fosc_line.get(_key)
    _acc = t.get('Acc') or nist_acc_line.get(_key, 'unknown')
    _delta = ACC_UNCERTAINTY.get(_acc, 0.50) if _acc else 0.50
    if _gA and _gA > 0 and g_upper:
        A = float(_gA) / g_upper
        if _f_line and _f_line > 0:
            fosc = float(_f_line)
        else:
            # f_lu from A: A = PREF/λ² * (g_l/g_u) * f
            fosc = A * (wl ** 2) / EINSTEIN_PREFACTOR * (g_upper / max(g_lower, 1))
    elif _fik and _fik > 0:
        fosc = float(_fik)
        A = EINSTEIN_PREFACTOR / (wl ** 2) * (g_lower / g_upper) * fosc
    elif _f_line and _f_line > 0:
        fosc = float(_f_line)
        A = EINSTEIN_PREFACTOR / (wl ** 2) * (g_lower / g_upper) * fosc
    else:
        return None
    if A <= 0 or fosc <= 0:
        return None
    return _row(t, fosc, 'NIST', _acc, _delta, g_upper, g_lower, A, A * (_delta or 0.50))


def _line_row(t, c: _Context) -> TransitionA | str:
    """The line with its A, or the reason it gets none."""
    nist = c.sources.nist
    nist_fosc = nist['fosc']
    nist_fosc_j = nist['fosc_j']
    nist_fosc_jt = nist.get('fosc_jt') or {}
    nist_acc = nist['acc']
    nist_acc_j = nist['acc_j']
    nist_acc_jt = nist.get('acc_jt') or {}
    nist_fosc_jl = nist.get('fosc_jl') or {}
    nist_fosc_true = c.nist_true
    literature_fosc = c.sources.literature['fosc']
    literature_fosc_j = c.sources.literature['fosc_j']
    literature_acc_j = c.sources.literature['acc_j']
    orca_fosc = c.sources.orca_fosc
    ee_fosc = c.sources.ee_fosc
    qd = c.qd
    gs_label = c.gs_label
    _is_ion = c.is_ion

    _drop_reason = None
    wl  = t['wavelength_nm']
    dE  = t['delta_E_eV']
    ul  = t['upper_label']
    ll  = t['lower_label']
    nu  = t['upper_n']
    nl  = t['lower_n']
    lu_char = t['upper_l']
    ll_char = t['lower_l']
    lu  = L_CHAR.get(lu_char, -1) if lu_char else -1
    ll_int = L_CHAR.get(ll_char, -1) if ll_char else -1

    g_upper = degeneracy(lu) if lu >= 0 else 1
    g_lower = degeneracy(ll_int) if ll_int >= 0 else 1

    fosc        = None
    fosc_source = None
    fosc_acc    = None    # NIST Acc string or None
    fosc_delta  = None    # fractional 1-sigma uncertainty

    # Priority 1: NIST experimental f-values
    # Prefer J-resolved lookup when the transition label carries J (e.g. 3p3/2).
    # Ions: term+J first so ¹P₁ / ³P₁ do not share one f.
    # Fall back to nl-averaged multiplet f_eff for states without J labels.
    _ll_nl = re.sub(r'(\d+[spdfg]).*', r'\1', ll)
    _ul_nl = re.sub(r'(\d+[spdfg]).*', r'\1', ul)
    _J_up  = t.get('upper_J') or parse_J_label(ul)
    _J_lo  = t.get('lower_J') or parse_J_label(ll) or default_lower_J(ll_char)
    _term_up = parse_term_from_label(ul) if _is_ion else None
    _term_lo = parse_term_from_label(ll) if _is_ion else None
    if (nist_fosc_jt or nist_fosc_j or nist_fosc) and not c.orca_only:
        _f = None
        if _is_ion and nist_fosc_jt and _J_up and _term_up:
            # Exact (term_lo, term_up) key; if lower has no term (e.g. '2p'), unique-match on (nl,nl,*,term_up,J).
            _jtkey = (_ll_nl, _ul_nl, _term_lo or '', _term_up, _J_up)
            _f = nist_fosc_jt.get(_jtkey)
            _hit_key = _jtkey if _f else None
            if not _f and not _term_lo:
                _cands = [k for k in nist_fosc_jt
                          if k[0] == _ll_nl and k[1] == _ul_nl
                          and k[3] == _term_up and k[4] == _J_up]
                if len(_cands) == 1:
                    _hit_key = _cands[0]
                    _f = nist_fosc_jt[_hit_key]
            if _f and _f > 0:
                fosc        = _f
                fosc_acc    = nist_acc_jt.get(_hit_key, 'unknown')
                fosc_source = 'NIST'
                fosc_delta  = ACC_UNCERTAINTY.get(fosc_acc, 0.50)
                _gJ_u = J_degeneracy(_J_up)
                _gJ_l = J_degeneracy(_J_lo)
                if _gJ_u:
                    g_upper = _gJ_u
                if _gJ_l:
                    g_lower = _gJ_l
        # Skip plain J lookup for ion multiplet labels, same J merges ¹P/³P.
        if fosc is None and _J_up and nist_fosc_j and not _term_up:
            _jkey = (_ll_nl, _ul_nl, _J_up)
            _f = nist_fosc_j.get(_jkey)
            if _f and _f > 0:
                fosc        = _f
                fosc_acc    = nist_acc_j.get(_jkey, 'unknown')
                fosc_source = 'precision' if fosc_acc == 'precision' else 'NIST'
                fosc_delta  = (0.01 if fosc_acc == 'precision'
                               else ACC_UNCERTAINTY.get(fosc_acc, 0.50))
                # Line-level NIST f + LS 6j completion over the lower fine-structure levels.
                if not _is_ion and fosc_source == 'NIST':
                    _fl = nist_line_f(_ll_nl, _ul_nl, lu, _J_up, ll_int, _J_lo, nist_fosc_jl)
                    if _fl is not None:
                        _f_line, _acc_line, _recon = _fl
                        fosc = _f_line
                        if _acc_line:
                            fosc_acc = _acc_line
                        fosc_delta = ACC_UNCERTAINTY.get(fosc_acc, 0.50)
                        if _recon:   # completed with the LS 6j ratio: add 5 % model error
                            fosc_delta = (fosc_delta**2 + 0.05**2) ** 0.5
                        if _f_line <= 0:
                            _drop_reason = 'forbidden_dJ (|J_u-J_l|>1)'
                _gJ_u = J_degeneracy(_J_up)
                _gJ_l = J_degeneracy(_J_lo)
                if _gJ_u:
                    g_upper = _gJ_u
                if _gJ_l:
                    g_lower = _gJ_l
        # Multiplet nl average also merges ion terms, skip when term is known.
        if fosc is None and nist_fosc and not _term_up:
            _key  = (_ll_nl, _ul_nl)
            _key2 = (_ul_nl, _ll_nl)
            _f    = nist_fosc.get(_key) or nist_fosc.get(_key2)
            _k    = _key if _key in nist_fosc else _key2
            _resolved = None
            if _f and _f > 0 and not _is_ion and lu >= 0 and ll_int >= 0:
                _ft = nist_fosc_true.get(_key) or nist_fosc_true.get(_key2)
                if _ft:
                    _f = _ft
                elif lu >= 1 and ll_int >= 1:
                    _f = None                      # overcounted multiplet, cannot be corrected
                if _f and _f > 0:
                    _resolved = resolve_multiplet_f(_f, lu, ll_int, _J_up, _J_lo, wl)
                    if _resolved is None:
                        _f = None
                        _drop_reason = 'forbidden_dJ (|J_u-J_l|>1 within the LS multiplet)'
            if _f and _f > 0:
                fosc        = _f
                fosc_source = 'NIST'
                fosc_acc    = nist_acc.get(_k, 'unknown')
                fosc_delta  = (0.01 if fosc_acc == 'precision'
                               else ACC_UNCERTAINTY.get(fosc_acc, 0.50))
                if _resolved is not None:
                    fosc, g_upper, g_lower = _resolved

    # Priority 1b: curated literature overlay
    if fosc is None and (literature_fosc_j or literature_fosc) and not c.orca_only:
        if _J_up and literature_fosc_j:
            _jkey = (_ll_nl, _ul_nl, _J_up)
            _f = literature_fosc_j.get(_jkey)
            if _f and _f > 0:
                fosc        = _f
                fosc_source = 'literature'
                fosc_acc    = literature_acc_j.get(_jkey, 'literature')
                fosc_delta  = ACC_UNCERTAINTY.get(fosc_acc, 0.20)
                _gJ_u = J_degeneracy(_J_up)
                _gJ_l = J_degeneracy(_J_lo)
                if _gJ_u:
                    g_upper = _gJ_u
                if _gJ_l:
                    g_lower = _gJ_l
        if fosc is None and not _J_up and literature_fosc_j and ll_int == 0 and not _is_ion:
            # unresolved upper level from an s lower level (single lower J): the multiplet f is the sum
            # of the J-resolved absorption f over the upper J, so the curated rows still apply
            _fs = [v for k, v in literature_fosc_j.items() if k[0] == _ll_nl and k[1] == _ul_nl]
            if _fs and sum(_fs) > 0:
                fosc        = float(sum(_fs))
                fosc_source = 'literature'
                _accs = {literature_acc_j.get(k, 'literature') for k in literature_fosc_j
                         if k[0] == _ll_nl and k[1] == _ul_nl}
                fosc_acc    = _accs.pop() if len(_accs) == 1 else 'literature'
                fosc_delta  = ACC_UNCERTAINTY.get(fosc_acc, 0.20)
        if fosc is None and literature_fosc:
            _key  = (_ll_nl, _ul_nl)
            _key2 = (_ul_nl, _ll_nl)
            _f    = literature_fosc.get(_key) or literature_fosc.get(_key2)
            _res_lit = None
            if _f and _f > 0 and not _is_ion and lu >= 0 and ll_int >= 0:
                # multiplet-averaged f: split over the fine-structure lines (g-weights) exactly as the
                # NIST multiplet path does, otherwise each J_lo sub-level would carry the full multiplet f
                _res_lit = resolve_multiplet_f(_f, lu, ll_int, _J_up, _J_lo, wl)
                if _res_lit is None:
                    _f = None
                    _drop_reason = 'forbidden_dJ (|J_u-J_l|>1 within the LS multiplet)'
            if _f and _f > 0:
                fosc        = _f
                fosc_source = 'literature'
                fosc_acc    = 'literature'
                fosc_delta  = 0.20
                if _res_lit is not None:
                    fosc, g_upper, g_lower = _res_lit

    # Priority 2: ORCA TD-DFT for ground state transitions
    # Match ONLY when a TD-DFT root lies close in energy to the spectroscopic level.
    # A fractional window that grows toward IE (old: up to 0.5 eV) wrongly
    # assigns high-n Rydberg np→gs lines to the highest valence/continuum-like
    # ORCA root, freezing their lifetime near ~0.4 us. Absolute 0.10 eV covers
    # typical CAM-B3LYP valence shifts (Na 3p ~0.07 eV, 4p ~0.08 eV) while
    # rejecting Rydberg latch-on. Roots within 0.25 eV of IE are discarded
    # standard bases do not describe those diffuse states.
    if fosc is None and not c.nist_only and ll == gs_label and orca_fosc:
        exc_ev = dE
        tolerance = 0.10  # eV, absolute
        continuum_cut = c.IE - 0.25
        candidates = [e for e in orca_fosc.keys() if e <= continuum_cut]
        best_match = min(candidates, key=lambda x: abs(x - exc_ev),
                         default=None) if candidates else None
        if best_match is not None and abs(best_match - exc_ev) < tolerance:
            fosc        = orca_fosc[best_match]
            fosc_source = 'ORCA'
            fosc_delta  = 0.05   # ~5% fosc accuracy for CAM-B3LYP

    # EOM excited->excited fosc (EOM-CCSD only)
    # Match upper and lower states by energy to their ORCA state indices, then look up
    # the excited->excited transition moment.
    if fosc is None and ee_fosc and ll != gs_label:
        upper_E = t['upper_energy_eV'] + c.IE    # convert abs to excitation eV
        lower_E = t['lower_energy_eV'] + c.IE

        # Find ORCA state indices closest in energy to upper and lower levels
        tol = max(0.1, min(0.4, 0.15 * abs(upper_E)))
        upper_idx = None
        lower_idx = None
        for idx, ex in enumerate(c.sources.orca_excitations):
            if abs(ex['energy_eV'] - upper_E) < tol:
                upper_idx = idx + 1   # 1-indexed
            if abs(ex['energy_eV'] - lower_E) < tol:
                lower_idx = idx + 1

        if upper_idx and lower_idx:
            f_ee = ee_fosc.get((upper_idx, lower_idx)) or \
                   ee_fosc.get((lower_idx, upper_idx))
            if f_ee and f_ee > 0:
                fosc        = f_ee
                fosc_source = 'EOM-CCSD'
                fosc_delta  = 0.05

    # QD Numerov radial (preferred theoretical fallback; alkalis only)
    # The Numerov f is a MULTIPLET (LS-averaged) absorption f for nl -> n'l'.
    # It is resolved into fine structure with LS 6j line strengths:
    #   A(Ju -> Jl) = A_mult * (2Lu+1)(2Jl+1){Lu Ju 1/2; Jl Ll 1}^2
    # where A_mult = PREF/lambda^2 * (2Ll+1)/(2Lu+1) * f is the rate of any one
    # upper level of the multiplet to the whole lower multiplet. A level label
    # without J (ns, 21p, 21d, ...) is an unresolved multiplet and is averaged
    # statistically over its fine-structure levels (no double counting).
    if fosc is None and not c.orca_only:
        delta_u = level_defect(qd, nu, lu_char)
        delta_l = level_defect(qd, nl, ll_char)
        n_eff_u = nu - delta_u
        n_eff_l = nl - delta_l
        f_num = 0.0
        if n_eff_u > 2.5 and n_eff_l > 2.5 and lu >= 0 and ll_int >= 0:
            try:
                f_num = numerov_fosc_scaled(
                    nu, lu, nl, ll_int, delta_u, delta_l,
                    scale=numerov_scale_for(c.numerov_cal or {}, ll_char, lu_char,
                                            gap=abs(n_eff_u - n_eff_l), n_star_upper=n_eff_u),
                    z_core=c.z_core)
            except Exception:
                f_num = 0.0
        if f_num and f_num > 0:
            fosc_source = 'Numerov'
            fosc_delta  = 0.15 if (n_eff_u > 4 and n_eff_l > 4) else 0.25
            _res = resolve_multiplet_f(f_num, lu, ll_int, _J_up, _J_lo, wl)
            if _res is None:
                fosc = None
                _drop_reason = 'forbidden_dJ (|J_u-J_l|>1 within the LS multiplet)'
            else:
                fosc, g_upper, g_lower = _res

    # Coulomb approximation (last resort; alkalis only)
    if fosc is None and _drop_reason is None and not c.orca_only:
        delta_u = level_defect(qd, nu, lu_char)
        delta_l = level_defect(qd, nl, ll_char)
        n_eff_u = nu - delta_u
        n_eff_l = nl - delta_l
        if n_eff_u > 4 and n_eff_l > 4:
            fosc        = coulomb_fosc(nu, lu, nl, ll_int, delta_u, delta_l)
            fosc_source = 'Coulomb'
            fosc_delta  = 0.30
        elif n_eff_u > 2.5 and n_eff_l > 2.5 and ll != gs_label:
            fosc        = coulomb_fosc(nu, lu, nl, ll_int, delta_u, delta_l)
            fosc_source = 'Coulomb(approx)'
            fosc_delta  = 0.50
        else:
            fosc        = 0.0
            fosc_source = 'unknown'
            fosc_delta  = 1.00

    if fosc is None or fosc <= 0:
        if _drop_reason is None:
            _du = level_defect(qd, nu, lu_char) if lu_char else 0.0
            _dl = level_defect(qd, nl, ll_char) if ll_char else 0.0
            _nsu, _nsl = nu - _du, nl - _dl
            if lu < 0 or ll_int < 0 or abs(lu - ll_int) != 1:
                _drop_reason = 'not_E1_or_unlabelled'
            elif not (_nsu > 2.5 and _nsl > 2.5):
                _drop_reason = 'below_numerov_gate (n*<=2.5, no NIST f)'
            elif (_nsu**2 - _nsl**2) < 0.5 * _nsu:
                _drop_reason = 'near_degenerate_guard (Coulomb/Numerov)'
            else:
                _drop_reason = 'f<=0 or radial integral failed'
        return _drop_reason

    # A_ul = PREFACTOR / lambda_nm^2 * (g_l/g_u) * f_lu
    A       = EINSTEIN_PREFACTOR / (wl**2) * (g_lower / g_upper) * fosc
    sigma_A = A * (fosc_delta or 0.50)   # 1-sigma uncertainty on A
    return _row(t, fosc, fosc_source, fosc_acc, fosc_delta, g_upper, g_lower, A, sigma_A)


# ============================================================================
# NIST SERIES EXTRAPOLATION
# ============================================================================

def _split_J_tag(label, n):
    m = re.match(rf'^{n}[spdfg](\d+/\d+)$', label or '')
    return m.group(1) if m else ''


def _j_value(J) -> float:
    """'3/2' -> 1.5, '2' -> 2.0; raises ValueError for anything that is not a number."""
    return float(Fraction(J)) if isinstance(J, str) else float(J)


def _fit_y(ns_, ys_, xs_):
    M = _np.vstack([_np.ones_like(xs_), 1.0 / xs_**2]).T
    c = _np.linalg.lstsq(M, ys_, rcond=None)[0]
    rms = float(_np.sqrt(_np.mean((M @ c / ys_ - 1.0)**2)))
    return c, rms


def _exclusion_index(exclusions: Sequence[Mapping]) -> dict:
    """{(lower, upper_l, J or '*', n): ref} from the curated ``nist_fit_exclusions`` list."""
    out: dict = {}
    for e in exclusions or ():
        for Jt in ([e['J']] if e.get('J') else None) or ['*']:
            for n in e['n']:
                out[(e['lower'], e['upper_l'], Jt, int(n))] = e.get('ref', '')
    return out


def _build_extrap_fits(lowers, transition_table: Sequence[TransitionA], qd,
                       exclusions: Sequence[Mapping] = ()):
    """Fit y = f n*^3 = a + b/n*^2 for every (lower_label, upper_l, J_tag) NIST series,
    plus the same fit on windows ending 2 and 4 points earlier (``alts``)."""
    series: dict = defaultdict(dict)
    excl = _exclusion_index(exclusions)
    for tt in transition_table:
        if tt.fosc_source not in ('NIST', 'precision') or tt.lower_label not in lowers:
            continue
        lc = tt.upper_l
        if lc not in qd or tt.upper_n is None:
            continue
        Jt = _split_J_tag(tt.upper_label, tt.upper_n) if lc != 's' else ''
        if ((tt.lower_label, lc, Jt, tt.upper_n) in excl
                or (tt.lower_label, lc, '*', tt.upper_n) in excl):
            continue
        series[(tt.lower_label, lc, Jt)][tt.upper_n] = (tt.fosc, tt.fosc_delta or 0.25)
    fits = {}
    for key, d in series.items():
        ns_all = sorted(d)
        if len(ns_all) < _EXTRAP_K:
            continue
        sel = ns_all[-_EXTRAP_K:]
        x = _np.array([n - qd[key[1]] for n in sel])
        y = _np.array([d[n][0] * (n - qd[key[1]])**3 for n in sel])
        c, rms = _fit_y(sel, y, x)
        dropped_pts: list = []
        if rms >= _EXTRAP_MAX_RMS and len(ns_all) >= _EXTRAP_K + 1:
            # robust retry: take a wider tail and leave out at most _EXTRAP_MAX_DROP worst points
            # (leave-one-out) so that one bad NIST entry does not disable a whole series.
            # NIST values are never altered; left-out points are only absent from the fit.
            pool = ns_all[-(_EXTRAP_K + _EXTRAP_MAX_DROP):]
            for _ in range(_EXTRAP_MAX_DROP):
                if len(pool) <= _EXTRAP_K:
                    break
                best = None
                for n_drop in pool:
                    cand = [n for n in pool if n != n_drop]
                    xc = _np.array([n - qd[key[1]] for n in cand])
                    yc = _np.array([d[n][0] * (n - qd[key[1]])**3 for n in cand])
                    cc, rc = _fit_y(cand, yc, xc)
                    if best is None or rc < best[0]:
                        best = (rc, n_drop, cand, cc)
                assert best is not None  #for mypy      
                pool = best[2]
                dropped_pts.append(best[1])
                if best[0] < _EXTRAP_MAX_RMS:
                    break
            sel = pool
            x = _np.array([n - qd[key[1]] for n in sel])
            y = _np.array([d[n][0] * (n - qd[key[1]])**3 for n in sel])
            c, rms = _fit_y(sel, y, x)
        if rms >= _EXTRAP_MAX_RMS:
            continue
        alts = []
        for back in _EXTRAP_BACKOFF[1:]:
            if len(ns_all) >= _EXTRAP_K + back:
                s2 = ns_all[len(ns_all) - _EXTRAP_K - back: len(ns_all) - back]
                x2 = _np.array([n - qd[key[1]] for n in s2])
                y2 = _np.array([d[n][0] * (n - qd[key[1]])**3 for n in s2])
                c2, _ = _fit_y(s2, y2, x2)
                alts.append(c2)
        fits[key] = dict(c=c, n_max=max(sel), rms=rms, delta=d[sel[-1]][1],
                         n_pts=len(sel), alts=alts, excluded=sorted(dropped_pts))
    return fits


def _extrap_channel(tt: Mapping, fits, qd):
    """Extrapolated (f_eff, A, g_up, g_lo, uncertainty) for channel *tt*, or None."""
    ul_, ll_, n_, lc_ = tt['upper_label'], tt['lower_label'], tt['upper_n'], tt['upper_l']
    wl_ = tt['wavelength_nm']
    lu_ = L_CHAR.get(lc_, -1)
    if lu_ < 0 or lc_ not in qd or n_ is None:
        return None
    Jtag = _split_J_tag(ul_, n_) if lc_ != 's' else ''
    if lc_ == 's':
        comps = [('', 2)]
    elif Jtag:
        comps = [(Jtag, J_degeneracy(Jtag))]
    else:
        comps = [(f'{j}/2', j + 1) for j in allowed_J2(lu_)]
    _J_lo_ = tt.get('lower_J') or parse_J_label(ll_) or default_lower_J(tt['lower_l'])
    g_l_ = J_degeneracy(_J_lo_) or (2 * L_CHAR.get(tt['lower_l'], 0) + 1)
    try:
        _Jlo_val = _j_value(_J_lo_)
    except (TypeError, ValueError, ZeroDivisionError):
        _Jlo_val = None
    ns_u = n_ - qd[lc_]
    A_sum, used, spread = 0.0, [], 0.0
    for Jc, gu_c in comps:
        # a dipole-allowed component without an accepted fit => do not extrapolate this channel
        allowed = True
        if Jc and _Jlo_val is not None:
            allowed = abs(_j_value(Jc) - _Jlo_val) <= 1.0 + 1e-9
        fit = fits.get((ll_, lc_, Jc))
        if fit is None:
            if allowed and Jc:
                return None
            continue
        if n_ <= fit['n_max']:
            return None
        y = fit['c'][0] + fit['c'][1] / ns_u**2
        if y <= 0:
            return None
        f_c = y / ns_u**3
        A_c = EINSTEIN_PREFACTOR / wl_**2 * (g_l_ / gu_c) * f_c
        w_c = gu_c if not Jtag and lc_ != 's' else 1.0
        A_sum += A_c * w_c
        used.append((Jc, fit))
        ys = [y] + [c2[0] + c2[1] / ns_u**2 for c2 in fit['alts']]
        if min(ys) > 0:
            spread = max(spread, (max(ys) / min(ys) - 1.0))
    if not used:
        return None
    if not Jtag and lc_ != 's':
        g_up = 2 * (2 * lu_ + 1)
        A_ch = A_sum / g_up
    else:
        g_up = comps[0][1]
        A_ch = A_sum
    f_eff = A_ch * wl_**2 / EINSTEIN_PREFACTOR * g_up / g_l_
    rms = max(fit['rms'] for _, fit in used)
    dnist = max(fit['delta'] for _, fit in used)
    d_ex = (dnist**2 + rms**2 + (0.5 * spread)**2)**0.5
    return dict(f=f_eff, A=A_ch, g_up=g_up, g_lo=g_l_, delta=d_ex, d_nist=dnist, rms=rms, spread=spread,
                n_fit_max=max(fit['n_max'] for _, fit in used))


def _model_replacement(row: TransitionA, res) -> TransitionA:
    """*row* (a Numerov/Coulomb value) with its f taken from the NIST extrapolation *res*."""
    f_model = row.fosc
    d_mod = abs(res['f'] / f_model - 1.0) / 2.0 if f_model else 0.0
    if f_model and abs(res['f'] / f_model - 1.0) < _EXTRAP_MODEL_AGREE:
        # Two different estimates (NIST-series extrapolation, NIST-calibrated grid-converged Numerov)
        # agree: the window-to-window spread only measures how poorly a SHORT window fixes the
        # asymptote a of y = a + b/n*^2 (it grows ~ linearly with n), not the error of the value.
        # Use NIST accuracy + fit rms + half the extrap/model difference + the Numerov scale spread floor.
        d_ex = (res['d_nist']**2 + res['rms']**2)**0.5
        d_tot = (d_ex**2 + d_mod**2 + _EXTRAP_MODEL_FLOOR**2)**0.5
        acc = (f"extrap(n>{res['n_fit_max']}, rms {100*res['rms']:.1f}%, "
               f"within {100*abs(res['f'] / f_model - 1.0):.0f}% of Numerov)")
    else:
        d_tot = (res['delta']**2 + d_mod**2)**0.5
        acc = (f"extrap(n>{res['n_fit_max']}, rms {100*res['rms']:.1f}%, "
               f"window spread {100*res['spread']:.0f}%)")
    return replace(row, fosc=res['f'], fosc_source='NIST-extrap', fosc_acc=acc, fosc_delta=d_tot,
                   g_upper=res['g_up'], g_lower=res['g_lo'], A_s=res['A'],
                   sigma_A_s=res['A'] * d_tot, fosc_model=f_model,
                   fosc_model_source=row.fosc_source)


def extrapolate_nist_series(rows: Sequence[TransitionA], dropped: Sequence[Dropped], qd,
                            exclusions: Sequence[Mapping] = ()):
    """Fill lines from the NIST series of their lower levels. Used ONLY for channels that have no
    NIST/literature/ORCA f:
      (1) lines dropped below the Numerov gate (recovered: appended as new rows),
      (2) Numerov/Coulomb model rows whose upper n lies beyond the last NIST point of a series
          with an accepted fit (replaced in place; the model value is kept as ``fosc_model``).
    Returns (rows, recovered rows, the drops still without an A, the series fits,
    n replaced, extrap/model f ratios per (lower, upper l))."""
    gate = [d for d in dropped if d.reason.startswith('below_numerov_gate')]
    model_rows = [i for i, r in enumerate(rows) if r.fosc_source in ('Numerov', 'Coulomb')]
    lit_rows = [r for r in rows if r.fosc_source == 'literature']
    if not (gate or model_rows or lit_rows):
        return list(rows), [], list(dropped), {}, 0, {}
    # fits are also built for the lower levels fed by the curated tier: they are not used there
    # (literature outranks extrapolation) but document the NIST series the literature continues
    lowers = ({d.transition['lower_label'] for d in gate}
              | {rows[i].lower_label for i in model_rows}
              | {r.lower_label for r in lit_rows})
    fits = _build_extrap_fits(lowers, rows, qd, exclusions)
    recovered, still_dropped = [], []
    for d in dropped:
        res = (_extrap_channel(d.transition, fits, qd)
               if d.reason.startswith('below_numerov_gate') else None)
        if res is None:
            still_dropped.append(d)
            continue
        recovered.append(_row(
            d.transition, res['f'], 'NIST-extrap',
            f"extrap(n>{res['n_fit_max']}, rms {100*res['rms']:.1f}%, "
            f"window spread {100*res['spread']:.0f}%)",
            res['delta'], res['g_up'], res['g_lo'], res['A'], res['A'] * res['delta']))
    out = list(rows)
    n_replaced = 0
    vs_model: dict[tuple[str, str | None], list[float]] = defaultdict(list)
    for i in model_rows:
        row = out[i]
        res = _extrap_channel(asdict(row), fits, qd)
        if res is None:
            continue
        vs_model[(row.lower_label, row.upper_l)].append(
            res['f'] / row.fosc if row.fosc else float('nan'))
        out[i] = _model_replacement(row, res)
        n_replaced += 1
    series = {f"{lo_}->{lc_}{Jc}": SeriesFit(n_fit_max=int(fit['n_max']), n_points=fit['n_pts'],
                                             a=float(fit['c'][0]), b=float(fit['c'][1]),
                                             rms=fit['rms'], excluded=tuple(fit.get('excluded', ())))
              for (lo_, lc_, Jc), fit in fits.items()}
    return out, recovered, still_dropped, series, n_replaced, dict(vs_model)


# ============================================================================
# NIST QUALITY FLAGS (uncertainty only; NIST f-values are never changed)
# ============================================================================

def flag_nist_vs_model(rows: Sequence[TransitionA], nist_cal_fosc: Mapping, numerov_cal,
                       qd, z_core):
    """(rows, flagged pairs, n flagged rows): NIST rows whose multiplet differs from the
    calibrated Numerov value by more than ×NIST_FLAG_RATIO get ``nist_model_ratio`` and,
    within ×NIST_FLAG_WIDEN_MAX, an uncertainty widened by half the discrepancy; beyond it the
    model is the unreliable estimate (radial-integral cancellation, e.g. Na 3d->np) and the
    row is tagged ``model_unreliable`` instead."""
    pairs: dict[frozenset, float] = {}
    for _key, _fn in nist_cal_fosc.items():
        if not isinstance(_key, tuple) or len(_key) != 2 or not _fn or _fn <= 0:
            continue
        _a, _b = _parse_nl_token(_key[0]), _parse_nl_token(_key[1])
        if _a is None or _b is None or abs(_a[1] - _b[1]) != 1:
            continue
        _da = level_defect(qd, _a[0], L_INT.get(_a[1], 's'))
        _db = level_defect(qd, _b[0], L_INT.get(_b[1], 'p'))
        _nsa, _nsb = n_star(_a[0], _da), n_star(_b[0], _db)
        _lo, _up, _dl, _du = (_a, _b, _da, _db) if _nsb > _nsa else (_b, _a, _db, _da)
        if min(_nsa, _nsb) <= 2.5:
            continue
        try:
            _fm = numerov_fosc_scaled(
                _up[0], _up[1], _lo[0], _lo[1], _du, _dl,
                scale=numerov_scale_for(numerov_cal, L_INT[_lo[1]], L_INT[_up[1]],
                                        gap=abs(_nsb - _nsa), n_star_upper=max(_nsa, _nsb)),
                z_core=z_core)
        except Exception:
            continue
        if not _fm or _fm <= 0:
            continue
        _r = _fn / _fm
        if _r > NIST_FLAG_RATIO or _r < 1.0 / NIST_FLAG_RATIO:
            pairs[frozenset(((_lo[0], L_INT[_lo[1]]), (_up[0], L_INT[_up[1]])))] = _r
    out, n_flagged = [], 0
    for row in rows:
        _r = None
        if row.fosc_source == 'NIST':
            _r = pairs.get(frozenset(((row.lower_n, row.lower_l), (row.upper_n, row.upper_l))))
        if _r is None:
            out.append(row)
            continue
        n_flagged += 1
        if 1.0 / NIST_FLAG_WIDEN_MAX <= _r <= NIST_FLAG_WIDEN_MAX:
            _d = (row.fosc_delta ** 2 + (0.5 * abs(_r - 1.0)) ** 2) ** 0.5
            out.append(replace(row, fosc_delta=_d, sigma_A_s=row.A_s * _d, nist_model_ratio=_r))
        else:
            out.append(replace(row, nist_model_ratio=_r, model_unreliable=True))
    return out, pairs, n_flagged


def apply_series_caveats(rows: Sequence[TransitionA], caveats: Sequence[Mapping]):
    """(rows, applied caveats): a curated caveat raises the quoted uncertainty of the NIST rows
    of one series (lower level, upper l, upper n >= n_min) to at least rel_unc and tags them."""
    out, applied = list(rows), []
    for _cv in caveats:
        _n_hit = 0
        for i, row in enumerate(out):
            if (row.fosc_source == 'NIST' and row.lower_label == _cv['lower']
                    and row.upper_l == _cv['upper_l'] and row.upper_n >= _cv['n_min']):
                _d = max(row.fosc_delta, float(_cv['rel_unc']))
                out[i] = replace(row, fosc_delta=_d, sigma_A_s=row.A_s * _d,
                                 nist_series_caveat=_cv['ref'])
                _n_hit += 1
        if _n_hit:
            applied.append({'lower': _cv['lower'], 'upper_l': _cv['upper_l'],
                            'n_min': _cv['n_min'], 'rel_unc': _cv['rel_unc'],
                            'n_rows': _n_hit, 'ref': _cv['ref']})
    return out, applied


# ============================================================================
# LIFETIMES AND LINE PARAMETERS
# ============================================================================

def lifetime_of(state_label, tlist: Sequence[TransitionA], mass_amu: float) -> Lifetime | None:
    """τ = 1/ΣA of one upper state and the line parameters of its dominant channel."""
    A_total     = sum(t.A_s for t in tlist)
    var_A_total = sum(t.sigma_A_s**2 for t in tlist)   # sum of variances
    if A_total <= 0:
        return None

    tau_s   = 1.0 / A_total
    sigma_A = var_A_total**0.5
    # σ(τ) = τ² × σ(A)   [error propagation: τ=1/A, dτ/dA=-1/A²]
    sigma_tau_s = sigma_A / A_total**2

    dominant = max(tlist, key=lambda t: t.A_s)

    # Overall quality grade: worst (largest delta) among channels weighted by A fraction
    # Channels contributing >5% of total A drive the uncertainty
    quality_acc   = None
    quality_delta = 0.0
    for ch in tlist:
        if ch.A_s / A_total > 0.05:
            d = ch.fosc_delta
            if d > quality_delta:
                quality_delta = d
                quality_acc   = ch.fosc_acc or ch.fosc_source

    # Einstein B coefficients
    # Computed for the DOMINANT decay channel (highest A).
    # B₂₁ = A₂₁ · c³ / (8πhν³)          stimulated emission [m³ J⁻¹ s⁻²]
    # B₁₂ = (g₂/g₁) · B₂₁                absorption
    # Both in SI: spectral energy density ρ(ν) units [J m⁻³ Hz⁻¹]
    #
    # Peak absorption cross-section at line centre (natural broadening):
    #   σ_peak = (λ²/2π) · (g₂/g₁) · (A₂₁/A_total) = (λ²/4π²) · (g₂/g₁) · A₂₁ / Δν_nat
    #   (a unit-normalised Lorentzian peaks at 2/(π·Δν_nat); P22: the old (λ²/8π)·A/Δν used 1/Δν
    #    and was π/2 too large. For A₂₁ = A_total: σ = (g₂/g₁)·λ²/2π, Steck's two-level value
    #    is 3λ²/2π for the stretched cycling transition.)
    #   Δν_nat = A_total / (2π)             [Hz]  natural linewidth (FWHM)
    #
    # Saturation intensity (two-level approximation):
    #   I_sat = hν · A₂₁² · (1/2) / (σ_peak · A_total)
    #   Simplified for a closed two-level: I_sat = hν/(2·σ_peak·τ)  [W m⁻²]
    #   (factor of 2 for equal degeneracies; exact prefactor depends on g ratio)

    dom_A    = dominant.A_s                 # s⁻¹
    dom_wl_m = dominant.wavelength_nm * 1e-9   # m
    dom_nu   = C_M_S / dom_wl_m          # Hz
    dom_E_J  = H_J_S * dom_nu            # J  (photon energy)
    g_upper  = dominant.g_upper
    g_lower  = dominant.g_lower

    # B₂₁  [m³ J⁻¹ s⁻²]
    B21_SI   = dom_A * C_M_S**3 / (8 * math.pi * H_J_S * dom_nu**3)

    # B₁₂  [m³ J⁻¹ s⁻²]
    B12_SI   = (g_upper / g_lower) * B21_SI

    # Natural linewidth Δν [Hz] = A_total / (2π)
    delta_nu_nat = A_total / (2 * math.pi)

    # Peak cross-section at line centre [m²] — natural broadening (Lorentzian)
    sigma_peak_m2 = (dom_wl_m**2 / (4 * math.pi**2)) * (g_upper / g_lower) * (dom_A / delta_nu_nat)

    # Saturation intensity [W m⁻²] — two-level closed transition approximation
    # I_sat = hν / (2 · σ_peak · τ)
    I_sat_Wm2  = dom_E_J / (2.0 * sigma_peak_m2 * tau_s)

    # Doppler-broadened quantities
    # The Doppler FWHM is:
    #   Δν_D = (ν₀/c) · √(8·k_B·T·ln2 / m)
    # Peak cross-section with Doppler broadening (CompTable / compare_elements):
    #   σ_peak_dop = σ_peak_nat · (Δν_nat / Δν_D) · √(π·ln2)
    # Derivation: σ(ν) = σ₀·(πΔν_nat/2)·φ(ν) with φ normalised; Lorentzian φ(0)=2/(πΔν_nat) gives σ₀,
    # Gaussian φ(0)=2√(ln2/π)/Δν_D gives σ₀·(Δν_nat/Δν_D)·√(π ln2). 

    # Saturation intensity with Doppler broadening (same formula, different σ):
    #   I_sat_dop = hν / (2 · σ_peak_dop · τ)
    # This is always LARGER than I_sat_nat because σ_peak_dop < σ_peak_nat.
    #
    # Recoil quantities (per absorbed photon):
    #   v_rec = ħk/m              recoil velocity [m/s]
    #   E_rec = ħ²k²/(2m) = ½m·v_rec²   recoil energy [J] → nK
    #
    # Doppler cooling limit:
    #   T_Dop = ħ·Γ/(2·k_B)      minimum temperature from Doppler cooling [K] → μK
    #
    # Capture velocity:
    #   v_cap = Γ/k               maximum velocity that can be captured [m/s]
    mass_kg = mass_amu * AMU_KG
    dom_k   = 2 * math.pi / dom_wl_m   # wave vector [rad/m]

    # Doppler FWHM [Hz]
    delta_nu_dop = (dom_nu / C_M_S) * math.sqrt(
        8 * KB_J_K * T_DOPPLER_K * math.log(2) / mass_kg
    )

    # Peak cross-section with Doppler broadening [m²]
    # Matches compare_elements.sigma_peak_doppler:
    #   σ_D = σ_nat · (Δν_nat / Δν_D) · √(π ln 2)
    sigma_peak_dop_m2  = (
        sigma_peak_m2 * (delta_nu_nat / delta_nu_dop)
        * math.sqrt(math.pi * math.log(2))
    )

    # Saturation intensity with Doppler broadening [W/m²]
    I_sat_dop_Wm2   = dom_E_J / (2.0 * sigma_peak_dop_m2 * tau_s)

    # Recoil velocity [m/s] and energy
    v_rec_ms   = (H_J_S / (2 * math.pi)) * dom_k / mass_kg    # ħk/m
    E_rec_J    = 0.5 * mass_kg * v_rec_ms**2

    # Doppler cooling temperature limit
    Gamma_s  = A_total
    T_dop_K  = (H_J_S / (2 * math.pi)) * Gamma_s / (2 * KB_J_K)

    return Lifetime(
        state=state_label,
        n=tlist[0].upper_n,
        l=tlist[0].upper_l,
        A_total_s=A_total,
        sigma_A_s=sigma_A,
        lifetime_s=tau_s,
        sigma_tau_s=sigma_tau_s,
        quality_acc=quality_acc,
        quality_delta=quality_delta,
        channels=sorted(tlist, key=lambda t: -t.A_s),
        B21_m3_J_s2=B21_SI,
        B12_m3_J_s2=B12_SI,
        delta_nu_nat_Hz=delta_nu_nat,
        sigma_peak_nat_m2=sigma_peak_m2,
        I_sat_nat_W_m2=I_sat_Wm2,
        mass_amu=mass_amu,
        delta_nu_dop_Hz=delta_nu_dop,
        sigma_peak_dop_m2=sigma_peak_dop_m2,
        I_sat_dop_W_m2=I_sat_dop_Wm2,
        v_rec_ms=v_rec_ms,
        E_rec_J=E_rec_J,
        E_rec_nK=E_rec_J / KB_J_K * 1e9,
        T_dop_uK=T_dop_K * 1e6,
        v_cap_ms=Gamma_s / dom_k,
    )


# ============================================================================
# ONE SPECIES
# ============================================================================

def compute(
    transitions: Sequence[Mapping], sources: FValueSources, *,
    qd: dict[str, float], z_core: float, IE: float, gs_label: str, mass_amu: float,
    is_ion: bool = False, is_metal: bool = False,
    orca_only: bool = False, nist_only: bool = False,
) -> LifetimesResult:
    """Einstein A of every transition, then τ and line parameters per upper state."""
    nist = sources.nist
    nist_true = {} if (is_ion or is_metal) else build_true_multiplet_f(nist.get('fosc_jl') or {})
    numerov_cal = None
    nist_cal_fosc: dict = {}
    if not is_metal:
        # Per (l_lower, l_upper) scale, fitted only on NIST pairs inside the Numerov
        # domain (n*_l, n*_u > 2.5); types with <5 inlier pairs use the pooled median.
        nist_cal_fosc = calibration_fvalues(nist['fosc'], nist_true)
        numerov_cal = fit_numerov_scales_by_type(qd, nist_cal_fosc, z_core=z_core)
    c = _Context(sources=sources, qd=qd, z_core=z_core, IE=IE, gs_label=gs_label,
                 is_ion=is_ion, is_metal=is_metal, orca_only=orca_only, nist_only=nist_only,
                 nist_true=nist_true, numerov_cal=numerov_cal)

    rows: list[TransitionA] = []
    dropped: list[Dropped] = []
    for t in transitions:
        if is_metal:
            metal_row = _metal_row(t, c)
            if metal_row is not None:
                rows.append(metal_row)
            continue
        line = _line_row(t, c)
        if isinstance(line, str):
            dropped.append(Dropped(t, line))
        else:
            rows.append(line)

    recovered: list[TransitionA] = []
    series: dict[str, SeriesFit] = {}
    n_replaced, vs_model = 0, {}
    if not is_metal:
        rows, recovered, dropped, series, n_replaced, vs_model = extrapolate_nist_series(
            rows, dropped, qd, sources.nist_fit_exclusions)
        rows.extend(recovered)
    flag_pairs: dict[frozenset, float] = {}
    n_flagged = 0
    if numerov_cal is not None:
        rows, flag_pairs, n_flagged = flag_nist_vs_model(rows, nist_cal_fosc, numerov_cal, qd, z_core)
    rows, caveats_applied = apply_series_caveats(rows, sources.nist_series_caveats)

    # τ = 1 / Σ A_ul over the channels of each upper state
    A_by_upper: dict[str, list[TransitionA]] = defaultdict(list)
    for row in rows:
        A_by_upper[row.upper_label].append(row)
    lifetimes = []
    for state_label, tlist in sorted(A_by_upper.items(),
                                     key=lambda x: parse_state_label(x[0])):
        lt = lifetime_of(state_label, tlist, mass_amu)
        if lt is not None:
            lifetimes.append(lt)

    return LifetimesResult(numerov_calibration=numerov_cal, transitions=rows,
                           dropped=dropped, n_extrapolated=len(recovered),
                           n_extrap_replaced=n_replaced, extrap_vs_model=vs_model,
                           series_fits=series, nist_flag_pairs=flag_pairs,
                           n_nist_flagged_rows=n_flagged,
                           series_caveats_applied=caveats_applied, lifetimes=lifetimes)
