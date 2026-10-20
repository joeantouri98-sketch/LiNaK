"""Hyperfine structure, Zeeman and Breit-Rabi: the pure computation.

For each state |n l J⟩ of an isotope with nuclear spin I, the hyperfine
energy is a polynomial in K = F(F+1) − I(I+1) − J(J+1) = 2 I·J (dipole A,
quadrupole B, octupole C; see ``_hf_polynomial``). g_F comes from g_J (Landé,
or an experimental override) and g_I; the Breit-Rabi levels diagonalise the
EXACT H_hf (the same polynomial applied to the operator 2 I·J) plus
(g_J μ_B J_z − g_I μ_N I_z)·B, so at B = 0 they equal the F-level energies.

A state absent from the constants table borrows them from a measured state
with the same (l, J), scaled as A ∝ n*⁻³ with the quantum defect of l; a
measured A = 0 is data, not "missing". The ground state (lowest n, l, J) gets
the microwave clock interval (with A, B and C) and, when a p state is in the
run, the cycling transition of its resonance line.

No files, prints or plots here: the ``hyperfine`` pipeline loads the nuclear
moments, the constants table and the quantum defects and passes them in.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from linak.core.constants import C_SI, MU_B_MHz_G, MU_N_MHz_G
from linak.physics.polarizability_core import L_CHAR

CLOCK_NOTE = 'Zero-field ground ΔF=1 interval (m_F=0 clock to first order in B)'


@dataclass(frozen=True)
class Nucleus:
    """One isotope's nuclear moments (a row of ``nuclear_data.json``)."""

    A: int
    I: float
    mu_I: float     # nuclear magnetons
    Q_barn: float
    gI: float       # μ_I / I

    @classmethod
    def from_json(cls, row: Mapping) -> Nucleus:
        # An unknown moment reads as 0: no nuclear Zeeman term.
        return cls(A=row['A'], I=row['I'], mu_I=row['mu_I'] or 0.0,
                   Q_barn=row['Q_barn'], gI=row['g_I'] or 0.0)


@dataclass(frozen=True)
class HfConstants:
    """Measured constants of one state (a row of ``<El>_hf_constants.json``)."""

    A_MHz: float
    B_MHz: float = 0.0    # electric quadrupole; absent reads as 0
    C_kHz: float = 0.0    # magnetic octupole; absent reads as 0
    gJ_exp: float | None = None   # overrides the Landé g_J
    S_exp: float | None = None    # effective spin for Landé (default 1/2)

    @classmethod
    def from_json(cls, row: Mapping) -> HfConstants:
        B = row.get('B_MHz')
        C = row.get('C_kHz')
        return cls(
            A_MHz=float(row['A_MHz']),
            B_MHz=0.0 if B is None else float(B),
            C_kHz=0.0 if C is None else float(C),
            gJ_exp=float(row['gJ_exp']) if 'gJ_exp' in row else None,
            S_exp=float(row['S_exp']) if 'S_exp' in row else None,
        )


@dataclass(frozen=True)
class FLevel:
    """One hyperfine level F of a state."""

    F: float
    E_MHz: float        # relative to the state's centroid
    gF: float
    mF: list[float]     # +F … −F


@dataclass(frozen=True)
class HyperfineState:
    label: str
    n: int
    l: str
    J: float
    gJ: float
    gJ_source: str      # 'experimental' or 'Lande'
    A_MHz: float
    B_MHz: float
    C_kHz: float
    extrapolated_from: str | None   # the measured state A, B, C were scaled from
    constants_missing: bool         # not in the table and nothing to scale from: A = B = C = 0
    levels: list[FLevel]
    BR_energies: np.ndarray         # (B point, level) Breit-Rabi eigenvalues, MHz
    BR_labels: list[tuple[float, float]]   # (F, m_F) of each level at B = 0


@dataclass(frozen=True)
class MicrowaveClock:
    """The zero-field ΔF = 1 interval between a state's two highest F levels."""

    F_upper: float
    F_lower: float
    frequency_MHz: float
    frequency_GHz: float
    frequency_Hz: float
    wavelength_m: float | None    # None for degenerate levels (A = 0)
    wavelength_cm: float | None
    note: str
    state: str


@dataclass(frozen=True)
class CyclingLine:
    """The closed F_max → F_max + 1 line from the ground state to its resonance p state."""

    lower: str
    upper: str
    F_lower: float
    F_upper: float


@dataclass(frozen=True)
class HyperfineResult:
    nucleus: Nucleus
    B_G: np.ndarray
    states: list[HyperfineState]
    clock: MicrowaveClock | None    # of the ground state
    cycling: CyclingLine | None
    skipped: list[str]              # labels that are not of the form '3p3/2'


# ============================================================================
# STATE LABELS AND F LEVELS
# ============================================================================

def parse_state_label(label):
    """
    Parse a state label like '3s1/2', '3p3/2', '4d5/2', '3s' into (n, l, J).
    Returns (n, l_char, J_float) or None.
    """
    m = re.match(r'^(\d+)([spdfgh])([\d/]+)?$', label.strip())
    if not m:
        return None
    n      = int(m.group(1))
    l_char = m.group(2)
    j_str  = m.group(3)
    if j_str:
        if '/' in j_str:
            jn, jd = j_str.split('/')
            J = int(jn) / int(jd)
        else:
            J = float(j_str)
    else:
        # Infer J from l and ground-state convention
        l_int = L_CHAR[l_char]
        J = l_int + 0.5   # default to J = l + 1/2 for doublet
    return n, l_char, J


def F_levels(I, J):
    """Return list of allowed F quantum numbers: |I-J|, ..., I+J."""
    F_min = abs(I - J)
    F_max = I + J
    Fs = []
    F = F_min
    while F <= F_max + 1e-9:
        Fs.append(round(F * 2) / 2)   # keep half-integer precision
        F += 1
    return Fs


# ============================================================================
# HYPERFINE ENERGIES AND g FACTORS
# ============================================================================

def _hf_polynomial(K, I, J, A_MHz, B_MHz, C_kHz, ident):
    """
    Hyperfine interaction as a polynomial in K = F(F+1) − I(I+1) − J(J+1) = 2 I·J.

    K may be a float (energy of one |F⟩ level) or a square matrix (the operator 2 I·J in the
    |m_I, m_J⟩ basis; `ident` is then the identity matrix). The SAME expression serves both, so the
    Breit–Rabi Hamiltonian at B = 0 reproduces `hyperfine_energy` to machine precision.

      dipole      E_A = A/2 · K
      quadrupole  E_B = B · [3/2 K(K+1) − 2 I(I+1) J(J+1)] / [(2I−1) 2I (2J−1) 2J]
      octupole    E_C = C · [5K²(K/4 + 1) + K(I(I+1) + J(J+1) + 3 − 3 I(I+1) J(J+1))
                             − 5 I(I+1) J(J+1)] / [I(I−1)(2I−1) J(J−1)(2J−1)]

    The octupole expression is Steck's (Rb/Cs D-line data); it equals, up to one constant, the
    exact rank-3 angular factor (−1)^{I+J+F} {I J F; J I 3} (checked to 1e-16 for several I, J).
    Quadrupole needs I ≥ 1 and J ≥ 1; octupole needs I ≥ 3/2 and J ≥ 3/2 (otherwise exactly 0).
    """
    mat = np.ndim(K) == 2
    II, JJ = I * (I + 1), J * (J + 1)
    E = 0.5 * A_MHz * K
    if B_MHz != 0.0 and I >= 1.0 - 1e-9 and J >= 1.0 - 1e-9:
        denom = (2*I - 1) * (2*I) * (2*J - 1) * (2*J)
        if abs(denom) > 1e-12:
            KK1 = K @ K + K if mat else K * (K + 1)
            E = E + B_MHz * (1.5 * KK1 - 2 * II * JJ * ident) / denom
    if C_kHz != 0.0 and I >= 1.5 - 1e-9 and J >= 1.5 - 1e-9:
        denom = I * (I - 1) * (2*I - 1) * J * (J - 1) * (2*J - 1)
        if abs(denom) > 1e-12:
            K2 = K @ K if mat else K * K
            K3 = K2 @ K if mat else K2 * K
            poly = (1.25 * K3 + 5.0 * K2 + (II + JJ + 3 - 3 * II * JJ) * K - 5 * II * JJ * ident)
            E = E + (C_kHz * 1e-3) * poly / denom
    return E


def hyperfine_energy(F, I, J, A_MHz, B_MHz=0.0, C_kHz=0.0):
    """
    Hyperfine energy of |I, J, F⟩ in MHz relative to the centroid, with magnetic dipole (A, MHz),
    electric quadrupole (B, MHz) and magnetic octupole (C, kHz) terms. See `_hf_polynomial` for the
    formulas (K = F(F+1) − I(I+1) − J(J+1)).

    References: Steck, Alkali D Line Data (hyperfine structure section); Armstrong (1971)
    Adv. At. Mol. Phys. 7:149.
    """
    K = F*(F+1) - I*(I+1) - J*(J+1)
    return _hf_polynomial(K, I, J, A_MHz, B_MHz, C_kHz, 1.0)


def lande_gJ(L, S, J, gJ_exp=None):
    """
    Landé g_J factor.
    g_J = 1 + [J(J+1) + S(S+1) − L(L+1)] / [2J(J+1)]

    For non-alkali atoms or when higher accuracy is needed, pass
    gJ_exp (experimental value from NIST or literature) to override
    the Landé formula. This handles:
      - Heavy atoms where LS coupling breaks down (jj coupling)
      - Multi-electron atoms (Yb, Sr, Ca, Hg, etc.)
      - States where QED corrections matter (>0.1% level)
    For J=0 returns 0 regardless.
    """
    if J < 1e-9:
        return 0.0
    if gJ_exp is not None:
        return float(gJ_exp)
    return 1.0 + (J*(J+1) + S*(S+1) - L*(L+1)) / (2*J*(J+1))


def gF_factor(F, I, J, gJ, gI=0.0):
    """
    g_F factor in the weak-field Zeeman regime.
    g_F = g_J·[F(F+1)+J(J+1)−I(I+1)] / [2F(F+1)]
          − g_I·(μ_N/μ_B)·[F(F+1)+I(I+1)−J(J+1)] / [2F(F+1)]
    μ_N/μ_B ≈ 1/1836.15 (ratio of nuclear to Bohr magneton)
    """
    if F < 1e-9:
        return 0.0
    FF = F*(F+1)
    JJ = J*(J+1)
    II = I*(I+1)
    gF  =  gJ * (FF + JJ - II) / (2*FF)
    gF -= gI * (1/1836.15267) * (FF + II - JJ) / (2*FF)
    return gF


# ============================================================================
# BREIT-RABI DIAGONALISATION
# ============================================================================

def breit_rabi_energies(I, J, A_MHz, B_MHz_quad, gJ, gI,
                        B_gauss_array, C_kHz=0.0):
    """
    Exact Breit-Rabi diagonalisation for arbitrary B field.

    Builds the full (2I+1)(2J+1) × (2I+1)(2J+1) Hamiltonian H = H_hf + H_Z and diagonalises it at
    every B.

    H_hf is the EXACT hyperfine operator (dipole + quadrupole + octupole, no approximation): the
    operator K = 2 I·J is built in the |m_I, m_J⟩ basis (I_zJ_z + ½(I_+J_- + I_-J_+)) and the same
    polynomial in K that gives the analytic F-level energies (`_hf_polynomial`) is applied to the
    matrix. At B = 0 the eigenvalues therefore equal `hyperfine_energy` exactly, with (2F+1)-fold
    degeneracy.
    H_Z = (g_J μ_B m_J − g_I μ_N m_I) B  (diagonal in |m_I, m_J⟩).

    Basis index: idx = (m_I index)·(2J+1) + (m_J index), m = −I..+I, −J..+J.

    Returns:
      energies[B_idx, level_idx]  — sorted eigenvalues in MHz
      labels[level_idx]           — approximate (F, m_F) label at B→0
    """
    dim_I = round(2*I + 1)
    dim_J = round(2*J + 1)
    dim   = dim_I * dim_J

    mI_vals = np.array([-I + i for i in range(dim_I)])
    mJ_vals = np.array([-J + j for j in range(dim_J)])

    def idx(mi_i, mj_i):
        return mi_i * dim_J + mj_i

    # K = 2 I·J 
    # ⟨m_I, m_J|I_zJ_z|m_I, m_J⟩ = m_I m_J
    # ⟨m_I+1, m_J−1|I_+J_-|m_I, m_J⟩ = sqrt[(I−m_I)(I+m_I+1)] sqrt[(J+m_J)(J−m_J+1)]
    IJ = np.zeros((dim, dim))
    for mi_i, mI in enumerate(mI_vals):
        for mj_i, mJ in enumerate(mJ_vals):
            r = idx(mi_i, mj_i)
            IJ[r, r] += mI * mJ
            if mi_i + 1 < dim_I and mj_i - 1 >= 0:
                r2 = idx(mi_i + 1, mj_i - 1)
                me = (math.sqrt((I - mI)*(I + mI + 1)) *
                      math.sqrt((J + mJ)*(J - mJ + 1)))
                IJ[r2, r] += 0.5 * me
                IJ[r, r2] += 0.5 * me
    K_mat = 2.0 * IJ
    H_hf = _hf_polynomial(K_mat, I, J, A_MHz, B_MHz_quad, C_kHz, np.eye(dim))

    # H_Z diagonal (per Gauss)
    H_Z_diag = np.array([
        gJ * MU_B_MHz_G * mJ_vals[mj_i]
        - gI * MU_N_MHz_G * mI_vals[mi_i]
        for mi_i in range(dim_I)
        for mj_i in range(dim_J)
    ])

    #(F, m_F) labels at B = 0 
    # A vanishing probe field along z splits each (2F+1)-fold level by m_F so that the eigenvectors
    # are m_F eigenstates (H_hf commutes with F_z = m_I + m_J); it is far below any energy scale.
    mF_diag = np.array([mI_vals[i] + mJ_vals[j] for i in range(dim_I) for j in range(dim_J)])
    # F² = I(I+1) + J(J+1) + 2 I·J is added as a tiny tie-breaker so that F is a good label even when
    # the hyperfine levels are degenerate (A = B = C = 0): the labels then come from <F²>, not from
    # the nearest F energy (which is arbitrary when every F has the same energy).
    F2_mat = (I*(I+1) + J*(J+1)) * np.eye(dim) + K_mat
    evals0, evecs0 = np.linalg.eigh(H_hf + np.diag(1e-9 * mF_diag) + 1e-7 * F2_mat)
    Fs = F_levels(I, J)
    labels = []
    for k in range(dim):
        v      = evecs0[:, k]
        dom_r  = int(np.argmax(np.abs(v)**2))
        mI     = mI_vals[dom_r // dim_J]
        mJ     = mJ_vals[dom_r %  dim_J]
        mF_approx = round(2*(mI + mJ)) / 2   # m_F = m_I + m_J
        F2_exp = float(np.real(v.conj() @ F2_mat @ v))
        best_F = min(Fs, key=lambda F_: abs(F_*(F_+1) - F2_exp))
        labels.append((best_F, mF_approx))

    # Diagonalise at each B 
    all_energies = np.zeros((len(B_gauss_array), dim))
    for bi, B in enumerate(B_gauss_array):
        H = H_hf.copy()
        np.fill_diagonal(H, H.diagonal() + H_Z_diag * B)
        all_energies[bi] = np.linalg.eigvalsh(H)

    return all_energies, labels


# ============================================================================
# CLOCK INTERVAL AND CYCLING TRANSITION
# ============================================================================

def allowed_transitions(Fs_lower, Fs_upper):
    """
    E1 selection rules between two sets of F levels:
      ΔF = 0, ±1  (F=0 → F=0 forbidden)
    Returns list of (F_lower, F_upper) pairs.
    """
    transitions = []
    for F_l in Fs_lower:
        for F_u in Fs_upper:
            dF = abs(F_u - F_l)
            if dF > 1.001:
                continue
            if F_l < 0.001 and F_u < 0.001:
                continue   # 0→0 forbidden
            transitions.append((F_l, F_u))
    return transitions


def cycling_transition(transitions_list, Fs_ground):
    """The closed F_max → F_max + 1 pair among the allowed (F_lower, F_upper) pairs of the
    resonance line (see ``allowed_transitions``), or None (also for an empty list)."""
    F_max_gs = max(Fs_ground)
    for F_l, F_u in transitions_list:
        if abs(F_l - F_max_gs) < 0.001 and abs(F_u - (F_max_gs + 1)) < 0.001:
            return F_l, F_u
    return None


def clock_interval(Fs, I, J, A_MHz, B_MHz=0.0, C_kHz=0.0):
    """(F_max, F_max − 1, |ΔE| in MHz) at zero field with the full A, B and C constants
    (B and C vanish for J = 1/2), or None with a single F."""
    if len(Fs) < 2:
        return None
    F1, F2 = Fs[-1], Fs[-2]
    dE = (hyperfine_energy(F1, I, J, A_MHz, B_MHz, C_kHz)
          - hyperfine_energy(F2, I, J, A_MHz, B_MHz, C_kHz))
    return F1, F2, abs(dE)


def microwave_clock(state: HyperfineState, I: float) -> MicrowaveClock | None:
    """The F_max ↔ F_max − 1 interval of *state* (m_F = 0 clock), or None with a single F."""
    interval = clock_interval([lv.F for lv in state.levels], I, state.J,
                              state.A_MHz, state.B_MHz, state.C_kHz)
    if interval is None:
        return None
    F1, F2, nu_MHz = interval
    return MicrowaveClock(
        F_upper=F1,
        F_lower=F2,
        frequency_MHz=nu_MHz,
        frequency_GHz=nu_MHz / 1e3,
        frequency_Hz=nu_MHz * 1e6,
        wavelength_m=C_SI / (nu_MHz * 1e6) if nu_MHz > 0 else None,
        wavelength_cm=C_SI * 1e-4 / nu_MHz if nu_MHz > 0 else None,
        note=CLOCK_NOTE,
        state=state.label,
    )


def ground_state(states: Sequence[HyperfineState]) -> HyperfineState:
    """Lowest n, then lowest l, then lowest J: independent of the order the states were listed
    (--states 3p 3s would otherwise take the 3p level as the "clock" state)."""
    order = 'spdfghik'
    return min(states, key=lambda s: (s.n, order.find(s.l) if s.l in order else 99, s.J))


# ============================================================================
# ONE STATE, ONE ISOTOPE
# ============================================================================

def _extrapolated(n, l_char, J, constants: Mapping[str, HfConstants],
                  quantum_defects: Mapping[str, float]):
    """(reference label, A, B, C) scaled from the first measured state with the
    same (l, J) as A ∝ n*⁻³, or None."""
    for ref_label, ref in constants.items():
        parsed = parse_state_label(ref_label)
        if parsed and parsed[1] == l_char and abs(parsed[2] - J) < 0.01 and ref.A_MHz != 0:
            delta_l = quantum_defects.get(l_char, 0.0)
            n_eff_ref = parsed[0] - delta_l
            n_eff_cur = n - delta_l
            if n_eff_ref > 0 and n_eff_cur > 0:
                scale = (n_eff_ref / n_eff_cur) ** 3
                return ref_label, ref.A_MHz * scale, ref.B_MHz * scale, ref.C_kHz * scale
            return None
    return None


def compute_state(label: str, nucleus: Nucleus, constants: Mapping[str, HfConstants],
                  quantum_defects: Mapping[str, float], B_G: np.ndarray) -> HyperfineState | None:
    """F levels, g factors and Breit-Rabi levels of one state; None for a bad label."""
    parsed = parse_state_label(label)
    if parsed is None:
        return None
    n, l_char, J = parsed
    I, gI = nucleus.I, nucleus.gI

    hfc = constants.get(label)
    gJ_exp = hfc.gJ_exp if hfc else None
    S = hfc.S_exp if hfc and hfc.S_exp is not None else 0.5   # electron spin
    gJ = lande_gJ(L_CHAR.get(l_char, 0), S, J, gJ_exp=gJ_exp)

    A_MHz, B_MHz, C_kHz = (hfc.A_MHz, hfc.B_MHz, hfc.C_kHz) if hfc else (0.0, 0.0, 0.0)
    source = None
    missing = hfc is None      # key presence, not value: a measured A of 0 is data
    if missing:
        found = _extrapolated(n, l_char, J, constants, quantum_defects)
        if found:
            source, A_MHz, B_MHz, C_kHz = found
            missing = False

    levels = [
        FLevel(F=F,
               E_MHz=hyperfine_energy(F, I, J, A_MHz, B_MHz, C_kHz),
               gF=gF_factor(F, I, J, gJ, gI),
               mF=[round((F-k)*2)/2 for k in range(round(2*F)+1)])
        for F in F_levels(I, J)
    ]
    BR_energies, BR_labels = breit_rabi_energies(I, J, A_MHz, B_MHz, gJ, gI, B_G, C_kHz)
    return HyperfineState(
        label=label, n=n, l=l_char, J=J,
        gJ=gJ,
        gJ_source='experimental' if gJ_exp is not None else 'Lande',
        A_MHz=A_MHz, B_MHz=B_MHz, C_kHz=C_kHz,
        extrapolated_from=source,
        constants_missing=missing,
        levels=levels,
        BR_energies=BR_energies,
        BR_labels=BR_labels,
    )


def compute(
    nucleus: Nucleus, constants: Mapping[str, HfConstants], state_labels: Sequence[str],
    quantum_defects: Mapping[str, float], *, B_max: float = 500.0, B_points: int = 500,
) -> HyperfineResult:
    """Every state's hyperfine structure on 0..B_max G, plus the ground state's clock and,
    with a p state in the run, the cycling line to its highest-J p state (the D2 line for
    J_gs = 1/2)."""
    B_G = np.linspace(0, B_max, B_points)
    states, skipped = [], []
    for label in state_labels:
        state = compute_state(label, nucleus, constants, quantum_defects, B_G)
        if state is None:
            skipped.append(label)
        else:
            states.append(state)
    clock = cycling = None
    if states:
        gs = ground_state(states)
        clock = microwave_clock(gs, nucleus.I)
        ex_candidates = [s for s in states if s.l == 'p']
        if ex_candidates and gs.l == 's':
            ex = max(ex_candidates, key=lambda s: s.J)
            Fs_gs = [lv.F for lv in gs.levels]
            pair = cycling_transition(allowed_transitions(Fs_gs, [lv.F for lv in ex.levels]), Fs_gs)
            if pair is not None:
                cycling = CyclingLine(lower=gs.label, upper=ex.label, F_lower=pair[0], F_upper=pair[1])
    return HyperfineResult(nucleus=nucleus, B_G=B_G, states=states, clock=clock,
                           cycling=cycling, skipped=skipped)
