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

from qali.core.constants import C_SI, MU_B_MHz_G, MU_N_MHz_G
from qali.physics.polarizability_core import L_CHAR

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
    BR_energies: np.ndarray         # (B point, level) Breit-Rabi eigenvalues, MHz; column = one level
    BR_labels: list[tuple[float, float]]   # (F, m_F) of each level at B = 0
    BR_high_field: list[tuple[float, float]] | None = None   # (m_J, m_I) of each level as B → ∞
    BR_slopes: np.ndarray | None = None     # dE/dB (MHz/G) on the same grid, exact (Hellmann–Feynman)
    zeeman: ZeemanSystem | None = None      # the Hamiltonian, for field-dependent lines


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
    microwave: list[MicrowaveLine] | None = None   # ground-state ΔF = ±1 lines vs B
    optical: list[OpticalLine] | None = None       # ground → each p state, Zeeman-resolved


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


G_S = 2.00231930436256     # free-electron |g| (CODATA 2018)
ELECTRON_MASS_U = 5.48579909065e-4


def lande_gJ(L, S, J, gJ_exp=None, mass_u=None):
    """
    Landé g_J with the free-electron spin g factor and the reduced-mass orbital g factor:

        g_J = g_L [J(J+1) − S(S+1) + L(L+1)] / [2J(J+1)] + g_S [J(J+1) + S(S+1) − L(L+1)] / [2J(J+1)],
        g_S = 2.00231930436 (CODATA 2018),  g_L = 1 − m_e / M_atom (1 when mass_u is not given).

    With g_S = 2 the ground state of every alkali would read g_J = 2 instead of ≈2.0023, a 0.1% error
    in every Zeeman slope. A measured value (gJ_exp) overrides the formula; it also carries the
    remaining many-body corrections (Cs 6s: 2.002542 against 2.002319). J = 0 returns 0.
    """
    if J < 1e-9:
        return 0.0
    if gJ_exp is not None:
        return float(gJ_exp)
    gL = 1.0 - (ELECTRON_MASS_U / mass_u if mass_u else 0.0)
    JJ, SS, LL = J*(J+1), S*(S+1), L*(L+1)
    return gL * (JJ - SS + LL) / (2*JJ) + G_S * (JJ + SS - LL) / (2*JJ)


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

class ZeemanSystem:
    """H_hf + H_Z of one state in the |m_I, m_J⟩ basis, solved per m_F block.

    m_F = m_I + m_J is exact at every field, so H is block-diagonal in m_F. Within a block levels
    only have avoided crossings, so the k-th lowest eigenvalue of a block is the same level at every
    B: each level keeps its identity, its B = 0 label |F, m_F⟩ and its high-field label |m_J, m_I⟩.
    (Sorting all eigenvalues together, as before, swapped levels at every crossing of different m_F:
    Na 3p3/2 has 22 such crossings below 100 G.)

    Level order (index k) is by zero-field energy, then m_F.
    """

    def __init__(self, I, J, A_MHz, B_MHz, gJ, gI, C_kHz=0.0):
        self.I, self.J = I, J
        dim_I, dim_J = round(2*I + 1), round(2*J + 1)
        self.dim = dim_I * dim_J
        mI = np.array([-I + i for i in range(dim_I)])
        mJ = np.array([-J + j for j in range(dim_J)])
        self.basis = [(float(mJ[j]), float(mI[i])) for i in range(dim_I) for j in range(dim_J)]   # (m_J, m_I)
        dim = self.dim
        IJ = np.zeros((dim, dim))
        for i, mi in enumerate(mI):
            for j, mj in enumerate(mJ):
                r = i * dim_J + j
                IJ[r, r] += mi * mj
                if i + 1 < dim_I and j - 1 >= 0:
                    r2 = (i + 1) * dim_J + (j - 1)
                    me = math.sqrt((I - mi)*(I + mi + 1)) * math.sqrt((J + mj)*(J - mj + 1))
                    IJ[r2, r] += 0.5 * me
                    IJ[r, r2] += 0.5 * me
        K = 2.0 * IJ
        self.H_hf = _hf_polynomial(K, I, J, A_MHz, B_MHz, C_kHz, np.eye(dim))
        # H_Z = (g_J μ_B m_J − g_I μ_N m_I) B, MHz per gauss
        self.Hz = np.array([gJ * MU_B_MHz_G * mj - gI * MU_N_MHz_G * mi for mj, mi in self.basis])
        mF = np.array([mj + mi for mj, mi in self.basis])
        F2 = (I*(I+1) + J*(J+1)) * np.eye(dim) + K
        Fs = F_levels(I, J)
        self.blocks = []               # (basis indices, mF)
        levels = []                    # (E0, mF, block, rank, F, mJ_hf, mI_hf)
        for m in sorted(set(np.round(2 * mF).astype(int) / 2)):
            idx = np.where(np.abs(mF - m) < 1e-9)[0]
            b = len(self.blocks)
            self.blocks.append((idx, float(m)))
            Hh = self.H_hf[np.ix_(idx, idx)]
            # F labels: a tiny <F^2> tie-breaker keeps F a good label even for degenerate levels (A = 0)
            e0, v0 = np.linalg.eigh(Hh + 1e-7 * F2[np.ix_(idx, idx)])
            Fv = [float(v0[:, k] @ F2[np.ix_(idx, idx)] @ v0[:, k]) for k in range(len(idx))]
            Flab = [min((F for F in Fs if F + 1e-9 >= abs(m)), key=lambda F_: abs(F_*(F_+1) - f2)) for f2 in Fv]
            # high-field labels: the k-th lowest level of a block ends on the k-th lowest |m_J, m_I⟩
            hf_order = sorted(range(len(idx)), key=lambda r: self.Hz[idx[r]] * 1e9 + Hh[r, r])
            for k in range(len(idx)):
                mj_hf, mi_hf = self.basis[idx[hf_order[k]]]
                levels.append((float(e0[k]), float(m), b, k, Flab[k], mj_hf, mi_hf))
        levels.sort(key=lambda t: (round(t[0], 9), t[1]))
        self.levels = levels
        self.labels = [(t[4], t[1]) for t in levels]          # (F, m_F) at B = 0
        self.high_field = [(t[5], t[6]) for t in levels]      # (m_J, m_I) as B → ∞
        self._pos = {(t[2], t[3]): n for n, t in enumerate(levels)}

    def solve(self, B_G, vectors=False):
        """Energies (MHz) of every level at field B (gauss), in level order; with vectors, also the
        |⟨m_J, m_I|level⟩|² composition rows (dim × dim, columns in basis order) and the eigenvectors."""
        E = np.empty(self.dim)
        V = np.zeros((self.dim, self.dim)) if vectors else None
        for b, (idx, _m) in enumerate(self.blocks):
            H = self.H_hf[np.ix_(idx, idx)] + np.diag(self.Hz[idx] * B_G)
            if vectors:
                e, v = np.linalg.eigh(H)
            else:
                e, v = np.linalg.eigvalsh(H), None
            for k in range(len(idx)):
                n = self._pos[(b, k)]
                E[n] = e[k]
                if vectors:
                    V[n, idx] = v[:, k]
        return (E, V) if vectors else E

    def energies(self, B_array):
        return np.array([self.solve(float(B)) for B in B_array])

    def slopes(self, B_G):
        """dE/dB (MHz/G) of every level at B, exact by Hellmann–Feynman: ⟨level|H_Z per gauss|level⟩.
        −dE/dB is the level's magnetic moment along the field; dE/dB > 0 is a low-field seeker."""
        _E, V = self.solve(float(B_G), vectors=True)
        return (V * V) @ self.Hz

    def energies_and_slopes(self, B_array):
        E = np.empty((len(B_array), self.dim))
        S = np.empty((len(B_array), self.dim))
        for k, B in enumerate(B_array):
            E[k], V = self.solve(float(B), vectors=True)
            S[k] = (V * V) @ self.Hz
        return E, S


def breit_rabi_energies(I, J, A_MHz, B_MHz_quad, gJ, gI,
                        B_gauss_array, C_kHz=0.0):
    """
    Exact Breit-Rabi levels of H = H_hf + (g_J μ_B J_z − g_I μ_N I_z) B on a field grid.

    H_hf is the exact hyperfine operator (dipole + quadrupole + octupole): the polynomial of
    `_hf_polynomial` applied to the operator K = 2 I·J in the |m_I, m_J⟩ basis, so at B = 0 the
    eigenvalues equal `hyperfine_energy` with (2F+1)-fold degeneracy. The levels are tracked per
    m_F block (see `ZeemanSystem`): column k is the same level at every B.

    Returns:
      energies[B_idx, level_idx]  — MHz, level order by zero-field energy then m_F
      labels[level_idx]           — (F, m_F) at B = 0
    """
    zs = ZeemanSystem(I, J, A_MHz, B_MHz_quad, gJ, gI, C_kHz)
    return zs.energies(B_gauss_array), zs.labels


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
    gJ = lande_gJ(L_CHAR.get(l_char, 0), S, J, gJ_exp=gJ_exp, mass_u=float(nucleus.A) or None)

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
    zs = ZeemanSystem(I, J, A_MHz, B_MHz, gJ, gI, C_kHz)
    BR_E, BR_S = zs.energies_and_slopes(B_G)
    return HyperfineState(
        label=label, n=n, l=l_char, J=J,
        gJ=gJ,
        gJ_source='experimental' if gJ_exp is not None else 'Lande',
        A_MHz=A_MHz, B_MHz=B_MHz, C_kHz=C_kHz,
        extrapolated_from=source,
        constants_missing=missing,
        levels=levels,
        BR_energies=BR_E,
        BR_labels=zs.labels,
        BR_high_field=zs.high_field,
        BR_slopes=BR_S,
        zeeman=zs,
    )


# ============================================================================
# FIELD-DEPENDENT LINES: GROUND MICROWAVE AND ZEEMAN-RESOLVED OPTICAL
# ============================================================================

@dataclass(frozen=True)
class FieldPoint:
    """A field where a line frequency is stationary (dν/dB = 0): a field-insensitive ("magic") point."""

    B_G: float
    nu_MHz: float
    curvature_Hz_per_G2: float      # ½ d²ν/dB² there


@dataclass(frozen=True)
class MicrowaveLine:
    """|F, m_F⟩ → |F', m_F'⟩ within the ground state, ΔF = ±1. kind: 'M1' for |Δm_F| ≤ 1 (single
    magnetic-dipole photon), 'two-photon' for |Δm_F| = 2 (e.g. the 87Rb |1,−1⟩ ↔ |2,+1⟩ qubit)."""

    state: str
    F_lower: float
    mF_lower: float
    F_upper: float
    mF_upper: float
    kind: str
    nu0_MHz: float                  # zero field
    linear_Hz_per_G: float          # dν/dB at B = 0
    quadratic_Hz_per_G2: float      # ½ d²ν/dB² at B = 0
    nu_MHz: np.ndarray              # on the result's field grid
    stationary: list[FieldPoint]


@dataclass(frozen=True)
class OpticalComponent:
    lower: int                      # level index in the ground state's level order
    upper: int                      # level index in the excited state's level order
    q: int                          # +1 σ+, 0 π, −1 σ− (absorption: m_F' = m_F + q)
    detuning_MHz: np.ndarray        # relative to the line's zero-field centroid, on B_G
    strength: np.ndarray            # (2J+1)|⟨e|d_q|g⟩|²/|⟨J'||d||J⟩|²: sums to 1 over each ground level


@dataclass(frozen=True)
class OpticalLine:
    lower: str
    upper: str
    B_G: np.ndarray
    components: list[OpticalComponent]
    strengths_F: dict[tuple[float, float], float]   # zero-field S_FF' (sum over F' = 1 for each F)


def _quadratic_fit(f, h):
    """(f0, slope, ½ curvature) at 0 from f(0), f(h), f(2h)."""
    f0, f1, f2 = f(0.0), f(h), f(2 * h)
    b = (f2 - 2 * f1 + f0) / (2 * h * h)
    return f0, (f1 - f0) / h - b * h, b


def microwave_lines(state: HyperfineState, B_G: np.ndarray, max_dmF: int = 2) -> list[MicrowaveLine]:
    """Every ΔF = ±1 line between the two highest F manifolds of *state* (the ground hyperfine
    doublet), |Δm_F| ≤ max_dmF, with its field dependence and its stationary (magic) fields."""
    from scipy.optimize import brentq

    zs = state.zeeman
    if zs is None:
        return []
    Fs = sorted({F for F, _ in zs.labels})
    if len(Fs) < 2:
        return []
    Fu, Fl = Fs[-1], Fs[-2]
    E = state.BR_energies
    lines = []
    for iu, (F1, m1) in enumerate(zs.labels):
        if abs(F1 - Fu) > 1e-9:
            continue
        for il, (F2, m2) in enumerate(zs.labels):
            if abs(F2 - Fl) > 1e-9 or abs(m1 - m2) > max_dmF + 1e-9:
                continue

            def nu(B, iu=iu, il=il):
                e = zs.solve(B)
                return e[iu] - e[il]

            def dnu(B, iu=iu, il=il):
                sl = zs.slopes(B)
                return sl[iu] - sl[il]

            curve = E[:, iu] - E[:, il]
            f0, _a, _b = _quadratic_fit(nu, 0.05)
            a = dnu(0.0)                                   # exact slope at B = 0
            h = 1e-3
            b = (dnu(h) - a) / (2 * h)                     # ½ d²ν/dB², from exact slopes
            stat = []
            dc = np.array([dnu(float(B)) for B in B_G])
            for k in range(1, len(B_G) - 1):
                if dc[k] == 0.0 or dc[k] * dc[k + 1] < 0:
                    Bs = float(brentq(dnu, B_G[k], B_G[k + 1], xtol=1e-9)) if dc[k] != 0.0 else float(B_G[k])
                    curv = (dnu(Bs + h) - dnu(max(Bs - h, 0.0))) / (2 * h) / 2
                    stat.append(FieldPoint(Bs, float(nu(Bs)), float(curv * 1e6)))
            lines.append(MicrowaveLine(
                state=state.label, F_lower=Fl, mF_lower=m2, F_upper=Fu, mF_upper=m1,
                kind='M1' if abs(m1 - m2) <= 1 + 1e-9 else 'two-photon',
                nu0_MHz=float(f0), linear_Hz_per_G=float(a * 1e6), quadratic_Hz_per_G2=float(b * 1e6),
                nu_MHz=curve, stationary=stat))
    return lines


def hyperfine_line_strengths(I, J, Jp) -> dict[tuple[float, float], float]:
    """Zero-field relative strengths S_FF' = (2F'+1)(2J+1){J J' 1; F' F I}² of the hyperfine
    components of a J → J' line (Steck's convention: sum over F' is 1 for every F)."""
    from qali.physics.atomic_structure.angular import sixj2
    out = {}
    for F in F_levels(I, J):
        for Fp in F_levels(I, Jp):
            s6 = sixj2(round(2 * J), round(2 * Jp), 2, round(2 * Fp), round(2 * F), round(2 * I))
            v = (2 * Fp + 1) * (2 * J + 1) * s6 * s6
            if v > 1e-14:
                out[(F, Fp)] = v
    return out


def optical_line(ground: HyperfineState, excited: HyperfineState, B_G: np.ndarray,
                 min_strength: float = 1e-6) -> OpticalLine | None:
    """Zeeman-resolved E1 components ground → excited on a field grid. The nuclear spin is a spectator:
    ⟨J' m_J' m_I'|d_q|J m_J m_I⟩ = δ_{m_I m_I'} (−1)^{J'−m_J'} (J' 1 J; −m_J' q m_J) ⟨J'||d||J⟩."""
    from qali.physics.atomic_structure.angular import threej2
    zg, ze = ground.zeeman, excited.zeeman
    if zg is None or ze is None or abs(abs(L_CHAR.get(ground.l, 0) - L_CHAR.get(excited.l, 0)) - 1) > 0:
        return None
    J, Jp = ground.J, excited.J
    if abs(J - Jp) > 1 + 1e-9:
        return None
    Dq = {}
    for q in (-1, 0, 1):
        D = np.zeros((ze.dim, zg.dim))
        for a, (mjp, mip) in enumerate(ze.basis):
            for b, (mj, mi) in enumerate(zg.basis):
                if abs(mip - mi) > 1e-9:
                    continue
                ph = -1.0 if round(Jp - mjp) % 2 else 1.0
                D[a, b] = ph * threej2(round(2 * Jp), 2, round(2 * J), round(-2 * mjp), 2 * q, round(2 * mj))
        Dq[q] = D
    nB = len(B_G)
    det = {}
    st = {}
    for k, B in enumerate(B_G):
        Eg, Vg = zg.solve(float(B), vectors=True)
        Ee, Ve = ze.solve(float(B), vectors=True)
        for q, D in Dq.items():
            amp = Ve @ D @ Vg.T                                   # (excited level, ground level)
            r = (2 * J + 1) * amp * amp
            for e_i, g_i in zip(*np.nonzero(r > min_strength)):
                key = (int(g_i), int(e_i), q)
                if key not in st:
                    st[key] = np.zeros(nB)
                    det[key] = np.full(nB, np.nan)
                st[key][k] = r[e_i, g_i]
        for key in st:
            g_i, e_i, _q = key
            det[key][k] = Ee[e_i] - Eg[g_i]
    comps = []
    for key in sorted(st):
        g_i, e_i, q = key
        # fill detunings at fields where the component was below threshold (keeps the curve continuous)
        if np.isnan(det[key]).any():
            for k, B in enumerate(B_G):
                if np.isnan(det[key][k]):
                    det[key][k] = ze.solve(float(B))[e_i] - zg.solve(float(B))[g_i]
        comps.append(OpticalComponent(g_i, e_i, q, det[key], st[key]))
    return OpticalLine(ground.label, excited.label, np.asarray(B_G, float), comps,
                       hyperfine_line_strengths(zg.I, J, Jp))


def compute(
    nucleus: Nucleus, constants: Mapping[str, HfConstants], state_labels: Sequence[str],
    quantum_defects: Mapping[str, float], *, B_max: float = 500.0, B_points: int = 500,
    lines: bool = False, optical_points: int = 121,
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
    microwave = optical = None
    if lines and states:
        gs = ground_state(states)
        microwave = microwave_lines(gs, B_G)
        B_opt = np.linspace(0, B_max, max(2, min(B_points, optical_points)))
        optical = [ol for ex in states if ex is not gs
                   for ol in [optical_line(gs, ex, B_opt)] if ol is not None]
    return HyperfineResult(nucleus=nucleus, B_G=B_G, states=states, clock=clock,
                           cycling=cycling, skipped=skipped, microwave=microwave, optical=optical)
