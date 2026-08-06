"""
constants.py — shared CODATA / SI / atomic-unit constants for the alkali pipeline
--------------------------------------------------------------------------------
Single source of truth for physical constants used across modules.
Element-specific physics tables that are not universal constants
(CORE_CONFIG, PRECISION_FOSC_J, ACC_UNCERTAINTY, L_CHAR) stay in alpha_core.

Usage:
    from constants import HC_EV_NM, HARTREE_TO_EV, ATOMIC_MASS_AMU, ...
"""

__version__ = '1.0'

import math

# ============================================================================
# FUNDAMENTAL (CODATA 2018 / exact SI)
# ============================================================================

C_SI            = 2.99792458e8          # m/s  (exact)
H_SI            = 6.62607015e-34        # J·s  (exact)
HBAR_SI         = 1.054571817e-34       # J·s  (ħ = h/2π)
KB_SI           = 1.380649e-23          # J/K  (exact)
E_CHARGE_C      = 1.602176634e-19       # C    (exact) — also J/eV
EV_TO_J         = E_CHARGE_C            # alias
AMU_KG          = 1.66053906660e-27     # kg
A0_SI           = 5.29177210903e-11     # m    Bohr radius
EPS0_SI         = 8.8541878128e-12      # F/m

# Aliases used in various modules
C_M_S           = C_SI
H_J_S           = H_SI
KB_J_K          = KB_SI
A0_M            = A0_SI
K_B             = KB_SI

# ============================================================================
# ENERGY / WAVELENGTH CONVERSIONS
# ============================================================================

HARTREE_TO_EV   = 27.211386245988       # E_h / e  (CODATA)
EV_TO_HARTREE   = 1.0 / HARTREE_TO_EV
HC_EV_NM        = 1239.841984           # eV·nm   (hc)
CM1_TO_EV       = HC_EV_NM / 1e7        # eV per cm⁻¹  (= 1.239841984e-4)
# ω_au = NM_TO_AU_OMEGA / λ_nm   (angular frequency in a.u.)
NM_TO_AU_OMEGA  = HC_EV_NM / HARTREE_TO_EV   # ≈ 45.56335

HBAR_EV_S       = 6.582119569e-16       # eV·s
KB_EV           = 8.617333262e-5        # eV/K

HARTREE_TO_HZ   = 6.579683920711515e15  # E_h / h  (Hz)
ATOMIC_FREQ_RAD_S = 4.134137333518e16   # E_h / ħ  (rad/s per a.u. of ω)
AU_TIME_S       = 2.4188843265857e-17   # ħ / E_h  (s)

# ============================================================================
# ATOMIC-UNIT → SI
# ============================================================================

# α_SI = α_au · 4πε₀ a₀³
AU_TO_SI_ALPHA  = 4.0 * math.pi * EPS0_SI * A0_SI**3
# Rounded compact form still used in some displays / docs
AU_TO_SI_ALPHA_ROUNDED = 1.6488e-41
# C₆_SI = C₆_au · E_h a₀⁶
AU_TO_SI_C6     = 9.573448e-80
AU_TO_ANGSTROM3 = (0.529177)**3

# ============================================================================
# SPECTROSCOPY
# ============================================================================

# A_ul = EINSTEIN_PREFACTOR / λ_nm² · (g_l/g_u) · f_lu
EINSTEIN_PREFACTOR = 6.6703e13          # s⁻¹·nm²
NIST_PREF          = EINSTEIN_PREFACTOR  # alias (alpha_core / ASD loader)

# ============================================================================
# MAGNETISM (hyperfine / Zeeman)
# ============================================================================

MU_B_MHz_G  = 1.399624604               # MHz/G  (μ_B / h)
MU_N_MHz_G  = 7.622593285e-4            # MHz/G  (μ_N / h)
MU_B_HZ_T   = 9.2740100783e9            # Hz/T

# ============================================================================
# BBR / THERMAL FIELD (Itano, Lewis & Wineland 1982)
# ============================================================================

E_FIELD_REF_V_M = 831.9                 # V/m RMS at T_REF
E_FIELD_REF_T_K = 300.0                 # K
EATOMIC_V_M     = 5.14220674763e11      # V/m  E_h/(e a₀)

# ============================================================================
# FESHBACH / SCATTERING
# ============================================================================

A_BAR_FACTOR = 0.955978                 # ā / β₆ (Gribakin–Flambaum)

# ============================================================================
# MASSES
# ============================================================================

# Elemental / IUPAC standard atomic weights (amu) — for Doppler, tweezers, etc.
ATOMIC_MASS_AMU = {
    'H':   1.008,   'He':  4.003,
    'Li':  6.941,   'Be':  9.012,   'B':  10.811,  'C':  12.011,
    'N':  14.007,   'O':  15.999,   'F':  18.998,  'Ne': 20.180,
    'Na': 22.990,   'Mg': 24.305,   'Al': 26.982,  'Si': 28.086,
    'P':  30.974,   'S':  32.060,   'Cl': 35.450,  'Ar': 39.948,
    'K':  39.098,   'Ca': 40.078,   'Sc': 44.956,  'Ti': 47.867,
    'V':  50.942,   'Cr': 51.996,   'Mn': 54.938,  'Fe': 55.845,
    'Co': 58.933,   'Ni': 58.693,   'Cu': 63.546,  'Zn': 65.380,
    'Ga': 69.723,   'Ge': 72.630,   'As': 74.922,  'Se': 78.971,
    'Br': 79.904,   'Kr': 83.798,
    'Rb': 85.468,   'Sr': 87.620,   'Y':  88.906,  'Zr': 91.224,
    'Nb': 92.906,   'Mo': 95.951,   'Tc': 98.000,  'Ru':101.070,
    'Rh':102.906,   'Pd':106.420,   'Ag':107.868,  'Cd':112.414,
    'In':114.818,   'Sn':118.710,   'Sb':121.760,  'Te':127.600,
    'I': 126.904,   'Xe':131.293,
    'Cs':132.905,   'Ba':137.327,   'La':138.905,  'Ce':140.116,
    'Pr':140.908,   'Nd':144.242,   'Pm':145.000,  'Sm':150.360,
    'Eu':151.964,   'Gd':157.250,   'Tb':158.925,  'Dy':162.500,
    'Ho':164.930,   'Er':167.259,   'Tm':168.934,  'Yb':173.045,
    'Lu':174.967,   'Hf':178.490,   'Ta':180.948,  'W': 183.840,
    'Re':186.207,   'Os':190.230,   'Ir':192.217,  'Pt':195.084,
    'Au':196.967,   'Hg':200.592,   'Tl':204.383,  'Pb':207.200,
    'Bi':208.980,   'Po':209.000,   'At':210.000,  'Rn':222.000,
    'Fr':223.000,   'Ra':226.025,   'Ac':227.000,  'Th':232.038,
    'Pa':231.036,   'U': 238.029,   'Np':237.000,  'Pu':244.000,
    'Am':243.000,   'Cm':247.000,   'Bk':247.000,  'Cf':251.000,
    'Es':252.000,   'Fm':257.000,   'Md':258.000,  'No':259.000,
    'Lr':262.000,   'Rf':267.000,   'Db':268.000,  'Sg':269.000,
    'Bh':270.000,   'Hs':269.000,   'Mt':278.000,  'Ds':281.000,
    'Rg':282.000,   'Cn':285.000,   'Nh':286.000,  'Fl':289.000,
    'Mc':290.000,   'Lv':293.000,   'Ts':294.000,  'Og':294.000,
}

# Precise isotope masses (amu) for Feshbach / isotope-resolved work
ISOTOPE_MASS_AMU = {
    ('Li', 6): 6.015122, ('Li', 7): 7.016004,
    ('Na', 23): 22.989769,
    ('K', 39): 38.963707, ('K', 40): 39.963998, ('K', 41): 40.961826,
    ('Rb', 85): 84.911790, ('Rb', 87): 86.909180,
    ('Cs', 133): 132.905452,
    ('Fr', 210): 209.996408, ('Fr', 223): 223.019736,
}

# Alkali isotope-preferred elemental fallback (most common lab isotope)
ALKALI_MASS_AMU = {
    'Li': 7.016004, 'Na': 22.989769, 'K': 38.963707,
    'Rb': 86.909180, 'Cs': 132.905452, 'Fr': 223.019736,
}

# ============================================================================
# EXPERIMENTAL REFERENCE TABLES (shared)
# ============================================================================

ALPHA0_EXPT = {
    'H':  4.500,  'He': 1.384,
    'Li': 164.0,  'Be': 37.74,  'B':  20.5,   'C':  11.67,
    'N':  7.54,   'O':  5.41,   'F':  3.76,   'Ne': 2.67,
    'Na': 162.7,  'Mg': 71.3,   'Al': 57.8,   'Si': 37.3,
    'P':  25.0,   'S':  19.6,   'Cl': 14.7,   'Ar': 11.08,
    'K':  290.6,  'Ca': 160.8,  'Rb': 318.8,  'Sr': 197.2,
    'Cs': 401.0,  'Ba': 268.0,  'Fr': 317.8,
}

C6_EXPT = {
    'Li': 1393.0,  'Na': 1556.0,  'K':  3897.0,
    'Rb': 4691.0,  'Cs': 6851.0,  'Fr': 5500.0,
}


# ============================================================================
# DISPLAY HELPERS
# ============================================================================

_SUPERSCRIPTS = str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹')


def nuclide_label(element, A=None):
    """Pretty nuclide string for plot titles, e.g. ('Li', 6) → '⁶Li'."""
    if A is None:
        return element
    return f'{str(int(A)).translate(_SUPERSCRIPTS)}{element}'


def nuclide_plain(element, A=None):
    """ASCII nuclide string, e.g. ('Li', 6) → 'Li-6'."""
    if A is None:
        return element
    return f'{element}-{int(A)}'
