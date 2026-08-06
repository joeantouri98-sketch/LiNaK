#!/usr/bin/env python3
"""
hyperfine.py  —  Hyperfine structure, Zeeman & Breit-Rabi for any atom
-----------------------------------------------------------------------
Computes for each atomic state (n, l, J, isotope):
  - Hyperfine F-level energies (magnetic dipole + electric quadrupole)
  - Weak-field Zeeman splitting (linear regime, all m_F sublevels)
  - Full Breit-Rabi diagram (exact diagonalisation, valid at any B field)
  - g_F factors, allowed transitions, clock/cycling transition identification

Physics:
  H_hf = A_hf·(I·J) + B_hf·[3(I·J)²+3(I·J)/2−I²J²] / [2I(2I−1)·2J(2J−1)]
         + C_hf · octupole operator  (optional, kHz level)
  E(F) = (A/2)·K + B·[3K(K+1)/4−I(I+1)J(J+1)] / [2I(2I−1)·2J(2J−1)]
         + C_kHz·1e-3 · [5K³+...] / denom_oct
         K = F(F+1)−I(I+1)−J(J+1)
  Breit-Rabi: exact diagonalisation of H_hf + g_J·μ_B·B·J_z − g_I·μ_N·B·I_z
  g_J: Landé formula (default) or experimental override via gJ_exp field

Data sources (no hardcoding):
  Nuclear data:  data_json/nuclear_data.json        (built from IAEA CSVs)
  HF constants:  data_json/<element>_hf_constants.json  (user provides / auto-fetch)

Usage:
  python hyperfine.py Na
  python hyperfine.py Na --isotope 23 --states 3s1/2 3p1/2 3p3/2
  python hyperfine.py Na --build-nuclear-data  (rebuild nuclear_data.json from CSVs)
  python hyperfine.py Na --B-max 1000          (Breit-Rabi up to 1000 Gauss)
  python hyperfine.py --all                    (run all elements with hf_constants)
"""

__version__ = '1.1'   # + octupole C term in hyperfine_energy (C_kHz field)
                      # + experimental gJ override (gJ_exp field in hf_constants)
                      # + works for non-alkali atoms with supplied gJ_exp

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("Plotly not installed. Run: pip install plotly")
    sys.exit(1)

from constants import (
    MU_B_MHz_G,
    MU_N_MHz_G,
    MU_B_HZ_T,
    HC_EV_NM,
    HBAR_EV_S,
    nuclide_label,
    nuclide_plain,
)

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(description='Hyperfine structure and Zeeman diagram')
parser.add_argument('element', nargs='?', default=None)
parser.add_argument('--isotope',   type=int,   default=None,
                    help='Mass number A (default: most abundant stable isotope)')
parser.add_argument('--states',    nargs='+',  default=None,
                    help='States to include e.g. 3s1/2 3p1/2 3p3/2')
parser.add_argument('--B-max',     type=float, default=500.0,
                    help='Max B field for Breit-Rabi diagram in Gauss (default: 500)')
parser.add_argument('--B-points',  type=int,   default=500,
                    help='Number of B-field points (default: 500)')
parser.add_argument('--data-dir',  default='data_json')
parser.add_argument('--output-dir',default=None,
                    help='Output directory (default: plots/<element>)')
parser.add_argument('--all',       action='store_true',
                    help='Run all elements with available hf_constants')
parser.add_argument('--build-nuclear-data', action='store_true',
                    help='Rebuild nuclear_data.json from IAEA CSV files')
parser.add_argument('--magn-csv',  default='Moments/magn_mom_recomm.csv')
parser.add_argument('--elec-csv',  default='Moments/elec_mom_recomm.csv')
args = parser.parse_args()

DATA_DIR = args.data_dir
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================================
# STEP 1 — BUILD nuclear_data.json FROM IAEA CSVs  (if requested or missing)
# ============================================================================

def build_nuclear_data(magn_csv, elec_csv, out_path):
    """Parse IAEA magnetic + quadrupole CSV files into nuclear_data.json."""
    import csv

    def parse_spin(s):
        s = s.strip().lstrip('(').rstrip(')').rstrip('+-')
        if '/' in s:
            try:
                n, d = s.split('/')
                return int(n) / int(d)
            except:
                return None
        try:
            v = float(s)
            return v if v >= 0 else None
        except:
            return None

    def parse_float(s):
        s = str(s).strip()
        s = re.sub(r'^\([\+\-]\)', '', s)
        s = re.sub(r'\(\d+\)', '', s)
        s = re.sub(r'\s', '', s)
        try:
            return float(s)
        except:
            return None

    def halflife_s(s):
        s = s.strip().lower()
        if s in ('stable', ''):
            return float('inf')
        m = re.match(r'([\d.]+(?:\s*x\s*10\^?\d+)?)\s*(s|m|h|d|y|ms|us|ns|ps|fs)?', s)
        if not m:
            return 0.0
        num_s = re.sub(r'\s*x\s*10\^?', 'e', m.group(1))
        try:
            num = float(num_s)
        except:
            num = 0.0
        unit = (m.group(2) or 's').strip()
        mult = {'fs':1e-15,'ps':1e-12,'ns':1e-9,'us':1e-6,'ms':1e-3,
                's':1,'m':60,'h':3600,'d':86400,'y':3.156e7}.get(unit, 1)
        return num * mult

    magn = {}
    with open(magn_csv, encoding='utf-8', errors='ignore') as f:
        for r in csv.DictReader(f):
            try:
                z  = int(r['z'])
                A  = int(r['n.n+n.z'].strip())
                if float(r['energy [keV]'].strip() or '0') != 0:
                    continue
                I  = parse_spin(r['spin'])
                mu = parse_float(r['magnetic dipole [nm]'])
                hl = halflife_s(r['halflife'])
                if I is None or I == 0:
                    continue
                key = (z, A)
                if key not in magn:
                    magn[key] = {
                        'Z': z, 'A': A,
                        'symbol': r['symbol'].strip(),
                        'I': I, 'mu_I': mu,
                        'halflife_s': hl if math.isfinite(hl) else -1,
                        'halflife_str': r['halflife'].strip(),
                        'stable': r['halflife'].strip().lower() == 'stable',
                    }
            except:
                pass

    quad = {}
    with open(elec_csv, encoding='utf-8', errors='ignore') as f:
        for r in csv.DictReader(f):
            try:
                z = int(r['z'])
                A = int(r['n.n+n.z'].strip())
                if float(r['energy [keV]'].strip() or '0') != 0:
                    continue
                Q = parse_float(r['electric quadrupole [b]'])
                if Q is not None:
                    quad[(z, A)] = Q
            except:
                pass

    isotopes = []
    for (z, A), d in sorted(magn.items()):
        d['Q_barn'] = quad.get((z, A), 0.0) or 0.0
        I  = d['I']
        mu = d['mu_I']
        d['g_I'] = round(mu / I, 7) if (mu is not None and I > 0) else None
        isotopes.append(d)

    out = {
        'source': 'IAEA Nuclear Moments — Stone (2025 update)',
        'files':  [magn_csv, elec_csv],
        'n_isotopes': len(isotopes),
        'isotopes': isotopes,
    }
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"✓ Built {out_path}: {len(isotopes)} isotopes ({sum(d['stable'] for d in isotopes)} stable)")
    return out


NUCLEAR_DATA_PATH = os.path.join(DATA_DIR, 'nuclear_data.json')

if args.build_nuclear_data:
    build_nuclear_data(args.magn_csv, args.elec_csv, NUCLEAR_DATA_PATH)
    if args.element is None:
        sys.exit(0)

if not os.path.exists(NUCLEAR_DATA_PATH):
    print(f"nuclear_data.json not found at {NUCLEAR_DATA_PATH}.")
    print(f"Run: python hyperfine.py --build-nuclear-data "
          f"--magn-csv {args.magn_csv} --elec-csv {args.elec_csv}")
    sys.exit(1)

with open(NUCLEAR_DATA_PATH) as f:
    _nd = json.load(f)

# Build lookup: symbol -> list of isotope dicts, sorted by A
NUCLEAR = defaultdict(list)
for iso in _nd['isotopes']:
    NUCLEAR[iso['symbol']].append(iso)
for sym in NUCLEAR:
    NUCLEAR[sym].sort(key=lambda x: x['A'])

# ============================================================================
# STEP 2 — LOAD / FETCH  <element>_hf_constants.json
# ============================================================================

def fetch_hf_constants_nist(element, out_path):
    """
    Attempt to scrape A_hf and B_hf from NIST ASD energy levels page.
    Falls back gracefully if network is unavailable.
    Saves whatever it finds to out_path as a starting template.
    """
    print(f"  Attempting NIST ASD fetch for {element}...")
    try:
        import urllib.request, html
        url = (
            f"https://physics.nist.gov/cgi-bin/ASD/energy1.pl?"
            f"spectrum={element}+I&units=1&format=2&output=0&"
            f"page_size=15&multiplet_ordered=0&conf_out=on&term_out=on&"
            f"level_out=on&unc_out=1&j_out=on&lande_out=on&perc_out=on&"
            f"biblio=on&hf_out=on&submit=Retrieve+Data"
        )
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            content = resp.read().decode('utf-8', errors='ignore')

        # Parse tab-separated lines for hf splitting
        hf = {}
        for line in content.split('\n'):
            if '\t' not in line:
                continue
            cols = [c.strip().strip('"') for c in line.split('\t')]
            if len(cols) < 5:
                continue
            # Look for lines with config + J + level energy
            # NIST ASD tab format varies; extract what we can
            # This is a best-effort parse — the user should verify
            pass

        print(f"  NIST ASD fetched but hf parsing requires manual verification.")
        print(f"  Creating template file at {out_path}")

    except Exception as e:
        print(f"  NIST fetch failed ({e}). Creating blank template.")

    # Write a template the user fills in
    template = {
        "_source": "Fill from Steck alkali data sheets (steck.us/alkalidata) or NIST ASD",
        "_format": {"A_MHz": "hyperfine A constant (MHz)", "B_MHz": "hyperfine B constant (MHz, 0 for J=1/2)"},
        "_example_Na23": {
            "3s1/2": {"A_MHz": 885.813064, "B_MHz": 0.0},
            "3p1/2": {"A_MHz":  94.44,     "B_MHz": 0.0},
            "3p3/2": {"A_MHz":  18.534,    "B_MHz": 2.724}
        }
    }
    with open(out_path, 'w') as f:
        json.dump(template, f, indent=2)
    print(f"  Template saved to {out_path}. Please fill in A_MHz and B_MHz values.")
    return None


def load_hf_constants(element, data_dir):
    """
    Load hyperfine constants from data_json/<element>_hf_constants.json.
    Returns dict: state_label -> {A_MHz, B_MHz} or None if file missing.
    """
    path = os.path.join(data_dir, f"{element}_hf_constants.json")
    if not os.path.exists(path):
        print(f"\n  No hf_constants file found for {element}.")
        fetch_hf_constants_nist(element, path)
        return None

    with open(path) as f:
        raw = json.load(f)

    # Filter out metadata keys (starting with '_')
    hf = {}
    for key, val in raw.items():
        if key.startswith('_'):
            continue
        if isinstance(val, dict) and 'A_MHz' in val:
            hf[key] = {
                'A_MHz':  float(val['A_MHz']),
                'B_MHz':  float(val.get('B_MHz',  0.0)),
                'C_kHz':  float(val.get('C_kHz',  0.0)),   # magnetic octupole
                'gJ_exp': float(val['gJ_exp']) if 'gJ_exp' in val else None,
                'S_exp':  float(val['S_exp'])  if 'S_exp'  in val else None,
            }
    return hf if hf else None

# ============================================================================
# STEP 3 — NUCLEAR DATA HELPERS
# ============================================================================

def get_isotope(element, A=None):
    """
    Return the isotope dict for (element, A).
    If A is None: return the most abundant stable isotope
    (or longest-lived if no stable isotope exists).
    """
    isos = NUCLEAR.get(element, [])
    if not isos:
        return None

    if A is not None:
        for iso in isos:
            if iso['A'] == A:
                return iso
        print(f"  Warning: {element}-{A} not found in nuclear_data.json")
        return None

    # Priority: stable isotope with highest abundance (proxy: smallest A among stable)
    stable = [i for i in isos if i['stable']]
    if stable:
        # For alkalis the most common stable isotope is usually the lightest stable one
        # Use the one with the most common isotope conventions:
        # Li-7 (92.5%), Na-23 (100%), K-39 (93.3%), Rb-87 (28%) or Rb-85 (72%)
        # Cs-133 (100%), Fr: none stable
        # Heuristic: pick the stable isotope with highest A (often most abundant for alkalis)
        # This is imperfect — user can override with --isotope
        return stable[-1]

    # No stable: return longest-lived
    return max(isos, key=lambda i: i['halflife_s'])


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
        l_int = {'s':0,'p':1,'d':2,'f':3,'g':4}[l_char]
        J = l_int + 0.5   # default to J = l + 1/2 for doublet
    return n, l_char, J

# ============================================================================
# STEP 4 — HYPERFINE PHYSICS
# ============================================================================

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


def hyperfine_energy(F, I, J, A_MHz, B_MHz=0.0, C_kHz=0.0):
    """
    Hyperfine energy of state |I, J, F⟩ in MHz relative to the centroid.

    Includes three multipole terms:
      Magnetic dipole (A):
        E_A = (A/2) · K
      Electric quadrupole (B):
        E_B = B · [3K(K+1)/4 − I(I+1)J(J+1)] / [2I(2I−1)·2J(2J−1)]
      Magnetic octupole (C):
        E_C = C · K(K+1)(K+2) · [5K − 6I(I+1) − 6J(J+1) + ... ] / denom_oct
        Simplified:  E_C = C · [5K³ + 3K²(1−2I(I+1)−2J(J+1)) + ...] / denom_oct
      where K = F(F+1) − I(I+1) − J(J+1)

    A in MHz, B in MHz, C in kHz (converted internally to MHz).
    B term requires I >= 1 AND J >= 1.
    C term requires I >= 3/2 AND J >= 3/2 (else identically zero).

    Reference for octupole formula:
      Armstrong (1971) Adv. At. Mol. Phys. 7:149
      Woodgate (1966) Proc. Roy. Soc. A293:117
    """
    K = F*(F+1) - I*(I+1) - J*(J+1)
    E = 0.5 * A_MHz * K

    # Quadrupole term — requires I >= 1 AND J >= 1
    if B_MHz != 0.0 and I >= 1.0 - 1e-9 and J >= 1.0 - 1e-9:
        denom_IJ = (2*I - 1) * (2*I) * (2*J - 1) * (2*J)
        if abs(denom_IJ) > 1e-12:
            E += B_MHz * (1.5*K*(K+1) - 2*I*(I+1)*J*(J+1)) / denom_IJ

    # Octupole term — requires I >= 3/2 AND J >= 3/2
    # Uses the Schwartz (1955) / Armstrong (1971) denominator:
    #   (2I−1)(2I)(2I+1)(2I+2) × (2J−1)(2J)(2J+1)(2J+2)
    # which is non-zero for I >= 3/2, J >= 3/2 (unlike the naive 2I-3 form).
    # Typical magnitude: sub-kHz shifts, informational for all but optical clocks.
    if C_kHz != 0.0 and I >= 1.5 - 1e-9 and J >= 1.5 - 1e-9:
        C_MHz = C_kHz * 1e-3
        denom_oct = (
            (2*I - 1) * (2*I) * (2*I + 1) * (2*I + 2) *
            (2*J - 1) * (2*J) * (2*J + 1) * (2*J + 2)
        )
        if abs(denom_oct) > 1e-12:
            II  = I * (I + 1)
            JJ  = J * (J + 1)
            # Armstrong (1971) Adv. At. Mol. Phys. 7:149, eq. (3.5)
            E_oct = (10*K*(K+1)*(K+2)
                     - 12*(K+1)*II*JJ
                     + 3*(K+1)*(II + JJ))
            E += C_MHz * E_oct / denom_oct
    return E


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


def zeeman_energy_weak(F, mF, E_hf_MHz, gF):
    """
    Weak-field Zeeman energy: E = E_hf + g_F·μ_B·B·m_F  [MHz, B in Gauss]
    """
    # g_F·μ_B·B·m_F = g_F · 1.399625 MHz/G · B · m_F
    # Returned as a function of B: caller multiplies by B_gauss
    return E_hf_MHz, gF * MU_B_MHz_G * mF   # (offset, slope_per_gauss)


# ============================================================================
# STEP 5 — BREIT-RABI EXACT DIAGONALISATION
# ============================================================================

def breit_rabi_energies(I, J, A_MHz, B_MHz_quad, gJ, gI,
                        B_gauss_array):
    """
    Exact Breit-Rabi diagonalisation for arbitrary B field.

    Builds the full (2I+1)(2J+1) × (2I+1)(2J+1) Hamiltonian matrix
    H = H_hf + H_Z   at each B value and diagonalises it.

    H_hf couples all |m_I, m_J⟩ states.
    H_Z  = (g_J·μ_B·m_J − g_I·μ_N·m_I)·B   (diagonal in |m_I, m_J⟩ basis)

    Basis: |m_I, m_J⟩ ordered as:
      m_I = −I, −I+1, ..., +I
      m_J = −J, −J+1, ..., +J
    Combined index: idx = (m_I_index)*(2J+1) + m_J_index

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

    # ── Build H_hf  (I·J operator in |m_I, m_J⟩ basis) ─────────────────────
    # I·J = I_z·J_z + (1/2)(I_+·J_- + I_-·J_+)
    # Matrix elements:
    #   ⟨m_I, m_J|I_z·J_z|m_I, m_J⟩ = m_I·m_J
    #   ⟨m_I+1, m_J-1|I_+·J_-|m_I, m_J⟩ = sqrt[(I-m_I)(I+m_I+1)(J+m_J)(J-m_J+1)]

    H_hf = np.zeros((dim, dim))

    # Diagonal: A/2 * K where K = F(F+1)-I(I+1)-J(J+1)
    # But in |m_I,m_J⟩ basis H_hf is not diagonal.
    # Build it directly from I·J:
    for mi_i, mI in enumerate(mI_vals):
        for mj_i, mJ in enumerate(mJ_vals):
            r = idx(mi_i, mj_i)
            # I_z J_z term
            H_hf[r, r] += A_MHz * mI * mJ

            # Quadrupole B term — diagonal in |m_I, m_J⟩ basis:
            # 3(I_z)²·(J_z)² - I²J²  ...  complex in this basis, use full tensor
            # For simplicity include leading quadrupole correction diagonally:
            # B·[3(m_I·m_J)² + 3(m_I·m_J)/2 - I(I+1)J(J+1)] / denom
            if B_MHz_quad != 0.0 and I >= 1 and J >= 1:
                denom = (2*I-1)*2*I*(2*J-1)*2*J
                if abs(denom) > 1e-12:
                    u = mI * mJ
                    H_hf[r, r] += B_MHz_quad * (3*u*u + 1.5*u - I*(I+1)*J*(J+1)) / denom

            # I_+ J_- term: ⟨m_I+1, m_J-1|...|m_I, m_J⟩
            if mi_i + 1 < dim_I and mj_i - 1 >= 0:
                mI_p = mI_vals[mi_i + 1]
                mJ_m = mJ_vals[mj_i - 1]
                r2   = idx(mi_i+1, mj_i-1)
                me   = (math.sqrt((I - mI)*(I + mI + 1)) *
                        math.sqrt((J + mJ)*(J - mJ + 1)))
                H_hf[r2, r] += 0.5 * A_MHz * me
                H_hf[r, r2] += 0.5 * A_MHz * me   # Hermitian

    # ── Compute H_Z diagonal at each B ──────────────────────────────────────
    # H_Z[r] = (gJ·μ_B·mJ − gI·μ_N·mI) · B
    H_Z_diag = np.array([
        gJ * MU_B_MHz_G * mJ_vals[mj_i]
        - gI * MU_N_MHz_G * mI_vals[mi_i]
        for mi_i in range(dim_I)
        for mj_i in range(dim_J)
    ])

    # ── Build |F, m_F⟩ basis ordering at B=0 for labelling ─────────────────
    # Eigenvectors of H_hf at B=0 → identify (F, m_F) quantum numbers
    evals0, evecs0 = np.linalg.eigh(H_hf)
    # For each eigenvector find the dominant |m_I, m_J⟩ and infer F, m_F
    labels = []
    for k in range(dim):
        v      = evecs0[:, k]
        dom_r  = int(np.argmax(np.abs(v)**2))
        mi_i   = dom_r // dim_J
        mj_i   = dom_r %  dim_J
        mI     = mI_vals[mi_i]
        mJ     = mJ_vals[mj_i]
        mF_approx = round(2*(mI + mJ)) / 2   # m_F = m_I + m_J
        # Estimate F from energy: E ≈ A/2·K
        # For the ground state family, use closest F level energy
        Fs   = F_levels(I, J)
        best_F = min(Fs, key=lambda F_: abs(evals0[k] - hyperfine_energy(F_, I, J, A_MHz, B_MHz_quad)))
        labels.append((best_F, mF_approx))

    # ── Diagonalise at each B ────────────────────────────────────────────────
    all_energies = np.zeros((len(B_gauss_array), dim))

    for bi, B in enumerate(B_gauss_array):
        H = H_hf.copy()
        np.fill_diagonal(H, H.diagonal() + H_Z_diag * B)
        evals = np.linalg.eigvalsh(H)
        all_energies[bi] = evals

    return all_energies, labels


# ============================================================================
# STEP 6 — TRANSITION SELECTION RULES
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


def identify_special_transitions(transitions_list, Fs_ground, I_gs, J_gs, A_MHz_gs):
    """
    Identify:
      - Cycling transition: F_max → F_max + 1  (closed for laser cooling)
      - Clock transition:   ΔF=1, Δm_F=0       (B-insensitive to 1st order)
      - Microwave transitions: within ground state (ΔF=1)
    """
    F_max_gs = max(Fs_ground)
    special  = {}

    # Cycling: ground F_max → excited F_max + 1
    for F_l, F_u in transitions_list:
        if abs(F_l - F_max_gs) < 0.001 and abs(F_u - (F_max_gs + 1)) < 0.001:
            special['cycling'] = (F_l, F_u)
            break

    # Ground-state microwave transition: ΔF=1 within ground multiplet
    # Most important for clocks: F_max ↔ F_max-1 at m_F=0
    if len(Fs_ground) >= 2:
        F1 = Fs_ground[-1]   # F_max
        F2 = Fs_ground[-2]   # F_max - 1
        dE = hyperfine_energy(F1, I_gs, J_gs, A_MHz_gs) - \
             hyperfine_energy(F2, I_gs, J_gs, A_MHz_gs)
        nu_MHz = abs(dE)
        special['microwave_clock'] = {
            'F_upper': F1, 'F_lower': F2,
            'frequency_MHz': nu_MHz,
            'frequency_GHz': nu_MHz / 1e3,
            'frequency_Hz':  nu_MHz * 1e6,
            'wavelength_m':  3e8 / (nu_MHz * 1e6) if nu_MHz > 0 else None,
            'wavelength_cm': 3e4 / nu_MHz if nu_MHz > 0 else None,
            'note': 'Zero-field ground ΔF=1 interval (m_F=0 clock to first order in B)',
        }

    return special


# ============================================================================
# STEP 7 — MAIN COMPUTATION PER ELEMENT
# ============================================================================

def run_element(element, isotope_A=None, state_labels=None,
                B_max=500.0, B_points=500, data_dir='data_json'):

    print(f"\n{'='*65}")
    print(f"  {element}  — Hyperfine Structure")
    print(f"{'='*65}")

    # ── Nuclear data ─────────────────────────────────────────────────────────
    iso = get_isotope(element, isotope_A)
    if iso is None:
        print(f"  ERROR: No nuclear data found for {element}")
        return None

    I    = iso['I']
    mu_I = iso['mu_I'] or 0.0
    Q    = iso['Q_barn']
    gI   = iso['g_I'] or 0.0
    A    = iso['A']
    print(f"  Isotope: {element}-{A}")
    print(f"  I = {I}  μ_I = {mu_I:.5f} nm  Q = {Q:.5f} b  g_I = {gI:.5f}")

    if I == 0:
        print(f"  I=0: no hyperfine structure. Exiting.")
        return None

    # ── HF constants ─────────────────────────────────────────────────────────
    hf_consts = load_hf_constants(element, data_dir)
    if hf_consts is None:
        print(f"  No hf_constants available. Fill in {data_dir}/{element}_hf_constants.json")
        return None

    # ── Rydberg data for state energies ──────────────────────────────────────
    ryd_file = os.path.join(data_dir, f"{element}_rydberg.json")
    ryd_data = {}
    if os.path.exists(ryd_file):
        with open(ryd_file) as f:
            ryd_data = json.load(f)

    # If no states specified, use all states in hf_constants
    if state_labels is None:
        state_labels = [k for k in hf_consts if not k.startswith('_')]

    print(f"  States: {', '.join(state_labels)}")

    # ── Process each state ───────────────────────────────────────────────────
    B_arr = np.linspace(0, B_max, B_points)
    state_results = []

    for slabel in state_labels:
        parsed = parse_state_label(slabel)
        if parsed is None:
            print(f"  Warning: could not parse state '{slabel}' — skipping")
            continue

        n, l_char, J = parsed
        l_int = {'s':0,'p':1,'d':2,'f':3,'g':4}.get(l_char, 0)
        S = 0.5   # electron spin — overridden by S_exp for non-alkalis
        L = l_int

        # HF constants — read all fields including new ones
        hfc    = hf_consts.get(slabel, {})
        A_MHz  = hfc.get('A_MHz', 0.0)
        B_MHz  = hfc.get('B_MHz', 0.0)
        C_kHz  = hfc.get('C_kHz', 0.0)    # magnetic octupole (optional)
        gJ_exp = hfc.get('gJ_exp', None)   # experimental g_J override (optional)
        S_exp  = hfc.get('S_exp',  None)   # effective S for Landé if not 0.5

        # S override for multi-electron atoms (e.g. triplet states S=1)
        if S_exp is not None:
            S = float(S_exp)

        # g_J: use experimental value if provided, else Landé formula
        gJ = lande_gJ(L, S, J, gJ_exp=gJ_exp)

        # Extrapolate if state not in file but same (l,J) exists
        if A_MHz == 0.0:
            # Find a reference state with same l and J
            for ref_label, ref_hfc in hf_consts.items():
                ref = parse_state_label(ref_label)
                if ref and ref[1] == l_char and abs(ref[2] - J) < 0.01 and ref_hfc.get('A_MHz', 0) != 0:
                    # Quantum defect scaling: A ∝ n*^{-3}
                    qd       = ryd_data.get('quantum_defects', {})
                    delta_l  = qd.get(l_char, 0.0)
                    n_eff_ref = ref[0] - delta_l
                    n_eff_cur = n     - delta_l
                    if n_eff_ref > 0 and n_eff_cur > 0:
                        scale = (n_eff_ref / n_eff_cur) ** 3
                        A_MHz = ref_hfc['A_MHz'] * scale
                        B_MHz = ref_hfc.get('B_MHz', 0.0) * scale
                        C_kHz = ref_hfc.get('C_kHz', 0.0) * scale
                        print(f"    {slabel}: A_hf extrapolated from {ref_label} "
                              f"× (n*/n*)³ = {A_MHz:.3f} MHz")
                    break

        if A_MHz == 0.0:
            print(f"    {slabel}: A_hf = 0 (not in hf_constants) — F levels degenerate")

        # F levels and energies (now including octupole C)
        Fs   = F_levels(I, J)
        E_hf = {F: hyperfine_energy(F, I, J, A_MHz, B_MHz, C_kHz) for F in Fs}

        # g_F factors
        gF  = {F: gF_factor(F, I, J, gJ, gI) for F in Fs}

        # m_F sublevels
        mF_levels = {}
        for F in Fs:
            mF_levels[F] = [round((F-k)*2)/2 for k in range(round(2*F)+1)]

        # Weak-field Zeeman slopes
        zeeman = {}
        for F in Fs:
            zeeman[F] = {mF: gF[F] * MU_B_MHz_G * mF for mF in mF_levels[F]}

        # Breit-Rabi
        BR_energies, BR_labels = breit_rabi_energies(
            I, J, A_MHz, B_MHz, gJ, gI, B_arr)

        print(f"\n  State {slabel}:  J={J}  g_J={gJ:.4f}"
              f"{'  [experimental]' if gJ_exp is not None else '  [Landé]'}"
              f"  A={A_MHz:.4f} MHz  B={B_MHz:.4f} MHz"
              f"{f'  C={C_kHz:.3f} kHz' if C_kHz != 0 else ''}")
        print(f"    F levels: {Fs}")
        for F in Fs:
            mFs = mF_levels[F]
            print(f"    F={F:.0f}:  E={E_hf[F]:+.3f} MHz  g_F={gF[F]:+.5f}  "
                  f"m_F={[int(2*m) for m in mFs]} (in units of 1/2)")

        state_results.append({
            'label':      slabel,
            'n': n, 'l': l_char, 'J': J, 'L': L, 'S': S,
            'gJ':         gJ,
            'gJ_source':  'experimental' if gJ_exp is not None else 'Lande',
            'A_MHz':      A_MHz,
            'B_MHz':      B_MHz,
            'C_kHz':      C_kHz,
            'Fs':         Fs,
            'E_hf':       E_hf,
            'gF':         gF,
            'mF_levels':  mF_levels,
            'zeeman':     zeeman,
            'BR_energies': BR_energies,
            'BR_labels':   BR_labels,
        })

    if not state_results:
        print("  No states processed.")
        return None

    # Ground state identification (lowest energy state = first in state_labels)
    gs = state_results[0]
    Fs_gs     = gs['Fs']
    A_MHz_gs  = gs['A_MHz']

    # Microwave / clock transition in ground state
    special = identify_special_transitions([], Fs_gs, I, gs['J'], A_MHz_gs)
    if 'microwave_clock' in special:
        special['microwave_clock']['state'] = gs['label']

    if 'microwave_clock' in special:
        mc = special['microwave_clock']
        print(f"\n  Ground-state microwave / clock transition ({gs['label']}):")
        print(f"    F={mc['F_upper']:.0f} ↔ F={mc['F_lower']:.0f}: "
              f"{mc['frequency_MHz']:.6f} MHz "
              f"({mc['frequency_GHz']:.9f} GHz)  "
              f"(λ ≈ {mc['wavelength_cm']:.3f} cm)")

    return {
        'element':       element,
        'isotope_A':     A,
        'I':             I,
        'mu_I':          mu_I,
        'Q_barn':        Q,
        'gI':            gI,
        'B_arr':         B_arr,
        'states':        state_results,
        'special':       special,
    }


# ============================================================================
# STEP 8 — PLOTS
# ============================================================================

ELEM_COLORS = {
    'Li':'#4df0b0','Na':'#f0c54d','K':'#4d9ff0',
    'Rb':'#f05a4d','Cs':'#c54df0','Fr':'#f04d9f',
}

# Colour palette for m_F sublevels: centre = white, ± red/blue
def mF_color(mF, F):
    """Map m_F in [-F, +F] to a colour."""
    if F < 1e-9:
        return '#ffffff'
    norm = mF / F   # -1 to +1
    # Diverging: blue (norm=-1) → white (0) → red (norm=+1)
    r = int(255 * max(0, norm))
    b = int(255 * max(0, -norm))
    g = int(255 * (1 - abs(norm)) * 0.7)
    return f'rgb({r},{g},{b})'


def plot_hyperfine(result, out_dir):
    """
    Hyperfine structure plot — NO Breit-Rabi (lives in zeeman.html).

    Layout:
      Row 1 (tall, ~68%): single shared-axis Grotrian-style level diagram —
             all states on one x-axis (one column each), F bars, m_F ticks,
             F/g_F labels, intra-state E1 connections. Shared y-axis (MHz)
             across all states. Ground-state microwave clock interval is
             annotated when special['microwave_clock'] is present.
      Row 2 (~32%): dropdown-toggled panel using native Plotly updatemenus
             (same pattern as the Breit-Rabi state selector in zeeman.html):
               - "Splittings" — log-scale bar chart of |dE(F,F+-1)| per state
                 (clock interval highlighted)
               - "Constants"  — table of A, B, C, g_J, g_F per state
               - "Clock"      — ground ΔF=1 microwave frequency card

    Scales to any number of states automatically — just add to hf_constants.json.
    """
    el     = result['element']
    A_iso  = result['isotope_A']
    nuc    = nuclide_label(el, A_iso)
    nuc_p  = nuclide_plain(el, A_iso)
    I      = result['I']
    states = result['states']
    mc     = result.get('special', {}).get('microwave_clock', {}) or {}
    mc_Fu  = mc.get('F_upper')
    mc_Fl  = mc.get('F_lower')
    mc_state = mc.get('state', states[0]['label'] if states else '')
    mc_nu  = mc.get('frequency_MHz')

    STATE_PALETTE = [
        '#f0c54d', '#4d9ff0', '#4df0b0', '#f05a4d',
        '#c54df0', '#f04d9f', '#7af07a', '#f0a44d',
    ]
    state_cols = [STATE_PALETTE[i % len(STATE_PALETTE)] for i in range(len(states))]
    CLOCK_COL = '#f0c54d'

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.68, 0.32],
        vertical_spacing=0.10,
        subplot_titles=[
            f'Hyperfine Level Structure — {nuc}  (I = {I})',
            'Hyperfine Splittings  |\u0394E(F, F\u00b11)|  (log scale)',
        ],
    )

    # ---- ROW 1: shared-axis level diagram (single x/y for all states) ------
    COL_W  = 1.4
    COL_SP = 2.4
    MF_DY  = 0.035

    all_E_vals = []
    for st in states:
        all_E_vals.extend(st['E_hf'].values())
    global_span = max(max(all_E_vals) - min(all_E_vals), 1.0)

    for si, st in enumerate(states):
        x_ctr   = si * COL_SP
        Fs      = st['Fs']
        E_hf    = st['E_hf']
        gF_map  = st['gF']
        mF_lev  = st['mF_levels']
        scol    = state_cols[si]
        A_MHz   = st['A_MHz']
        E_vals  = list(E_hf.values())
        E_span  = max(E_vals) - min(E_vals) if len(E_vals) > 1 else global_span * 0.1
        mF_drop = max(global_span * MF_DY, E_span * 0.08)

        fig.add_annotation(
            x=x_ctr, y=max(E_vals) + global_span * 0.05,
            text=f'<b>{st["label"]}</b>',
            showarrow=False, font=dict(size=12, color=scol),
            xref='x', yref='y', xanchor='center', yanchor='bottom',
        )

        fig.add_trace(go.Scatter(
            x=[x_ctr, x_ctr],
            y=[min(E_vals) - mF_drop * 1.6, max(E_vals)],
            mode='lines',
            line=dict(color=f'rgba({int(scol[1:3],16)},'
                            f'{int(scol[3:5],16)},'
                            f'{int(scol[5:7],16)},0.16)', width=1),
            showlegend=False, hoverinfo='skip',
        ), row=1, col=1)

        for F in Fs:
            E     = E_hf[F]
            x0    = x_ctr - COL_W * 0.42
            x1    = x_ctr + COL_W * 0.42
            mFs   = mF_lev[F]
            gF    = gF_map[F]
            F_lbl = int(F) if F == int(F) else F
            n_mF  = len(mFs)
            is_clock_F = (
                mc_nu is not None
                and st['label'] == mc_state
                and (abs(F - mc_Fu) < 1e-6 or abs(F - mc_Fl) < 1e-6)
            )
            line_w = 4.5 if is_clock_F else 3
            line_c = CLOCK_COL if is_clock_F else scol

            hover = (f'<b>{nuc}  {st["label"]}  F={F_lbl}</b><br>'
                     f'E = {E:+.4f} MHz<br>'
                     f'g<sub>F</sub> = {gF:+.5f}<br>'
                     f'A = {A_MHz:.4f} MHz<br>'
                     f'm<sub>F</sub> = {int(min(mFs))}\u2026{int(max(mFs))}<br>'
                     f'2F+1 = {n_mF} sublevels')
            if is_clock_F:
                hover += (f'<br><b>Clock manifold</b> '
                          f'({mc_nu:.6f} MHz gs interval)')

            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[E, E],
                mode='lines',
                line=dict(color=line_c, width=line_w),
                name=f'{st["label"]} F={F_lbl}',
                legendgroup=f'st{si}',
                showlegend=(F == Fs[0]),
                hovertemplate=hover + '<extra></extra>',
            ), row=1, col=1)

            fig.add_annotation(
                x=x0 - 0.05, y=E, text=f'F={F_lbl}',
                showarrow=False, xanchor='right',
                font=dict(size=9, color=line_c), xref='x', yref='y',
            )
            fig.add_annotation(
                x=x1 + 0.05, y=E, text=f'g<sub>F</sub>={gF:+.3f}',
                showarrow=False, xanchor='left',
                font=dict(size=8, color='#7a9acd'), xref='x', yref='y',
            )

            y_tick  = E - mF_drop
            tick_hw = COL_W * 0.36 / max(n_mF - 1, 1) if n_mF > 1 else 0
            for k, mF in enumerate(sorted(mFs)):
                tx = (x_ctr - COL_W * 0.36 + k * tick_hw * 2
                      if n_mF > 1 else x_ctr)
                fig.add_trace(go.Scatter(
                    x=[tx, tx], y=[E, y_tick], mode='lines',
                    line=dict(color=mF_color(mF, F), width=1.5),
                    showlegend=False,
                    hovertemplate=f'm<sub>F</sub>={int(mF):+d}<extra></extra>',
                ), row=1, col=1)
                if mF == max(mFs) or mF == min(mFs) or mF == 0:
                    fig.add_annotation(
                        x=tx, y=y_tick - global_span * 0.014,
                        text=(f'{int(mF):+d}' if mF == int(mF) else f'{mF:+.1f}'),
                        showarrow=False, xanchor='center', yanchor='top',
                        font=dict(size=7, color='#5a6a8a'), xref='x', yref='y',
                    )

        # Ground-state microwave clock arrow (ΔF=1 interval)
        if (mc_nu is not None and st['label'] == mc_state
                and mc_Fu in E_hf and mc_Fl in E_hf):
            E_hi = E_hf[mc_Fu]
            E_lo = E_hf[mc_Fl]
            x_clk = x_ctr + COL_W * 0.55
            Fu_i = int(mc_Fu) if mc_Fu == int(mc_Fu) else mc_Fu
            Fl_i = int(mc_Fl) if mc_Fl == int(mc_Fl) else mc_Fl
            fig.add_trace(go.Scatter(
                x=[x_clk, x_clk], y=[E_lo, E_hi],
                mode='lines+markers',
                line=dict(color=CLOCK_COL, width=2.5),
                marker=dict(symbol='line-ew', size=10, color=CLOCK_COL,
                            line=dict(width=2, color=CLOCK_COL)),
                name='Microwave clock',
                legendgroup='clock',
                showlegend=True,
                hovertemplate=(
                    f'<b>Microwave clock</b><br>'
                    f'{st["label"]}  F={Fu_i} \u2194 F={Fl_i}<br>'
                    f'\u03bd = {mc_nu:.6f} MHz<br>'
                    f'= {mc.get("frequency_GHz", mc_nu/1e3):.9f} GHz<br>'
                    f'\u03bb \u2248 {mc.get("wavelength_cm", 0):.3f} cm'
                    f'<extra></extra>'
                ),
            ), row=1, col=1)
            fig.add_annotation(
                x=x_clk + 0.12, y=0.5 * (E_lo + E_hi),
                text=(f'<b>\u03bd<sub>clock</sub></b><br>'
                      f'{mc_nu:.3f} MHz<br>'
                      f'({mc.get("frequency_GHz", mc_nu/1e3):.6f} GHz)'),
                showarrow=False, xanchor='left', yanchor='middle',
                font=dict(size=10, color=CLOCK_COL),
                xref='x', yref='y',
                bgcolor='rgba(10,13,20,0.75)',
                bordercolor=CLOCK_COL, borderwidth=1, borderpad=4,
            )

    # Inter-state E1 transition connectors
    for si in range(len(states) - 1):
        st_l = states[si]
        st_u = states[si + 1]
        x_l  = si * COL_SP + COL_W * 0.42
        x_u  = (si + 1) * COL_SP - COL_W * 0.42
        for F_l in st_l['Fs']:
            for F_u in st_u['Fs']:
                dF = abs(F_u - F_l)
                if dF > 1.001 or (F_l < 0.001 and F_u < 0.001):
                    continue
                E_l   = st_l['E_hf'][F_l]
                E_u   = st_u['E_hf'][F_u]
                dE_hf = E_u - E_l
                F_l_i = int(F_l) if F_l == int(F_l) else F_l
                F_u_i = int(F_u) if F_u == int(F_u) else F_u
                fig.add_trace(go.Scatter(
                    x=[x_l, x_u], y=[E_l, E_u], mode='lines',
                    line=dict(color='rgba(200,200,200,0.18)', width=1, dash='dot'),
                    showlegend=False,
                    hovertemplate=(
                        f'<b>E1: {st_l["label"]} F={F_l_i} \u2192 '
                        f'{st_u["label"]} F={F_u_i}</b><br>'
                        f'\u0394F = {int(F_u - F_l):+d}<br>'
                        f'E({st_l["label"]} F={F_l_i}) = {E_l:+.4f} MHz<br>'
                        f'E({st_u["label"]} F={F_u_i}) = {E_u:+.4f} MHz<br>'
                        f'\u0394E_hf = {dE_hf:+.4f} MHz<extra></extra>'
                    ),
                ), row=1, col=1)

    x_lo = -COL_W * 0.8
    x_hi = (len(states) - 1) * COL_SP + COL_W * 0.8
    if mc_nu is not None:
        x_hi += COL_W * 0.55
    fig.update_xaxes(range=[x_lo, x_hi], showticklabels=False,
                     showgrid=False, zeroline=False, row=1, col=1)
    fig.update_yaxes(title_text='E_hf (MHz, rel. centroid)',
                     gridcolor='#1a2540', zeroline=True,
                     zerolinecolor='rgba(255,255,255,0.1)', row=1, col=1)

    # ---- ROW 2 TRACE SET A: Splittings bar chart (log scale) ---------------
    bar_x, bar_y, bar_col, bar_hover, bar_line_w, bar_line_c = [], [], [], [], [], []
    for si, st in enumerate(states):
        Fs   = st['Fs']
        E_hf = st['E_hf']
        scol = state_cols[si]
        for k in range(len(Fs) - 1):
            F_lo, F_hi = Fs[k], Fs[k + 1]
            dE     = abs(E_hf[F_hi] - E_hf[F_lo])
            F_lo_i = int(F_lo) if F_lo == int(F_lo) else F_lo
            F_hi_i = int(F_hi) if F_hi == int(F_hi) else F_hi
            lam_cm = 3e4 / dE if dE > 0 else 0
            is_clock_bar = (
                mc_nu is not None
                and st['label'] == mc_state
                and ((abs(F_lo - mc_Fl) < 1e-6 and abs(F_hi - mc_Fu) < 1e-6)
                     or (abs(F_lo - mc_Fu) < 1e-6 and abs(F_hi - mc_Fl) < 1e-6))
            )
            bar_x.append(f'{st["label"]}<br>F={F_lo_i}\u2194{F_hi_i}')
            bar_y.append(max(dE, 1e-3))
            bar_col.append(CLOCK_COL if is_clock_bar else scol)
            bar_line_w.append(2 if is_clock_bar else 1)
            bar_line_c.append('#ffffff' if is_clock_bar else 'rgba(255,255,255,0.2)')
            hover = (
                f'<b>{st["label"]}  F={F_lo_i}\u2194F={F_hi_i}</b><br>'
                f'\u0394E = {dE:.5f} MHz<br>'
                f'\u03bb \u2248 {lam_cm:.2f} cm  ({lam_cm*10:.1f} mm)'
            )
            if is_clock_bar:
                hover += (f'<br><b>Ground microwave clock</b><br>'
                          f'\u03bd = {dE:.6f} MHz = {dE/1e3:.9f} GHz')
            bar_hover.append(hover)

    bar_trace_idx = len(fig.data)
    fig.add_trace(go.Bar(
        x=bar_x, y=bar_y,
        marker=dict(
            color=bar_col,
            line=dict(color=bar_line_c, width=bar_line_w),
        ),
        text=[f'{v:.4f}' for v in bar_y],
        textposition='outside',
        textfont=dict(size=9, color='#c8d0e0'),
        showlegend=False,
        customdata=bar_hover,
        hovertemplate='%{customdata}<extra></extra>',
        visible=True,
    ), row=2, col=1)

    fig.update_xaxes(gridcolor='#1a2540', tickfont=dict(size=9), row=2, col=1)
    fig.update_yaxes(title_text='|\u0394E| (MHz) (log scale)',
                     type='log', gridcolor='#1a2540', row=2, col=1)

    # ---- Constants table data — rendered as an HTML overlay panel below ----
    # (go.Table sharing a row with go.Bar via visible-toggling is unreliable:
    # Plotly's default template draws a residual background for hidden Table
    # traces. An HTML/JS overlay panel is robust and matches the dropdown
    # button styling used for the Bar/Table switch.)
    tbl_state, tbl_A, tbl_B, tbl_C = [], [], [], []
    tbl_gJ, tbl_gJ_src, tbl_F_gF  = [], [], []
    for st in states:
        tbl_state.append(st['label'])
        tbl_A.append(f'{st["A_MHz"]:.5f}')
        tbl_B.append(f'{st["B_MHz"]:.4f}' if st['B_MHz'] != 0 else '\u2014')
        tbl_C.append(f'{st.get("C_kHz", 0):.3f}'
                     if st.get('C_kHz', 0) != 0 else '\u2014')
        tbl_gJ.append(f'{st["gJ"]:.5f}')
        tbl_gJ_src.append(st.get('gJ_source', 'Land\u00e9'))
        gF_lines = ', '.join(
            f'F={int(F) if F==int(F) else F}: {st["gF"][F]:+.4f}'
            for F in st['Fs']
        )
        tbl_F_gF.append(gF_lines)

    # ---- Overall layout ------------------------------------------------------
    mc_str = ''
    if mc_nu is not None:
        Fu_i = int(mc_Fu) if mc_Fu == int(mc_Fu) else mc_Fu
        Fl_i = int(mc_Fl) if mc_Fl == int(mc_Fl) else mc_Fl
        mc_str = (f' \u00b7 Clock {mc_state} F={Fu_i}\u2194F={Fl_i}'
                  f' @ {mc_nu:.6f} MHz'
                  f' ({mc.get("frequency_GHz", mc_nu/1e3):.6f} GHz)')

    fig.update_layout(
        title=dict(
            text=(f'<b>{nuc} Hyperfine Structure</b>'
                  f'<br><sup>{nuc_p}  ·  I={I}  '
                  f'\u03bc_I={result["mu_I"]:.4f} nm  '
                  f'Q={result["Q_barn"]:.5f} b'
                  f'{mc_str}</sup>'),
            x=0.5, xanchor='center',
            font=dict(size=18, color='#e8edf5'),
        ),
        plot_bgcolor='#0f1628',
        paper_bgcolor='#0a0d14',
        font=dict(color='#c8d0e0',
                  family="'Helvetica Neue',Arial,sans-serif"),
        hovermode='closest',
        autosize=True,
        margin=dict(t=130, r=40, b=40, l=80),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(10,13,20,0.88)',
            bordercolor='#1e2840', borderwidth=1,
            x=1.01, xanchor='left', y=0.99, yanchor='top',
            font=dict(size=10),
            title=dict(text='<b>State (F levels)</b>',
                       font=dict(size=10, color='#8a9ac0')),
        ),
    )

    for ann in fig.layout.annotations:
        if ann.text:
            ann.font.color = '#8a9ac0'
            ann.font.size  = 11

    out_file = os.path.join(out_dir, f"{el}_hyperfine.html")
    raw_html = fig.to_html(include_plotlyjs='cdn', full_html=True)
    if '<title>' in raw_html:
        raw_html = raw_html.replace('<title></title>', f'<title>{nuc_p} Hyperfine</title>', 1)
        raw_html = raw_html.replace('<title>Plotly</title>', f'<title>{nuc_p} Hyperfine</title>', 1)
    else:
        raw_html = raw_html.replace('</head>', f'<title>{nuc_p} Hyperfine</title></head>', 1)
    _fs_css  = ('<style>'
                'html,body{margin:0;padding:0;width:100%;height:100%;'
                'overflow:hidden;background:#0a0d14;}'
                '.plotly-graph-div{width:100vw!important;height:100vh!important;}'
                '#hf-toggle-wrap{position:fixed;z-index:9999;'
                'font-family:Arial,sans-serif;font-size:12px;}'
                '#hf-toggle-select{padding:6px 10px;background:#111830;'
                'color:#c8d0e0;border:1px solid #2a3060;border-radius:5px;'
                'cursor:pointer;font-size:12px;}'
                '#hf-const-overlay,#hf-clock-overlay{position:fixed;left:0;right:0;'
                'bottom:0;max-height:34vh;overflow-y:auto;background:#0a0d14;'
                'border-top:2px solid #2a3555;padding:10px 16px;'
                'font-family:Arial,sans-serif;font-size:12px;color:#c8d0e0;'
                'display:none;z-index:9998;}'
                '#hf-const-overlay table{width:100%;border-collapse:collapse;}'
                '#hf-const-overlay th{background:#1a2540;color:#e8edf5;'
                'padding:7px 10px;text-align:center;position:sticky;top:0;'
                'border-bottom:1px solid #2a3555;}'
                '#hf-const-overlay td{padding:6px 10px;text-align:center;'
                'border-bottom:1px solid #1e2840;color:#c8d0e0;}'
                '#hf-const-overlay tr:nth-child(even) td{background:#111a2f;}'
                '#hf-const-overlay tr:nth-child(odd) td{background:#0f1628;}'
                '#hf-const-overlay td.state-col{color:#4df0b0;font-weight:bold;}'
                '.hf-clock-card{display:flex;flex-wrap:wrap;gap:14px;'
                'align-items:stretch;}'
                '.hf-clock-box{flex:1 1 180px;background:#111a2f;'
                'border:1px solid #2a3555;border-radius:8px;padding:12px 14px;}'
                '.hf-clock-box.accent{border-color:#f0c54d;}'
                '.hf-clock-k{color:#8a9ac0;font-size:11px;margin-bottom:4px;}'
                '.hf-clock-v{color:#e8edf5;font-size:18px;font-weight:bold;'
                'font-variant-numeric:tabular-nums;}'
                '.hf-clock-v.gold{color:#f0c54d;}'
                '.hf-clock-note{color:#8a9ac0;font-size:11px;margin-top:8px;'
                'line-height:1.45;max-width:720px;}'
                '</style>')
    raw_html = raw_html.replace('</head>', _fs_css + '</head>', 1)

    # ---- Inject dropdown toggle + HTML constants / clock panels --------------
    import json as _json_hf
    _tbl_rows_js = _json_hf.dumps([
        {'state': s, 'A': a, 'B': b, 'C': c, 'gJ': gj, 'src': src, 'gF': gf}
        for s, a, b, c, gj, src, gf in zip(
            tbl_state, tbl_A, tbl_B, tbl_C, tbl_gJ, tbl_gJ_src, tbl_F_gF)
    ])
    _clock_js = _json_hf.dumps({
        'present': bool(mc_nu is not None),
        'state': mc_state,
        'F_upper': int(mc_Fu) if mc_Fu is not None and mc_Fu == int(mc_Fu) else mc_Fu,
        'F_lower': int(mc_Fl) if mc_Fl is not None and mc_Fl == int(mc_Fl) else mc_Fl,
        'frequency_MHz': mc.get('frequency_MHz'),
        'frequency_GHz': mc.get('frequency_GHz'),
        'frequency_Hz':  mc.get('frequency_Hz'),
        'wavelength_cm': mc.get('wavelength_cm'),
        'wavelength_m':  mc.get('wavelength_m'),
        'note': mc.get('note', ''),
        'isotope': nuc_p,
        'I': I,
    }, allow_nan=False)
    _clock_opt = (
        '<option value="clock">Clock frequency</option>'
        if mc_nu is not None else ''
    )
    _panel_html = (
        '<div id="hf-toggle-wrap" style="left:14px;top:62px;">'
        '<select id="hf-toggle-select" onchange="hfOnToggleChange(this.value)">'
        '<option value="splittings">Splittings</option>'
        '<option value="constants">Constants</option>'
        f'{_clock_opt}'
        '</select></div>'
        '<div id="hf-const-overlay"></div>'
        '<div id="hf-clock-overlay"></div>'
        '<script>'
        f'const HF_TABLE_DATA = {_tbl_rows_js};'
        f'const HF_CLOCK = {_clock_js};'
        '''
        function hfFmt(n, digits) {
          if (n === null || n === undefined) return '-';
          return Number(n).toLocaleString(undefined, {
            minimumFractionDigits: digits, maximumFractionDigits: digits
          });
        }
        function hfBuildTable() {
          const panel = document.getElementById('hf-const-overlay');
          let html = '<table><thead><tr>'
            + '<th>State</th><th>A (MHz)</th><th>B (MHz)</th><th>C (kHz)</th>'
            + '<th>g_J</th><th>g_J source</th><th>g_F per F</th>'
            + '</tr></thead><tbody>';
          HF_TABLE_DATA.forEach(r => {
            html += `<tr><td class="state-col">${r.state}</td>`
              + `<td>${r.A}</td><td>${r.B}</td><td>${r.C}</td>`
              + `<td>${r.gJ}</td><td>${r.src}</td><td>${r.gF}</td></tr>`;
          });
          html += '</tbody></table>';
          panel.innerHTML = html;
        }
        function hfBuildClock() {
          const panel = document.getElementById('hf-clock-overlay');
          if (!HF_CLOCK.present) {
            panel.innerHTML = '<div class="hf-clock-note">No ground-state '
              + 'microwave clock interval (need I>0 and at least two F levels).</div>';
            return;
          }
          panel.innerHTML = `
            <div class="hf-clock-card">
              <div class="hf-clock-box accent">
                <div class="hf-clock-k">${HF_CLOCK.isotope} · ${HF_CLOCK.state}
                  · F=${HF_CLOCK.F_upper} ↔ F=${HF_CLOCK.F_lower}</div>
                <div class="hf-clock-v gold">${hfFmt(HF_CLOCK.frequency_MHz, 6)} MHz</div>
              </div>
              <div class="hf-clock-box">
                <div class="hf-clock-k">Frequency (GHz)</div>
                <div class="hf-clock-v">${hfFmt(HF_CLOCK.frequency_GHz, 9)} GHz</div>
              </div>
              <div class="hf-clock-box">
                <div class="hf-clock-k">Frequency (Hz)</div>
                <div class="hf-clock-v">${hfFmt(HF_CLOCK.frequency_Hz, 0)} Hz</div>
              </div>
              <div class="hf-clock-box">
                <div class="hf-clock-k">Wavelength</div>
                <div class="hf-clock-v">${hfFmt(HF_CLOCK.wavelength_cm, 4)} cm</div>
              </div>
            </div>
            <div class="hf-clock-note">${HF_CLOCK.note}
              Stored in data_json as special.microwave_clock. The m_F=0
              component is first-order B-insensitive (clock transition);
              full Zeeman / Breit-Rabi curves are in the companion zeeman HTML.</div>`;
        }
        function hfOnToggleChange(value) {
          const gd         = document.querySelector('.plotly-graph-div');
          const constPanel = document.getElementById('hf-const-overlay');
          const clockPanel = document.getElementById('hf-clock-overlay');
          const showBar = (value === 'splittings');
          Plotly.restyle(gd, {visible: showBar}, [BAR_TRACE_IDX]);
          constPanel.style.display = (value === 'constants') ? 'block' : 'none';
          clockPanel.style.display = (value === 'clock') ? 'block' : 'none';
        }
        hfBuildTable();
        hfBuildClock();
        '''
        .replace('BAR_TRACE_IDX', str(bar_trace_idx))
        + '</script>'
    )
    raw_html = raw_html.replace('</body>', _panel_html + '</body>', 1)

    with open(out_file, 'w', encoding='utf-8') as fh:
        fh.write(raw_html)
    print(f"\n  \u2713 {out_file}")
    return out_file



def plot_zeeman(result, out_dir):
    """
    Standalone full-page Breit-Rabi / Zeeman diagram with dropdown
    to switch between states.
    """
    el     = result['element']
    A      = result['isotope_A']
    nuc    = nuclide_label(el, A)
    nuc_p  = nuclide_plain(el, A)
    I      = result['I']
    states = result['states']
    B_arr  = result['B_arr']
    col    = ELEM_COLORS.get(el, '#7777ff')

    fig = go.Figure()

    buttons = []
    traces_per_state = []

    for si, st in enumerate(states):
        BR_E   = st['BR_energies']
        BR_lab = st['BR_labels']
        dim    = BR_E.shape[1]
        t_indices = []

        for li in range(dim):
            F_approx, mF_approx = BR_lab[li]
            c  = mF_color(mF_approx, F_approx)
            ls = f'F={F_approx:.0f}, m<sub>F</sub>={mF_approx:+.0f}'
            fig.add_trace(go.Scatter(
                x=B_arr.tolist(), y=BR_E[:, li].tolist(),
                mode='lines', line=dict(color=c, width=2),
                name=ls, visible=(si == 0),
                hovertemplate=(f'<b>{st["label"]}</b> {ls}<br>'
                               f'B=%{{x:.1f}} G  E=%{{y:.3f}} MHz<extra></extra>'),
            ))
            t_indices.append(len(fig.data) - 1)

        traces_per_state.append(t_indices)

    # Build dropdown
    total_traces = len(fig.data)
    for si, st in enumerate(states):
        vis = [False] * total_traces
        for ti in traces_per_state[si]:
            vis[ti] = True
        buttons.append(dict(
            label=st['label'],
            method='update',
            args=[{'visible': vis},
                  {'title.text': (f'<b>{nuc} Breit-Rabi: {st["label"]}</b><br>'
                                  f'<sup>{nuc_p}  ·  J={st["J"]}  g_J={st["gJ"]:.4f}  '
                                  f'A={st["A_MHz"]:.3f} MHz</sup>')}],
        ))

    fig.update_layout(
        title=dict(
            text=(f'<b>{nuc} Breit-Rabi Diagram: {states[0]["label"]}</b><br>'
                  f'<sup>{nuc_p}  ·  J={states[0]["J"]}  g_J={states[0]["gJ"]:.4f}  '
                  f'A={states[0]["A_MHz"]:.3f} MHz  |  '
                  f'I={I}  μ_I={result["mu_I"]:.4f} nm</sup>'),
            x=0.5, xanchor='center', font=dict(size=18, color='#e8edf5'),
        ),
        xaxis=dict(title='B field (Gauss)', gridcolor='#1a2540',
                   tickfont=dict(color='#6b7a99')),
        yaxis=dict(title='Energy (MHz)', gridcolor='#1a2540',
                   zeroline=True, zerolinecolor='#2a3555',
                   tickfont=dict(color='#6b7a99')),
        plot_bgcolor='#0f1628', paper_bgcolor='#0a0d14',
        font=dict(color='#c8d0e0'),
        hovermode='x unified',
        autosize=True,
        margin=dict(t=110, r=220, b=60, l=80),
        legend=dict(
            bgcolor='rgba(10,13,20,0.88)',
            bordercolor='#1e2840', borderwidth=1,
            x=1.01, xanchor='left', y=1, yanchor='top',
            font=dict(size=11),
            title=dict(text='<b>|F, m_F⟩</b><br><sub>colour: m_F (blue→red)</sub>',
                       font=dict(size=10, color='#8a9ac0')),
        ),
        updatemenus=[dict(
            type='dropdown', direction='down',
            buttons=buttons,
            x=0.01, xanchor='left', y=1.08, yanchor='top',
            bgcolor='#111830', bordercolor='#2a3060',
            font=dict(color='#c8d0e0'),
        )],
    )

    out_file = os.path.join(out_dir, f"{el}_zeeman.html")
    raw_html = fig.to_html(include_plotlyjs='cdn', full_html=True)
    if '<title>' in raw_html:
        raw_html = raw_html.replace('<title></title>', f'<title>{nuc_p} Zeeman</title>', 1)
        raw_html = raw_html.replace('<title>Plotly</title>', f'<title>{nuc_p} Zeeman</title>', 1)
    else:
        raw_html = raw_html.replace('</head>', f'<title>{nuc_p} Zeeman</title></head>', 1)
    _fs_css  = ('<style>html,body{margin:0;padding:0;width:100%;height:100%;'
                'background:#0a0d14;}'
                '.plotly-graph-div{width:100vw!important;height:100vh!important;}'
                '</style>')
    raw_html = raw_html.replace('</head>', _fs_css + '</head>', 1)
    with open(out_file, 'w', encoding='utf-8') as fh:
        fh.write(raw_html)
    print(f"  ✓ {out_file}")
    return out_file


# ============================================================================
# STEP 9 — SAVE JSON
# ============================================================================

def save_json(result, data_dir):
    el  = result['element']
    A   = result['isotope_A']
    out = {
        'element':    el,
        'isotope_A':  A,
        'I':          result['I'],
        'mu_I':       result['mu_I'],
        'Q_barn':     result['Q_barn'],
        'gI':         result['gI'],
        'special':    result['special'],
        'states': [
            {
                'label':      st['label'],
                'n': st['n'], 'l': st['l'], 'J': st['J'],
                'gJ':         st['gJ'],
                'gJ_source':  st.get('gJ_source', 'Lande'),
                'A_MHz':      st['A_MHz'],
                'B_MHz':      st['B_MHz'],
                'C_kHz':      st.get('C_kHz', 0.0),
                'F_levels': [
                    {'F': F, 'E_MHz': st['E_hf'][F],
                     'gF': st['gF'][F],
                     'mF': st['mF_levels'][F]}
                    for F in st['Fs']
                ],
            }
            for st in result['states']
        ],
    }
    path = os.path.join(data_dir, f"{el}_hyperfine.json")
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  ✓ {path}")


# ============================================================================
# MAIN
# ============================================================================

def get_available_elements(data_dir):
    from species import is_ion_id
    avail = []
    if os.path.isdir(data_dir):
        for fname in os.listdir(data_dir):
            m = re.match(r'^(\w+)_hf_constants\.json$', fname)
            if m:
                el = m.group(1)
                if not is_ion_id(el):
                    avail.append(el)
    return sorted(avail)


if __name__ == '__main__':
    from species import require_neutral_species

    if args.all:
        elements = get_available_elements(DATA_DIR)
        if not elements:
            print(f"No _hf_constants.json files found in {DATA_DIR}.")
            sys.exit(1)
    else:
        if args.element is None:
            avail = get_available_elements(DATA_DIR)
            print("Available elements:", ', '.join(avail) if avail else 'none')
            args.element = input("Enter element symbol: ").strip()
        elements = [require_neutral_species(args.element, "hyperfine.py")["species_id"]]

    for el in elements:
        out_dir = args.output_dir or os.path.join('plots', el)
        os.makedirs(out_dir, exist_ok=True)

        result = run_element(
            el,
            isotope_A    = args.isotope,
            state_labels = args.states,
            B_max        = args.B_max,
            B_points     = args.B_points,
            data_dir     = DATA_DIR,
        )
        if result is None:
            continue

        save_json(result, DATA_DIR)
        plot_hyperfine(result, out_dir)
        plot_zeeman(result, out_dir)

    print("\n✓ Done.")
