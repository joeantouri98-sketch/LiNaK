"""
Fallback values for the neutral alkalis

Used by compare_elements.py (cross-element table) and spectra.py (hard-coded D-line fallback), which
previously kept two different copies. Edit values here only.
"""

# LITERATURE / FALLBACK VALUES
# Used when pipeline JSON is not available.
# Sources: Steck D-line data sheets, NIST ASD, Sansonetti 2011.
# fosc_* are J-resolved absorption oscillator strengths (not multiplet totals).

LITERATURE = {
    'Li': {
        'Z': 3, 'mass_amu': 6.941, 'gs': '2s', 'np': '2p',
        'IE_eV': 5.391714,
        'D1_nm': 670.976,   'D2_nm': 670.962,
        'tau_D1_ns': 27.10,  'tau_D2_ns': 27.10,
        'fosc_D1': 0.2490,   'fosc_D2': 0.4980,
        'g_upper_D1': 2, 'g_lower_D1': 2,
        'g_upper_D2': 4, 'g_lower_D2': 2,
        'delta_s': 0.3995,   'delta_p': 0.0472,
        'soc_meV': 0.043,
        'source': 'literature',
    },
    'Na': {
        'Z': 11, 'mass_amu': 22.990, 'gs': '3s', 'np': '3p',
        'IE_eV': 5.139076,
        'D1_nm': 589.7558, 'D2_nm': 589.1583,
        'tau_D1_ns': 16.299, 'tau_D2_ns': 16.249,
        'fosc_D1': 0.3200,   'fosc_D2': 0.6410,
        'g_upper_D1': 2, 'g_lower_D1': 2,
        'g_upper_D2': 4, 'g_lower_D2': 2,
        'delta_s': 1.3478,   'delta_p': 0.8551,
        'soc_meV': 2.19,
        'source': 'literature',
    },
    'K': {
        'Z': 19, 'mass_amu': 39.098, 'gs': '4s', 'np': '4p',
        'IE_eV': 4.340664,
        'D1_nm': 770.1083, 'D2_nm': 766.7000,
        'tau_D1_ns': 26.72,  'tau_D2_ns': 26.37,
        'fosc_D1': 0.3320,   'fosc_D2': 0.6660,
        'g_upper_D1': 2, 'g_lower_D1': 2,
        'g_upper_D2': 4, 'g_lower_D2': 2,
        'delta_s': 2.1800,   'delta_p': 1.7130,
        'soc_meV': 7.23,
        'source': 'literature',
    },
    'Rb': {
        'Z': 37, 'mass_amu': 85.468, 'gs': '5s', 'np': '5p',
        'IE_eV': 4.177128,
        'D1_nm': 794.9788, 'D2_nm': 780.2414,
        'tau_D1_ns': 27.70,  'tau_D2_ns': 26.24,
        'fosc_D1': 0.3424,   'fosc_D2': 0.6958,
        'g_upper_D1': 2, 'g_lower_D1': 2,
        'g_upper_D2': 4, 'g_lower_D2': 2,
        'delta_s': 3.1311,   'delta_p': 2.6549,
        'soc_meV': 29.43,
        'source': 'literature',
    },
    'Cs': {
        'Z': 55, 'mass_amu': 132.905, 'gs': '6s', 'np': '6p',
        'IE_eV': 3.893905,
        'D1_nm': 894.5930, 'D2_nm': 852.3471,
        'tau_D1_ns': 34.894, 'tau_D2_ns': 30.462,
        'fosc_D1': 0.3435,   'fosc_D2': 0.7142,
        'g_upper_D1': 2, 'g_lower_D1': 2,
        'g_upper_D2': 4, 'g_lower_D2': 2,
        'delta_s': 4.0493,   'delta_p': 3.5915,
        'soc_meV': 68.70,   # 554.04 cm^-1 NIST ASD
        'source': 'literature',
    },
    'Fr': {
        'Z': 87, 'mass_amu': 223.000, 'gs': '7s', 'np': '7p',
        'IE_eV': 4.072741,
        'D1_nm': 817.2,    'D2_nm': 718.2,
        'tau_D1_ns': 29.45, 'tau_D2_ns': 21.02,   # Simsarian 1999
        'fosc_D1': 0.340,   'fosc_D2': 0.736,
        'g_upper_D1': 2, 'g_lower_D1': 2,
        'g_upper_D2': 4, 'g_lower_D2': 2,
        'delta_s': 4.9731,  'delta_p': 4.4965,
        'soc_meV': 209.10,  # 1686.6 cm^-1 NIST ASD / Sansonetti 2007
        'source': 'literature',
    },
}


def dline_transitions(element):
    """Spectral-bar records [{wl, fosc, label, src}] for the two fine-structure D lines of `element`
    (J-resolved f from LITERATURE), or [] if the element is not tabulated."""
    rec = LITERATURE.get(element)
    if not rec:
        return []
    n = rec['gs'].rstrip('s')
    out = []
    for tag, J in (('D2', '3/2'), ('D1', '1/2')):
        out.append({"wl": rec[f'{tag}_nm'], "fosc": rec[f"fosc_{tag}"],
                    "label": f"{rec['np']}{J}\u2192{rec['gs']}", "src": "lit"})
    return out
