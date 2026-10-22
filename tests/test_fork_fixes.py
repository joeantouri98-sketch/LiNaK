# -*- coding: utf-8 -*-
"""Regression tests for the fork-review fixes (P18, P28, P33, P39-P45)."""
from __future__ import annotations

import json
import math
import os
import shutil

import numpy as np
import pytest

import hyperfine as hf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── P41: nuclear-data parsing ────────────────────────────────────────────────
@pytest.mark.parametrize('txt,val', [('22.5 ms', 0.0225), ('4.9 m', 294.0), ('3.18 m', 190.8),
                                     ('146 ns', 1.46e-7), ('2 h', 7200.0), ('', 0.0)])
def test_halflife_units(txt, val):
    assert hf.halflife_s(txt) == pytest.approx(val)


def test_halflife_stable():
    assert math.isinf(hf.halflife_s('stable'))


@pytest.mark.parametrize('txt,val', [('3/2(-)', 1.5), ('1(-)', 1.0), ('(3+)', 3.0), ('9/2-', 4.5), ('5+', 5.0)])
def test_spin_parsing(txt, val):
    assert hf.parse_spin(txt) == val


def test_bundled_nuclear_data_fr():
    d = json.load(open(os.path.join(ROOT, 'data_json', 'nuclear_data.json')))
    fr = {i['A']: i for i in d['isotopes'] if i['Z'] == 87}
    assert fr[219]['halflife_s'] == pytest.approx(0.0225)     # was 1350 s (ms read as minutes)
    assert fr[223]['I'] == 1.5 and fr[224]['I'] == 1.0        # spins written "3/2(-)", "1(-)"


# ── P45: Breit-Rabi labels with A = 0, colour clamp ──────────────────────────
def test_breit_rabi_labels_degenerate_levels():
    _, labels = hf.breit_rabi_energies(1.5, 0.5, 0.0, 0.0, 2.0, -0.001, np.array([0.0, 5.0]))
    assert all(abs(m) <= F + 1e-9 for F, m in labels)
    assert sorted(labels) == sorted(set(labels))
    assert sum(1 for F, _ in labels if F == 1.0) == 3 and sum(1 for F, _ in labels if F == 2.0) == 5


def test_mF_color_never_invalid():
    for m, F in [(4, 1), (-4, 1), (0, 0), (1, 2)]:
        rgb = hf.mF_color(m, F)
        vals = [int(x) for x in rgb[4:-1].split(',')] if rgb.startswith('rgb') else [0]
        assert all(0 <= v <= 255 for v in vals)


# ── P40: clock transition comes from the lowest state, whatever the order ─────
def test_clock_state_independent_of_state_order(tmp_path, monkeypatch):
    shutil.copytree(os.path.join(ROOT, 'data_json'), tmp_path / 'data_json')
    monkeypatch.chdir(tmp_path)
    res = []
    for order in (['3s1/2', '3p3/2'], ['3p3/2', '3s1/2']):
        r = hf.run_element('Na', state_labels=order)
        res.append(r['special']['microwave_clock'])
    assert res[0]['state'] == res[1]['state'] == '3s1/2'
    assert res[0]['frequency_MHz'] == pytest.approx(res[1]['frequency_MHz'])
    assert res[0]['frequency_MHz'] == pytest.approx(1771.6, abs=1.0)


def test_clock_wavelength_uses_exact_c():
    sp = hf.identify_special_transitions([], [1.0, 2.0], 1.5, 0.5, 885.8, 0.0, 0.0)
    mc = sp['microwave_clock']
    assert mc['wavelength_m'] == pytest.approx(299792458.0 / mc['frequency_Hz'], rel=1e-12)


# ── P43: GAMESS L (SP) shells ────────────────────────────────────────────────
def test_gamess_L_shell_becomes_S_and_P():
    import orca_templates as ot
    txt = "$DATA\nLITHIUM\nS   2\n1 10.0 0.5\n2 2.0 0.6\nL   2\n1 0.9 0.1 0.2\n2 0.3 0.4 0.5\nD 1\n1 0.2 1.0\n$END"
    out = ot._basis_block_from_gamess(txt, 'Li')
    kinds = [ln.split()[0] for ln in out.splitlines() if ln.strip()[:1] in 'SPD' and len(ln.split()) == 2]
    assert kinds == ['S', 'S', 'P', 'D']
    assert '0.2000000000' in out and '0.5000000000' in out     # the p coefficients


# ── P18: one D-line table ────────────────────────────────────────────────────
def test_dline_tables_share_one_source():
    from dline_literature import LITERATURE, dline_transitions
    for el, rec in LITERATURE.items():
        tr = {t['label'].split('\u2192')[0]: t for t in dline_transitions(el)}
        assert tr[rec['np'] + '3/2']['fosc'] == rec['fosc_D2']
        assert tr[rec['np'] + '1/2']['fosc'] == rec['fosc_D1']
    for fname in ('spectra.py', 'compare_elements.py'):
        src = open(os.path.join(ROOT, fname), encoding='utf-8').read()
        assert 'dline_literature' in src
    assert '"fosc": 0.' not in open(os.path.join(ROOT, 'spectra.py'), encoding='utf-8').read().split('SRC_COLORS')[0]


# ── P28: non-LS downward channel carries g_lower/g_upper ─────────────────────
def test_non_ls_downward_channel_has_g_ratio():
    import polarizability as p
    d = p.load_element('Na', os.path.join(ROOT, 'data_json'))
    d['ls_mode'] = False
    tr = p.build_transitions_for_state('3p3/2', d, d['transitions'])
    down = [t for t in tr if t['other_label'] == '3s' and t['sign'] == -1]
    assert down and down[0]['fosc'] == pytest.approx(0.6408 * 2 / 4, rel=5e-3)


# ── P39 / P44: species without a QDT scheme ──────────────────────────────────
def test_transition_metal_rejected_for_alpha():
    import alpha_core as ac
    with pytest.raises(SystemExit):
        ac.require_alpha_species('Fe', 'test')
    assert ac.require_alpha_species('Na', 'test')['species_id'] == 'Na'


def test_nist_only_rydberg_json_selects_line_list_path(tmp_path, monkeypatch):
    import alpha_core as ac
    (tmp_path / 'data_json').mkdir()
    monkeypatch.chdir(tmp_path)
    assert not ac.is_transition_metal_species('Na')
    (tmp_path / 'data_json' / 'Na_rydberg.json').write_text(json.dumps({'method': 'NIST ASD levels (no QDT)'}))
    assert ac.is_transition_metal_species('Na')
    assert 'Fe' not in ac.get_available_elements(str(tmp_path / 'data_json'))


# ── P42: GUI roots field is blank by default ─────────────────────────────────
def test_gui_nroots_default_blank():
    src = open(os.path.join(ROOT, 'linak_gui.py'), encoding='utf-8').read()
    i = src.index('name="--nroots"')
    assert 'default=""' in src[i:i + 120]


# ── P19/P20/P34/P35/P36/P37/P38: plotting-script fixes (source-level; the script runs at import) ──
def _src(name):
    return open(os.path.join(ROOT, name), encoding='utf-8').read()


def test_plotinteractive_fixes_present():
    s = _src('plotinteractive.py')
    assert 'np.sqrt(KB_EV * T_K / _mc2_sp)' in s                 # P34: sigma, not FWHM
    assert 'Math.sqrt(kB * T_K / mc2)' in s or 'Math.sqrt(kB*T_K/mc2)' in s and '{KB_EV!r}' in s  # P34/P35: JS sigma with injected kB
    assert '_natural_gamma_nm(r.get' in s                         # P20
    assert 'def ionization_energy_eV' in s                    # P37
    assert '_REGION_COLORS4 = REGION_COLORS' in s             # P19
    assert "_ee_nxt" in s and "'NIST-extrap': 'dash'" in s   # P38 (plot 3)


def test_spectra_coulomb_approx_colour_distinct():
    from importlib import util  # noqa: F401
    s = _src('spectra.py')
    assert '"Coulomb(approx)": "#52be80"' in s               # P36


# ── P22: peak cross-section = Lorentzian peak (lambda^2/2pi)(g_u/g_l)(A_dom/A_tot) ───────────
def test_peak_cross_section_lorentzian_convention():
    d = json.load(open(os.path.join(ROOT, 'data_json', 'Na_lifetimes.json')))['lifetimes']
    for e in d[:20]:
        lam = e['dominant_wl_nm'] * 1e-9
        ratio = e['dominant_A_s'] / e['A_total_s']
        # sigma_nat / (lambda^2/2pi * A_dom/A_tot) must be the integer-ish g ratio g_u/g_l
        g = e['sigma_peak_nat_m2'] / (lam ** 2 / (2 * math.pi) * ratio)
        assert g > 0 and min(abs(g * k - round(g * k)) for k in range(1, 15)) < 1e-6   # a ratio of level degeneracies
        # Doppler peak: sigma_nat * (dnu_nat/dnu_D) * sqrt(pi ln2)
        want = e['sigma_peak_nat_m2'] * e['delta_nu_nat_Hz'] / e['delta_nu_dop_Hz'] * math.sqrt(math.pi * math.log(2))
        assert e['sigma_peak_dop_m2'] == pytest.approx(want, rel=1e-9)
        # I_sat = h nu / (2 sigma tau)
        hnu = 6.62607015e-34 * 299792458.0 / lam
        assert e['I_sat_nat_W_m2'] == pytest.approx(hnu / (2 * e['sigma_peak_nat_m2'] * e['lifetime_s']), rel=1e-6)


# ── P26: Coulomb f hybrid (Whittaker <= n*=12, Numerov above) ────────────────
@pytest.mark.parametrize('args,exact', [((2, 1, 1, 0), 0.4161967), ((3, 2, 2, 1), 0.6957847),
                                        ((4, 3, 3, 2), 1.0175189), ((3, 1, 2, 0), 0.4348654)])
def test_coulomb_fosc_hydrogen_limit_low_n(args, exact):
    import alpha_core as ac
    assert ac.coulomb_fosc(*args, 0.0, 0.0) == pytest.approx(exact, rel=1e-5)


@pytest.mark.parametrize('args,exact', [((30, 2, 29, 1), 1.658130), ((60, 2, 59, 1), 3.058680),
                                        ((100, 2, 99, 1), 4.934668), ((150, 1, 5, 0), 3.7547e-6)])
def test_coulomb_fosc_hydrogen_limit_high_n(args, exact):
    import alpha_core as ac
    assert ac.coulomb_fosc(*args, 0.0, 0.0) == pytest.approx(exact, rel=0.02)   # Numerov, log-space start


def test_coulomb_fosc_alkali_close_to_nist():
    import alpha_core as ac
    nist = ac.load_nist_fvalues('Na', os.path.join(ROOT, 'f_values'), verbose=False)['fosc']
    q = json.load(open(os.path.join(ROOT, 'data_json', 'Na_rydberg.json')))['quantum_defects']
    r = []
    for (lo, up), f in nist.items():
        if (lo[-1], up[-1]) != ('p', 'd') or int(lo[:-1]) > 12 or int(up[:-1]) > 12:
            continue
        r.append(ac.coulomb_fosc(int(up[:-1]), 2, int(lo[:-1]), 1, q['d'], q['p']) / f)
    assert len(r) > 10 and 0.8 < float(np.median(r)) < 1.1        # the old formula gave 0.66 (p->d), 15 (s->p)


def test_tail_partner_interpolation_matches_exact():
    import blackbody as bb
    import alpha_core as ac
    import polarizability as p
    data = p.load_element('Na', os.path.join(ROOT, 'data_json'))
    tail = bb.discrete_tail_partners('40d', data, 51, 200)
    assert tail
    for x in tail[::53]:
        n = int(''.join(c for c in x['other_label'] if c.isdigit())); lc = x['other_label'][-1]
        ex = ac.coulomb_fosc(n, bb.L_CHAR[lc], 40, 2, bb._qd_delta(data['qd'], lc), bb._qd_delta(data['qd'], 'd'))
        assert x['fosc'] == pytest.approx(ex, rel=0.03)
