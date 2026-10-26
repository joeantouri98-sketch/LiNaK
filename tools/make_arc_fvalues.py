#!/usr/bin/env python3
"""
Generate f_values/<El>_literature.json: ARC model-potential ground-state ns -> np absorption f
(J-resolved), for the np states where NIST lists no f, plus an uncertainty caveat on the NIST
series tail where it departs from the smooth ARC series.

Requires ARC (pip install ARC-Alkali-Rydberg-Calculator). Run from the repo root:
    python tools/make_arc_fvalues.py Li            # writes f_values/Li_literature.json
    python tools/make_arc_fvalues.py Na --check    # compare with the existing Na file

Only elements whose ARC series is validated are enabled (see ARC_OK).
Li, Na: ground-state ns -> np (and for Li 2p -> ns/nd) where NIST lists no f.
K, Rb, Cs: first-excited np -> ns/nd ('scaled_p' mode). Published reduced matrix elements
(f_values/<El>_published_anchors.json) are written as PUBLISHED rows and the ARC model potential
(useLiterature=False, i.e. NOT ARC's bundled literature values) is scaled by the mean published/model
ratio of the highest published n and used beyond the last published n. The model reproduces the
published values to 1-4 % (ns), 1-7 % (Rb/Cs nd) and 15 % after scaling (K nd, strong cancellation).
The ground-state ns -> np of K, Rb, Cs is NOT generated: the model potential misses the Cooper-minimum
suppression (3-100x above NIST).
Francium is not in ARC.
"""
import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PREF = 6.6703e13   # A = PREF/lambda_nm^2 * (g_l/g_u) * f   (constants.EINSTEIN_PREFACTOR)
ARC_OK = {
    'Na': dict(cls='Sodium', n0=3, nmax=50,
               caveat=dict(n_min=12, rel_unc=0.4,
                           ref="Theodosiou, arXiv:2212.00577 (2022): NBS/NIST 3s->np deviates from the smooth "
                               "df/dE curve for n>11 and carries 25-50% error; NIST/ARC = 1.3-1.45 at n=12-18, "
                               "0.8 at n=8-10")),
    'Li': dict(cls='Lithium7', n0=2, nmax=50, p_series=True,
               caveat=dict(n_min=10, rel_unc=0.5,
                           ref="NIST 2s->np f*n*^3 jumps by 48% between n=9 and n=10 (0.233 -> 0.344) while the "
                               "ARC model-potential series is smooth (0.2326 -> 0.2345); NIST/ARC = 1.46-1.63 at "
                               "n=10-13, 0.97-1.05 at n<=9 (this repository's NIST/ARC comparison)")),
}
    # np -> ns/nd scaled to published matrix elements; floor = minimum relative uncertainty of the scaled tail
SCALED = {
    'K':  dict(cls='Potassium',   n0=4, nmax=50, floor=dict(s=0.05, d=0.12)),
    'Rb': dict(cls='Rubidium87',  n0=5, nmax=50, floor=dict(s=0.05, d=0.08)),
    'Cs': dict(cls='Caesium',     n0=6, nmax=50, floor=dict(s=0.05, d=0.08)),
}
for _k, _v in SCALED.items():
    ARC_OK[_k] = dict(_v, scaled_p=True)
REFTXT_PUB = ("PUBLISHED: {src}. Multiplet-averaged absorption f from the J-resolved reduced matrix elements, "
              "{lo}->{up}; NIST lists no f for this transition")
REFTXT_SC = ("ARC 3.10 model potential (useLiterature=False; Sibalic et al., CPC 220, 319 (2017)) scaled by "
             "published/model = {R:.3f} (mean of {anch}), multiplet-averaged absorption f, {lo}->{up}; "
             "NIST lists no f and the published tables end at n={npub}")
REFTXT_P = ("ARC 3.10 model-potential (Sibalic et al., CPC 220, 319 (2017); Marinescu 1994) multiplet-averaged "
            "absorption f, {lo}->n{ul}; NIST lists no f for n>={nfirst}. Smooth series; NIST/ARC = 0.95-1.05 "
            "where both exist. Fills the channels the n*<=2.5 Coulomb gate would otherwise drop")
REFTXT = ("ARC 3.10 model-potential (Sibalic et al., CPC 220, 319 (2017); Marinescu 1994) absorption f, "
          "{gs}->np high n; NIST lists no f for n>={nfirst}. Smooth f*n*^3 series (NIST/ARC = 1.0 for "
          "n<={nsafe})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('element')
    ap.add_argument('--check', action='store_true', help='compare with the existing file instead of writing')
    ap.add_argument('--f-values-dir', default='f_values')
    ap.add_argument('--refresh-caveats', action='store_true',
                    help="replace this tool's own nist_series_caveats entry with the built-in text (default: keep the file's)")
    a = ap.parse_args()
    el = a.element
    if el not in ARC_OK:
        sys.exit(f"{el}: ARC series not validated against NIST (see the module docstring); no file written.")
    cfg = ARC_OK[el]
    import arc
    import alpha_core as ac
    if cfg.get('scaled_p'):
        return main_scaled(a, el, cfg, arc, ac)
    atom = getattr(arc, cfg['cls'])()
    n0 = cfg['n0']
    gs = f'{n0}s'
    nist = ac.load_nist_fvalues(el, a.f_values_dir, verbose=False)
    listed = sorted(int(''.join(c for c in up if c.isdigit())) for (lo, up) in nist['fosc']
                    if lo == gs and up.endswith('p'))
    nfirst = max(listed) + 1
    rows = []
    for n in range(nfirst, cfg['nmax'] + 1):
        for j, jk in ((0.5, '1/2'), (1.5, '3/2')):
            A = atom.getTransitionRate(n, 1, j, n0, 0, 0.5, temperature=0)
            lam = abs(atom.getTransitionWavelength(n0, 0, 0.5, n, 1, j)) * 1e9
            f = A * lam ** 2 * (2 * j + 1) / (PREF * 2)
            if f > 0:
                rows.append(dict(lower=gs, upper=f'{n}p', upper_J=jk, f_abs=float(f'{f:.5g}'), acc='C+',
                                 ref=REFTXT.format(gs=gs, nfirst=nfirst, nsafe=cfg['caveat']['n_min'] - 1)))
    # first-excited np -> ns / nd multiplet rows (absorption f averaged over the lower fine-structure
    # multiplet), only for upper states where NIST lists no f at all. Enabled only where the NIST series is
    # too short for the NIST-extrapolation (Li: 2p has n*<=2.5 and 6 NIST points); for Na the extrapolation
    # of the real NIST 3p->ns/nd series is the better source and is kept
    LET = {0: 's', 1: 'p', 2: 'd'}
    lo_p = f'{n0}p'
    def js(l):
        return [0.5] if l == 0 else [l - 0.5, l + 0.5]
    for lu in ((0, 2) if cfg.get('p_series') else ()):
        listed_u = {int(''.join(c for c in up if c.isdigit())) for (lo, up) in nist['fosc']
                    if lo == lo_p and up.endswith(LET[lu])}
        nfirst_u = max(listed_u) + 1
        for n in range(nfirst_u, cfg['nmax'] + 1):
            num = 0.0
            ok = True
            for jl in js(1):
                for ju in js(lu):
                    if abs(ju - jl) > 1:
                        continue
                    try:
                        A = atom.getTransitionRate(n, lu, ju, n0, 1, jl, temperature=0)
                        lam = abs(atom.getTransitionWavelength(n0, 1, jl, n, lu, ju)) * 1e9
                    except Exception:
                        ok = False
                        break
                    num += A * lam ** 2 * (2 * ju + 1) / PREF
            if ok and num > 0:
                f = num / 6.0          # sum over the lower multiplet of g_l*f / sum g_l (=6)
                rows.append(dict(lower=lo_p, upper=f'{n}{LET[lu]}', f_abs=float(f'{f:.5g}'), acc='C+',
                                 ref=REFTXT_P.format(lo=lo_p, ul=LET[lu], nfirst=nfirst_u)))
    # never overwrite anything that is not this generator's own ARC rows: curated rows, the note, hand-edited
    # caveats and any other top-level key are kept as they are in the existing file
    path = os.path.join(a.f_values_dir, f'{el}_literature.json')
    old = json.load(open(path, encoding='utf-8')) if os.path.exists(path) else {}
    kept = [t for t in old.get('transitions', []) if not str(t.get('ref', '')).startswith('ARC ')]
    keys_kept = {(t['lower'], t['upper'], t.get('upper_J')) for t in kept}
    rows = kept + [t for t in rows if (t['lower'], t['upper'], t.get('upper_J')) not in keys_kept]

    mine = dict(lower=gs, upper_l='p', **cfg['caveat'])
    cav = list(old.get('nist_series_caveats', []))
    idx = [i for i, c in enumerate(cav) if (c.get('lower'), c.get('upper_l')) == (mine['lower'], mine['upper_l'])]
    if not idx:
        cav.append(mine)
    elif a.refresh_caveats:
        cav[idx[0]] = mine
    cav_drift = bool(idx) and cav[idx[0]] != mine
    out = dict(old)                                    # keeps unknown keys
    out.update(
        element=el,
        note=old.get('note') or ("Curated absorption oscillator strengths preferred over Coulomb/Numerov where "
                                 "NIST lists no f. Rows citing 'ARC' are independent model-potential calculations "
                                 "used only where NIST lists no f; NIST values are never overwritten. The 'literature' tier "
                                 "mixes published values (NIST-ASD-cited or lifetime-derived rows) with independent ARC "
                                 "model-potential calculations; ARC rows carry a ref starting 'ARC ' and are computed "
                                 "estimates, not measurements."),
        generated_with=dict(tool='tools/make_arc_fvalues.py', arc_version=getattr(arc, '__version__', 'unknown'),
                            note='rows whose ref starts with "ARC " are regenerated by the tool; all other rows, '
                                 'the note and existing caveats are never modified by it'),
        transitions=rows,
        nist_series_caveats=cav)
    if a.check:
        oldv = {(t['upper'], t.get('upper_J')): t['f_abs'] for t in old.get('transitions', [])
                if str(t.get('ref', '')).startswith('ARC')}
        worst = 0.0
        for t in [r for r in rows if 'upper_J' in r]:
            k = (t['upper'], t['upper_J'])
            if k in oldv:
                worst = max(worst, abs(t['f_abs'] / oldv[k] - 1))
        print(f'{el}: {len(rows)} rows, {sum((t["upper"], t["upper_J"]) in oldv for t in rows if "upper_J" in t)} '
              f'match the file, worst rel diff {worst:.2e}; ARC {getattr(arc, "__version__", "?")} '
              f'(file records {old.get("generated_with", {}).get("arc_version", "none")}); '
              f'caveat differs from the tool default: {cav_drift}')
        return
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:       # atomic: a failed run never leaves a half-written file
        json.dump(out, fh, indent=1, ensure_ascii=False)
    os.replace(tmp, path)
    print(f'wrote {path}: {len(rows)} rows (n = {nfirst}..{cfg["nmax"]})'
          + ('; kept existing caveat (differs from tool default; --refresh-caveats to replace)' if cav_drift and not a.refresh_caveats else ''))


def _acc(u):
    return 'B+' if u <= 0.07 else 'B' if u <= 0.10 else 'C+' if u <= 0.18 else 'C'


def main_scaled(a, el, cfg, arc, ac):
    import functools
    import numpy as np
    n0 = cfg['n0']
    lo_p = f'{n0}p'
    atom = getattr(arc, cfg['cls'])()
    atom.getRadialMatrixElement = functools.partial(atom.getRadialMatrixElement, useLiterature=False)
    nist = ac.load_nist_fvalues(el, a.f_values_dir, verbose=False)
    apath = os.path.join(a.f_values_dir, f'{el}_published_anchors.json')
    anchors = json.load(open(apath, encoding='utf-8'))['transitions']
    HA_EV = 27.211386245988
    LET = {0: 's', 2: 'd'}
    PJ = {'1/2': 0.5, '3/2': 1.5, '5/2': 2.5}

    def parse(lab):
        m = __import__('re').match(r'(\d+)([spd])(\d/2)?$', lab)
        return int(m.group(1)), 'spd'.index(m.group(2)), PJ[m.group(3) or '1/2']

    def f_model(n, lu):
        num = 0.0
        for jl in (0.5, 1.5):
            for ju in ((0.5,) if lu == 0 else (1.5, 2.5)):
                if abs(ju - jl) > 1 or (lu == 2 and ju < 1.5):
                    pass
                if abs(ju - jl) > 1:
                    continue
                A = atom.getTransitionRate(n, lu, ju, n0, 1, jl, temperature=0)
                lam = abs(atom.getTransitionWavelength(n0, 1, jl, n, lu, ju)) * 1e9
                num += A * lam ** 2 * (2 * ju + 1) / PREF
        return num / 6.0

    def need(lu):
        return [(jl, ju) for jl in (0.5, 1.5) for ju in ((0.5,) if lu == 0 else (1.5, 2.5)) if abs(ju - jl) <= 1]

    pub = {}          # (n, lu) -> [num, set(J pairs), refs]
    for t in anchors:
        _, _, jl = parse(t['lower'])
        n, lu, ju = parse(t['upper'])
        lam = abs(atom.getTransitionWavelength(n0, 1, jl, n, lu, ju))
        E = 6.62607015e-34 * 299792458 / lam / 1.602176634e-19 / HA_EV
        f = 2.0 / 3.0 * E * t['D'] ** 2 / (2 * jl + 1)
        d = pub.setdefault((n, lu), dict(num=0.0, pairs=set(), refs=set()))
        d['num'] += (2 * jl + 1) * f
        d['pairs'].add((jl, ju))
        d['refs'].add(t['ref'])
    fpub = {k: v['num'] / 6.0 for k, v in pub.items() if v['pairs'] == set(need(k[1]))}
    rows, diag = [], {}
    for lu in (0, 2):
        let = LET[lu]
        listed = {int(''.join(c for c in up if c.isdigit())) for (lo, up) in nist['fosc']
                  if lo == lo_p and up.endswith(let)}
        pn = sorted(n for (n, l) in fpub if l == lu and n >= n0)
        if not pn:
            continue
        ratios = {n: fpub[(n, lu)] / f_model(n, lu) for n in pn}
        cand = [n for n in pn if n >= n0 + 2] or pn      # skip n close to n0: strongest cancellation / unscaled SD values
        sel = cand[-3:]
        Rm = float(np.mean([ratios[n] for n in sel]))
        spread = float(np.std([ratios[n] for n in sel])) / Rm if len(sel) > 1 else 0.0
        u_sc = (spread ** 2 + cfg['floor'][let] ** 2) ** 0.5
        diag[let] = dict(ratios={n: round(ratios[n], 3) for n in pn}, R=round(Rm, 4), n_pub_max=pn[-1],
                         rel_unc=round(u_sc, 3))
        for (n, l), f in sorted(fpub.items()):
            if l == lu and n not in listed and n > n0 - 1:
                src = '; '.join(sorted(pub[(n, l)]['refs']))
                rows.append(dict(lower=lo_p, upper=f'{n}{let}', f_abs=float(f'{f:.5g}'), acc='B',
                                 ref=REFTXT_PUB.format(src=src, lo=lo_p, up=f'{n}{let}')))
        for n in range(pn[-1] + 1, cfg['nmax'] + 1):
            if n in listed:
                continue
            f = Rm * f_model(n, lu)
            if f > 0:
                rows.append(dict(lower=lo_p, upper=f'{n}{let}', f_abs=float(f'{f:.5g}'), acc=_acc(u_sc),
                                 ref=REFTXT_SC.format(R=Rm, anch=', '.join(f'{m}{let}' for m in sel),
                                                      lo=lo_p, up=f'{n}{let}', npub=pn[-1])))
    # published J-resolved f (f_values/<El>_published_f.json), e.g. K 4s->np: only where NIST lists no f
    pf_path = os.path.join(a.f_values_dir, f'{el}_published_f.json')
    if os.path.exists(pf_path):
        pf = json.load(open(pf_path, encoding='utf-8'))
        nj = {(lo, up, str(J)) for (lo, up, J) in nist.get('fosc_j', {})}
        nm = set(nist['fosc'])
        n_pf = 0
        for t in pf['transitions']:
            if (t['lower'], t['upper'], str(t['upper_J'])) in nj or (t['lower'], t['upper']) in nm:
                continue
            if t['lower'] == f'{n0}s' and int(''.join(c for c in t['upper'] if c.isdigit())) <= n0:
                continue
            rows.append(dict(lower=t['lower'], upper=t['upper'], upper_J=t['upper_J'], f_abs=float(f"{t['f_abs']:.5g}"),
                             acc=t.get('acc', 'C'), ref=pf['ref']))
            n_pf += 1
        print(f'{el}: {n_pf} published J-resolved rows from {pf_path}')
    path = os.path.join(a.f_values_dir, f'{el}_literature.json')
    old = json.load(open(path, encoding='utf-8')) if os.path.exists(path) else {}
    mine = ('ARC ', 'PUBLISHED')
    kept = [t for t in old.get('transitions', []) if not str(t.get('ref', '')).startswith(mine)]
    keys_kept = {(t['lower'], t['upper'], t.get('upper_J')) for t in kept}
    rows = kept + [t for t in rows if (t['lower'], t['upper'], t.get('upper_J')) not in keys_kept]
    out = dict(old)
    out.update(
        element=el,
        note=(old.get('note') or '') + ('' if 'scaled' in (old.get('note') or '') else
              " Rows citing 'PUBLISHED' are transcribed published matrix elements (f_values/<El>_published_anchors.json); "
              "rows citing 'ARC' are the ARC model potential scaled to them. Both are used only where NIST lists no f."),
        generated_with=dict(tool='tools/make_arc_fvalues.py', arc_version=getattr(arc, '__version__', 'unknown'),
                            mode='scaled_p', diagnostics=diag,
                            note='rows whose ref starts with "ARC " or "PUBLISHED" are regenerated by the tool; all other '
                                 'rows, the note, caveats and fit exclusions are never modified by it'),
        transitions=rows)
    for let, d in diag.items():
        print(f"{el} {lo_p}->n{let}: model->published ratios {d['ratios']}; scale {d['R']}; "
              f"rel. unc {d['rel_unc']}; tail from n={d['n_pub_max'] + 1}")
    if a.check:
        print(f'{el}: {len(rows)} rows would be written (check only)')
        return
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    os.replace(tmp, path)
    print(f'wrote {path}: {len(rows)} rows')


if __name__ == '__main__':
    main()
