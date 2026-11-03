"""NIST-series extrapolation: robust outlier retry, curated exclusions, dropped-by-family summary."""
from types import SimpleNamespace

from linak.physics import lifetimes_core as lc

QD = {'p': 2.0}


def _rows(ys, lower='5s', ns=range(10, 20)):
    # f n*^3 = a + b/n*^2 with a=1, b=2; ys overrides single points
    out = []
    for n in ns:
        ns_ = n - QD['p']
        y = ys.get(n, 1.0 + 2.0 / ns_**2)
        out.append(SimpleNamespace(fosc_source='NIST', lower_label=lower, upper_l='p',
                                   upper_n=n, upper_label=f'{n}p3/2', fosc=y / ns_**3,
                                   fosc_delta=0.1))
    return out


def test_clean_series_accepted_without_drops():
    fits = lc._build_extrap_fits({'5s'}, _rows({}), QD)
    assert fits[('5s', 'p', '3/2')]['excluded'] == []


def test_single_bad_point_is_dropped_not_fatal():
    rows = _rows({17: 3.0})                         # one wildly wrong NIST entry in the tail
    fits = lc._build_extrap_fits({'5s'}, rows, QD)
    fit = fits[('5s', 'p', '3/2')]
    assert fit['excluded'] == [17]
    assert abs(fit['c'][0] - 1.0) < 1e-6 and abs(fit['c'][1] - 2.0) < 1e-6


def test_curated_exclusion_removes_point_from_fit_only():
    rows = _rows({17: 3.0})
    excl = [{'lower': '5s', 'upper_l': 'p', 'J': '3/2', 'n': [17], 'ref': 'test'}]
    fits = lc._build_extrap_fits({'5s'}, rows, QD, excl)
    assert fits[('5s', 'p', '3/2')]['excluded'] == []
    assert rows[7].fosc * (17 - 2.0)**3 == 3.0     # the NIST row itself is untouched
