"""
A structure model's E1 network as an oscillator-strength source.

The ``dhf`` (DHF + core polarisation) and ``brueckner`` stages store J-resolved reduced dipoles
<a||D||b> (``data/<El>_dhf.json``, ``data/<El>_brueckner.json``; ``source`` says which).
Other stages use them through :class:`DHFLines`, combining the ab initio dipole with
the line's measured (NIST) energy: f = 2 dE |D|^2 / (3 g_lower), the usual way of
pairing theoretical matrix elements with experimental energies.

Levels without J (an unresolved multiplet, or NIST's merged '5/2,7/2') are handled
statistically, as elsewhere in Qali: absorption f averaged over the lower group,
summed over the upper group,

    f(group_l -> group_u) = 2 dE / (3 g_L) * sum_{jl, ju} |<jl||D||ju>|^2,
    g_L = sum_jl (2 jl + 1),

so A = PREF / lambda^2 * (g_L / g_U) * f is the group rate the lifetimes and
polarizability code already use. Pure: no files, no printing.

Two kinds of line are never supplied (the caller falls back to its other sources):

* cancellation < MIN_CANCELLATION: the radial integral is a small difference of
  large parts (lines near a Cooper or Rydberg-Rydberg zero, f ~ 1e-7..1e-9), where
  numerics and missing correlation dominate the small result;
* ground ns -> np beyond the resonance doublet: the Cooper-minimum series. Against
  the validated extrapolation model tables (all-order checked), DHF is -15% (Na),
  -5..-17% (K), -11% (Rb), +15..+23% (Cs p3/2) and x1.5..x12 (Cs p1/2, n = 12..30)
  in f; that series belongs to NIST and the extrapolator. The resonance lines
  themselves are accurate (Cs 6s-6p D within 0.2% of all-order).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

# Relative 1-sigma given to a DHF f-value. D-line f agree with lifetime-derived
# precision values to 0.4% and alpha(0) to 1.3% (Na-Fr); weaker and higher lines
# are less tested, so a uniform 5% is used.
DHF_REL_UNCERTAINTY = 0.05
MIN_CANCELLATION = 0.05

_LABEL = re.compile(r'^(\d+)([spdfgh])(\d+)/2$')


def _l_of(nl: str) -> int:
    return 'spdfgh'.index(nl[-1])


@dataclass(frozen=True)
class DHFLines:
    """|<a||D||b>| (atomic units) for every computed E1 pair, keyed in both orders."""

    element: str
    dipoles: Mapping[tuple[str, int, str, int], float]   # (nl_a, 2j_a, nl_b, 2j_b) -> |D|
    levels: frozenset                                     # {(nl, 2j)} the model computed
    ground: str = ''                                      # valence nl, e.g. '3s'
    rejected: frozenset = frozenset()                     # pair keys (both orders) not supplied
    source: str = 'DHF'                                   # 'DHF' or 'Brueckner': the f-value source label

    @classmethod
    def from_payload(cls, payload: Mapping) -> DHFLines:
        """From the parsed ``<El>_dhf.json``."""
        dip: dict = {}
        lv: set = set()
        rejected: set = set()
        g = _LABEL.match(str(payload.get('valence', '')))
        ground = g.group(1) + g.group(2) if g else ''
        n_ground = int(g.group(1)) if g else 0
        for t in payload.get('transitions') or []:
            a, b = _LABEL.match(t['lower']), _LABEL.match(t['upper'])
            if not (a and b):
                continue
            ka = (a.group(1) + a.group(2), int(a.group(3)))
            kb = (b.group(1) + b.group(2), int(b.group(3)))
            d = abs(float(t['reduced_dipole_au']))
            dip[(*ka, *kb)] = d
            dip[(*kb, *ka)] = d
            cooper = ground and {ka[0], kb[0]} & {ground} and {a.group(2), b.group(2)} == {'s', 'p'} \
                and max(int(a.group(1)), int(b.group(1))) > n_ground
            if float(t.get('cancellation', 1.0)) < MIN_CANCELLATION or cooper:
                rejected.update({(*ka, *kb), (*kb, *ka)})
        for level in payload.get('levels') or []:
            m = _LABEL.match(level['label'])
            if m:
                lv.add((m.group(1) + m.group(2), int(m.group(3))))
        source = 'Brueckner' if payload.get('model') == 'brueckner' else 'DHF'
        return cls(str(payload.get('element', '')), dip, frozenset(lv), ground, frozenset(rejected), source)

    def f_absorption(self, lo_nl: str, up_nl: str, JL2: Sequence[int], JU2: Sequence[int],
                     dE_au: float) -> float | None:
        """
        Absorption f of lower group (nl, J set) -> upper group at the energy ``dE_au``;
        None when the model does not cover every allowed (|dJ| <= 1) pair.
        """
        if dE_au <= 0 or not JL2 or not JU2 or abs(_l_of(lo_nl) - _l_of(up_nl)) != 1:
            return None
        total = 0.0
        for jl in JL2:
            if (lo_nl, jl) not in self.levels:
                return None
            for ju in JU2:
                if abs(ju - jl) > 2:
                    continue          # forbidden: |dJ| > 1
                key = (lo_nl, jl, up_nl, ju)
                d = self.dipoles.get(key)
                if d is None or key in self.rejected:
                    return None
                total += d * d
        if total <= 0:
            return None
        gL = sum(j + 1 for j in JL2)
        return 2.0 * dE_au * total / (3.0 * gL)
