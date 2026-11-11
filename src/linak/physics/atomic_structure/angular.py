"""Angular-momentum algebra for relativistic (j-j) one-electron orbitals.

An orbital is labelled by Dirac's kappa:  kappa = -(l+1) for j = l+1/2,
kappa = +l for j = l-1/2.  Wigner 3j symbols take *doubled* arguments so half
integers stay integers (Racah's formula, e.g. Edmonds, Angular Momentum in
Quantum Mechanics, eq. 3.7.3).
"""
from __future__ import annotations

import math

L_LETTERS = "spdfghik"


def kappa_lj(kappa: int) -> tuple[int, int]:
    """kappa -> (l, 2j)."""
    if kappa < 0:
        return -kappa - 1, -2 * kappa - 1
    return kappa, 2 * kappa - 1


def kappas_of_l(l: int) -> list[int]:
    """The kappa values of one l: s has only j=1/2."""
    return [-1] if l == 0 else [l, -l - 1]


def kappa_label(n: int, kappa: int) -> str:
    """LiNaK-style J-resolved label, e.g. (6, -2) -> '6p3/2'."""
    l, j2 = kappa_lj(kappa)
    return f"{n}{L_LETTERS[l]}{j2}/2"


def _f(n2: int) -> float:
    """(n2/2)! for even n2."""
    return float(math.factorial(n2 // 2))


def threej2(j1: int, j2: int, j3: int, m1: int, m2: int, m3: int) -> float:
    """Wigner 3j symbol with every argument doubled."""
    if m1 + m2 + m3 != 0:
        return 0.0
    if not (abs(j1 - j2) <= j3 <= j1 + j2) or (j1 + j2 + j3) % 2:
        return 0.0
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0
    if (j1 + m1) % 2 or (j2 + m2) % 2 or (j3 + m3) % 2:
        return 0.0
    tri = _f(j1 + j2 - j3) * _f(j1 - j2 + j3) * _f(-j1 + j2 + j3) / _f(j1 + j2 + j3 + 2)
    pref = math.sqrt(tri * _f(j1 + m1) * _f(j1 - m1) * _f(j2 + m2) * _f(j2 - m2)
                     * _f(j3 + m3) * _f(j3 - m3))
    total = 0.0
    for t2 in range(0, 2 * (j1 + j2 + j3) + 2, 2):
        args = (t2, j3 - j2 + t2 + m1, j3 - j1 + t2 - m2, j1 + j2 - j3 - t2,
                j1 - t2 - m1, j2 - t2 + m2)
        if min(args) < 0:
            continue
        den = 1.0
        for a in args:
            den *= _f(a)
        total += (-1) ** (t2 // 2) / den
    return (-1) ** ((j1 - j2 - m3) // 2) * pref * total


def reduced_c(kappa_a: int, kappa_b: int, k: int = 1) -> float:
    """<kappa_a || C^k || kappa_b> (W. R. Johnson, Atomic Structure Theory, 2007, ch. 3)
    = (-1)^(ja+1/2) sqrt((2ja+1)(2jb+1)) (ja k jb; -1/2 0 1/2), parity la+k+lb even."""
    la, ja2 = kappa_lj(kappa_a)
    lb, jb2 = kappa_lj(kappa_b)
    if (la + k + lb) % 2:
        return 0.0
    return ((-1) ** ((ja2 + 1) // 2) * math.sqrt((ja2 + 1) * (jb2 + 1))
            * threej2(ja2, 2 * k, jb2, -1, 0, 1))


def exchange_coefficients(kappa_a: int, kappa_b: int) -> list[tuple[int, float]]:
    """[(k, Lambda)] for the exchange of orbital a with a closed kappa_b subshell:
    Lambda = (2jb+1) (ja jb k; -1/2 1/2 0)^2 with la+lb+k even (Johnson, ch. 3)."""
    la, ja2 = kappa_lj(kappa_a)
    lb, jb2 = kappa_lj(kappa_b)
    out = []
    for k in range(abs(ja2 - jb2) // 2, (ja2 + jb2) // 2 + 1):
        if (la + lb + k) % 2:
            continue
        lam = (jb2 + 1) * threej2(ja2, jb2, 2 * k, -1, 1, 0) ** 2
        if lam > 0:
            out.append((k, lam))
    return out


# 6j symbol (doubled arguments)
def _tri(a: int, b: int, c: int) -> float:
    return _f(a + b - c) * _f(a - b + c) * _f(-a + b + c) / _f(a + b + c + 2)


def _ok(a: int, b: int, c: int) -> bool:
    return abs(a - b) <= c <= a + b and (a + b + c) % 2 == 0


def sixj2(j1: int, j2: int, j3: int, j4: int, j5: int, j6: int) -> float:
    """{j1 j2 j3; j4 j5 j6} with doubled arguments (Racah formula)."""
    if not (_ok(j1, j2, j3) and _ok(j1, j5, j6) and _ok(j4, j2, j6) and _ok(j4, j5, j3)):
        return 0.0
    pre = math.sqrt(_tri(j1, j2, j3) * _tri(j1, j5, j6) * _tri(j4, j2, j6) * _tri(j4, j5, j3))
    a1, a2, a3, a4 = j1 + j2 + j3, j1 + j5 + j6, j4 + j2 + j6, j4 + j5 + j3
    b1, b2, b3 = j1 + j2 + j4 + j5, j2 + j3 + j5 + j6, j3 + j1 + j6 + j4
    tot = 0.0
    for t in range(max(a1, a2, a3, a4), min(b1, b2, b3) + 1, 2):
        den = (_f(t - a1) * _f(t - a2) * _f(t - a3) * _f(t - a4)
               * _f(b1 - t) * _f(b2 - t) * _f(b3 - t))
        tot += (-1) ** (t // 2) * _f(t + 2) / den
    return pre * tot
