"""Atomic structure engines for one-valence atoms and ions (no files, prints or plots).

``dhf_cp``       Dirac-Hartree-Fock frozen core + fitted core-polarisation potential
``dhf_sigma``    second-order correlation potential Sigma(2)(E), Brueckner levels
``dhf_rpa``      RPA core response: dressed E1 operator, core polarisability
``brueckner``    Brueckner-orbital levels + RPA E1 network (the ``brueckner`` stage)
``dhf_lines``    a model's E1 network as an f-value source for other stages
``dirac_basis``  dual-kinetic-balance B-spline Dirac basis and quadrature
``angular``      j-j angular-momentum algebra
"""
