import numpy as np, math
from sympy.physics.wigner import wigner_6j, wigner_3j
from sympy import Rational as Q
def ang(l,j,l2,j2):
    """|<l j||C^1... D||l2 j2>|/R  (reduced, Edmonds convention, magnitude)."""
    s6=float(wigner_6j(l,Q(j*2,2)*1 if False else Q(int(round(j*2)),2),Q(1,2),Q(int(round(j2*2)),2),l2,1))
    c1=math.sqrt((2*l+1)*(2*l2+1))*abs(float(wigner_3j(l,1,l2,0,0,0)))
    return math.sqrt((2*j+1)*(2*j2+1))*abs(s6)*c1
def radial(m, n1,l1,j1, n2,l2,j2, lam=0.0, om=None):
    b1=m._bound(n1,l1,j1); b2=m._bound(n2,l2,j2)
    r1,P1=b1[1],b1[2]; r2,P2=b2[1],b2[2]
    k=min(len(r1),len(r2)); r=r1[:k]; x=b1[3][:k]
    base=P1[:k]*np.interp(r,r2,P2)*r
    i0=np.trapezoid(base*r,x)
    from linak.physics.extrapolation.exact import polarization_term
    al,rc=m.params["singor_alpha_D"],m.params["singor_rc"]
    i1=np.trapezoid(base*r*polarization_term(m.params["form"],r,al,rc),x)
    return i0,i1
def D(m,n1,l1,j1,n2,l2,j2,lam=0.0,wc=None):
    i0,i1=radial(m,n1,l1,j1,n2,l2,j2)
    sd=1.0
    if wc:
        om=abs(m.energy_ha(n2,l2,j2)-m.energy_ha(n1,l1,j1)); sd=1/(1-(om/(wc/27.211386))**2)
    return abs(i0-lam*sd*i1)*ang(l1,j1,l2,j2)
