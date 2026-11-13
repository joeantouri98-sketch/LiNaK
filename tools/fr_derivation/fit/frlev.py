import json, numpy as np
from pathlib import Path
from qali.core.paths import reference_path
R=json.loads(Path(str(reference_path("data_json","Fr_rydberg.json"))).read_text(encoding="utf-8"))
HA=27.211386245988
RY=13.605693122994
IE=R["ionization_energy_eV"]
LN={'s':0,'p':1,'d':2,'f':3,'g':4}
NIST={}
for e in R["excitations"]:
    if e["source"]=="NIST" and e["J"]:
        NIST[(e["n"],LN[e["l"]],float(eval(e["J"])))]=e["energy_eV"]
g=R["ground_state"]; NIST[(7,0,0.5)]=-IE
# Ritz per (l,j): delta(n) = d0 + d2/(n-d0)^2 from n>=9
RITZ={}
for (l,j) in {(k[1],k[2]) for k in NIST}:
    pts=[(n,E) for (n,ll,jj),E in NIST.items() if ll==l and jj==j and n>=9]
    if len(pts)<4: continue
    n=np.array([p[0] for p in pts],float); E=np.array([p[1] for p in pts])/RY
    nu=1/np.sqrt(-E); d=n-nu
    # iterate d = d0 + d2/(n-d)^2
    A=np.vstack([np.ones_like(n),1/(n-d)**2]).T
    for _ in range(20):
        c,*_=np.linalg.lstsq(A,d,rcond=None); dd=c[0]+c[1]/(n-d)**2; A=np.vstack([np.ones_like(n),1/(n-dd)**2]).T; d=dd
    RITZ[(l,j)]=c
class Atom:
    def getEnergy(self,n,l,j):
        k=(n,l,float(j))
        if k in NIST: return NIST[k]
        c=RITZ.get((l,float(j)))
        if c is None: 
            c=RITZ.get((l,0.5)) if l==0 else None
        if c is None: raise KeyError(k)
        d=c[0]+c[1]/(n-c[0])**2
        return -RY/(n-d)**2
if __name__=="__main__":
    a=Atom()
    for k,v in RITZ.items(): print(k,v)
    for n in (8,9,30,31,50): print(n,[a.getEnergy(n,1,j) for j in (0.5,1.5)],a.getEnergy(n,0,0.5))
