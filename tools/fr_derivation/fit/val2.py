import sys, json, numpy as np
from pathlib import Path
import os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from frmodel import *; from frD import *
from scipy.optimize import brentq
mp=json.loads(Path(os.path.join(HERE,'mp_sp.json')).read_text(encoding="utf-8"))
CM=219474.63
def fs(sof):
    p=dict(PARAMS); p["so_scale"]=sof
    m=FrModel(mp,p); m._tune()
    e=[m.bound_state(7,1,j,m.nu(7,j))[0] for j in (.5,1.5)]
    return m,(e[1]-e[0])*CM-1686.6
sof=brentq(lambda s:fs(s)[1],1.0,1.8,xtol=1e-4)
m,_=fs(sof); print("so_scale",sof,"rc1 tuned",m.mp["rc"][1])
Path(os.path.join(HERE,'tuned_sp.json')).write_text(json.dumps(dict(so_scale=sof,mp=m.mp)), encoding="utf-8")
e=[m.bound_state(7,1,j,m.nu(7,j))[0]*HA_EV-m._atom.getEnergy(7,1,j) for j in (.5,1.5)]
print("7p errs eV",e, "8p FS exp", (m._atom.getEnergy(8,1,1.5)-m._atom.getEnergy(8,1,.5))*8065.54, "model", (m.bound_state(8,1,1.5,m.nu(8,1.5))[0]-m.bound_state(8,1,.5,m.nu(8,.5))[0])*CM)
A=[((7,0,.5),(7,1,.5),4.2895,.0055),((7,0,.5),(7,1,1.5),5.9065,.0085),((7,0,.5),(8,1,.5),.3041,.008),((7,0,.5),(8,1,1.5),.9070,.0085),((8,0,.5),(8,1,.5),10.079,.016),((8,0,.5),(8,1,1.5),13.338,.024),((8,0,.5),(7,1,.5),4.231,.023),((8,0,.5),(7,1,1.5),7.455,.019)]
for rcs in (4.4,4.8,5.2):
  for wc in (None,12.0,8.0):
    m.params["singor_rc"]=rcs
    row=[D(m,*a,*b,lam=1.0,wc=wc)/ref-1 for a,b,ref,u in A]
    print(rcs,wc,np.round(np.array(row)*100,1))
