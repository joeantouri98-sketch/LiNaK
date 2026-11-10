import sys, json, numpy as np
from pathlib import Path
import os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from frmodel import *
t=json.loads(Path(os.path.join(HERE,'tuned_sp.json')).read_text(encoding="utf-8"))
res={}
for lab,rcs,wc in [("A 4.2 none",4.2,None),("B 4.4 16eV",4.4,16.0),("C 4.4 12eV",4.4,12.0),("D 4.6 9eV",4.6,9.0)]:
    p=dict(PARAMS); p["so_scale"]=t["so_scale"]; p["singor_rc"]=rcs; p["wcore_eV"]=wc
    m=FrModel(t["mp"],p); m._tuned=True
    row=[]
    for n in (8,9,12,20,40):
        for J in (.5,1.5):
            nu=m.nu(n,J); row.append(m.f(n,J)*nu**3)
    res[lab]=np.array(row); sg=m.sigma_th_Mb() if True else 0
    print(lab,"sigma_th",round(sg,4)); 
ref=res["B 4.4 16eV"]
labs=[f"{n}p{J}" for n in (8,9,12,20,40) for J in ("1/2","3/2")]
print(labs)
for k,v in res.items(): print(k,np.round(v/ref,3))
print("F(B)",np.round(ref,5))
