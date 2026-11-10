import sys, json, numpy as np
from pathlib import Path
import os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from frmodel import *; from frD import *
t=json.loads(Path(os.path.join(HERE,'tuned_sp.json')).read_text(encoding="utf-8"))
p=dict(PARAMS); p["so_scale"]=t["so_scale"]
m=FrModel(t["mp"],p); m._tuned=True
A=[((7,0,.5),(7,1,.5),4.2895,.0055),((7,0,.5),(7,1,1.5),5.9065,.0085),((7,0,.5),(8,1,.5),.3041,.008),((7,0,.5),(8,1,1.5),.9070,.0085)]
FLOOR=0.01
print("rc  wc  chi2  rel.dev%")
for wc in (None,16.0,12.0,9.0):
  for rcs in np.arange(3.8,5.01,0.2):
    m.params["singor_rc"]=rcs
    dev=[D(m,*a,*b,lam=1.0,wc=wc)/ref-1 for a,b,ref,u in A]
    chi=sum((d/np.hypot(u/ref,FLOOR))**2 for d,(a,b,ref,u) in zip(dev,A))
    print(round(rcs,1),wc,round(chi,2),np.round(np.array(dev)*100,1))
