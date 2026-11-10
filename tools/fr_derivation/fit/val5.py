import sys, json, csv, re
from pathlib import Path
import os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from frmodel import *
cl=lambda s:re.sub(r'^="?|"?$','',s or '').strip('"')
nist={}
for r in csv.DictReader(Path(os.path.join(HERE,'..','..','..','src','linak','resources','f_values','Fr_lines.csv')).read_text(encoding="utf-8").splitlines()):
    g={k:cl(v) for k,v in r.items() if k}
    if g.get('conf_i')=='7s' and g.get('fik') and g['conf_k'].endswith('p'):
        nist[(int(g['conf_k'][:-1]),float(eval(g['J_k'])))]=float(g['fik'])
t=json.loads(Path(os.path.join(HERE,'tuned_sp.json')).read_text(encoding="utf-8"))
ns=[7,8,9,10,12,14,16,20]
print("n   J:",[ (n,J) for n in ns for J in (.5,1.5)][:0])
out={}
for lab,rcs,wc in [("A 4.2 none",4.2,None),("B 4.4 16eV",4.4,16.0),("C 4.4 12eV",4.4,12.0),("D 4.6 9eV",4.6,9.0)]:
    p=dict(PARAMS); p["so_scale"]=t["so_scale"]; p["singor_rc"]=rcs; p["wcore_eV"]=wc
    m=FrModel(t["mp"],p); m._tuned=True
    out[lab]=[(n,J,m.f(n,J)/nist[(n,J)]) for n in ns for J in (.5,1.5)]
for lab,v in out.items():
    print(lab); print("  p1/2:",[ (n,round(r,2)) for n,J,r in v if J==.5]); print("  p3/2:",[ (n,round(r,2)) for n,J,r in v if J==1.5])
