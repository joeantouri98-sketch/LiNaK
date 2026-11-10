import sys, json, numpy as np, time
from pathlib import Path
import os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from frmodel import *
from scipy.optimize import least_squares
CM=219474.63
LEV={0:[(n,[0.5]) for n in range(7,15)],1:[(n,[0.5,1.5]) for n in range(7,15)],2:[(n,[1.5,2.5]) for n in range(6,15)]}
def avg_exp(m,n,l,js):
    w=[2*j+1 for j in js]; return sum(wi*m.energy_ha(n,l,j) for wi,j in zip(w,js))/sum(w)
def avg_mod(m,n,l,js):
    w=[2*j+1 for j in js]
    return sum(wi*m.bound_state(n,l,j,1/np.sqrt(-2*m.energy_ha(n,l,j)))[0] for wi,j in zip(w,js))/sum(w)
def fit_l(l, p0, free, prior_sd, sofs=1.0):
    base=base_mp()
    def mk(x):
        mp={k:(list(v) if isinstance(v,list) else v) for k,v in base.items()}
        for name,val in zip(free,x):
            key,idx=name; mp[key][idx]=val
        return mp
    p=dict(PARAMS); p["so_scale"]=sofs
    def res(x):
        m=FrModel(mk(x),p); out=[]
        for n,js in LEV[l]:
            try: out.append((avg_mod(m,n,l,js)-avg_exp(m,n,l,js))*CM)
            except RuntimeError: out.append(3000.0)
        # soft prior on (x - p0)/sd
        out+=list((np.array(x)-np.array(p0))/np.array(prior_sd))
        return np.array(out)
    r=least_squares(res,p0,diff_step=1e-3,x_scale=np.maximum(np.abs(p0),0.3),max_nfev=40)
    return r, mk(r.x)
if __name__=="__main__":
    l=int(sys.argv[1])
    free=[("a1",l),("a2",l),("a3",l),("a4",l),("rc",l)]
    p0=[CS[k][l] for k,_ in free]
    sd=[0.8*abs(p0[0]),0.8*abs(p0[1]),3*abs(p0[2])+5,1.0,0.5*abs(p0[4])]
    t=time.time()
    r,mp=fit_l(l,p0,free,sd)
    print("l",l,"x",list(np.round(r.x,5)),"cost",r.cost,"nfev",r.nfev,time.time()-t)
    m=FrModel(mp,PARAMS)
    print("resid cm-1:",np.round(r.fun[:len(LEV[l])],2))
    Path(os.path.join(HERE,f"fit1_l{l}.json")).write_text(json.dumps(dict(l=l,x=list(r.x))), encoding="utf-8")
