import sys
import os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from linak.physics.extrapolation.exact import ExactModel, MARINESCU, HA_EV
import frlev
CS=MARINESCU["Cs"]
class FrModel(ExactModel):
    def __init__(self, mp, params):
        self.element="Fr"; self.params=dict(params); self.hash="x"
        self.mp={k:(list(v) if isinstance(v,list) else v) for k,v in mp.items()}
        self.nl=7; self._atom=frlev.Atom(); self._bs={}; self._f={}; self._sigma=None; self._tuned=False; self._cache_path=None
    def energy_ha(self,n,l,j): return float(self._atom.getEnergy(n,l,j))/HA_EV
def base_mp():
    mp={k:(list(v) if isinstance(v,list) else v) for k,v in CS.items()}
    mp["Z"]=87; mp["alphaC"]=20.41
    return mp
PARAMS=dict(nl=7,singor_alpha_D=20.41,singor_rc=4.8,lam=1.0,wcore_eV=None,so_scale=1.0,form="E_sqrt(1-exp(-(r/rc)^6))")
if __name__=="__main__":
    import time
    m=FrModel(base_mp(),PARAMS)
    t=time.time()
    for (n,l,j) in [(7,0,.5),(8,0,.5),(7,1,.5),(7,1,1.5),(6,2,1.5),(8,1,.5)]:
        E=m.bound_state(n,l,j,None)[0]; print(n,l,j,E*HA_EV, m._atom.getEnergy(n,l,j))
    print(time.time()-t)
