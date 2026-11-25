"""Can the integral term beat a TUNED P governor? numpy only (runs on Colab as is).
Both arms: smoothed sensor (beta_rho=.9), c=.1, M=2, Tw=50. Lagged sensor (paper). Score = mean true objective, steps 201-2000, lower better.
P  arm: tune (eta, kp)        over eta in RATES, kp in {.5,1,2,4},            ki=0
PI arm: tune (eta, kp, ki)    over eta in RATES, kp in {.25,.5,1,2}, ki in {.01,.02,.05}
Tuning ONLY on 'jump x5' (10 seeds); evaluation on 20 fresh seeds across regimes (incl. persistent/saturating and oscillating noise).
"""
import numpy as np
from math import sqrt
N,T=50,2000; A=np.geomspace(1,np.sqrt(1000),N)
REG={'jump x5 (tuned)':('jump',1,5,.5),'jump x10':('jump',1,10,.5),'jump x50 (sat.)':('jump',1,50,.5),'x5 from t=0.1T':('jump',1,5,.1),
     'drift 1->10':('drift',1,10),'oscillate 1<->5':('osc',1,5),'stationary':('jump',1,1,.5),'DROP 5->1':('jump',5,1,.5)}
def sig(r,t):
    if r[0]=='jump': return r[1] if t<r[3]*T else r[2]
    if r[0]=='osc': return r[1] if (t//100)%2==0 else r[2]
    return r[1]+(r[2]-r[1])*t/(T-1)
def run(reg,kp,ki,eta,seeds,b1=.9,b2=.999,brho=.9,c=.1,M=2.,Tw=50,Imax=20.,ea=1e-8,ev=1e-16):
    S=len(seeds); th=np.empty((S,N)); xi=np.empty((T,S,N))
    for i,s in enumerate(seeds):
        r=np.random.default_rng(s); th[i]=r.normal(size=N); xi[:,i,:]=r.normal(size=(T,N))
    m=np.zeros((S,N)); v=np.zeros((S,N)); z=np.ones(S); I=np.zeros(S); sc=0.; cnt=0
    for t in range(1,T+1):
        g=A*th+sig(REG[reg],t-1)*xi[t-1]
        vp=v/(1-b2**(t-1)) if t>1 else None
        m=b1*m+(1-b1)*g; v=b2*v+(1-b2)*g*g; mh=m/(1-b1**t); vh=v/(1-b2**t); mu=np.ones(S)
        if t>Tw:
            z=brho*z+(1-brho)*np.mean(g*g/(vp+ev),axis=1); e=np.log(np.maximum(z,1e-30)); I=np.clip(I+e,-Imax,Imax)
            mu=np.clip(np.exp(-(kp*e+ki*I)),c,M)
        th=th-eta*mu[:,None]*mh/(np.sqrt(vh)+ea)
        if t>200: sc+=0.5*np.sum(A*th*th,axis=1); cnt+=1
    sc=sc/cnt; return np.where(np.isfinite(sc),sc,1e9)
RATES=[.0025,.005,.01,.02,.04,.08]
GRID={'tuned P (no I)':[(kp,0.) for kp in (.5,1,2,4)],'tuned PI':[(kp,ki) for kp in (.25,.5,1,2) for ki in (.01,.02,.05)],
      'PI paper (kp.5,ki.02)':[(.5,.02)]}
tune=list(range(7000,7010)); ev=list(range(8000,8020)); raw={}
for name,gains in GRID.items():
    kp,ki,eta=min(((k,i,e) for (k,i) in gains for e in RATES),key=lambda p: run('jump x5 (tuned)',p[0],p[1],p[2],tune).mean())
    print(f"{name:24s} chosen kp={kp} ki={ki} eta={eta}",flush=True)
    raw[name]={r:run(r,kp,ki,eta,ev) for r in REG}
print(f"\n{'':24s}"+''.join(f"{c[:15]:>16s}" for c in REG))
for n in raw: print(f"{n:24s}"+''.join(f"{raw[n][r].mean():16.3f}" for r in REG))
print("\npaired diff  tuned PI minus tuned P  (negative = I helps):")
print(f"{'':24s}"+''.join(f"{c[:15]:>16s}" for c in REG))
row=[]
for r in REG:
    d=raw['tuned PI'][r]-raw['tuned P (no I)'][r]; row.append(f"{d.mean():+.3f}+/-{2.093*d.std(ddof=1)/sqrt(len(d)):.3f}")
print(f"{'PI - P':24s}"+''.join(f"{x:>16s}" for x in row))
