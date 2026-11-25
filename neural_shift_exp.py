import numpy as np, sys
from scipy import stats
from digits_numpy import Xtr,Ytr,Xva,yva,Xte,yte,init,grads,ce_acc,LAM
# Neural version of the "tune once, face noise shifts" test. Shift = BATCH SIZE change mid-training (natural noise-scale change,
# no injected gradients). Digits MLP 64-64-10, 600 steps, wd 1e-4. Tune eta (and beta2) ONLY on scenario 1, evaluate on all.
T=600; NTR=len(Xtr)
def sched(kind):
    if kind=='stat': return [64]*T
    a,b,f=kind; return [a if t<f*T else b for t in range(T)]
SC={'bs 256->8 (tuned here)':(256,8,.5),'bs 256->16':(256,16,.5),'bs 256->4':(256,4,.5),'bs 256->8 early':(256,8,.25),
    'bs 256->8 late':(256,8,.75),'stationary bs 64':'stat','DROP: bs 8->256':(8,256,.5)}
def train(seed,cfg,eta,b2,scen,b1=.9,brho=.9,c=.1,Tw=50,Imax=20.,ea=1e-8,ev=1e-16):
    rng=np.random.default_rng(seed); P=init(rng); S=sched(SC[scen]); idxs=[rng.choice(NTR,B,replace=False) for B in S]
    m=[np.zeros_like(p) for p in P]; v=[np.zeros_like(p) for p in P]; z=np.ones(4); I=np.zeros(4); k=cfg['kind']
    for t in range(1,T+1):
        ix=idxs[t-1]; g=grads(P,Xtr[ix],Ytr[ix])
        if k=='gclip':
            nrm=np.sqrt(sum((a**2).sum() for a in g)); g=[a*min(1,1/max(nrm,1e-30)) for a in g]
        vp=[vi/(1-b2**(t-1)) if t>1 else None for vi in v]
        for i in range(4): m[i]=b1*m[i]+(1-b1)*g[i]; v[i]=b2*v[i]+(1-b2)*g[i]**2
        mh=[a/(1-b1**t) for a in m]; vh=[a/(1-b2**t) for a in v]; mu=np.ones(4)
        for i in range(4):
            if k=='stable': mu[i]=1/max(1,np.sqrt(np.mean(g[i]**2/np.maximum(vh[i],ea**2))))
            elif k=='lagclip' and t>1: mu[i]=1/max(1,np.sqrt(np.mean(g[i]**2/(vp[i]+ev))))
            elif k=='gov' and t>Tw:
                kp,ki,M=cfg['kp'],cfg['ki'],cfg.get('M',2.); rho=np.mean(g[i]**2/(vp[i]+ev))
                if cfg.get('delay'): mu[i]=np.clip(np.exp(-(kp*np.log(max(z[i],1e-30))+ki*I[i])),c,M)
                z[i]=brho*z[i]+(1-brho)*rho; e=np.log(max(z[i],1e-30)); I[i]=np.clip(I[i]+e,-Imax,Imax)
                if not cfg.get('delay'): mu[i]=np.clip(np.exp(-(kp*e+ki*I[i])),c,M)
        for i in range(4): P[i]=(1-eta*mu[i]*LAM[i])*P[i]-eta*mu[i]*mh[i]/(np.sqrt(vh[i])+ea)
    return ce_acc(P,Xte,yte)[0], ce_acc(P,Xva,yva)[0]
RATES=[.0003,.001,.003,.01,.03]
M_={'AdamW':dict(kind='adam'),'AdamW tuned beta2':dict(kind='adam'),'Global clip':dict(kind='gclip'),'StableAdamW':dict(kind='stable'),
    'Lagged clip':dict(kind='lagclip'),'Governor PI (paper)':dict(kind='gov',kp=.5,ki=.02),'Governor P only':dict(kind='gov',kp=.5,ki=0.),
    'Governor PI predictable':dict(kind='gov',kp=.5,ki=.02,delay=True),'Governor P kp=2':dict(kind='gov',kp=2.,ki=0.)}
tune=list(range(100,105)); ev=list(range(200,210)); cal='bs 256->8 (tuned here)'
chosen={}
for n,cfg in M_.items():
    b2s=[.9,.99,.999] if n=='AdamW tuned beta2' else [.999]
    chosen[n]=min(((e,b) for e in RATES for b in b2s),key=lambda p: np.mean([train(s,cfg,p[0],p[1],cal)[1] for s in tune]))
R={}
for n,cfg in M_.items():
    for sc in SC: R[(n,sc)]=np.array([train(s,cfg,chosen[n][0],chosen[n][1],sc)[0] for s in ev])
print('final TEST cross-entropy (mean of 10 held-out seeds), lower = better. eta/beta2 tuned ONLY on "bs 256->8", by validation CE.')
print('chosen (eta,beta2):',chosen)
cols=list(SC); print(f"{'':26s}"+''.join(f"{c[:14]:>15s}" for c in cols))
for n in M_: print(f"{n:26s}"+''.join(f"{R[(n,c)].mean():15.4f}" for c in cols))
print('\nPaired difference (Governor PI minus baseline) in test CE; negative = governor better; 95% t-interval, 10 seeds')
for base in ['AdamW','AdamW tuned beta2','Global clip','StableAdamW']:
    out=[]
    for c in cols:
        d=R[('Governor PI (paper)',c)]-R[(base,c)]; h=stats.t.ppf(.975,9)*d.std(ddof=1)/np.sqrt(10); out.append(f"{d.mean():+.4f}±{h:.4f}")
    print(f"vs {base:18s}"+''.join(f"{x:>17s}" for x in out))
