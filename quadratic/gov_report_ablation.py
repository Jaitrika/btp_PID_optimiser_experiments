"""Ablation + D-term probe for BTP report v2.  Python 3 + numpy only.
Quadratic benchmark: 50 params, curvature 1..sqrt(1000), 2000 steps, score = mean true objective
over steps 201-2000 (sum form). eta (and beta2 for the tuned-beta2 arm) tuned ONLY on 'jump x5' (10 seeds);
evaluated on 20 fresh seeds. Lower is better.
Sensors: 'lag' g^2/vhat_{t-1} (default) ; 'cur' g^2/vhat_t ; 'diff' (g_t-g_{t-1})^2/2 over its own lagged EMA (D-idea).
"""
import numpy as np, sys
from math import sqrt
N,T=50,2000
A=np.geomspace(1,np.sqrt(1000),N)
REG={'jump x5 (tuned)':('jump',1,5,.5),'jump x10':('jump',1,10,.5),'drift 1->10':('drift',1,10),
     'stationary':('jump',1,1,.5),'drop 5->1':('jump',5,1,.5),'rare spikes':('spike',)}
def sig(r,t):
    if r[0]=='jump': return r[1] if t<r[3]*T else r[2]
    if r[0]=='drift': return r[1]+(r[2]-r[1])*t/(T-1)
    return 1.
def run(reg,cfg,eta,seeds,b1=.9,c=.1,Tw=50,Imax=20.,ea=1e-8,ev=1e-16):
    b2=cfg.get('b2',.999); S=len(seeds); r_=REG[reg]
    th=np.empty((S,N)); xi=np.empty((T,S,N)); sm=np.ones((T,S))
    for i,s in enumerate(seeds):
        r=np.random.default_rng(s); th[i]=r.normal(size=N); xi[:,i,:]=r.normal(size=(T,N))
        if r_[0]=='spike': sm[:,i]=np.where(r.random(T)<1/250,40.,1.)
    m=np.zeros((S,N)); v=np.zeros((S,N)); z=np.ones(S); I=np.zeros(S); ep=np.zeros(S)
    gp=np.zeros((S,N)); vd=np.zeros((S,N)); kobs=0; sc=0.; cnt=0
    cc=cfg.get('c',c)
    for t in range(1,T+1):
        g=A*th+sig(r_,t-1)*sm[t-1][:,None]*xi[t-1]
        if cfg['kind']=='clip':
            nrm=np.linalg.norm(g,axis=1,keepdims=True); g=g/np.maximum(1,nrm)
        vp=v/(1-b2**(t-1)) if t>1 else None
        m=b1*m+(1-b1)*g; v=b2*v+(1-b2)*g*g
        mh=m/(1-b1**t); vh=v/(1-b2**t); mu=np.ones(S)
        rho_d=None
        if t>=2:
            dg=(g-gp)**2/2
            if kobs>=1: rho_d=np.mean(dg/(vd/(1-b2**kobs)+ev),axis=1)
            vd=b2*vd+(1-b2)*dg; kobs+=1
        gp=g.copy()
        if cfg['kind']=='gov' and t>Tw:
            sn=cfg['sensor']
            if sn=='lag': rho=np.mean(g*g/(vp+ev),axis=1)
            elif sn=='cur': rho=np.mean(g*g/(vh+ev),axis=1)
            else: rho=rho_d
            br=cfg['brho']; z=br*z+(1-br)*rho
            e=np.log(np.maximum(z,1e-30)); I=np.clip(I+e,-Imax,Imax)
            u=cfg['kp']*e+cfg['ki']*I+cfg['kd']*(e-ep); ep=e
            mu=np.clip(np.exp(-u),cc,cfg['M'])
        th=th-eta*mu[:,None]*mh/(np.sqrt(vh)+ea)
        if t>200: sc+=0.5*np.sum(A*th*th,axis=1); cnt+=1
    sc=sc/cnt
    return np.where(np.isfinite(sc),sc,1e9)
def G(sensor,brho,kp,ki,kd,M,c=None):
    d=dict(kind='gov',sensor=sensor,brho=brho,kp=kp,ki=ki,kd=kd,M=M)
    if c is not None: d['c']=c
    return d
ARMS={
 'AdamW':dict(kind='adam'),
 'AdamW, tuned beta2':dict(kind='adam',tuneb2=True),
 'Global grad clip (norm 1)':dict(kind='clip'),
 'Update clipping (StableAdamW-style)':G('cur',0.,.5,0,0,1.,c=0.),
 'Governor P':G('lag',.9,.5,0,0,2.),
 'Governor PI (default)':G('lag',.9,.5,.02,0,2.),
 'Governor PID (default, kd=.05)':G('lag',.9,.5,.02,.05,2.),
 'Governor PI + big D (kd=1)':G('lag',.9,.5,.02,1.,2.),
 'Governor P + big D (kd=1)':G('lag',.9,.5,0,1.,2.),
 'Governor P, DIFF sensor':G('diff',.9,.5,0,0,2.),
 'Governor PI, DIFF sensor':G('diff',.9,.5,.02,0,2.),
}
RATES=[.0025,.005,.01,.02,.04,.08]
tune=list(range(7000,7010)); ev_=list(range(8000,8020))
raw={}; info={}
for n,cfg in ARMS.items():
    cands=[(e,b) for e in RATES for b in ([.9,.99,.999] if cfg.get('tuneb2') else [.999])]
    best=min(cands,key=lambda eb:run('jump x5 (tuned)',{**cfg,'b2':eb[1]},eb[0],tune).mean())
    cf={**cfg,'b2':best[1]}
    raw[n]={r:run(r,cf,best[0],ev_) for r in REG}; info[n]=best
    print(f"{n:38s} eta={best[0]:<6} b2={best[1]:<5}"+''.join(f"{raw[n][r].mean():9.3f}" for r in REG),flush=True)
print(f"{'':58s}"+''.join(f"{k[:9]:>9s}" for k in REG))
def diff(a,b):
    out=[]
    for r in REG:
        d=raw[a][r]-raw[b][r]; out.append(f"{d.mean():+.3f}+/-{2.093*d.std(ddof=1)/sqrt(len(d)):.3f}")
    print(f"{a[:30]:30s} - {b[:30]:30s}  "+'  '.join(out),flush=True)
print("\nPAIRED (negative = first is better)")
diff('Governor PI (default)','AdamW'); diff('Governor PI (default)','AdamW, tuned beta2'); diff('Governor PI (default)','Global grad clip (norm 1)')
diff('Governor PI (default)','Update clipping (StableAdamW-style)')
diff('Governor P','Update clipping (StableAdamW-style)')
diff('Governor PI (default)','Governor P'); diff('Governor PID (default, kd=.05)','Governor PI (default)')
diff('Governor PI + big D (kd=1)','Governor PI (default)'); diff('Governor P + big D (kd=1)','Governor P')
diff('Governor P, DIFF sensor','Governor P'); diff('Governor PI, DIFF sensor','Governor PI (default)')
