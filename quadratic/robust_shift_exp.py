import numpy as np
# ONE experiment: "tune once, then face noise-scale shifts you did not tune for" on the paper's quadratic benchmark.
# Every method gets the same eta grid; tuned ONLY on jump x5 (10 seeds); evaluated on 20 new seeds on 9 regimes.
N=50; T=2000; A=np.geomspace(1,np.sqrt(1000),N)
REG={'jump x5 (tuned here)':('jump',1,5,.5),'jump x2':('jump',1,2,.5),'jump x10':('jump',1,10,.5),'jump x20':('jump',1,20,.5),
     'jump x5 early':('jump',1,5,.25),'jump x5 late':('jump',1,5,.75),'drift 1->10':('drift',1,10),'stationary':('jump',1,1,.5),'DROP 5->1':('jump',5,1,.5)}
def sig(r,t):
    if r[0]=='jump': return r[1] if t<r[3]*T else r[2]
    return r[1]+(r[2]-r[1])*t/(T-1)
def run(reg,cfg,eta,b2,seeds,b1=.9,brho=.9,c=.1,Tw=50,Imax=20.,ea=1e-8,ev=1e-16):
    S=len(seeds); th=np.empty((S,N)); xi=np.empty((T,S,N))
    for i,s in enumerate(seeds):
        r=np.random.default_rng(s); th[i]=r.normal(size=N); xi[:,i,:]=r.normal(size=(T,N))
    m=np.zeros((S,N)); v=np.zeros((S,N)); z=np.ones(S); I=np.zeros(S); sc=0.; cnt=0; k=cfg['kind']
    for t in range(1,T+1):
        g=A*th+sig(REG[reg],t-1)*xi[t-1]
        if k=='gclip': g=g*np.minimum(1,1/np.maximum(np.linalg.norm(g,axis=1,keepdims=True),1e-30))
        vp=v/(1-b2**(t-1)) if t>1 else None
        m=b1*m+(1-b1)*g; v=b2*v+(1-b2)*g*g; mh=m/(1-b1**t); vh=v/(1-b2**t); mu=np.ones(S)
        if k=='stable': mu=1/np.maximum(1,np.sqrt(np.mean(g*g/np.maximum(vh,ea**2),axis=1)))
        elif k=='lagclip' and t>1: mu=1/np.maximum(1,np.sqrt(np.mean(g*g/(vp+ev),axis=1)))
        elif k=='gov' and t>Tw:
            rho=np.mean(g*g/(vp+ev),axis=1); kp,ki,M=cfg['kp'],cfg['ki'],cfg.get('M',2.)
            if cfg.get('delay'): mu=np.clip(np.exp(-(kp*np.log(np.maximum(z,1e-30))+ki*I)),c,M)
            z=brho*z+(1-brho)*rho; e=np.log(np.maximum(z,1e-30)); I=np.clip(I+e,-Imax,Imax)
            if not cfg.get('delay'): mu=np.clip(np.exp(-(kp*e+ki*I)),c,M)
        th=th-eta*mu[:,None]*mh/(np.sqrt(vh)+ea)
        if t>200: sc+=0.5*np.sum(A*th*th,axis=1); cnt+=1
    sc=sc/cnt; return float(np.mean(np.where(np.isfinite(sc),sc,1e9)))
RATES=[.0025,.005,.01,.02,.04,.08]
M_={'AdamW':dict(kind='adam'),'AdamW tuned beta2':dict(kind='adam'),'Global clip':dict(kind='gclip'),'StableAdamW':dict(kind='stable'),
    'Lagged clip':dict(kind='lagclip'),'Governor PI (paper)':dict(kind='gov',kp=.5,ki=.02),'Governor P only':dict(kind='gov',kp=.5,ki=0.),
    'Governor PI predictable':dict(kind='gov',kp=.5,ki=.02,delay=True)}
tune=list(range(1000,1010)); ev=list(range(2000,2020)); cal='jump x5 (tuned here)'
def tune_cfg(name,reg):
    b2s=[.9,.99,.999] if name=='AdamW tuned beta2' else [.999]
    best=min(((e,b) for e in RATES for b in b2s),key=lambda p: run(reg,M_[name],p[0],p[1],tune)); return best
res={}; chosen={}
for name in M_:
    chosen[name]=tune_cfg(name,cal)
    res[name]=[run(r,M_[name],chosen[name][0],chosen[name][1],ev) for r in REG]
# oracle controls: retuned separately for EACH regime (the baselines get an unfair advantage)
for name in ['AdamW','Global clip']:
    row=[]
    for r in REG:
        e,b=tune_cfg(name,r); row.append(run(r,M_[name],e,b,ev))
    res[name+'  [retuned per regime: unfair oracle]']=row
cols=list(REG)
print('mean objective, steps 201-2000, 20 held-out seeds, lower = better.  eta (and beta2) tuned ONLY on "jump x5".')
print('chosen:',{k:v for k,v in chosen.items()})
print(f"{'':44s}"+''.join(f"{c[:11]:>12s}" for c in cols))
for n,row in res.items(): print(f"{n:44s}"+''.join(f"{x:12.3f}" for x in row))
print('\nbest NON-oracle method per regime:')
for j,c in enumerate(cols):
    cand={n:res[n][j] for n in M_}; b=min(cand,key=cand.get); print(f"  {c:22s} {b:26s} {cand[b]:.3f}")
