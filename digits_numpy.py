import numpy as np, itertools
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from scipy import stats
# ---- data: 1797 digits, fixed stratified 60/20/20, standardize with train stats ----
D=load_digits(); X=D.data.astype(np.float64); y=D.target
Xtr,Xtmp,ytr,ytmp=train_test_split(X,y,test_size=.4,stratify=y,random_state=0)
Xva,Xte,yva,yte=train_test_split(Xtmp,ytmp,test_size=.5,stratify=ytmp,random_state=0)
mu_,sd_=Xtr.mean(0),Xtr.std(0); sd_[sd_==0]=1
Xtr,Xva,Xte=[(a-mu_)/sd_ for a in (Xtr,Xva,Xte)]
def onehot(a): return np.eye(10)[a]
Ytr=onehot(ytr)
def fwd(P,Xb):
    h=np.maximum(Xb@P[0]+P[1],0); return h,h@P[2]+P[3]
def ce_acc(P,Xb,yb):
    _,z=fwd(P,Xb); z=z-z.max(1,keepdims=True); lp=z-np.log(np.exp(z).sum(1,keepdims=True))
    return -lp[np.arange(len(yb)),yb].mean(), (z.argmax(1)==yb).mean()
def grads(P,Xb,Yb):
    h,z=fwd(P,Xb); z=z-z.max(1,keepdims=True); p=np.exp(z); p/=p.sum(1,keepdims=True)
    dz=(p-Yb)/len(Xb); dh=(dz@P[2].T)*(h>0)
    return [Xb.T@dh,dh.sum(0),h.T@dz,dz.sum(0)]
def init(rng):
    sh=[(64,64),(64,),(64,10),(10,)]; fan=[64,64,64,64]
    return [rng.uniform(-1/np.sqrt(f),1/np.sqrt(f),size=s) for s,f in zip(sh,fan)]
STEPS=400; BS=64; LAM=[1e-4,0,1e-4,0]
def scale_at(cond,t):  # t is 1-based
    if cond=="spikes" and t in (STEPS//4,STEPS//2,3*STEPS//4): return 40.
    if cond=="jump" and t>STEPS//2: return 5.
    return 1.
def train(seed,cfg,eta,cond="clean",b1=.9,b2=.999,brho=.9,c=.1,M=2.,Tw=50,Imax=20.,ea=1e-8,ev=1e-16):
    rng=np.random.default_rng(seed); P=init(rng)
    perm=np.concatenate([rng.permutation(len(Xtr)) for _ in range(STEPS*BS//len(Xtr)+2)])
    m=[np.zeros_like(p) for p in P]; v=[np.zeros_like(p) for p in P]
    z=np.ones(4); I=np.zeros(4); L=np.zeros(4); mus=[]
    kind=cfg["kind"]
    for t in range(1,STEPS+1):
        idx=perm[(t-1)*BS:t*BS]; g=grads(P,Xtr[idx],Ytr[idx]); s=scale_at(cond,t); g=[s*a for a in g]
        vh_prev=[vi/(1-b2**(t-1)) if t>1 else None for vi in v]
        for i in range(4): m[i]=b1*m[i]+(1-b1)*g[i]; v[i]=b2*v[i]+(1-b2)*g[i]**2
        mh=[a/(1-b1**t) for a in m]; vh=[a/(1-b2**t) for a in v]
        mu=np.ones(4)
        if kind=="stable":
            for i in range(4):
                r=np.mean(g[i]**2/np.maximum(vh[i],ea**2)); mu[i]=1/max(1,np.sqrt(r))
        elif kind=="gov" and t>Tw:
            rs=cfg.get("rs",1.0); kp,ki=cfg["kp"],cfg["ki"]
            for i in range(4):
                rho=np.mean(g[i]**2/(vh_prev[i]+ev))
                if cfg.get("delay"): mu[i]=np.clip(np.exp(-(kp*(np.log(max(z[i],1e-30))-L[i])+ki*I[i])),c,M)
                z[i]=brho*z[i]+(1-brho)*rho; lz=np.log(max(z[i],1e-30))
                if isinstance(rs,float): L[i]=np.log(rs)
                else: L[i]+=rs[1]*(lz-L[i])
                e=lz-L[i]; I[i]=np.clip(I[i]+e,-Imax,Imax)
                if not cfg.get("delay"): mu[i]=np.clip(np.exp(-(kp*e+ki*I[i])),c,M)
        mus.append(mu.mean())
        for i in range(4):
            P[i]=(1-eta*mu[i]*LAM[i])*P[i]-eta*mu[i]*mh[i]/(np.sqrt(vh[i])+ea)
    te=ce_acc(P,Xte,yte); va=ce_acc(P,Xva,yva)
    return dict(test_ce=te[0],test_acc=te[1],val_ce=va[0],mu=np.mean(mus))
methods={"AdamW":dict(kind="adamw"),"StableAdamW":dict(kind="stable"),
 "PI (paper)":dict(kind="gov",kp=.5,ki=.02),
 "PI rho*=EMA .001":dict(kind="gov",kp=.5,ki=.02,rs=("ema",.001)),
 "PI rho*=EMA .01":dict(kind="gov",kp=.5,ki=.02,rs=("ema",.01)),
 "P kp=2":dict(kind="gov",kp=2,ki=0),
 "P kp=2 rho*=EMA .001":dict(kind="gov",kp=2,ki=0,rs=("ema",.001)),
 "PI predictable":dict(kind="gov",kp=.5,ki=.02,delay=True),
 "PI pred + EMA .001":dict(kind="gov",kp=.5,ki=.02,delay=True,rs=("ema",.001))}
rates=[.0003,.001,.003,.01,.03]; tune=[100,101,102]; ev=list(range(200,210))
if __name__=="__main__":
    best={}
    for name,cfg in methods.items():
        sc=[np.mean([train(s,cfg,e)["val_ce"] for s in tune]) for e in rates]
        best[name]=rates[int(np.argmin(sc))]
    res={}
    for name,cfg in methods.items():
        for cond in ["clean","spikes","jump"]:
            res[(name,cond)]=[train(s,cfg,best[name],cond) for s in ev]
    print("rates:",best)
    print(f"{'method':24s} | "+" | ".join(f"{c:^26s}" for c in ["clean CE (acc, mean mu)","spikes CE","jump CE"]))
    for name in methods:
        row=[]
        for cond in ["clean","spikes","jump"]:
            r=res[(name,cond)]; ce=np.mean([a["test_ce"] for a in r])
            extra=f" ({100*np.mean([a['test_acc'] for a in r]):.1f}%, mu {np.mean([a['mu'] for a in r]):.2f})" if cond=="clean" else ""
            row.append(f"{ce:.4f}{extra}")
        print(f"{name:24s} | "+" | ".join(f"{x:^26s}" for x in row))
    print("\nPaired difference vs StableAdamW (test CE; negative = better), 95% t-interval:")
    for name in methods:
        if name in("StableAdamW",): continue
        out=[]
        for cond in ["clean","spikes","jump"]:
            d=np.array([a["test_ce"] for a in res[(name,cond)]])-np.array([a["test_ce"] for a in res[("StableAdamW",cond)]])
            h=stats.t.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d)); out.append(f"{d.mean():+.4f}±{h:.4f}")
        print(f"{name:24s} | "+" | ".join(f"{x:^20s}" for x in out))
