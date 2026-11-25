# Toy: first-order descent per unit step norm, D(U)=<S,U>/||U||_F, momentum M=S+N, N iid Gaussian.
# Compares linear (M), polar (sigma^0), power maps sigma^p, hard-threshold oracle, and hard-threshold by rank-1 (top SV only).
import numpy as np
rng = np.random.default_rng(0)
def D(S,U): return float((S*U).sum()/np.linalg.norm(U))
def spec(M,f):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return (u*f(s))@vt
def run(n,r,c,trials=40):
    out={k:[] for k in ["lin","polar","p=.5","p=.25","top1","top_r","oracle_r"]}; nf=[]
    for _ in range(trials):
        U0,_=np.linalg.qr(rng.standard_normal((n,r))); V0,_=np.linalg.qr(rng.standard_normal((n,r)))
        s=c*np.sqrt(n)/np.sqrt(r)               # per-spike strength; ||S||_F = c*sqrt(n)
        S=(U0*s)@V0.T; N=rng.standard_normal((n,n)); M=S+N
        nf.append(np.linalg.norm(N)**2/np.linalg.norm(M)**2)
        out["lin"].append(D(S,M)); out["polar"].append(D(S,spec(M,lambda x:np.ones_like(x))))
        out["p=.5"].append(D(S,spec(M,lambda x:x**.5))); out["p=.25"].append(D(S,spec(M,lambda x:x**.25)))
        out["top1"].append(D(S,spec(M,lambda x:(np.arange(len(x))==0)*1.0)))
        out["top_r"].append(D(S,spec(M,lambda x:(np.arange(len(x))<r)*1.0)))
        out["oracle_r"].append(D(S,spec(M,lambda x:(np.arange(len(x))<r)*x)))
    return np.mean(nf),{k:(np.mean(v),np.std(v)/np.sqrt(trials)) for k,v in out.items()}
for r in (1,4):
  for c in (0.5,1.0,1.5,2.0,3.0,5.0):
    nf,o=run(128,r,c)
    print(f"r={r} c={c:3.1f} noise_frac={nf:.3f} | "+" ".join(f"{k}={v[0]:.3f}" for k,v in o.items()))
