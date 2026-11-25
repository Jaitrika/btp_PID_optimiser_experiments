import numpy as np
# (tuning uses 10 seeds here, since 3 seeds picked a wrong rate for AdamW stat; scores x50 = paper's sum-form objective)
# Paper's quadratic benchmark (50 params, curvature 1..sqrt(1000), 2000 steps, score = mean true f over steps 201-2000).
# Question: does the integral term (PI) beat simply turning up the P gain? Rates tuned per arm per regime on 3 seeds, eval on 20 new seeds.
N=50; T=2000; A=np.geomspace(1,np.sqrt(1000),N)
def sigma(regime,t):
    if regime=="noiseless": return 0.0
    if regime=="stat": return 1.0
    if regime=="drift": return 1+9*t/(T-1)
    if regime=="jump": return 1.0 if t<T//2 else 5.0
def run(regime,eta,kp,ki,seeds,b1=.9,b2=.999,brho=.9,c=.1,M=2.,Tw=50,Imax=20.,ea=1e-8,ev=1e-16,gov=True):
    S=len(seeds); th=np.empty((S,N)); xi=np.empty((T,S,N))
    for i,s in enumerate(seeds):
        r=np.random.default_rng(s); th[i]=r.normal(size=N); xi[:,i,:]=r.normal(size=(T,N))
    m=np.zeros((S,N)); v=np.zeros((S,N)); z=np.ones(S); I=np.zeros(S); sc=0.0; cnt=0; hit={1e-2:np.full(S,float(T)),1e-4:np.full(S,float(T))}; mus=0.0; pk=np.zeros(S); fin=[]
    for t in range(1,T+1):
        g=A*th+sigma(regime,t-1)*xi[t-1]
        vh_prev=v/(1-b2**(t-1)) if t>1 else None
        m=b1*m+(1-b1)*g; v=b2*v+(1-b2)*g*g
        mh=m/(1-b1**t); vh=v/(1-b2**t)
        mu=np.ones(S)
        if gov and t>Tw:
            rho=np.mean(g*g/(vh_prev+ev),axis=1)
            z=brho*z+(1-brho)*rho
            e=np.log(np.maximum(z,1e-30)); I=np.clip(I+e,-Imax,Imax)
            mu=np.clip(np.exp(-(kp*e+ki*I)),c,M)
        th=th-eta*mu[:,None]*mh/(np.sqrt(vh)+ea)
        fsum=0.5*np.sum(A*th*th,axis=1); mus=mus+mu
        if t>=500: pk=np.maximum(pk,fsum)
        for thr in hit: hit[thr]=np.where((fsum<thr)&(hit[thr]==T),t,hit[thr])
        if t>200: sc+=0.5*np.mean(A*th*th,axis=1); cnt+=1
    run.final=fsum.mean(); run.peak=pk.mean(); run.finmed=np.median(fsum); run.hit={k:v.mean() for k,v in hit.items()}; run.mu=(mus/T).mean(); sc=sc/cnt; return np.where(np.isfinite(sc),sc,1e9)
ev=list(range(2000,2020))
cfgs=[("AdamW eta=.04",.04,dict(gov=False,kp=0,ki=0)),("AdamW eta=.02",.02,dict(gov=False,kp=0,ki=0)),("AdamW eta=.01",.01,dict(gov=False,kp=0,ki=0)),
 ("P kp=.5 M=2 eta=.02",.02,dict(kp=.5,ki=0,M=2.)),("P kp=.5 M=5 eta=.01",.01,dict(kp=.5,ki=0,M=5.)),("PI paper M=2 eta=.02",.02,dict(kp=.5,ki=.02,M=2.))]
print("peak f during steps 500-2000 (mean over seeds) | final f median | final f mean")
for n,e,kw in cfgs:
    r=run("noiseless",e,seeds=ev,**kw); print(f"  {n:22s} peak {run.peak:.2e} | final median {run.finmed:.2e} | final mean {run.final:.2e}")
