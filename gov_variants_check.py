import numpy as np
# (tuning uses 10 seeds here, since 3 seeds picked a wrong rate for AdamW stat; scores x50 = paper's sum-form objective)
# Paper's quadratic benchmark (50 params, curvature 1..sqrt(1000), 2000 steps, score = mean true f over steps 201-2000).
# Question: does the integral term (PI) beat simply turning up the P gain? Rates tuned per arm per regime on 3 seeds, eval on 20 new seeds.
N=50; T=2000; A=np.geomspace(1,np.sqrt(1000),N)
def sigma(regime,t):
    if regime=="stat" or regime=="spk_t": return 1.0
    if regime=="drift": return 1+9*t/(T-1)
    if regime=="jump": return 1.0 if t<T//2 else 5.0
def run(regime,eta,kp,ki,seeds,delay=False,leak=1.0,b1=.9,b2=.999,brho=.9,c=.1,M=2.,Tw=50,Imax=20.,ea=1e-8,ev=1e-16,gov=True):
    S=len(seeds); th=np.empty((S,N)); xi=np.empty((T,S,N))
    for i,s in enumerate(seeds):
        r=np.random.default_rng(s); th[i]=r.normal(size=N); xi[:,i,:]=r.normal(size=(T,N))
        if regime=='spk_t': xi[:,i,:]+=(r.random((T,1))<1/250)*r.normal(size=(T,N))*40
    m=np.zeros((S,N)); v=np.zeros((S,N)); z=np.ones(S); I=np.zeros(S); sc=0.0; cnt=0
    for t in range(1,T+1):
        g=A*th+sigma(regime,t-1)*xi[t-1]
        vh_prev=v/(1-b2**(t-1)) if t>1 else None
        m=b1*m+(1-b1)*g; v=b2*v+(1-b2)*g*g
        mh=m/(1-b1**t); vh=v/(1-b2**t)
        mu=np.ones(S)
        if gov and t>Tw:
            rho=np.mean(g*g/(vh_prev+ev),axis=1)
            if delay:   # mu_t uses only past information (predictable): compute from previous state first
                e_old=np.log(np.maximum(z,1e-30)); mu=np.clip(np.exp(-(kp*e_old+ki*I)),c,M)
            z=brho*z+(1-brho)*rho
            e=np.log(np.maximum(z,1e-30)); I=np.clip(leak*I+e,-Imax,Imax)
            if not delay: mu=np.clip(np.exp(-(kp*e+ki*I)),c,M)
        th=th-eta*mu[:,None]*mh/(np.sqrt(vh)+ea)
        if t>200: sc+=0.5*np.mean(A*th*th,axis=1); cnt+=1
    sc=sc/cnt; return np.where(np.isfinite(sc),sc,1e9)
rates=[.0025,.005,.01,.02,.04,.08]
arms={"AdamW":dict(gov=False,kp=0,ki=0),
 "PI (paper)":dict(kp=.5,ki=.02),
 "P kp=2":dict(kp=2,ki=0),
 "PI predictable":dict(kp=.5,ki=.02,delay=True),
 "P kp=2 predictable":dict(kp=2,ki=0,delay=True),
 "PI leaky .99":dict(kp=.5,ki=.02,leak=.99),
 "kp=1 +I.02 leaky, pred":dict(kp=1,ki=.02,leak=.99,delay=True)}
tune=list(range(1000,1010)); ev=list(range(2000,2020))
for reg in ["stat","drift","jump","spk_t"]:
    print("==",reg)
    for name,kw in arms.items():
        best=min(rates,key=lambda e: run(reg,e,seeds=tune,**kw).mean())
        r=run(reg,best,seeds=ev,**kw)
        print(f"  {name:26s} eta={best:<6} f={50*r.mean():.4f}")
