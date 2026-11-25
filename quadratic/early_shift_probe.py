"""Why does I help when the noise jump happens early (t=200)?  numpy only (Colab ok).
Part 1: traces (mean multiplier mu and mean loss per 200-step window) for P kp=4 vs PI kp=2,ki=.05, early jump x5 at t=200.
Part 2: is it a tuning artifact? re-tune P and PI (grid as before) on (a) early jump only, (b) early+mid jumps together; eval on fresh seeds.
"""
import numpy as np
from math import sqrt
N,T=50,2000; A=np.geomspace(1,np.sqrt(1000),N)
REG={'x5 mid (t=1000)':(1,5,.5),'x5 early (t=200)':(1,5,.1)}
def run(reg,kp,ki,eta,seeds,trace=False,b1=.9,b2=.999,brho=.9,c=.1,M=2.,Tw=50,Imax=20.,ea=1e-8,ev=1e-16):
    s0,s1,fr=REG[reg]; S=len(seeds); th=np.empty((S,N)); xi=np.empty((T,S,N))
    for i,s in enumerate(seeds):
        r=np.random.default_rng(s); th[i]=r.normal(size=N); xi[:,i,:]=r.normal(size=(T,N))
    m=np.zeros((S,N)); v=np.zeros((S,N)); z=np.ones(S); I=np.zeros(S); sc=0.; cnt=0; MU=[]; LS=[]
    for t in range(1,T+1):
        g=A*th+(s0 if t-1<fr*T else s1)*xi[t-1]
        vp=v/(1-b2**(t-1)) if t>1 else None
        m=b1*m+(1-b1)*g; v=b2*v+(1-b2)*g*g; mh=m/(1-b1**t); vh=v/(1-b2**t); mu=np.ones(S)
        if t>Tw:
            z=brho*z+(1-brho)*np.mean(g*g/(vp+ev),axis=1); e=np.log(np.maximum(z,1e-30)); I=np.clip(I+e,-Imax,Imax)
            mu=np.clip(np.exp(-(kp*e+ki*I)),c,M)
        th=th-eta*mu[:,None]*mh/(np.sqrt(vh)+ea); L=0.5*np.sum(A*th*th,axis=1)
        if t>200: sc+=L; cnt+=1
        MU.append(mu.mean()); LS.append(L.mean())
    sc=np.where(np.isfinite(sc/cnt),sc/cnt,1e9)
    return (sc,np.array(MU),np.array(LS)) if trace else sc
ev=list(range(8000,8020)); tune=list(range(7000,7010))
print("PART 1: early jump x5 at t=200. window = steps [a,b). mu = mean multiplier, loss = mean true objective (sum form)")
tr={}
for name,(kp,ki) in {'P kp=4':(4,0.),'PI kp=2,ki=.05':(2,.05)}.items():
    sc,MU,LS=run('x5 early (t=200)',kp,ki,.01,ev,trace=True); tr[name]=(MU,LS); print(f"  {name:16s} score {sc.mean():.3f}")
print(f"  {'window':>12s}"+''.join(f"{n+' mu':>20s}{n+' loss':>20s}" for n in tr))
for a in range(0,T,200):
    print(f"  {a:5d}-{a+200:<5d}"+''.join(f"{tr[n][0][a:a+200].mean():20.3f}{tr[n][1][a:a+200].mean():20.3f}" for n in tr))
print("\nPART 2: tuning artifact?  best (kp,ki,eta) per arm under different tuning sets; score on eval seeds")
RATES=[.0025,.005,.01,.02,.04,.08]
GR={'P':[(kp,0.) for kp in (.5,1,2,4)],'PI':[(kp,ki) for kp in (.25,.5,1,2) for ki in (.01,.02,.05)]}
for label,regs in {'tuned on early only':['x5 early (t=200)'],'tuned on mid+early':['x5 mid (t=1000)','x5 early (t=200)']}.items():
    print(f" {label}")
    for arm,gains in GR.items():
        kp,ki,eta=min(((k,i,e) for (k,i) in gains for e in RATES),key=lambda p: sum(run(r,p[0],p[1],p[2],tune).mean() for r in regs))
        out=[run(r,kp,ki,eta,ev) for r in REG]
        print(f"   {arm:3s} kp={kp} ki={ki} eta={eta:<6}"+'   '.join(f"{r}: {o.mean():.3f}" for r,o in zip(REG,out)))
        if arm=='P': Pout=out
        else:
            print("   PI - P: "+'   '.join(f"{r}: {(o-p).mean():+.3f}+/-{2.093*(o-p).std(ddof=1)/sqrt(len(o)):.3f}" for r,o,p in zip(REG,out,Pout)))
