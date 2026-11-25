"""Real-governor worked examples for the BTP report v2.  Python 3 + numpy.
Equations of the governor: rho=g^2/vhat_{t-1}; z=EMA(rho); e=log(z/rho*);
I=clip(I+e); u=kp*e+ki*I+kd*(e-e_prev); mu=clip(exp(-u),c,M).
Teaching setting: beta1=.9, beta2=.99, no warm-up, state starts at steady state (m=v=1)."""
import numpy as np
from math import log, exp, sqrt
B1,B2,BR=.9,.99,.9; KP,KI,KD=.5,.02,.05; C,M=.1,2.

def gov(rhos, kp, ki, kd):
    z=1.;I=0.;ep=0.;out=[]
    for r in rhos:
        z=BR*z+(1-BR)*r; e=log(z); I=max(-20,min(20,I+e)); D=kd*(e-ep); ep=e
        u=kp*e+ki*I+D; mu=min(M,max(C,exp(-u))); out.append((r,z,e,I,kp*e,ki*I,D,u,mu))
    return out

def exA(n=40):
    g=[10.]+[1.]*(n-1); m=1.;v=1.;rows=[]
    adam=[];clip=[];rho=[]
    for gt in g:
        rho.append(gt*gt/v)           # lagged sensor: old v
        m=B1*m+(1-B1)*gt; v=B2*v+(1-B2)*gt*gt
        d=m/sqrt(v); adam.append(d)
        r=gt*gt/v; clip.append(d/max(1,sqrt(r)))   # StableAdamW-style: current v, brake only
    P=gov(rho,KP,0,0); PI=gov(rho,KP,KI,0); PID=gov(rho,KP,KI,KD)
    return g,adam,clip,rho,P,PI,PID
g,adam,clip,rho,P,PI,PID=exA()
print("EXAMPLE A  (spike of 10 at t=1, then 1s).  step = eta*mu*d with eta=1")
print(" t   g   rho    z      e     Pterm  Iterm  Dterm   u     muP   muPI  muPID | Adam   Clip   GovP   GovPI  GovPID")
for t in list(range(10))+[14,19,29,39]:
    r,z,e,I,pp,ii,dd,u,mu=PID[t]
    print(f"{t+1:2d} {g[t]:4.0f} {rho[t]:6.2f} {z:6.3f} {e:6.3f} {pp:6.3f} {ii:6.3f} {dd:6.3f} {u:6.3f} {P[t][8]:.3f} {PI[t][8]:.3f} {mu:.3f} | {adam[t]:.3f} {clip[t]:.3f} {adam[t]*P[t][8]:.3f} {adam[t]*PI[t][8]:.3f} {adam[t]*mu:.3f}")
def S(a,lo,hi): return sum(a[lo:hi])
for nm,a in [("Adam",adam),("clip",clip),("GovP",[adam[i]*P[i][8] for i in range(40)]),("GovPI",[adam[i]*PI[i][8] for i in range(40)]),("GovPID",[adam[i]*PID[i][8] for i in range(40)])]:
    print(f"{nm:7s} sum steps1-3 {S(a,0,3):.3f}  sum steps4-40 {S(a,3,40):.3f} (calm run would be 3.000 and 37.000)")

print("\nEXAMPLE B  (noise std 1 -> 3 at t=1, stays).  scale = mu/sqrt(v) ; ideal = 1/3")
v=1.;rows=[];rho=[]
vs=[]
for k in range(1,401):
    rho.append(9/v); vs.append(v)   # sensor uses old v
    v=B2*v+(1-B2)*9
P=gov(rho,KP,0,0);PI=gov(rho,KP,KI,0);PID=gov(rho,KP,KI,KD)
print(" k    v     rho    z      e     Pterm Iterm  Dterm   muP  muPI  muPID | Adam 1/sqrtv  P-scale  PI-scale  PID-scale")
for k in list(range(1,9))+[10,15,20,30,50,100,200,400]:
    i=k-1; r,z,e,I,pp,ii,dd,u,mu=PID[i]; s=1/sqrt(vs[i])
    print(f"{k:3d} {vs[i]:6.3f} {rho[i]:6.3f} {z:6.3f} {e:6.3f} {pp:6.3f} {ii:6.3f} {dd:6.3f} {P[i][8]:.3f} {PI[i][8]:.3f} {mu:.3f} | {s:.3f}   {s*P[i][8]:.3f}  {s*PI[i][8]:.3f}  {s*mu:.3f}")
print("I at k=400:",PI[399][3], "muPI at 400:",PI[399][8])
