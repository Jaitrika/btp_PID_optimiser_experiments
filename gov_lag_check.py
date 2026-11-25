import numpy as np, sys
# Scalar-separable quadratic f = k/2 * sum theta_j^2 (n coords = one tensor), noise std jumps s1 -> s2 at t=0.
# Compares AdamW vs P-governor (kp=1/2, rho*=1, beta_rho=.9, floors c,M) against closed-form quasi-stationary prediction.
def run(s1, s2, eta=0.01, k=1.0, n=50, R=2000, b1=0.9, b2=0.999, brho=0.9, c=0.1, M=2.0,
        burn=4000, T=3000, seed=0, gov=True):
    rng=np.random.default_rng(seed)
    th=rng.normal(size=(R,n))*np.sqrt(eta/s1/(2*k))*s1   # approx stationary start
    m=np.zeros((R,n)); v=np.full((R,n), s1**2); z=np.ones(R)
    L=[]; MU=[]
    for t in range(burn+T):
        s = s1 if t<burn else s2
        g = k*th + s*rng.normal(size=(R,n))
        vprev=v.copy()
        m=b1*m+(1-b1)*g; v=b2*v+(1-b2)*g*g
        # steady-state: bias correction ~1 (we start both at stationarity)
        rho=np.mean(g*g/(vprev+1e-16),axis=1)
        z=brho*z+(1-brho)*rho
        mu=np.clip(np.exp(-0.5*np.log(z)),c,M) if gov else np.ones(R)
        th=th-eta*mu[:,None]*m/(np.sqrt(v)+1e-8)
        if t>=burn:
            L.append(0.5*k*np.mean(th**2)); MU.append(mu.mean())
    return np.array(L), np.array(MU)
if __name__=="__main__":
    eta=0.01
    for s2 in [3.0,5.0,0.7]:
        La,_=run(1.0,s2,eta=eta,gov=False); Lg,MUg=run(1.0,s2,eta=eta,gov=True,seed=0)
        kk=np.arange(len(La)); v=s2**2-(s2**2-1)*0.999**kk
        predA=eta*s2**2/(4*np.sqrt(v)); predG=np.full_like(predA,eta*s2/4)
        pre=eta*1.0/4
        print(f"--- jump 1 -> {s2}  (pre-jump floor eta*s1/4={pre:.5f})")
        for k_ in [50,300,1000,2000,2999]:
            print(f" k={k_:5d}  Adam sim {La[k_]:.5f} pred {predA[k_]:.5f} | Gov sim {Lg[k_]:.5f} pred {predG[k_]:.5f} | mu sim {MUg[k_]:.3f} pred {np.sqrt(v[k_])/s2:.3f}")
        print(f" window mean (3000 steps): Adam {La.mean():.5f}  Gov {Lg.mean():.5f}  ratio {Lg.mean()/La.mean():.3f}  pred ratio {predG.mean()/predA.mean():.3f}")
