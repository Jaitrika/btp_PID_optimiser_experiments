# v1 was degenerate: with a STATIONARY fixed point, target-network drift is self-extinguishing
# (q_eff ~ E[(e-em)^2] -> 0), so eta->0 monotonically wins. No optimum.
# Real persistent non-stationarity in RL = POLICY IMPROVEMENT moving the critic's target.
#   theta*_{t+1} = theta*_t + nu_t, Var(nu)=qp   (persistent)
#   e_{t+1}  = e_t - eta h (e_t - gamma em_t) + eta xi_t - nu_t
#   em_{t+1} = (1-tau) em_t + tau e_t - nu_t
# Cov(w) = [[eta^2 r + qp, qp],[qp, qp]].  Interior optima in BOTH eta and tau?
import numpy as np

def steady(eta, tau, gamma, h, r, qp):
    M = np.array([[1 - eta * h, eta * h * gamma], [tau, 1 - tau]])
    if max(abs(np.linalg.eigvals(M))) >= 1 - 1e-9: return None
    W = np.array([eta * eta * r + qp, qp, qp, qp])
    try: S = np.linalg.solve(np.eye(4) - np.kron(M, M), W).reshape(2, 2)
    except np.linalg.LinAlgError: return None
    return S if S[0, 0] > 0 else None

def best_of(gamma, h, r, qp, ETAS, TAUS):
    b = None
    for eta in ETAS:
        for tau in TAUS:
            S = steady(eta, tau, gamma, h, r, qp)
            if S is None: continue
            if b is None or S[0, 0] < b[2]: b = (eta, tau, S[0, 0])
    return b

ETAS = np.logspace(-3.5, -0.32, 110)
TAUS = np.logspace(-3.5, -0.01, 110)
h, r, gamma = 1.0, 1.0, 0.99
qs = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]

print("Joint optimum over (eta,tau) under persistent policy drift qp.  h=r=1, gamma=0.99")
print("      qp |    eta*     tau* |  E[e^2]* | interior?")
print("-" * 62)
res = {}
for qp in qs:
    b = best_of(gamma, h, r, qp, ETAS, TAUS)
    if b is None: print(f" {qp:7.1e} |  none stable"); continue
    e, t, v = b; res[qp] = b
    inter = "yes" if (ETAS[0] < e < ETAS[-1] and TAUS[0] < t < TAUS[-1]) else "EDGE"
    print(f" {qp:7.1e} | {e:8.5f} {t:8.5f} | {v:8.2e} | {inter}")

ks = sorted(res); lq = np.log(ks)
ae = np.polyfit(lq, np.log([res[k][0] for k in ks]), 1)[0]
at = np.polyfit(lq, np.log([res[k][1] for k in ks]), 1)[0]
print(f"\nScaling:  eta* ~ qp^{ae:.3f}    tau* ~ qp^{at:.3f}")
print("  (exogenous single-walk theory gives eta* ~ q^0.5 at small q)")

qp = 1e-5; eo, to, vo = res[qp]
print(f"\nDoes tau matter? penalty for wrong tau at optimal eta={eo:.5f} (qp={qp:g}):")
for tau in (to/10, to/3, to, to*3, min(to*10, 0.97)):
    S = steady(eo, tau, gamma, h, r, qp)
    print(f"  tau={tau:8.5f} unstable" if S is None else
          f"  tau={tau:8.5f} -> E[e^2]={S[0,0]:.3e}  ({S[0,0]/vo:5.2f}x optimal)")

print(f"\nDoes gamma shift the optimum? (qp=1e-5)")
for g in (0.0, 0.5, 0.9, 0.99, 0.999):
    b = best_of(g, h, r, 1e-5, ETAS, TAUS)
    print(f"  gamma={g:6.3f} -> eta*={b[0]:.5f} tau*={b[1]:.5f} E[e^2]*={b[2]:.3e}")
