"""Unlagged-sensor PI governor vs baselines on the paper's quadratic benchmark.
Python 3 + numpy only.

"Unlagged" = mu_t is computed from z_t, which already contains the CURRENT gradient g_t
(rho_t = mean g_t^2 / vhat_{t-1}; z_t = b*z_{t-1} + (1-b)*rho_t; mu_t from z_t).
"Predictable" (reference only) = mu_t from z_{t-1}, i.e. it only sees g_{t-1}.

Setup (same as robust_shift_exp.py): 50 params, curvature 1..sqrt(1000), 2000 steps,
score = mean true objective (sum form) over steps 201-2000, lower is better.
Each method's eta (and beta2 for 'AdamW tuned beta2') is tuned ONLY on 'jump x5'
using 10 tuning seeds, then evaluated on 20 fresh seeds on every regime.
Seeds differ from earlier runs (3000+/4000+) so this is an independent replication.
"""
import numpy as np

N, T = 50, 2000
A = np.geomspace(1, np.sqrt(1000), N)

# regime = (kind, sigma_before, sigma_after, switch_fraction)
REG = {
    'jump x5 (tuned here)': ('jump', 1, 5, .5),
    'jump x10':             ('jump', 1, 10, .5),
    'drift 1->10':          ('drift', 1, 10),
    'stationary':           ('jump', 1, 1, .5),
    'DROP 5->1':            ('jump', 5, 1, .5),
}


def sig(r, t):
    if r[0] == 'jump':
        return r[1] if t < r[3] * T else r[2]
    return r[1] + (r[2] - r[1]) * t / (T - 1)


def run(reg, cfg, eta, b2, seeds, b1=.9, brho=.9, c=.1, M=2., Tw=50, Imax=20., ea=1e-8, ev=1e-16):
    S = len(seeds)
    th = np.empty((S, N)); xi = np.empty((T, S, N))
    for i, s in enumerate(seeds):
        r = np.random.default_rng(s)
        th[i] = r.normal(size=N)
        xi[:, i, :] = r.normal(size=(T, N))
    m = np.zeros((S, N)); v = np.zeros((S, N))
    z = np.ones(S); I = np.zeros(S)
    sc = 0.; cnt = 0; k = cfg['kind']
    for t in range(1, T + 1):
        g = A * th + sig(REG[reg], t - 1) * xi[t - 1]
        if k == 'gclip':
            g = g * np.minimum(1, 1 / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-30))
        vp = v / (1 - b2 ** (t - 1)) if t > 1 else None          # vhat_{t-1}
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** t); vh = v / (1 - b2 ** t)
        mu = np.ones(S)
        if k == 'stable':
            mu = 1 / np.maximum(1, np.sqrt(np.mean(g * g / np.maximum(vh, ea ** 2), axis=1)))
        elif k == 'gov' and t > Tw:
            rho = np.mean(g * g / (vp + ev), axis=1)
            kp, ki = cfg['kp'], cfg['ki']
            if cfg.get('delay'):       # predictable: gain from state BEFORE g_t
                mu = np.clip(np.exp(-(kp * np.log(np.maximum(z, 1e-30)) + ki * I)), c, M)
            z = brho * z + (1 - brho) * rho
            e = np.log(np.maximum(z, 1e-30))
            I = np.clip(I + e, -Imax, Imax)
            if not cfg.get('delay'):   # UNLAGGED: gain uses z_t, which contains g_t
                mu = np.clip(np.exp(-(kp * e + ki * I)), c, M)
        th = th - eta * mu[:, None] * mh / (np.sqrt(vh) + ea)
        if t > 200:
            sc += 0.5 * np.sum(A * th * th, axis=1); cnt += 1
    sc = sc / cnt
    return np.where(np.isfinite(sc), sc, 1e9)       # per-seed scores


RATES = [.0025, .005, .01, .02, .04, .08]
METHODS = {
    'AdamW':                       dict(kind='adam'),
    'AdamW tuned beta2':           dict(kind='adam'),
    'Global clip':                 dict(kind='gclip'),
    'StableAdamW':                 dict(kind='stable'),
    'PI governor, UNLAGGED (g_t)': dict(kind='gov', kp=.5, ki=.02),
    'PI governor, predictable (g_t-1) [ref]': dict(kind='gov', kp=.5, ki=.02, delay=True),
}
tune = list(range(3000, 3010)); ev = list(range(4000, 4020)); cal = 'jump x5 (tuned here)'

chosen, res, raw = {}, {}, {}
for name, cfg in METHODS.items():
    b2s = [.9, .99, .999] if name == 'AdamW tuned beta2' else [.999]
    chosen[name] = min(((e, b) for e in RATES for b in b2s),
                       key=lambda p: run(cal, cfg, p[0], p[1], tune).mean())
    raw[name] = {r: run(r, cfg, chosen[name][0], chosen[name][1], ev) for r in REG}
    res[name] = [raw[name][r].mean() for r in REG]

cols = list(REG)
print('mean objective (steps 201-2000), 20 fresh seeds, lower = better; eta tuned only on jump x5')
print('chosen (eta, beta2):', chosen)
print(f"{'':42s}" + ''.join(f"{c[:12]:>14s}" for c in cols))
for n, row in res.items():
    print(f"{n:42s}" + ''.join(f"{x:14.3f}" for x in row))

# paired differences: UNLAGGED PI minus each baseline (negative = PI better), 95% t-interval
from math import sqrt
tcrit = 2.093  # t_{0.975, df=19}
print('\nPaired difference  (UNLAGGED PI) minus baseline;  negative = PI better.  mean +/- 95% half-width')
pi = raw['PI governor, UNLAGGED (g_t)']
for b in ['AdamW', 'AdamW tuned beta2', 'Global clip', 'StableAdamW',
          'PI governor, predictable (g_t-1) [ref]']:
    out = []
    for r in REG:
        d = pi[r] - raw[b][r]
        out.append(f"{d.mean():+.3f}+/-{tcrit * d.std(ddof=1) / sqrt(len(d)):.3f}")
    print(f"vs {b:40s}" + ''.join(f"{o:>20s}" for o in out))
