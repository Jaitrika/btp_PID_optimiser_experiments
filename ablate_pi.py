"""Ablation: where does the governor's win come from?  Python 3 + numpy only (also runs on Colab as is).
Quadratic benchmark: 50 params, curvature 1..sqrt(1000), 2000 steps, score = mean true objective
(sum form) over steps 201-2000, lower is better. eta tuned ONLY on 'jump x5' (10 seeds), eval on 20 fresh seeds.

Ingredients toggled:
  sensor  : 'lag' = g_t^2 / vhat_{t-1}   (what AdaGC/StableAdamW-style lagged ideas use)
            'cur' = g_t^2 / vhat_t       (current sensor, includes g_t in the denominator)
  smooth  : z = EMA of the sensor (beta_rho=.9)   vs   raw sensor (beta_rho=0)
  kp, ki  : P and I gains
  M       : multiplier cap. M=1 means the governor can only BRAKE, never speed up.
"""
import numpy as np
from math import sqrt

N, T = 50, 2000
A = np.geomspace(1, np.sqrt(1000), N)
REG = {'jump x5 (tuned)': ('jump', 1, 5, .5), 'jump x10': ('jump', 1, 10, .5),
       'drift 1->10': ('drift', 1, 10), 'stationary': ('jump', 1, 1, .5), 'DROP 5->1': ('jump', 5, 1, .5)}


def sig(r, t):
    return (r[1] if t < r[3] * T else r[2]) if r[0] == 'jump' else r[1] + (r[2] - r[1]) * t / (T - 1)


def run(reg, cfg, eta, seeds, b1=.9, b2=.999, c=.1, Tw=50, Imax=20., ea=1e-8, ev=1e-16):
    S = len(seeds); th = np.empty((S, N)); xi = np.empty((T, S, N))
    for i, s in enumerate(seeds):
        r = np.random.default_rng(s); th[i] = r.normal(size=N); xi[:, i, :] = r.normal(size=(T, N))
    m = np.zeros((S, N)); v = np.zeros((S, N)); z = np.ones(S); I = np.zeros(S); sc = 0.; cnt = 0
    for t in range(1, T + 1):
        g = A * th + sig(REG[reg], t - 1) * xi[t - 1]
        vp = v / (1 - b2 ** (t - 1)) if t > 1 else None
        m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** t); vh = v / (1 - b2 ** t); mu = np.ones(S)
        if cfg['kind'] == 'gov' and t > Tw:
            den = (vp if cfg['sensor'] == 'lag' else vh) + ev
            rho = np.mean(g * g / den, axis=1)
            br = cfg['brho']
            z = br * z + (1 - br) * rho
            e = np.log(np.maximum(z, 1e-30)); I = np.clip(I + e, -Imax, Imax)
            mu = np.clip(np.exp(-(cfg['kp'] * e + cfg['ki'] * I)), c, cfg['M'])
        th = th - eta * mu[:, None] * mh / (np.sqrt(vh) + ea)
        if t > 200: sc += 0.5 * np.sum(A * th * th, axis=1); cnt += 1
    sc = sc / cnt
    return np.where(np.isfinite(sc), sc, 1e9)


def G(sensor, brho, kp, ki, M): return dict(kind='gov', sensor=sensor, brho=brho, kp=kp, ki=ki, M=M)


ARMS = {
    'AdamW (no governor)':                          dict(kind='adam'),
    '1. clip-only, lagged sensor (no smooth/I)':    G('lag', 0.0, .5, 0, 1.),
    '2. clip-only, CURRENT sensor (StableAdamW-like)': G('cur', 0.0, .5, 0, 1.),
    '3. + smoothing, lagged, brake-only (M=1)':     G('lag', .9, .5, 0, 1.),
    '4. + smoothing, lagged, can boost (M=2) = P':  G('lag', .9, .5, 0, 2.),
    '5. P, CURRENT sensor':                         G('cur', .9, .5, 0, 2.),
    '6. PI, lagged (paper)':                        G('lag', .9, .5, .02, 2.),
    '7. PI, CURRENT sensor':                        G('cur', .9, .5, .02, 2.),
    '8. PI, lagged, brake-only (M=1)':              G('lag', .9, .5, .02, 1.),
    '9. P kp=2, lagged':                            G('lag', .9, 2., 0, 2.),
    '10. P kp=2, CURRENT sensor':                   G('cur', .9, 2., 0, 2.),
}
RATES = [.0025, .005, .01, .02, .04, .08]
tune = list(range(5000, 5010)); ev = list(range(6000, 6020))
res, raw = {}, {}
for n, cfg in ARMS.items():
    eta = min(RATES, key=lambda e: run('jump x5 (tuned)', cfg, e, tune).mean())
    raw[n] = {r: run(r, cfg, eta, ev) for r in REG}
    res[n] = [raw[n][r].mean() for r in REG]
    print(f"{n:50s} eta={eta:<6}" + ''.join(f"{x:10.3f}" for x in res[n]), flush=True)
print(f"\n{'':57s}" + ''.join(f"{c[:10]:>10s}" for c in REG))
base = raw['AdamW (no governor)']
print('\nPaired diff vs AdamW at the same protocol is large for all governors; the interesting contrasts are:')
def diff(a, b):
    out = []
    for r in REG:
        d = raw[a][r] - raw[b][r]; out.append(f"{d.mean():+.3f}+/-{2.093 * d.std(ddof=1) / sqrt(len(d)):.3f}")
    print(f"  {a[:34]:34s} minus {b[:34]:34s}", '  '.join(out))
diff('4. + smoothing, lagged, can boost (M=2) = P', '5. P, CURRENT sensor')   # does the lag matter?
diff('6. PI, lagged (paper)', '7. PI, CURRENT sensor')
diff('6. PI, lagged (paper)', '4. + smoothing, lagged, can boost (M=2) = P')   # does the I term matter?
diff('6. PI, lagged (paper)', '8. PI, lagged, brake-only (M=1)')               # does boosting matter?
diff('4. + smoothing, lagged, can boost (M=2) = P', '3. + smoothing, lagged, brake-only (M=1)')
diff('3. + smoothing, lagged, brake-only (M=1)', '1. clip-only, lagged sensor (no smooth/I)')  # does smoothing matter?
