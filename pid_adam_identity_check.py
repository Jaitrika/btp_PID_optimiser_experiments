"""Checks on AdaPID (Weng et al. 2022; as written in Saini et al. 2605.21968, Eqs 11-16).

1. D_t = (1-g) * (grad_t - m_{t-1})          (D term is the 'innovation')
2. I_t = m_t / (1-g)                         (I term is Adam's momentum, rescaled)
3. AdaPID with K_d = 0  ==  Adam(beta1=g, beta2=beta, lr=eta*K_i/(1-g), no m bias-correction)
4. Size of the D term vs the I term at the paper's default gains.
numpy only. Python 3.10+.
"""
import numpy as np

g_, beta, eta, Ki, Kd, sig = 0.9, 0.99, 1e-3, 0.5, 1.0, 1e-8


def adapid(grads, Ki=Ki, Kd=Kd):
    """Returns per-step (I-term contribution, D-term contribution, total step) for a fixed gradient stream."""
    n = grads.shape[1]
    I = np.zeros(n); D = np.zeros(n); v = np.zeros(n); d = np.zeros(n); gp = np.zeros(n)
    out = []
    for t, g in enumerate(grads, 1):
        dg = g - gp; gp = g
        I = g_ * I + g
        D = g_ * D + (1 - g_) * dg
        v = beta * v + (1 - beta) * g * g
        d = beta * d + (1 - beta) * dg * dg
        vh, dh = v / (1 - beta ** t), d / (1 - beta ** t)
        a = Ki * I / (np.sqrt(vh) + sig)
        b = Kd * D / (np.sqrt(dh) + sig)
        out.append((a, b, eta * (a + b)))
    return out


def adam_nobc(grads, lr, b1, b2):
    n = grads.shape[1]
    m = np.zeros(n); v = np.zeros(n); steps = []
    for t, g in enumerate(grads, 1):
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        steps.append(lr * m / (np.sqrt(v / (1 - b2 ** t)) + sig))
    return steps


rng = np.random.RandomState(0)
T, n = 5000, 200
# gradient stream: weak signal + noise, plus an AR(1) wobble so it is not white
signal = 0.1 * np.sin(np.linspace(0, 20, T))[:, None] * rng.randn(1, n)
noise = rng.randn(T, n)
grads = signal + noise

# 1 + 2: identities
m = np.zeros(n); I = np.zeros(n); D = np.zeros(n); gp = np.zeros(n)
e1 = e2 = 0.0
for g in grads:
    D = g_ * D + (1 - g_) * (g - gp); gp = g
    innov = (1 - g_) * (g - m)            # uses m_{t-1}
    e1 = max(e1, np.abs(D - innov).max())
    m = g_ * m + (1 - g_) * g
    I = g_ * I + g
    e2 = max(e2, np.abs(I - m / (1 - g_)).max())
print("1. max |D_t - (1-g)(grad_t - m_{t-1})| over all t, coords: %.2e" % e1)
print("2. max |I_t - m_t/(1-g)|:                                  %.2e" % e2)

# 3: K_d = 0 equivalence with Adam
ada = adapid(grads, Kd=0.0)
adm = adam_nobc(grads, lr=eta * Ki / (1 - g_), b1=g_, b2=beta)
e3 = max(np.abs(a[2] - s).max() for a, s in zip(ada, adm))
print("3. AdaPID(K_d=0) step vs Adam(b1=.9,b2=.99,lr=%.4f): max abs diff %.2e (step size ~%.1e)" % (
    eta * Ki / (1 - g_), e3, np.abs(adm[-1]).mean()))

# 4: relative size of D vs I contribution
full = adapid(grads)
rI = np.sqrt(np.mean([np.mean(a[0] ** 2) for a in full[500:]]))
rD = np.sqrt(np.mean([np.mean(a[1] ** 2) for a in full[500:]]))
print("4. RMS of K_i I/sqrt(v) = %.3f   RMS of K_d D/sqrt(d) = %.3f   ratio D/I = %.1f%%" % (rI, rD, 100 * rD / rI))

# 4b: ratio d_hat / v_hat (how the D normaliser compares to v)
v = np.zeros(n); d = np.zeros(n); gp = np.zeros(n); ratios = []
for t, g in enumerate(grads, 1):
    dg = g - gp; gp = g
    v = beta * v + (1 - beta) * g * g; d = beta * d + (1 - beta) * dg * dg
    if t > 500: ratios.append(np.mean(d) / np.mean(v))
print("   d_hat / v_hat (noise-dominated stream): %.2f   (2.0 = white noise)" % np.mean(ratios))

# 5: closed form of the AdaPID numerator as  c_g * g_t + c_m * m_{t-1}  (if both normalisers were sqrt(v))
a = Ki / (1 - g_)                      # weight on m_t
b = Kd * (1 - g_) / np.sqrt(2.0)       # weight on innovation, using d ~ 2v
cg, cm = a * (1 - g_) + b, a * g_ - b
print("5. numerator ~ %.4f * g_t + %.4f * m_{t-1}  -> fraction on current gradient nu = %.3f (Adam b1=.9 would be %.3f)" % (
    cg, cm, cg / (cg + cm), 1 - g_))
