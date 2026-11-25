#!/usr/bin/env python3
"""
gate_drift.py -- Gate 0 for "optimizers under drift" (plan: direction_nonstationary_2026-11-10.md)

Needs numpy; torch is optional (GPU strongly recommended for `pmnist`). Python >= 3.8.

Subcommands
  selftest   gradient check + optimizer checks on the numpy backend (seconds). RUN THIS FIRST.
  kalman     closed-form check of the theory seed (numpy, seconds):
             EMA with gain K tracking a random walk (var q) in noise (var r):
               steady MSE  P(K) = ((1-K)^2 q + K^2 r) / (K (2-K)),   optimal K solves K^2/(1-K) = q/r.
             Also prints how much fixed Adam-style betas (beta1 = 1-K) lose as q/r changes.
  quad       Adam on a drifting noisy quadratic (numpy, ~1 min). Does the best beta1 move with drift speed,
             and how much does per-regime tuning beat one fixed config?
  pmnist     the main test: continual permuted-MNIST, tasks switch every S steps (S = drift speed).
             Prints Gate 0a / 0b / 0c verdicts. Resumable (stage results cached in --out).

Gates (thresholds fixed BEFORE seeing results; the numbers are my proposals)
  0a problem exists : default Adam's online loss on the last 10% of tasks exceeds the first 10% of tasks
                      by > 2 SE AND > GAP_MIN, in >= 1 regime.
  0b headroom       : R (robustness) : per-regime tuned Adam beats BOTH one globally-fixed config AND default+reset
                                       by > 2 SE and > REL_MIN (relative), in >= 2 regimes; or
                      L (layerwise)  : per-layer (beta1,beta2) beats the tuned global one the same way, in >= 2 regimes.
  0c sensor         : Spearman( measured q/r , oracle K = 1-beta1 ) >= RHO_MIN across regimes,
                      q/r measured with the split-batch noise copy (D=(G1-G2)/2 gives r; q = excess gradient-increment energy).
  Selection uses seeds 0..sel-1; all gate numbers use disjoint evaluation seeds 100.. (no selection bias).

Not included (needs checking against the authors' code): Adam-Rel (2412.17113). Add it before Gate 1.
Timing/GPU numbers in the plan are estimates, not measured.
"""
import argparse, copy, gzip, itertools, json, math, struct, sys, time, urllib.request
from pathlib import Path
import numpy as np

GAP_MIN, REL_MIN, RHO_MIN = 0.03, 0.03, 0.7
DEFAULT = dict(lr=1e-3, b1=0.9, b2=0.999, eps=1e-8)


# ----------------------------------------------------------------------------- stats helpers
def pdiff(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    return float(d.mean()), (float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"))


def rank(x):
    x = np.asarray(x, float)
    o = np.argsort(x)
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(x, y):
    rx, ry = rank(x), rank(y)
    if len(rx) < 3 or rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def passes(mean, se, base, rel_min=REL_MIN):
    """mean>0 means 'ours better'. True if > 2 SE and > rel_min * base."""
    return bool(np.isfinite(se) and mean > 2 * se and mean > rel_min * abs(base))


# ----------------------------------------------------------------------------- kalman
def p_formula(K, q, r):
    return ((1 - K) ** 2 * q + K ** 2 * r) / (K * (2 - K))


def k_star(q, r):
    s = q / r
    return (-s + math.sqrt(s * s + 4 * s)) / 2


def cmd_kalman(a):
    rng = np.random.RandomState(0)
    Ks = np.linspace(0.005, 0.95, 190)
    n, T = 400, a.steps
    print("EMA tracking of a random walk: simulated vs closed-form steady-state MSE (r=1)")
    print("%9s | %8s %8s | %12s %12s | %10s | %12s %12s" % (
        "q/r", "K*_form", "K*_sim", "P(K*)_form", "P(K*)_sim", "max rel.err", "P(b1=.9)/P*", "P(b1=.99)/P*"))
    for ratio in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        q, r = ratio, 1.0
        mu = np.zeros(n)
        est = np.zeros((len(Ks), n))
        acc, cnt = np.zeros(len(Ks)), 0
        for t in range(T):
            mu += rng.randn(n) * math.sqrt(q)
            y = mu + rng.randn(n) * math.sqrt(r)
            est += Ks[:, None] * (y[None] - est)
            if t >= T // 2:
                acc += ((mu[None] - est) ** 2).mean(1)
                cnt += 1
        sim = acc / cnt
        form = p_formula(Ks, q, r)
        ks = k_star(q, r)
        ps = p_formula(ks, q, r)
        rel = np.max(np.abs(sim - form) / form)
        print("%9.0e | %8.4f %8.4f | %12.5f %12.5f | %10.3f | %12.2f %12.2f" % (
            ratio, ks, Ks[np.argmin(sim)], ps, sim.min(), rel, p_formula(0.1, q, r) / ps, p_formula(0.01, q, r) / ps))
    print("\nReading: K*_sim should sit near K*_form (grid step 0.005); P ratio columns = how much MSE Adam's beta1 loses")
    print("vs the optimal gain at that drift/noise ratio. Large ratios = the failure region the theory quantifies.")


# ----------------------------------------------------------------------------- quad
def cmd_quad(a):
    rng = np.random.RandomState(a.seed)
    d, S, T = a.dim, a.nseeds, a.steps
    h = np.logspace(-1, 0, d)
    lrs, b1s, b2s = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2], [0.0, 0.25, 0.5, 0.8, 0.9, 0.95, 0.99], [0.9, 0.99, 0.999]
    cfg = list(itertools.product(lrs, b1s, b2s))
    C = len(cfg)
    lr = np.array([c[0] for c in cfg])[:, None, None]
    b1 = np.array([c[1] for c in cfg])[:, None, None]
    b2 = np.array([c[2] for c in cfg])[:, None, None]
    rhos = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    sig = 1.0
    L = {}
    for rho in rhos:
        th = np.zeros((C, S, d))
        tg = np.zeros((S, d))
        m, v = np.zeros_like(th), np.zeros_like(th)
        acc, cnt = np.zeros((C, S)), 0
        for t in range(1, T + 1):
            tg += rng.randn(S, d) * rho
            g = h * (th - tg[None]) + (rng.randn(S, d) * sig)[None]
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            th -= lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + 1e-8)
            if t > T // 2:
                acc += 0.5 * (h * (th - tg[None]) ** 2).sum(2)
                cnt += 1
        L[rho] = acc / cnt
        print("regime rho=%g done" % rho, flush=True)
    sel, ev = slice(0, S // 2), slice(S // 2, S)
    best = {r: int(np.argmin(L[r][:, sel].mean(1))) for r in rhos}
    rel = np.mean([L[r][:, sel].mean(1) / L[r][:, sel].mean(1).min() for r in rhos], axis=0)
    gfix = int(np.argmin(rel))
    print("\nglobally fixed config (lr,b1,b2) = %s" % (cfg[gfix],))
    print("%8s | %-22s | %6s %8s | %10s %10s | %s" % ("rho", "tuned (lr,b1,b2)", "b1*", "K_pred", "loss tuned", "loss fixed", "fixed-tuned (paired, eval seeds)"))
    for r in rhos:
        K = k_star(np.mean(h ** 2) * r * r, sig ** 2)
        lt, lf = L[r][best[r], ev], L[r][gfix, ev]
        m_, se = pdiff(lf, lt)
        print("%8.0e | %-22s | %6.2f %8.4f | %10.4f %10.4f | %+.4f +/- %.4f  (%s)" % (
            r, str(cfg[best[r]]), cfg[best[r]][1], K, lt.mean(), lf.mean(), m_, se,
            ">2SE" if passes(m_, se, lt.mean()) else "no clear gap"))
    print("\nK_pred = Kalman gain for q/r = mean(h^2) rho^2 / sigma^2 (gradient-mean drift vs gradient noise).")
    print("Compare 1-b1* with K_pred only for ORDER (feedback and Adam's v make it inexact). Idealised sanity test, not a gate.")


# ----------------------------------------------------------------------------- backends
class NPBE:
    def __init__(self, dtype=np.float32):
        self.name, self.dt = "numpy", dtype

    def f(self, a): return np.asarray(a, dtype=self.dt)
    def i(self, a): return np.asarray(a, dtype=np.int64)
    def zeros(self, *s): return np.zeros(s, self.dt)
    def sqrt(self, a): return np.sqrt(a)
    def exp(self, a): return np.exp(a)
    def log(self, a): return np.log(a)
    def sum(self, a, ax, keep=False): return a.sum(axis=ax, keepdims=keep)
    def max(self, a, ax, keep=False): return a.max(axis=ax, keepdims=keep)
    def mean(self, a, ax): return a.mean(axis=ax)
    def argmax(self, a, ax): return a.argmax(axis=ax)
    def fl(self, a): return a.astype(self.dt)
    def npy(self, a): return np.asarray(a)


class THBE:
    def __init__(self, device):
        import torch
        torch.set_grad_enabled(False)
        self.t, self.dev, self.name = torch, device, "torch:" + device

    def f(self, a): return self.t.as_tensor(np.ascontiguousarray(a, dtype=np.float32), device=self.dev)
    def i(self, a): return self.t.as_tensor(np.ascontiguousarray(a, dtype=np.int64), device=self.dev)
    def zeros(self, *s): return self.t.zeros(s, device=self.dev)
    def sqrt(self, a): return self.t.sqrt(a)
    def exp(self, a): return self.t.exp(a)
    def log(self, a): return self.t.log(a)
    def sum(self, a, ax, keep=False): return a.sum(dim=ax, keepdim=keep)
    def max(self, a, ax, keep=False): return a.amax(dim=ax, keepdim=keep)
    def mean(self, a, ax): return a.mean(dim=ax)
    def argmax(self, a, ax): return a.argmax(dim=ax)
    def fl(self, a): return a.float()
    def npy(self, a): return a.detach().cpu().numpy()


def get_backend(a):
    if a.backend in ("auto", "torch"):
        try:
            import torch
            return THBE("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            if a.backend == "torch":
                raise
    print("!! torch not found: using numpy backend (slow; fine only for --quick)")
    return NPBE()


# ----------------------------------------------------------------------------- data
MNIST_BASES = ["https://ossci-datasets.s3.amazonaws.com/mnist/", "https://storage.googleapis.com/cvdf-datasets/mnist/"]


def _fetch(d, name):
    p = d / name
    if p.exists():
        return p
    for base in MNIST_BASES:
        try:
            urllib.request.urlretrieve(base + name, p)
            return p
        except Exception as e:
            print("download failed from %s (%s)" % (base, e))
            if p.exists():
                p.unlink()
    return None


def load_mnist(data_dir):
    d = Path(data_dir)
    d.mkdir(parents=True, exist_ok=True)
    pi, pl = _fetch(d, "train-images-idx3-ubyte.gz"), _fetch(d, "train-labels-idx1-ubyte.gz")
    if pi is not None and pl is not None:
        with gzip.open(pi) as f:
            _, n, r, c = struct.unpack(">IIII", f.read(16))
            X = np.frombuffer(f.read(), np.uint8).reshape(n, r * c).astype(np.float32)
        with gzip.open(pl) as f:
            struct.unpack(">II", f.read(8))
            Y = np.frombuffer(f.read(), np.uint8).astype(np.int64)
        return (X / 255.0 - 0.1307) / 0.3081, Y, True
    print("!! MNIST unavailable -> synthetic 10-class data. Results are NOT meaningful for the gates.")
    rs = np.random.RandomState(0)
    protos = rs.randn(10, 784).astype(np.float32)
    Y = rs.randint(0, 10, 20000)
    return (protos[Y] + rs.randn(20000, 784).astype(np.float32)).astype(np.float32), Y.astype(np.int64), False


# ----------------------------------------------------------------------------- batched MLP + Adam (K independent models)
GROUP = [0, 0, 1, 1, 2, 2]  # params W1,b1,W2,b2,W3,b3 -> layer group


def init_params(be, K, H, seed):
    rs = np.random.RandomState(10_000 + seed)
    P = []
    for li, (fi, fo) in enumerate([(784, H), (H, H), (H, 10)]):
        W = (rs.randn(fi, fo) * math.sqrt((2.0 if li < 2 else 1.0) / fi))
        P.append(be.f(np.repeat(W[None], K, axis=0)))
        P.append(be.zeros(K, 1, fo))
    return P


def fwd_bwd(be, P, x, Y, yi):
    """x (B,784) shared; P batched over K models. Returns loss (K,), acc (K,), grads (list like P)."""
    W1, b1, W2, b2, W3, b3 = P
    B = x.shape[0]
    x3 = x[None]
    z1 = x3 @ W1 + b1
    a1 = z1 * (z1 > 0)
    z2 = a1 @ W2 + b2
    a2 = z2 * (z2 > 0)
    z3 = a2 @ W3 + b3
    m = be.max(z3, 2, True)
    e = be.exp(z3 - m)
    s = be.sum(e, 2, True)
    p = e / s
    logp = (z3 - m) - be.log(s)
    loss = -be.mean(be.sum(Y[None] * logp, 2), 1)
    acc = be.mean(be.fl(be.argmax(z3, 2) == yi[None]), 1)
    d3 = (p - Y[None]) / B
    gW3 = a2.swapaxes(1, 2) @ d3
    gb3 = be.sum(d3, 1, True)
    dz2 = (d3 @ W3.swapaxes(1, 2)) * (z2 > 0)
    gW2 = a1.swapaxes(1, 2) @ dz2
    gb2 = be.sum(dz2, 1, True)
    dz1 = (dz2 @ W2.swapaxes(1, 2)) * (z1 > 0)
    gW1 = x3.swapaxes(1, 2) @ dz1
    gb1 = be.sum(dz1, 1, True)
    return loss, acc, [gW1, gb1, gW2, gb2, gW3, gb3]


def make_cfg(rows):
    K = len(rows)
    out = {k: np.zeros((K, 3)) for k in ("lr", "b1", "b2", "eps")}
    out["reset"] = np.zeros(K, bool)
    for i, r in enumerate(rows):
        for k in ("lr", "b1", "b2", "eps"):
            out[k][i, :] = np.broadcast_to(np.asarray(r[k], dtype=np.float64), (3,))
        out["reset"][i] = bool(r.get("reset", False))
    return out


def row(lr, b1, b2, eps, reset=False):
    return dict(lr=float(lr), b1=[float(b1)] * 3, b2=[float(b2)] * 3, eps=float(eps), reset=bool(reset))


class BAdam:
    """Adam for K independent models, per-model and per-layer-group (lr, b1, b2, eps); optional state reset at task boundaries."""

    def __init__(self, be, P, cfg):
        self.be, self.K = be, P[0].shape[0]
        self.m = [be.zeros(*p.shape) for p in P]
        self.v = [be.zeros(*p.shape) for p in P]
        self.t = be.zeros(self.K, 1, 1)
        self.h = []
        for g in range(3):
            self.h.append({k: be.f(cfg[k][:, g]).reshape(self.K, 1, 1) for k in ("lr", "b1", "b2", "eps")})
        self.keep = 1.0 - be.f(cfg["reset"].astype(np.float64)).reshape(self.K, 1, 1)

    def reset(self):
        self.t = self.t * self.keep
        self.m = [x * self.keep for x in self.m]
        self.v = [x * self.keep for x in self.v]

    def step(self, P, G):
        be = self.be
        self.t = self.t + 1.0
        for g in range(3):
            h = self.h[g]
            c1 = 1.0 - h["b1"] ** self.t
            c2 = 1.0 - h["b2"] ** self.t
            for j in (2 * g, 2 * g + 1):
                self.m[j] = h["b1"] * self.m[j] + (1.0 - h["b1"]) * G[j]
                self.v[j] = h["b2"] * self.v[j] + (1.0 - h["b2"]) * G[j] * G[j]
                P[j] -= h["lr"] * (self.m[j] / c1) / (be.sqrt(self.v[j] / c2) + h["eps"])


def _task_stream(be, X, Yl, rs, N, B, steps):
    sub = rs.choice(len(X), N, replace=False)
    perm = rs.permutation(784)
    Xt = be.f(X[sub][:, perm])
    yt = Yl[sub]
    Yt = be.f(np.eye(10)[yt])
    yti = be.i(yt)
    bidx = be.i(rs.randint(0, N, size=(steps, B)))
    return Xt, Yt, yti, bidx


def run_batch(be, X, Yl, cfg, seed, S, T, N, B, H):
    """Train K models on a stream of permuted tasks (S steps each) for T steps. Returns step losses (T,K), accs (T,K)."""
    K = cfg["lr"].shape[0]
    rs = np.random.RandomState(seed)
    P = init_params(be, K, H, seed)
    opt = BAdam(be, P, cfg)
    L, A = be.zeros(T, K), be.zeros(T, K)
    t = 0
    for k in range(int(math.ceil(T / S))):
        steps = min(S, T - t)
        Xt, Yt, yti, bidx = _task_stream(be, X, Yl, rs, N, B, steps)
        if k > 0:
            opt.reset()
        for s in range(steps):
            ix = bidx[s]
            loss, acc, G = fwd_bwd(be, P, Xt[ix], Yt[ix], yti[ix])
            L[t], A[t] = loss, acc
            opt.step(P, G)
            t += 1
    return be.npy(L), be.npy(A)


def run_diag(be, X, Yl, rw, seed, S, T, N, B, H):
    """Default-style run (K=1) with the split-batch noise copy. Returns per-layer-group q/r estimates."""
    cfg = make_cfg([rw])
    rs = np.random.RandomState(seed)
    P = init_params(be, 1, H, seed)
    opt = BAdam(be, P, cfg)
    R, Q = be.zeros(T, 3), be.zeros(T, 3)
    Gprev, t, hb = None, 0, B // 2
    for k in range(int(math.ceil(T / S))):
        steps = min(S, T - t)
        Xt, Yt, yti, bidx = _task_stream(be, X, Yl, rs, N, B, steps)
        if k > 0:
            opt.reset()
        for s in range(steps):
            ix = bidx[s]
            _, _, G1 = fwd_bwd(be, P, Xt[ix[:hb]], Yt[ix[:hb]], yti[ix[:hb]])
            _, _, G2 = fwd_bwd(be, P, Xt[ix[hb:]], Yt[ix[hb:]], yti[ix[hb:]])
            G = [(u + w) * 0.5 for u, w in zip(G1, G2)]
            D = [(u - w) * 0.5 for u, w in zip(G1, G2)]
            for g in range(3):
                j = (2 * g, 2 * g + 1)
                R[t, g] = (D[j[0]] * D[j[0]]).sum() + (D[j[1]] * D[j[1]]).sum()
                if Gprev is not None:
                    Q[t, g] = ((G[j[0]] - Gprev[j[0]]) ** 2).sum() + ((G[j[1]] - Gprev[j[1]]) ** 2).sum()
            Gprev = G
            opt.step(P, G)
            t += 1
    R, Q = be.npy(R), be.npy(Q)
    n = np.array([784 * H + H, H * H + H, H * 10 + 10], float)
    t0 = T // 2
    r_g, dq_g = R[t0:].mean(0) / n, Q[t0:].mean(0) / n
    q_g = np.maximum(dq_g - 2 * r_g, 1e-30)
    tot_r, tot_dq = R[t0:].mean(0).sum(), Q[t0:].mean(0).sum()
    return dict(r=r_g.tolist(), q=q_g.tolist(), qr_group=(q_g / r_g).tolist(),
                qr_overall=float(max(tot_dq - 2 * tot_r, 1e-30) / tot_r))


def metric(L, frac=0.5):
    L = np.minimum(np.nan_to_num(L, nan=10.0, posinf=10.0, neginf=10.0), 10.0)
    return L[int(len(L) * (1 - frac)):].mean(0)


def task_gap(L, S):
    """late-minus-early mean online loss over tasks, per model (K,). Needs >= 10 tasks, else nan."""
    L = np.minimum(np.nan_to_num(L, nan=10.0, posinf=10.0, neginf=10.0), 10.0)
    nt = int(math.ceil(len(L) / S))
    if nt < 10:
        return np.full(L.shape[1], np.nan)
    per = np.stack([L[k * S:(k + 1) * S].mean(0) for k in range(nt)])
    q = max(1, int(0.1 * nt))
    return per[-q:].mean(0) - per[:q].mean(0)


# ----------------------------------------------------------------------------- selftest
def cmd_selftest(a):
    be = NPBE(np.float64)
    rs = np.random.RandomState(1)
    K, H, B = 2, 5, 7
    P = init_params(be, K, H, 0)
    for j in (1, 3, 5):
        P[j] = P[j] + 0.1 * rs.randn(*P[j].shape)
    P[0] = P[0][:, :, :]  # keep
    x = rs.randn(B, 784)
    yl = rs.randint(0, 10, B)
    Y, yi = np.eye(10)[yl], yl
    P[1], P[3], P[5] = P[1] + 0.05, P[3] + 0.05, P[5] + 0.05
    P[0] = P[0] + 0.01 * rs.randn(*P[0].shape)
    loss, acc, G = fwd_bwd(be, P, x, Y, yi)
    worst = 0.0
    for j in range(6):
        for _ in range(4):
            k = rs.randint(K)
            idx = tuple(rs.randint(s) for s in P[j].shape[1:])
            e = 1e-6
            P[j][(k,) + idx] += e
            lp = fwd_bwd(be, P, x, Y, yi)[0][k]
            P[j][(k,) + idx] -= 2 * e
            lm = fwd_bwd(be, P, x, Y, yi)[0][k]
            P[j][(k,) + idx] += e
            num = (lp - lm) / (2 * e)
            worst = max(worst, abs(num - G[j][(k,) + idx]) / (abs(num) + abs(G[j][(k,) + idx]) + 1e-12))
    print("gradient check: worst relative error %.2e  -> %s" % (worst, "OK" if worst < 1e-5 else "FAIL"))
    ok = worst < 1e-5
    # Adam vs textbook (K=1, per-group betas differ so check each group's formula on one param)
    be32 = NPBE(np.float64)
    Pa = init_params(be32, 1, H, 0)
    rowx = dict(lr=1e-2, b1=[0.9, 0.5, 0.99], b2=[0.999, 0.9, 0.99], eps=1e-8)
    opt = BAdam(be32, Pa, make_cfg([rowx]))
    ref = [p.copy() for p in Pa]
    m = [np.zeros_like(p) for p in Pa]
    v = [np.zeros_like(p) for p in Pa]
    for t in range(1, 6):
        _, _, G = fwd_bwd(be32, Pa, x, Y, yi)
        opt.step(Pa, [g.copy() for g in G])
        for j in range(6):
            g_ = GROUP[j]
            b1_, b2_ = rowx["b1"][g_], rowx["b2"][g_]
            m[j] = b1_ * m[j] + (1 - b1_) * G[j]
            v[j] = b2_ * v[j] + (1 - b2_) * G[j] ** 2
            ref[j] = ref[j] - rowx["lr"] * (m[j] / (1 - b1_ ** t)) / (np.sqrt(v[j] / (1 - b2_ ** t)) + 1e-8)
    err = max(float(np.abs(Pa[j] - ref[j]).max()) for j in range(6))
    print("Adam vs textbook (per-group betas): max abs diff %.2e -> %s" % (err, "OK" if err < 1e-9 else "FAIL"))
    ok = ok and err < 1e-9
    # identical rows -> identical trajectories; reset flag changes behaviour; reset zeroes state
    X = rs.randn(300, 784).astype(np.float32)
    Yl = rs.randint(0, 10, 300)
    be32 = NPBE(np.float32)
    cfg = make_cfg([row(1e-3, .9, .999, 1e-8), row(1e-3, .9, .999, 1e-8), row(1e-3, .9, .999, 1e-8, reset=True)])
    L, _ = run_batch(be32, X, Yl, cfg, 0, 20, 100, 100, 16, 8)
    same = np.allclose(L[:, 0], L[:, 1])
    diff = not np.allclose(L[:, 0], L[:, 2])
    print("identical configs give identical runs: %s ; reset config differs: %s" % (same, diff))
    ok = ok and same and diff
    d = run_diag(be32, X, Yl, row(1e-3, .9, .999, 1e-8), 0, 20, 100, 100, 16, 8)
    print("diag runs: qr_overall=%.3g, per-group q/r=%s" % (d["qr_overall"], np.round(d["qr_group"], 3).tolist()))
    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


# ----------------------------------------------------------------------------- pmnist
def cached(path, fn):
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text())
    out = fn()
    p.write_text(json.dumps(out))
    return out


def grid_axes(a):
    if a.grid == "full":
        return [3e-4, 1e-3, 3e-3, 1e-2], [0.0, 0.5, 0.8, 0.9, 0.95, 0.99], [0.9, 0.99, 0.999], [1e-8, 1e-5, 1e-3]
    return [1e-3], [0.8, 0.95], [0.99, 0.999], [1e-8, 1e-3]


def cmd_pmnist(a):
    if a.quick:
        a.steps, a.regimes, a.grid = 600, "25,100", "small"
        a.H, a.N, a.sel_seeds, a.eval_seeds = 32, 500, 1, 2
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    be = get_backend(a)
    X, Yl, real = load_mnist(a.data_dir)
    LRS, B1S, B2S, EPS = grid_axes(a)
    grid = [row(l, b1, b2, e) for l, b1, b2, e in itertools.product(LRS, B1S, B2S, EPS)]
    sel = list(range(a.sel_seeds))
    ev = [100 + i for i in range(a.eval_seeds)]
    T, N, B, H = a.steps, a.N, a.B, a.H
    regimes = [int(s) for s in a.regimes.split(",")]
    print("backend %s | real MNIST %s | grid %d configs | steps %d | regimes %s | sel seeds %s | eval seeds %s" % (
        be.name, real, len(grid), T, regimes, sel, ev), flush=True)

    # ---- stage 1: grid on selection seeds
    gm = {}
    for S in regimes:
        def stage():
            t0, ms = time.time(), []
            cfg = make_cfg(grid)
            for sd in sel:
                L, _ = run_batch(be, X, Yl, cfg, sd, S, T, N, B, H)
                ms.append(metric(L))
            print("  grid S=%d done in %.0fs" % (S, time.time() - t0), flush=True)
            return dict(metric=np.mean(ms, 0).tolist())
        gm[S] = np.array(cached(out / ("grid_S%d.json" % S), stage)["metric"])
    rel = np.mean([gm[S] / gm[S].min() for S in regimes], axis=0)
    gfix = grid[int(np.argmin(rel))]
    tuned = {S: grid[int(np.argmin(gm[S]))] for S in regimes}
    print("\nglobally-fixed config: lr=%g b1=%g b2=%g eps=%g" % (gfix["lr"], gfix["b1"][0], gfix["b2"][0], gfix["eps"]))

    # ---- stage 2: layerwise coordinate descent, stage 3: finalists on eval seeds, stage 4: q/r diagnostics
    fin_names = ["default", "default+reset", "global_fixed", "tuned", "tuned+reset", "layerwise"]
    R = {}
    for S in regimes:
        def lw_stage():
            cur = copy.deepcopy(tuned[S])
            t0 = time.time()
            for _ in range(a.cd_passes):
                for g in range(3):
                    cands = []
                    for b1, b2 in itertools.product(B1S, B2S):
                        c = copy.deepcopy(cur)
                        c["b1"][g], c["b2"][g] = b1, b2
                        cands.append(c)
                    cfg = make_cfg(cands)
                    m = np.mean([metric(run_batch(be, X, Yl, cfg, sd, S, T, N, B, H)[0]) for sd in sel], 0)
                    cur = cands[int(np.argmin(m))]
            print("  layerwise S=%d done in %.0fs: b1=%s b2=%s" % (S, time.time() - t0, cur["b1"], cur["b2"]), flush=True)
            return cur
        lw = cached(out / ("lw_S%d.json" % S), lw_stage)
        tr = copy.deepcopy(tuned[S])
        tr["reset"] = True
        dr = row(DEFAULT["lr"], DEFAULT["b1"], DEFAULT["b2"], DEFAULT["eps"], reset=True)
        fin_rows = [row(**DEFAULT), dr, gfix, tuned[S], tr, lw]

        def fin_stage():
            cfg = make_cfg(fin_rows)
            mets, gaps = [], []
            for sd in ev:
                L, _ = run_batch(be, X, Yl, cfg, sd, S, T, N, B, H)
                mets.append(metric(L))
                gaps.append(task_gap(L, S))
            return dict(metric=np.array(mets).tolist(), gap=np.array(gaps).tolist())
        fin = cached(out / ("final_S%d.json" % S), fin_stage)
        diag = cached(out / ("diag_S%d.json" % S), lambda: run_diag(be, X, Yl, row(**DEFAULT), ev[0], S, T, N, B, H))
        R[S] = dict(met=np.array(fin["metric"]), gap=np.array(fin["gap"]), diag=diag, lw=lw, tuned=tuned[S])

    # ---- report
    ix = {n: i for i, n in enumerate(fin_names)}
    print("\n== final online loss (mean of last 50%% of steps; eval seeds %s; lower is better)" % ev)
    print("%6s | %s" % ("S", " ".join("%14s" % n for n in fin_names)))
    for S in regimes:
        m = R[S]["met"]
        print("%6d | %s" % (S, " ".join("%14.4f" % m[:, ix[n]].mean() for n in fin_names)))
    print("\ntuned configs per regime (selection seeds):")
    for S in regimes:
        t, lw = R[S]["tuned"], R[S]["lw"]
        edge = "".join(" [!%s at grid edge]" % nm for nm, v, ax in (("lr", t["lr"], LRS), ("b1", t["b1"][0], B1S), ("b2", t["b2"][0], B2S), ("eps", t["eps"], EPS))
                       if len(ax) > 1 and v in (min(ax), max(ax)))
        print("  S=%-4d tuned lr=%g b1=%g b2=%g eps=%g | layerwise b1=%s b2=%s%s" % (
            S, t["lr"], t["b1"][0], t["b2"][0], t["eps"], lw["b1"], lw["b2"], edge))

    print("\n== Gate 0a: plasticity loss of default Adam (late-task minus early-task online loss; >0 = worse)")
    ok_a = False
    for S in regimes:
        g = R[S]["gap"]
        if np.isnan(g[:, 0]).all():
            print("  S=%-4d n/a (<10 tasks)" % S)
            continue
        for n in ("default", "tuned"):
            mm, se = g[:, ix[n]].mean(), (g[:, ix[n]].std(ddof=1) / math.sqrt(len(g)) if len(g) > 1 else float("nan"))
            hit = bool(np.isfinite(se) and mm > 2 * se and mm > GAP_MIN)
            if n == "default":
                ok_a = ok_a or hit
            print("  S=%-4d %-8s gap %+.4f +/- %.4f  %s" % (S, n, mm, se, "PLASTICITY LOSS" if hit else ""))
    print("  -> Gate 0a: %s (needs gap > 2 SE and > %.2f in >= 1 regime)" % ("PASS" if ok_a else "FAIL", GAP_MIN))

    print("\n== Gate 0b: headroom (paired over eval seeds; positive = first is WORSE, i.e. second has headroom)")
    nR = nL = 0
    for S in regimes:
        m = R[S]["met"]
        base = m[:, ix["tuned"]].mean()
        d1, s1 = pdiff(m[:, ix["global_fixed"]], m[:, ix["tuned"]])
        d2, s2 = pdiff(m[:, ix["default+reset"]], m[:, ix["tuned"]])
        d3, s3 = pdiff(m[:, ix["tuned"]], m[:, ix["layerwise"]])
        rR = passes(d1, s1, base) and passes(d2, s2, base)
        rL = passes(d3, s3, base)
        nR, nL = nR + rR, nL + rL
        print("  S=%-4d global_fixed-tuned %+.4f+/-%.4f | default+reset-tuned %+.4f+/-%.4f | tuned-layerwise %+.4f+/-%.4f | R:%s L:%s" % (
            S, d1, s1, d2, s2, d3, s3, "yes" if rR else "no", "yes" if rL else "no"))
    ok_b = nR >= 2 or nL >= 2
    print("  -> Gate 0b: %s  (R robustness in %d regimes, L layerwise in %d; need >= 2). rel. threshold %.0f%%" % (
        "PASS" if ok_b else "FAIL", nR, nL, 100 * REL_MIN))

    print("\n== Gate 0c: does measured q/r track the oracle gain K = 1-beta1?  (gate = beta1 only; b2/lr columns are exploratory)")
    print("%6s | %12s %26s | %9s %9s | %9s %9s" % ("S", "q/r overall", "q/r per layer (L1,L2,L3)", "K oracle", "K Kalman", "b2 oracle", "lr oracle"))
    qr, kor, b2or, lror = [], [], [], []
    for S in regimes:
        dg, tn = R[S]["diag"], R[S]["tuned"]
        qr.append(dg["qr_overall"])
        kor.append(1 - tn["b1"][0])
        b2or.append(tn["b2"][0])
        lror.append(tn["lr"])
        print("%6d | %12.4g %26s | %9.3f %9.3f | %9g %9g" % (
            S, dg["qr_overall"], np.array2string(np.array(dg["qr_group"]), precision=3), kor[-1], k_star(dg["qr_overall"], 1.0), b2or[-1], lror[-1]))
    rho = spearman(qr, kor)
    ok_c = bool(np.isfinite(rho) and rho >= RHO_MIN)
    print("  Spearman(q/r, K oracle) = %.2f over %d regimes -> Gate 0c: %s (need >= %.1f; n is small, read the table too)" % (
        rho, len(regimes), "PASS" if ok_c else "FAIL", RHO_MIN))
    print("  exploratory, post-hoc (NOT gate-eligible): Spearman(q/r, b2 oracle) = %.2f ; Spearman(q/r, lr oracle) = %.2f" % (
        spearman(qr, b2or), spearman(qr, lror)))
    if len(set(round(x, 3) for x in kor)) == 1:
        print("  note: oracle K is identical in all regimes -> nothing for the sensor to track.")

    print("\n== SUMMARY: 0a %s | 0b %s | 0c %s" % tuple("PASS" if x else "FAIL" for x in (ok_a, ok_b, ok_c)))
    print("   ALL PASS -> go to Gate 1 (build the adaptive rule). Any FAIL -> read the table above, then decide; plan says kill.")
    if not real:
        print("   !! synthetic data was used: ignore the verdicts.")
    (out / "summary.json").write_text(json.dumps(dict(gate_a=ok_a, gate_b=ok_b, gate_c=ok_c, rho=rho, nR=int(nR), nL=int(nL))))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("selftest")
    s.set_defaults(fn=cmd_selftest)
    s = sp.add_parser("kalman")
    s.add_argument("--steps", type=int, default=6000)
    s.set_defaults(fn=cmd_kalman)
    s = sp.add_parser("quad")
    s.add_argument("--dim", type=int, default=50)
    s.add_argument("--nseeds", type=int, default=16)
    s.add_argument("--steps", type=int, default=20000)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_quad)
    s = sp.add_parser("pmnist")
    s.add_argument("--out", default="out_drift")
    s.add_argument("--data_dir", default="data")
    s.add_argument("--backend", default="auto", choices=["auto", "torch", "numpy"])
    s.add_argument("--grid", default="full", choices=["full", "small"])
    s.add_argument("--steps", type=int, default=10000)
    s.add_argument("--regimes", default="25,50,100,200,400", help="steps per task (drift speed)")
    s.add_argument("--N", type=int, default=2000, help="examples per task")
    s.add_argument("--B", type=int, default=64)
    s.add_argument("--H", type=int, default=128)
    s.add_argument("--sel_seeds", type=int, default=2)
    s.add_argument("--eval_seeds", type=int, default=3)
    s.add_argument("--cd_passes", type=int, default=1)
    s.add_argument("--quick", action="store_true", help="tiny smoke test")
    s.set_defaults(fn=cmd_pmnist)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
