import json, sys

cells = []
def md(s): cells.append(("markdown", s))
def code(s): cells.append(("code", s))

md(r'''# Fashion-MNIST CNN: does the governor win after a natural noise increase (batch-size drop)?

**One test, short.** Halfway through training the batch size drops (256 -> 8), so gradient noise rises. Nothing is injected. Every method gets the same learning-rate grid; learning rate (and beta2 for the "tuned beta2" baseline) is tuned **only** on the scenario `bs 256->8`, using validation loss on 3 tuning seeds. Then each method is evaluated on 5 scenarios with 10 new seeds, final **test** cross-entropy.

**Methods:** AdamW, AdamW with tuned beta2, global clip (norm 1), StableAdamW, lagged clip, governor PI (paper settings), governor P-only, governor PI with predictable gain.

**What to look for:** the paired-difference table at the bottom. Negative = governor better. If an interval includes 0 it is a tie, not a win.

**How to run:** Runtime -> Change runtime type -> T4 GPU (CPU also works for this tiny network, just slower). Run all. The cell headed "SELF-TEST + SMOKE TEST" prints a time estimate; the sweep saves `fmnist_shift_results.csv` after every run and is resumable. Download that CSV at the end and upload it to the chat.''')

code(r'''import os, sys, math, time
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from scipy import stats

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device, '| torch', torch.__version__)''')

code(r'''# ---------------- CONFIG ----------------
T = 600                                   # training steps
RATES = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]    # same grid for every method (paper's grid)
TUNE_SEEDS = [100, 101, 102]
EVAL_SEEDS = list(range(200, 210))        # use list(range(200, 205)) for a quicker run
WD = 1e-4                                 # decoupled weight decay on weights (not biases)
CSV = 'fmnist_shift_results.csv'

# scenario = (batch size before, batch size after, fraction of training at which it changes); 'stat' = constant 64
SC = {'bs 256->8 (tuned here)': (256, 8, .5), 'bs 256->4': (256, 4, .5), 'bs 256->8 late': (256, 8, .75),
      'stationary bs 64': 'stat', 'DROP: bs 8->256': (8, 256, .5)}
CAL = 'bs 256->8 (tuned here)'
def sched(s):
    if s == 'stat': return [64] * T
    a, b, f = s; return [a if t < f * T else b for t in range(T)]

METHODS = {
    'AdamW':                   dict(kind='adam'),
    'AdamW tuned beta2':       dict(kind='adam'),                 # beta2 chosen from {0.9, 0.99, 0.999}
    'Global clip':             dict(kind='gclip'),
    'StableAdamW':             dict(kind='stable'),
    'Lagged clip':             dict(kind='lagclip'),
    'Governor PI (paper)':     dict(kind='gov', kp=0.5, ki=0.02, M=2.0),
    'Governor P only':         dict(kind='gov', kp=0.5, ki=0.0, M=2.0),
    'Governor PI predictable': dict(kind='gov', kp=0.5, ki=0.02, M=2.0, delay=True),
}
GOV = 'Governor PI (paper)'
BASELINES = ['AdamW', 'AdamW tuned beta2', 'Global clip', 'StableAdamW']''')

code(r'''# ---------------- DATA: class-balanced subset as in the paper (10,000 train / 2,000 val from the official train set; full 10,000 test) ----------------
tr = torchvision.datasets.FashionMNIST('data', train=True, download=True)
te = torchvision.datasets.FashionMNIST('data', train=False, download=True)
Xall, yall = tr.data.float() / 255.0, tr.targets
rng0 = np.random.default_rng(0); tr_idx, va_idx = [], []
for c in range(10):
    ids = np.where(yall.numpy() == c)[0]; ids = rng0.permutation(ids)
    tr_idx += list(ids[:1000]); va_idx += list(ids[1000:1200])
tr_idx, va_idx = np.array(tr_idx), np.array(va_idx)
mean, std = Xall[tr_idx].mean(), Xall[tr_idx].std()       # normalization uses selected TRAIN images only
prep = lambda X: ((X - mean) / std).unsqueeze(1).to(device)
Xtr, ytr = prep(Xall[tr_idx]), yall[tr_idx].to(device)
Xva, yva = prep(Xall[va_idx]), yall[va_idx].to(device)
Xte, yte = prep(te.data.float() / 255.0), te.targets.to(device)
NTR = len(Xtr); print('train', tuple(Xtr.shape), 'val', tuple(Xva.shape), 'test', tuple(Xte.shape))

class CNN(nn.Module):
    # two ReLU conv layers (8 and 16 channels, stride 2) + linear classifier (~9.1k params; the paper reports 9,226, exact layers unknown)
    def __init__(self):
        super().__init__(); self.c1 = nn.Conv2d(1, 8, 3, 2, 1); self.c2 = nn.Conv2d(8, 16, 3, 2, 1); self.fc = nn.Linear(16 * 7 * 7, 10)
    def forward(self, x):
        return self.fc(F.relu(self.c2(F.relu(self.c1(x)))).flatten(1))
print('params:', sum(p.numel() for p in CNN().parameters()))

@torch.no_grad()
def evaluate(model, X, y):
    ls, ac = 0.0, 0
    for i in range(0, len(X), 5000):
        lg = model(X[i:i + 5000]); ls += F.cross_entropy(lg, y[i:i + 5000], reduction='sum').item(); ac += (lg.argmax(1) == y[i:i + 5000]).sum().item()
    return ls / len(X), ac / len(X)''')

code(r'''# ---------------- OPTIMIZER: AdamW / StableAdamW / lagged clip / governor (one controller per parameter tensor) ----------------
class Opt:
    def __init__(self, params, cfg, b2=0.999, wd=WD, b1=0.9, ea=1e-8, ev=1e-16, Tw=50, brho=0.9, c=0.1, Imax=20.0):
        self.cfg = cfg; self.b1 = b1; self.b2 = b2; self.ea = ea; self.ev = ev; self.Tw = Tw; self.brho = brho; self.c = c; self.Imax = Imax
        self.t = 0; self.kp = cfg.get('kp', 0.5); self.ki = cfg.get('ki', 0.02); self.M = cfg.get('M', 2.0)
        f64 = torch.float64; self.S = []
        for p in params:
            d = p.device
            self.S.append(dict(p=p, m=torch.zeros_like(p), v=torch.zeros_like(p), z=torch.ones((), device=d, dtype=f64),
                               I=torch.zeros((), device=d, dtype=f64), wd=(wd if p.ndim >= 2 else 0.0)))
        d = self.S[0]['p'].device
        self.mu_acc = torch.zeros((), device=d, dtype=f64); self.n = 0; self.last_mu = []

    @torch.no_grad()
    def step(self, lr):
        self.t += 1; t = self.t; b1 = self.b1; b2 = self.b2; kind = self.cfg['kind']; mus = []
        for st in self.S:
            p = st['p']; g = p.grad; m = st['m']; v = st['v']
            vh_prev = v / (1 - b2 ** (t - 1)) if t > 1 else None
            m.mul_(b1).add_(g, alpha=1 - b1); v.mul_(b2).addcmul_(g, g, value=1 - b2)
            mh = m / (1 - b1 ** t); vh = v / (1 - b2 ** t)
            mu = torch.ones((), device=p.device, dtype=torch.float64)
            if kind == 'stable':                                   # StableAdamW: current normalized gradient RMS
                r = (g * g / torch.clamp(vh, min=self.ea ** 2)).mean(dtype=torch.float64)
                mu = 1.0 / torch.clamp(torch.sqrt(r), min=1.0)
            elif kind == 'lagclip' and t > 1:                      # clip using the LAGGED ratio, no smoothing
                r = (g * g / (vh_prev + self.ev)).mean(dtype=torch.float64)
                mu = 1.0 / torch.clamp(torch.sqrt(r), min=1.0)
            elif kind == 'gov' and t > self.Tw:                    # governor: frozen for the first Tw steps
                rho = (g * g / (vh_prev + self.ev)).mean(dtype=torch.float64); delay = self.cfg.get('delay', False)
                if delay:
                    mu = torch.clamp(torch.exp(-(self.kp * torch.log(torch.clamp(st['z'], min=1e-30)) + self.ki * st['I'])), self.c, self.M)
                st['z'] = self.brho * st['z'] + (1 - self.brho) * rho
                e = torch.log(torch.clamp(st['z'], min=1e-30))
                st['I'] = torch.clamp(st['I'] + e, -self.Imax, self.Imax)
                if not delay: mu = torch.clamp(torch.exp(-(self.kp * e + self.ki * st['I'])), self.c, self.M)
            mus.append(mu)
            s = (lr * mu).to(p.dtype)                              # multiplier scales BOTH decay and step (paper eq. 3)
            p.mul_(1 - s * st['wd']); p.sub_(s * mh / (vh.sqrt() + self.ea))
        if mus:
            self.mu_acc += torch.stack(mus).mean(); self.n += 1; self.last_mu = mus

def selftest():
    torch.manual_seed(0)
    a = nn.Parameter(torch.randn(5, 4, device=device)); b = nn.Parameter(a.detach().clone())
    o1 = Opt([a], dict(kind='adam'), b2=0.999, wd=0.1); o2 = torch.optim.AdamW([b], lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)
    for _ in range(20):
        g = torch.randn(5, 4, device=device); a.grad = g.clone(); b.grad = g.clone(); o1.step(1e-2); o2.step()
    assert torch.allclose(a, b, atol=1e-5), 'AdamW mismatch: ' + str((a - b).abs().max().item())
    p = nn.Parameter(torch.zeros(10, device=device)); o = Opt([p], dict(kind='gov', kp=0.5, ki=0.02, M=2.0))
    for _ in range(300): p.grad = torch.full((10,), 0.5, device=device); o.step(1e-3)
    mu_flat = o.last_mu[0].item()
    p.grad = torch.full((10,), 50.0, device=device); o.step(1e-3); mu_spike = o.last_mu[0].item()
    assert 0.9 < mu_flat < 1.1, 'mu on flat gradient: ' + str(mu_flat)
    assert mu_spike < 0.5, 'mu after spike: ' + str(mu_spike)
    print('self-test OK | mu flat = %.3f, mu after 100x spike = %.3f' % (mu_flat, mu_spike))''')

code(r'''# ---------------- TRAINING ----------------
def train(method, scen, seed, eta, b2):
    cfg = METHODS[method]; t0 = time.time()
    torch.manual_seed(seed); model = CNN().to(device)                  # same init for every method (per seed)
    rng = np.random.default_rng(seed); S = sched(SC[scen])
    idxs = [torch.from_numpy(rng.choice(NTR, B, replace=False)).to(device) for B in S]   # same batches for every method
    opt = Opt(list(model.parameters()), cfg, b2=b2)
    for t in range(1, T + 1):
        ix = idxs[t - 1]; loss = F.cross_entropy(model(Xtr[ix]), ytr[ix])
        model.zero_grad(set_to_none=True); loss.backward()
        if cfg['kind'] == 'gclip': torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(eta)
    va, _ = evaluate(model, Xva, yva); te_ce, te_acc = evaluate(model, Xte, yte)
    return dict(val_ce=va, test_ce=te_ce, test_acc=te_acc, mu=float(opt.mu_acc) / max(opt.n, 1), secs=time.time() - t0)

# ---------------- SELF-TEST + SMOKE TEST (run this first) ----------------
selftest()
for m in METHODS:
    r = train(m, CAL, 0, 1e-3, 0.999)
    print('%-26s val CE %.3f | test CE %.3f | mean mu %.2f | %.1fs' % (m, r['val_ce'], r['test_ce'], r['mu'], r['secs']))
n_runs = (len(METHODS) - 1) * len(RATES) * len(TUNE_SEEDS) + 3 * len(RATES) * len(TUNE_SEEDS) + len(METHODS) * len(SC) * len(EVAL_SEEDS)
print('\nFull experiment: ~%d runs, about %.0f minutes at this speed.' % (n_runs, n_runs * r['secs'] / 60))''')

code(r'''# ---------------- RUN (resumable: every run is appended to the CSV) ----------------
cols = ['phase', 'method', 'scen', 'seed', 'eta', 'b2', 'val_ce', 'test_ce', 'test_acc', 'mu']
cache = {}
if os.path.exists(CSV):
    for r in pd.read_csv(CSV).to_dict('records'): cache[(r['phase'], r['method'], r['scen'], int(r['seed']), float(r['eta']), float(r['b2']))] = r
def get(phase, method, scen, seed, eta, b2):
    k = (phase, method, scen, seed, float(eta), float(b2))
    if k in cache: return cache[k]
    r = train(method, scen, seed, eta, b2)
    row = dict(phase=phase, method=method, scen=scen, seed=seed, eta=eta, b2=b2, val_ce=r['val_ce'], test_ce=r['test_ce'], test_acc=r['test_acc'], mu=r['mu'])
    pd.DataFrame([row]).to_csv(CSV, mode='a', header=not os.path.exists(CSV), index=False)
    cache[k] = row; return row

t0 = time.time(); chosen = {}
for m in METHODS:                                                       # 1) tune on the calibration scenario only, by VALIDATION loss
    b2s = [0.9, 0.99, 0.999] if m == 'AdamW tuned beta2' else [0.999]
    best = None
    for e in RATES:
        for b in b2s:
            v = np.mean([get('tune', m, CAL, s, e, b)['val_ce'] for s in TUNE_SEEDS])
            if best is None or v < best[0]: best = (v, e, b)
    chosen[m] = (best[1], best[2]); print('tuned %-26s eta=%g beta2=%g  (val CE %.4f)  [%.0fs]' % (m, best[1], best[2], best[0], time.time() - t0))
for m in METHODS:                                                       # 2) evaluate on every scenario with fresh seeds
    for sc in SC:
        for s in EVAL_SEEDS: get('eval', m, sc, s, *chosen[m])
    print('evaluated %-26s [%.0fs]' % (m, time.time() - t0))''')

code(r'''# ---------------- RESULTS ----------------
df = pd.read_csv(CSV); ev = df[df.phase == 'eval']
order = list(METHODS); scs = list(SC)
tab = ev.pivot_table(index='method', columns='scen', values='test_ce', aggfunc='mean').loc[order, scs]
acc = ev.pivot_table(index='method', columns='scen', values='test_acc', aggfunc='mean').loc[order, scs]
print('final TEST cross-entropy, mean over eval seeds (lower = better); eta/beta2 tuned only on "bs 256->8":'); print(tab.round(4).to_string())
print('\nfinal TEST accuracy (%):'); print((100 * acc).round(2).to_string())
print('\nmean governor multiplier:'); print(ev.pivot_table(index='method', columns='scen', values='mu', aggfunc='mean').loc[[m for m in order if m.startswith('Governor')], scs].round(2).to_string())

def paired(a, b, sc):
    x = ev[(ev.method == a) & (ev.scen == sc)].sort_values('seed').test_ce.values
    y = ev[(ev.method == b) & (ev.scen == sc)].sort_values('seed').test_ce.values
    d = x - y; h = stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)); return d.mean(), h

print('\nPaired difference, %s minus baseline (negative = governor better; WIN if interval is entirely below 0), 95%% t-interval:' % GOV)
for b in BASELINES:
    out = []
    for sc in scs:
        d, h = paired(GOV, b, sc); tag = 'WIN' if d + h < 0 else ('LOSS' if d - h > 0 else 'tie')
        out.append('%-4s %+.4f+/-%.4f' % (tag, d, h))
    print('vs %-18s ' % b + ' | '.join(out))
print('columns:', scs)
print('\nIntegral question: PI minus P-only (negative = integral helps):')
print(' | '.join('%s: %+.4f+/-%.4f' % ((sc[:14],) + paired(GOV, 'Governor P only', sc)) for sc in scs))
print('\nchosen (eta, beta2):', chosen)''')

code(r'''# ---------------- DOWNLOAD (upload this CSV back to the chat) ----------------
try:
    from google.colab import files
    files.download(CSV)
except ImportError:
    print('Not in Colab; the CSV is in the working directory.')''')

nb = {"nbformat": 4, "nbformat_minor": 4,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}}, "cells": []}
for kind, src in cells:
    lines = src.split("\n"); srcl = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    c = {"cell_type": kind, "metadata": {}, "source": srcl}
    if kind == "code": c["outputs"] = []; c["execution_count"] = None
    nb["cells"].append(c)
json.dump(nb, open(sys.argv[1], "w"), indent=1)
print("wrote", sys.argv[1], "cells:", len(nb["cells"]))
