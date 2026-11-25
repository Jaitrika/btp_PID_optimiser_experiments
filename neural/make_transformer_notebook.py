import json, sys

cells = []
def md(s): cells.append(("markdown", s))
def code(s): cells.append(("code", s))

md(r'''# Governor vs. baselines on a small GPT: do natural loss spikes appear, and does the governor help?

**What this tests.** A character-level GPT (~3M params) is trained at a sweep of learning rates, starting safe and going to unsafe. Near the edge of stability, Adam's stale second moment causes *natural* loss spikes (no injected gradients). For each optimizer we record: which learning rates diverge, how many spikes happen, and final validation loss.

**Arms.** `adamw`, `adamw_gclip` (global clip 1.0), `stable` (StableAdamW), `gov_PI` (paper's governor, speed-up cap M=2), `gov_PI_brake` (cap M=1, brake only), `gov_PI_brake_pred` (cap M=1, multiplier computed from past information only), and two **P-only** arms with a bigger gain, `gov_P2` (M=2) and `gov_P2_brake` (M=1).

**The PI question.** Is the integral term doing anything a bigger P gain doesn't? Compare `gov_PI` vs `gov_P2` and `gov_PI_brake` vs `gov_P2_brake` (the analysis cell prints this). If P-only is as good or better, the integral is not earning its place.

**Decided in advance (pass/fail).** The governor "passes" only if, over all seeds, it (A) stays stable at a higher learning rate than BOTH `adamw` and `stable`, or (B) has at least 30% fewer spikes than both at learning rates where nobody diverges. Otherwise we report a null result.

**How to run.** Runtime -> Change runtime type -> T4 GPU, then Run all. The cell headed "SELF-TEST + SMOKE TEST" runs first and prints a time estimate. The cell headed "MAIN SWEEP" saves `results.csv` after every run, so you can stop and resume. When done, download `results.csv` and upload it back to the chat.''')

code(r'''import os, sys, math, time, glob, sysconfig, urllib.request
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device, '| torch', torch.__version__)
if device == 'cpu':
    print('WARNING: no GPU. Runtime -> Change runtime type -> T4 GPU.')''')

code(r'''# ---------------- CONFIG (edit here) ----------------
STEPS    = 1000                           # training steps per run
LRS      = [1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2]   # learning-rate sweep (safe -> unsafe)
SEEDS    = [0, 1]                         # use [0,1,2] if you have time
WARMUP   = 20                             # linear LR warm-up steps (short on purpose)
BETA2    = 0.999                          # Adam beta2 (0.999 makes stale-moment spikes more likely)
BATCH, BLOCK = 64, 128
DIVERGE_LOSS = 10.0                       # run counts as diverged if loss is non-finite or above this
SPIKE_THR, SPIKE_WIN = 0.5, 50            # a spike = loss exceeds median of previous 50 steps by 0.5 nats

ARMS = {
    'adamw':             dict(kind='adamw'),
    'adamw_gclip':       dict(kind='adamw', gclip=1.0),
    'stable':            dict(kind='stable'),
    'gov_PI':            dict(kind='gov', kp=0.5, ki=0.02, M=2.0),
    'gov_PI_brake':      dict(kind='gov', kp=0.5, ki=0.02, M=1.0),
    'gov_PI_brake_pred': dict(kind='gov', kp=0.5, ki=0.02, M=1.0, delay=True),
    # P-only arms with a bigger gain: tests whether the integral term does anything a bigger kp doesn't.
    # (delete either line to save ~12% runtime each)
    'gov_P2':            dict(kind='gov', kp=2.0, ki=0.0, M=2.0),
    'gov_P2_brake':      dict(kind='gov', kp=2.0, ki=0.0, M=1.0),
}
CSV = 'results.csv'
os.makedirs('curves', exist_ok=True)''')

code(r'''# ---------------- DATA ----------------
SHK_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
if not os.path.exists('shakespeare.txt'):
    urllib.request.urlretrieve(SHK_URL, 'shakespeare.txt')
text_a = open('shakespeare.txt', encoding='utf-8').read()

# second corpus (only used by the optional distribution-shift test): python stdlib source
buf, n = [], 0
for f in sorted(glob.glob(sysconfig.get_paths()['stdlib'] + '/*.py')):
    try:
        s = open(f, encoding='utf-8').read()
    except Exception:
        continue
    buf.append(s); n += len(s)
    if n >= len(text_a): break
text_b = ''.join(buf)[:len(text_a)]

chars = sorted(set(text_a) | set(text_b)); stoi = {c: i for i, c in enumerate(chars)}; V = len(chars)
def enc(t): return torch.tensor([stoi[c] for c in t], dtype=torch.long)
def split(t):
    d = enc(t).to(device); k = int(0.9 * len(d)); return d[:k], d[k:]
A_tr, A_va = split(text_a); B_tr, B_va = split(text_b)
print('vocab', V, '| shakespeare chars', len(text_a), '| code chars', len(text_b), '| initial loss ~', round(math.log(V), 2))''')

code(r'''# ---------------- MODEL (small GPT, ~3M params) ----------------
class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d); self.fc = nn.Linear(d, 4 * d); self.out = nn.Linear(4 * d, d); self.h = h
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = [t.view(B, T, self.h, C // self.h).transpose(1, 2) for t in (q, k, v)]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, C)
        x = x + self.proj(y)
        return x + self.out(F.gelu(self.fc(self.ln2(x))))

class GPT(nn.Module):
    def __init__(self, V, d=256, L=4, h=4, T=BLOCK):
        super().__init__()
        self.tok = nn.Embedding(V, d); self.pos = nn.Embedding(T, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(L)])
        self.lnf = nn.LayerNorm(d); self.head = nn.Linear(d, V, bias=False)
    def forward(self, idx, tgt=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks: x = b(x)
        lg = self.head(self.lnf(x))
        if tgt is None: return lg
        return F.cross_entropy(lg.view(-1, lg.size(-1)), tgt.view(-1))

print('params (M):', round(sum(p.numel() for p in GPT(V).parameters()) / 1e6, 2))''')

code(r'''# ---------------- OPTIMIZER: AdamW / StableAdamW / governor (one controller per parameter tensor) ----------------
class Opt:
    def __init__(self, params, cfg, wd=0.1, b1=0.9, ea=1e-8, ev=1e-16, Tw=50, brho=0.9, c=0.1, Imax=20.0):
        self.cfg = cfg; self.b1 = b1; self.b2 = BETA2; self.ea = ea; self.ev = ev
        self.Tw = Tw; self.brho = brho; self.c = c; self.Imax = Imax; self.t = 0
        self.kp = cfg.get('kp', 0.5); self.ki = cfg.get('ki', 0.02); self.M = cfg.get('M', 2.0)
        self.rs = cfg.get('rs_ema', None)          # optional: slow-EMA target for rho* (None = rho* = 1, as in the paper)
        f64 = torch.float64; self.S = []
        for p in params:
            d = p.device
            self.S.append(dict(p=p, m=torch.zeros_like(p), v=torch.zeros_like(p),
                               z=torch.ones((), device=d, dtype=f64), I=torch.zeros((), device=d, dtype=f64),
                               L=torch.zeros((), device=d, dtype=f64), wd=(wd if p.ndim >= 2 else 0.0)))
        d = self.S[0]['p'].device
        self.mu_acc = torch.zeros((), device=d, dtype=f64); self.brake_acc = torch.zeros((), device=d, dtype=f64)
        self.boost_acc = torch.zeros((), device=d, dtype=f64); self.n = 0; self.last_mu = []

    @torch.no_grad()
    def step(self, lr):
        self.t += 1; t = self.t; b1 = self.b1; b2 = self.b2; kind = self.cfg['kind']; mus = []
        for st in self.S:
            p = st['p']; g = p.grad
            if g is None: continue
            m = st['m']; v = st['v']
            vh_prev = v / (1 - b2 ** (t - 1)) if t > 1 else None      # bias-corrected v BEFORE this gradient
            m.mul_(b1).add_(g, alpha=1 - b1); v.mul_(b2).addcmul_(g, g, value=1 - b2)
            mh = m / (1 - b1 ** t); vh = v / (1 - b2 ** t)
            mu = torch.ones((), device=p.device, dtype=torch.float64)
            if kind == 'stable':                                       # StableAdamW: clip by CURRENT normalized gradient RMS
                r = (g * g / torch.clamp(vh, min=self.ea ** 2)).mean(dtype=torch.float64)
                mu = 1.0 / torch.clamp(torch.sqrt(r), min=1.0)
            elif kind == 'gov' and t > self.Tw:                        # governor: frozen for the first Tw steps
                rho = (g * g / (vh_prev + self.ev)).mean(dtype=torch.float64)
                delay = self.cfg.get('delay', False)
                if delay:                                              # predictable gain: uses only past state
                    mu = torch.clamp(torch.exp(-(self.kp * (torch.log(torch.clamp(st['z'], min=1e-30)) - st['L'])
                                                 + self.ki * st['I'])), self.c, self.M)
                st['z'] = self.brho * st['z'] + (1 - self.brho) * rho
                lz = torch.log(torch.clamp(st['z'], min=1e-30))
                if self.rs is not None: st['L'] = st['L'] + self.rs * (lz - st['L'])
                e = lz - st['L']
                st['I'] = torch.clamp(st['I'] + e, -self.Imax, self.Imax)
                if not delay:
                    mu = torch.clamp(torch.exp(-(self.kp * e + self.ki * st['I'])), self.c, self.M)
            mus.append(mu)
            s = (lr * mu).to(p.dtype)                                  # multiplier scales BOTH decay and step (paper eq. 3)
            p.mul_(1 - s * st['wd'])
            p.sub_(s * mh / (vh.sqrt() + self.ea))
        if mus:
            ms = torch.stack(mus)
            self.mu_acc += ms.mean(); self.brake_acc += (ms < 0.5).double().mean(); self.boost_acc += (ms > 1.5).double().mean()
            self.n += 1; self.last_mu = mus''')

code(r'''# ---------------- TRAIN / EVAL / METRICS ----------------
def get_batch(data, gen):
    ix = torch.randint(len(data) - BLOCK - 1, (BATCH,), generator=gen, device=data.device)
    idx = ix[:, None] + torch.arange(BLOCK, device=data.device)[None, :]
    return data[idx], data[idx + 1]

@torch.no_grad()
def evaluate(model, data, n=20):
    model.eval(); gen = torch.Generator(device=device); gen.manual_seed(1234)
    tot = sum(model(*get_batch(data, gen)).item() for _ in range(n)) / n
    model.train(); return tot

def spike_stats(losses):
    l = np.asarray(losses); n = 0; ex = 0.0; inside = False
    for t in range(SPIKE_WIN, len(l)):
        base = np.median(l[t - SPIKE_WIN:t])
        if l[t] > base + SPIKE_THR:
            if not inside: n += 1; inside = True
        else:
            inside = False
        ex = max(ex, l[t] - base)
    return n, ex

def train_run(arm, lr, seed, steps=None, shift_at=None):
    steps = STEPS if steps is None else steps
    cfg = ARMS[arm]; t0 = time.time()
    torch.manual_seed(seed); model = GPT(V).to(device)          # same init for every arm (per seed)
    gen = torch.Generator(device=device); gen.manual_seed(seed)  # same batches for every arm (per seed)
    opt = Opt(list(model.parameters()), cfg)
    losses = []; diverged = False; div_step = -1
    for t in range(1, steps + 1):
        data = A_tr if (shift_at is None or t <= shift_at) else B_tr
        cur = lr * min(1.0, t / WARMUP) if WARMUP > 0 else lr
        x, y = get_batch(data, gen); loss = model(x, y)
        lv = loss.item(); losses.append(lv)
        if (not math.isfinite(lv)) or lv > DIVERGE_LOSS:
            diverged = True; div_step = t; break
        model.zero_grad(set_to_none=True); loss.backward()
        if cfg.get('gclip'): torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gclip'])
        opt.step(cur)
    n_sp, ex = spike_stats(losses)
    va = vb = float('nan')
    if not diverged:
        va = evaluate(model, A_va); vb = evaluate(model, B_va)
    nn_ = max(opt.n, 1)
    return dict(arm=arm, lr=lr, seed=seed, diverged=diverged, div_step=div_step,
                final_train=float(np.mean(losses[-50:])), final_val=va, val_b=vb,
                n_spikes=n_sp, max_excursion=ex, mu_mean=float(opt.mu_acc) / nn_,
                frac_brake=float(opt.brake_acc) / nn_, frac_boost=float(opt.boost_acc) / nn_,
                secs=time.time() - t0, losses=losses)''')

code(r'''# ---------------- SELF-TEST + SMOKE TEST (run this first) ----------------
def selftest():
    torch.manual_seed(0)
    # 1) our AdamW must match torch.optim.AdamW
    a = nn.Parameter(torch.randn(5, 4, device=device)); b = nn.Parameter(a.detach().clone())
    o1 = Opt([a], dict(kind='adamw'), wd=0.1)
    o2 = torch.optim.AdamW([b], lr=1e-2, betas=(0.9, BETA2), eps=1e-8, weight_decay=0.1)
    for _ in range(20):
        g = torch.randn(5, 4, device=device); a.grad = g.clone(); b.grad = g.clone(); o1.step(1e-2); o2.step()
    assert torch.allclose(a, b, atol=1e-5), 'AdamW mismatch: ' + str((a - b).abs().max().item())
    # 2) governor: ~1 on a constant gradient, brakes hard after a 100x spike
    p = nn.Parameter(torch.zeros(10, device=device)); o = Opt([p], dict(kind='gov', kp=0.5, ki=0.02, M=2.0))
    for _ in range(300): p.grad = torch.full((10,), 0.5, device=device); o.step(1e-3)
    mu_flat = o.last_mu[0].item()
    p.grad = torch.full((10,), 50.0, device=device); o.step(1e-3); mu_spike = o.last_mu[0].item()
    assert 0.9 < mu_flat < 1.1, 'mu on flat gradient: ' + str(mu_flat)
    assert mu_spike < 0.5, 'mu after spike: ' + str(mu_spike)
    print('self-test OK | mu flat = %.3f, mu after 100x spike = %.3f' % (mu_flat, mu_spike))

selftest()
res = {}
for arm in ARMS:
    r = train_run(arm, 1e-3, 0, steps=60); res[arm] = r
    print('%-18s loss@60 = %.3f | mean mu = %.2f | %.1f it/s' % (arm, r['losses'][-1], r['mu_mean'], 60 / r['secs']))
ips = 60 / max(res['gov_PI']['secs'], 1e-9)
total_runs = len(ARMS) * len(LRS) * len(SEEDS)
print('\nFull sweep: %d runs x %d steps -> at most ~%.0f minutes (diverged runs stop early).' % (total_runs, STEPS, total_runs * STEPS / ips / 60))''')

code(r'''# ---------------- MAIN SWEEP (resumable: results.csv is appended after every run) ----------------
done = set()
if os.path.exists(CSV):
    d0 = pd.read_csv(CSV); done = set(zip(d0.arm, d0.lr, d0.seed))
total = len(ARMS) * len(LRS) * len(SEEDS); t0 = time.time()
for lr in LRS:
    for seed in SEEDS:
        for arm in ARMS:
            if (arm, lr, seed) in done: continue
            r = train_run(arm, lr, seed)
            np.save('curves/%s_%g_%d.npy' % (arm, lr, seed), np.array(r['losses']))
            row = {k: v for k, v in r.items() if k != 'losses'}
            pd.DataFrame([row]).to_csv(CSV, mode='a', header=not os.path.exists(CSV), index=False)
            done.add((arm, lr, seed))
            print('[%d/%d] %-18s lr=%-7g seed=%d  %s  val=%.3f spikes=%d  mu=%.2f  (%.0fs elapsed)' % (
                len(done), total, arm, lr, seed, 'DIVERGED@%d' % r['div_step'] if r['diverged'] else 'ok', r['final_val'],
                r['n_spikes'], r['mu_mean'], time.time() - t0))''')

code(r'''# ---------------- ANALYSIS ----------------
df = pd.read_csv(CSV)
rows = []
for (arm, lr), s in df.groupby(['arm', 'lr']):
    rows.append(dict(arm=arm, lr=lr, n=len(s), div_rate=s.diverged.mean(),
                     val=np.inf if s.diverged.any() else s.final_val.mean(),
                     spikes=s.n_spikes.mean(), mu=s.mu_mean.mean(), brake=s.frac_brake.mean(), boost=s.frac_boost.mean()))
tab = pd.DataFrame(rows).set_index(['arm', 'lr'])
order = [a for a in ARMS if a in df.arm.unique()]
for col, title in [('div_rate', 'fraction of seeds that diverged'), ('val', 'final val loss (inf = at least one seed diverged)'),
                   ('spikes', 'mean number of loss spikes'), ('mu', 'mean multiplier (1.0 = no change)')]:
    print('\n== ' + title); print(tab[col].unstack('lr').loc[order].round(3).to_string())

def max_stable(arm):
    best = None
    for lr in LRS:
        sub = df[(df.arm == arm) & (df.lr == lr)]
        if len(sub) < len(SEEDS) or sub.diverged.any(): break
        best = lr
    return best

print('\n== highest learning rate with no divergence (contiguous from the smallest)')
for a in order: print('%-18s %s' % (a, max_stable(a)))

base = [a for a in ('adamw', 'stable') if a in order]
common = [lr for lr in LRS if all((tab.loc[(a, lr), 'div_rate'] == 0) for a in order if (a, lr) in tab.index)]
print('\n== VERDICT (rule fixed in advance)')
for a in order:
    if not a.startswith('gov'): continue
    ms = max_stable(a) or 0; mb = max((max_stable(b) or 0) for b in base)
    A = ms > mb
    B = None
    if common:
        sg = sum(tab.loc[(a, lr), 'spikes'] for lr in common)
        sb = min(sum(tab.loc[(b, lr), 'spikes'] for lr in common) for b in base)
        B = (sb > 0) and (sg <= 0.7 * sb)
    print('%-18s (A) wider stable range: %s  [%s vs best baseline %s] | (B) >=30%% fewer spikes: %s' % (
        a, 'YES' if A else 'no', ms, mb, 'n/a (no common stable LR)' if B is None else ('YES' if B else 'no')))

print('\n== PI QUESTION: does the integral beat a bigger P gain? (same cap, matched pairs)')
for pi_arm, p_arm in [('gov_PI', 'gov_P2'), ('gov_PI_brake', 'gov_P2_brake')]:
    if pi_arm not in order or p_arm not in order: continue
    both_ok = [lr for lr in LRS if (pi_arm, lr) in tab.index and (p_arm, lr) in tab.index
               and tab.loc[(pi_arm, lr), 'div_rate'] == 0 and tab.loc[(p_arm, lr), 'div_rate'] == 0]
    print('%s vs %s | max stable LR: %s vs %s' % (pi_arm, p_arm, max_stable(pi_arm), max_stable(p_arm)))
    if both_ok:
        for nm, col in [('mean final val loss', 'val'), ('mean spikes', 'spikes')]:
            print('   %-20s %s = %.3f | %s = %.3f   (over LRs where both are stable)' % (
                nm, pi_arm, np.mean([tab.loc[(pi_arm, lr), col] for lr in both_ok]), p_arm,
                np.mean([tab.loc[(p_arm, lr), col] for lr in both_ok])))
    else:
        print('   no learning rate where both are stable')

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for a in order:
    t_ = tab.loc[a].sort_index()
    ax[0].plot(t_.index, t_.val.replace(np.inf, np.nan), marker='o', label=a)
    ax[1].plot(t_.index, t_.spikes, marker='o', label=a)
    ax[2].plot(t_.index, t_.div_rate, marker='o', label=a)
for x_, t_ in zip(ax, ['final val loss (missing = diverged)', 'mean # spikes', 'fraction diverged']):
    x_.set_xscale('log'); x_.set_xlabel('learning rate'); x_.set_title(t_)
ax[0].legend(fontsize=7); plt.tight_layout(); plt.show()

# loss curves at the first LR where plain AdamW has trouble (or the largest LR)
bad = [lr for lr in LRS if ((df.arm == 'adamw') & (df.lr == lr) & df.diverged).any()]
lr_show = bad[0] if bad else LRS[-1]
plt.figure(figsize=(10, 4))
for a in order:
    fn = 'curves/%s_%g_%d.npy' % (a, lr_show, SEEDS[0])
    if os.path.exists(fn): plt.plot(np.load(fn), label=a, lw=1)
plt.ylim(1.0, 6); plt.xlabel('step'); plt.ylabel('train loss'); plt.title('seed %d, lr=%g' % (SEEDS[0], lr_show)); plt.legend(fontsize=7); plt.show()''')

code(r'''# ---------------- OPTIONAL: natural distribution shift (Shakespeare -> Python source halfway) ----------------
RUN_SHIFT = False                 # set True to run (adds ~ len(ARMS)*len(SHIFT_LRS)*len(SEEDS) runs)
SHIFT_LRS, SHIFT_STEPS, SHIFT_AT = [2e-3, 4e-3], 1000, 500
if RUN_SHIFT:
    out = []
    for lr in SHIFT_LRS:
        for seed in SEEDS:
            for arm in ARMS:
                r = train_run(arm, lr, seed, steps=SHIFT_STEPS, shift_at=SHIFT_AT)
                L = np.array(r['losses']); post = L[SHIFT_AT:SHIFT_AT + 100]
                out.append(dict(arm=arm, lr=lr, seed=seed, diverged=r['diverged'],
                                peak_after_shift=post.max() if len(post) else np.nan,
                                last100=L[-100:].mean(), val_shakespeare=r['final_val'], val_code=r['val_b'], mu=r['mu_mean']))
                print(out[-1])
    sh = pd.DataFrame(out); sh.to_csv('shift_results.csv', index=False)
    print(sh.groupby(['arm', 'lr'])[['peak_after_shift', 'last100', 'val_code']].mean().round(3).to_string())''')

code(r'''# ---------------- DOWNLOAD (upload these back to the chat) ----------------
import shutil
shutil.make_archive('curves', 'zip', 'curves')
try:
    from google.colab import files
    files.download(CSV); files.download('curves.zip')
    if os.path.exists('shift_results.csv'): files.download('shift_results.csv')
except ImportError:
    print('Not in Colab; files are in the working directory.')''')

nb = {
    "nbformat": 4, "nbformat_minor": 4,
    "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                 "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
    "cells": [],
}
for kind, src in cells:
    lines = src.split("\n")
    srcl = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    c = {"cell_type": kind, "metadata": {}, "source": srcl}
    if kind == "code":
        c["outputs"] = []; c["execution_count"] = None
    nb["cells"].append(c)

out = sys.argv[1]
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out, "cells:", len(nb["cells"]))
