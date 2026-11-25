#!/usr/bin/env python3
"""
gate0.py -- Gate-0 tournament for the Muon noise-calibration ideas (T1, T2, T3).

Needs: torch (>=2.0), numpy.  Tested syntax only on the authoring machine (no torch there),
so run the smoke test first (see README in the chat).

Subcommands
  sweep      train a grid of runs (resumable: finished runs are skipped)
  summarize  print T1 / T2 / oracle tables from result json files
  t3         offline checkpoint-averaging test (needs runs made with --save_ckpt_every)

Methods
  adamw       AdamW on every parameter (lr = --lr)
  muon        Muon (Newton-Schulz, 5 steps) on hidden matrices, AdamW (lr = --adam_lr) on the rest
  muon_svd    same but exact polar factor via SVD
  muon_gain   ORACLE for T1: U diag(g) V^T with g_i = clamp(1 - (sN_i / s_i)^2, 0, 1),
              s = singular values of momentum M, sN = singular values of the signal-free noise copy M_D

Noise copy: every step the batch is split into two halves, giving gradients G1, G2.
  G = (G1+G2)/2 feeds momentum M;  D = (G1-G2)/2 feeds momentum M_D.
  D has zero signal and the same noise law as G, so M_D has the noise law of M's noise part.
"""
import argparse, itertools, json, math, os, sys, time, urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
NS_COEF = (3.4445, -4.7750, 2.0315)


# ----------------------------------------------------------------------------- data
def load_text(cache_dir):
    p = Path(cache_dir) / "tinyshakespeare.txt"
    if p.exists():
        return p.read_text()
    try:
        txt = urllib.request.urlopen(DATA_URL, timeout=30).read().decode()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt)
        return txt
    except Exception as e:  # smoke-test fallback only
        print("!! download failed (%s) -> synthetic text, results are NOT meaningful" % e)
        rng = np.random.RandomState(0)
        words = ["the", "king", "lord", "and", "of", "to", "my", "thou", "love", "death"]
        return " ".join(rng.choice(words, 200000)) + "\n"


class Data:
    def __init__(self, text, ctx, device):
        chars = sorted(set(text))
        self.vocab = len(chars)
        stoi = {c: i for i, c in enumerate(chars)}
        ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
        n = int(0.9 * len(ids))
        self.train, self.val = ids[:n].to(device), ids[n:].to(device)
        h = len(self.val) // 2
        self.valA, self.valB = self.val[:h], self.val[h:]
        self.ctx, self.device = ctx, device

    def batch(self, t, B, gen):
        ix = torch.randint(0, len(t) - self.ctx - 1, (B,), generator=gen).to(self.device)
        idx = ix[:, None] + torch.arange(self.ctx + 1, device=self.device)[None, :]
        w = t[idx]
        return w[:, :-1], w[:, 1:]


# ----------------------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__()
        s.h = h
        s.ln1, s.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.qkv = nn.Linear(d, 3 * d, bias=False)
        s.proj = nn.Linear(d, d, bias=False)
        s.fc = nn.Linear(d, 4 * d, bias=False)
        s.fc2 = nn.Linear(4 * d, d, bias=False)

    def forward(s, x):
        B, T, C = x.shape
        q, k, v = s.qkv(s.ln1(x)).split(C, dim=2)
        q, k, v = [z.view(B, T, s.h, C // s.h).transpose(1, 2) for z in (q, k, v)]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.proj(y.transpose(1, 2).contiguous().view(B, T, C))
        return x + s.fc2(F.gelu(s.fc(s.ln2(x))))


class GPT(nn.Module):
    def __init__(s, vocab, ctx, d, L, h):
        super().__init__()
        s.wte, s.wpe = nn.Embedding(vocab, d), nn.Embedding(ctx, d)
        s.blocks = nn.ModuleList([Block(d, h) for _ in range(L)])
        s.lnf = nn.LayerNorm(d)
        s.head = nn.Linear(d, vocab, bias=False)
        for m in s.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)

    def forward(s, x):
        pos = torch.arange(x.shape[1], device=x.device)
        h = s.wte(x) + s.wpe(pos)[None]
        for b in s.blocks:
            h = b(h)
        return s.head(s.lnf(h))


def loss_fn(model, x, y):
    logits = model(x)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


@torch.no_grad()
def evaluate(model, data, t, n_batches=50, B=64):
    model.eval()
    gen = torch.Generator().manual_seed(0)
    ls = [loss_fn(model, *data.batch(t, B, gen)).item() for _ in range(n_batches)]
    model.train()
    return float(np.mean(ls))


# ----------------------------------------------------------------------------- matrix optimizer
def newton_schulz(M, steps=5, eps=1e-7):
    X = M / (M.norm() + eps)
    tr = X.shape[0] > X.shape[1]
    if tr:
        X = X.T
    a, b, c = NS_COEF
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X
    return X.T if tr else X


def svd(M):
    try:
        return torch.linalg.svd(M, full_matrices=False)
    except Exception:  # rare cusolver failures
        U, S, Vh = torch.linalg.svd(M.cpu(), full_matrices=False)
        return U.to(M.device), S.to(M.device), Vh.to(M.device)


def svdvals(M):
    try:
        return torch.linalg.svdvals(M)
    except Exception:
        return torch.linalg.svdvals(M.cpu()).to(M.device)


def noise_gain(S, SN, eps=1e-12):
    return torch.clamp(1.0 - (SN / (S + eps)) ** 2, 0.0, 1.0)


class MatrixOpt:
    def __init__(self, params, mode, beta, wd, sw=-1, kappa_every=10):
        self.params, self.mode, self.beta, self.wd = list(params), mode, beta, wd
        self.sw, self.kappa_every = sw, kappa_every
        self.M = [torch.zeros_like(p) for p in self.params]
        self.MD = [torch.zeros_like(p) for p in self.params]
        self.kappa = [None] * len(self.params)

    @torch.no_grad()
    def step(self, G, D, lr, want_diag, t=0):
        stats = []
        b = self.beta
        for i, p in enumerate(self.params):
            M, MD = self.M[i], self.MD[i]
            M.mul_(b).add_(G[i], alpha=1 - b)
            MD.mul_(b).add_(D[i], alpha=1 - b)
            S = SN = None
            if self.mode == "muon":
                O = newton_schulz(M)
            elif self.mode == "nsgd":  # Frobenius-normalised momentum, same update norm as Muon
                O = M * (math.sqrt(min(M.shape)) / (M.norm() + 1e-12))
            elif self.mode == "muon_sw":  # Muon until step sw, then response-matched linear step kappa*M
                linear = t >= self.sw
                if (not linear) or self.kappa[i] is None or t % self.kappa_every == 0:
                    On = newton_schulz(M)  # kappa = <polar(M),M>/||M||_F^2 = best linear fit of the Muon update
                    self.kappa[i] = float((On * M).sum() / (M.pow(2).sum() + 1e-30))
                O = self.kappa[i] * M if linear else On
            else:
                U, S, Vh = svd(M)
                if self.mode == "muon_svd":
                    O = U @ Vh
                else:  # muon_gain
                    SN = svdvals(MD)
                    O = (U * noise_gain(S, SN)[None, :]) @ Vh
            if want_diag:
                S = svdvals(M) if S is None else S
                SN = svdvals(MD) if SN is None else SN
                stats.append(dict(
                    noise_frac=min(1.0, float(MD.pow(2).sum() / (M.pow(2).sum() + 1e-30))),
                    bulk_frac=float((S <= SN[0]).float().mean()),
                    gain_mean=float(noise_gain(S, SN).mean())))
            scale = max(1.0, p.shape[0] / p.shape[1]) ** 0.5
            p.mul_(1 - lr * self.wd)
            p.add_(O, alpha=-lr * scale)
        return stats


# ----------------------------------------------------------------------------- training
def lr_mult(step, total, warm, sched):
    if step < warm:
        return (step + 1) / warm
    if sched == "const":
        return 1.0
    prog = (step - warm) / max(1, total - warm)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog))


def make_tag(c):
    b = "-" if c["method"] == "adamw" else str(c["beta"])
    sw = "_sw%d" % c["sw"] if c["method"] == "muon_sw" else ""
    return "%s%s_B%d_lr%g_b%s_s%d_%s" % (c["method"], sw, c["B"], c["lr"], b, c["seed"], c["sched"])


def run_one(c, data, device, out_dir):
    tag = make_tag(c)
    path = out_dir / (tag + ".json")
    if path.exists():
        print("skip", tag)
        return
    torch.manual_seed(c["seed"])
    gen = torch.Generator().manual_seed(1000 + c["seed"])
    model = GPT(data.vocab, c["ctx"], c["d"], c["layers"], c["heads"]).to(device)
    named = list(model.named_parameters())
    is_mat = lambda n, p: p.ndim == 2 and n.startswith("blocks")
    plist = [p for _, p in named]
    mats = [p for n, p in named if is_mat(n, p)]
    mat_ids = {id(p) for p in mats}
    others = [p for p in plist if id(p) not in mat_ids]
    if c["method"] == "adamw":
        adam = torch.optim.AdamW(plist, lr=c["lr"], betas=(0.9, 0.95), weight_decay=c["wd"])
        mopt = None
        base_adam_lr = c["lr"]
    else:
        adam = torch.optim.AdamW(others, lr=c["adam_lr"], betas=(0.9, 0.95), weight_decay=c["wd"])
        mopt = MatrixOpt(mats, c["method"], c["beta"], c["wd"], c.get("sw", -1), c.get("kappa_every", 10))
        base_adam_lr = c["adam_lr"]
    ckdir = out_dir / ("ckpt_" + tag)
    curve, trl, diag, diverged = [], [], [], False
    run_loss, t0 = [], time.time()
    for step in range(c["steps"]):
        m = lr_mult(step, c["steps"], c["warm"], c["sched"])
        x, y = data.batch(data.train, c["B"], gen)
        grads, losses = [], []
        for xi, yi in zip(x.chunk(2), y.chunk(2)):
            model.zero_grad(set_to_none=True)
            loss = loss_fn(model, xi, yi)
            loss.backward()
            losses.append(loss.item())
            grads.append([p.grad.detach().clone() for p in plist])
        if not all(np.isfinite(losses)):
            diverged = True
            break
        run_loss.append(float(np.mean(losses)))
        idx = {id(p): j for j, p in enumerate(plist)}
        for p in others if mopt else plist:
            j = idx[id(p)]
            p.grad = 0.5 * (grads[0][j] + grads[1][j])
        for g in adam.param_groups:
            g["lr"] = base_adam_lr * m
        adam.step()
        if mopt:
            G = [0.5 * (grads[0][idx[id(p)]] + grads[1][idx[id(p)]]) for p in mats]
            D = [0.5 * (grads[0][idx[id(p)]] - grads[1][idx[id(p)]]) for p in mats]
            want = (step % c["diag_every"] == 0) or step == c["steps"] - 1
            st = mopt.step(G, D, c["lr"] * m, want, step)
            if st:
                diag.append(dict(step=step, **{k: float(np.median([s[k] for s in st])) for k in st[0]}))
        if (step + 1) % 25 == 0:
            trl.append((step + 1, float(np.mean(run_loss[-25:]))))
        if (step + 1) % c["eval_every"] == 0 or step + 1 == c["steps"]:
            curve.append((step + 1, evaluate(model, data, data.val)))
        k = c["save_ckpt_every"]
        if k > 0 and step + 1 >= 0.4 * c["steps"] and ((step + 1) % k == 0 or step + 1 == c["steps"]):
            ckdir.mkdir(parents=True, exist_ok=True)
            torch.save({n: p.detach().cpu() for n, p in named}, ckdir / ("%06d.pt" % (step + 1)))
    final = float("nan") if diverged else curve[-1][1]
    res = dict(c, tag=tag, final_val=final, diverged=diverged, curve=curve, train_loss=trl,
               diag=diag, secs=time.time() - t0)
    path.write_text(json.dumps(res))
    print("%-48s final_val=%.4f  %.0fs%s" % (tag, final, res["secs"], "  DIVERGED" if diverged else ""), flush=True)


def csv(s, typ):
    return [typ(v) for v in s.split(",") if v != ""]


def cmd_sweep(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Data(load_text(a.data_dir), a.ctx, device)
    print("device", device, "vocab", data.vocab, "train chars", len(data.train))
    base = dict(steps=a.steps, warm=a.warm, sched=a.sched, d=a.d, layers=a.layers, heads=a.heads,
                ctx=a.ctx, adam_lr=a.adam_lr, wd=a.wd, diag_every=a.diag_every,
                eval_every=a.eval_every, save_ckpt_every=a.save_ckpt_every, kappa_every=a.kappa_every)
    betas = csv(a.betas, float)
    for method in csv(a.methods, str):
        bs = betas[:1] if method == "adamw" else betas  # beta irrelevant for adamw
        sws = csv(a.switches, int) if method == "muon_sw" else [-1]
        for B, lr, beta, seed, sw in itertools.product(
                csv(a.Bs, int), csv(a.lrs, float), bs, csv(a.seeds, int), sws):
            run_one(dict(base, method=method, B=B, lr=lr, beta=beta, seed=seed, sw=sw), data, device, out)


# ----------------------------------------------------------------------------- summarize
def load_results(dirs):
    rs = []
    for d in dirs:
        for f in sorted(Path(d).glob("*.json")):
            try:
                r = json.loads(f.read_text())
            except Exception:
                continue
            if "final_val" in r:
                r["mkey"] = r["method"] + ("@%d" % r.get("sw", -1) if r["method"] == "muon_sw" else "")
                rs.append(r)
    return rs


def mean_se(xs):
    xs = np.array(xs, dtype=float)
    return xs.mean(), (xs.std(ddof=1) / np.sqrt(len(xs)) if len(xs) > 1 else float("nan"))


def fmt(m, se):
    return "%.4f +/- %s" % (m, "n/a" if np.isnan(se) else "%.4f" % se)


def late(r, key, frac=0.5):
    v = [d[key] for d in r["diag"] if d["step"] >= frac * r["steps"]]
    return float(np.mean(v)) if v else float("nan")


def best_lr_table(rs, method, B, beta_ref):
    """returns (best_lr, {seed: val}, all_lrs) with lowest seed-mean final_val"""
    by = {}
    for r in rs:
        if r["mkey"] == method and r["B"] == B and (method == "adamw" or abs(r["beta"] - beta_ref) < 1e-9):
            by.setdefault(r["lr"], {})[r["seed"]] = r["final_val"]
    if not by:
        return None
    best = min(by, key=lambda lr: np.nanmean(list(by[lr].values())))
    return best, by[best], sorted(by)


def cmd_summarize(a):
    rs = load_results(a.dirs)
    print("loaded %d runs\n" % len(rs))
    # --- T1 problem test
    print("== T1 problem test: how much of the momentum is noise? (late = last half of training)")
    print("   noise_frac = ||M_D||^2/||M||^2 (share of momentum energy that is noise);"
          " bulk_frac = share of M's singular values <= top noise singular value")
    print("%-10s %4s %7s %5s %4s | %10s %10s %10s | %s" % ("method", "B", "lr", "beta", "seed", "noise_frac", "bulk_frac", "gain_mean", "final_val"))
    for r in sorted((r for r in rs if r["diag"]), key=lambda r: (r["mkey"], r["B"], r["lr"], r["beta"], r["seed"])):
        print("%-10s %4d %7g %5g %4d | %10.3f %10.3f %10.3f | %.4f" % (
            r["mkey"], r["B"], r["lr"], r["beta"], r["seed"],
            late(r, "noise_frac"), late(r, "bulk_frac"), late(r, "gain_mean"), r["final_val"]))
    # --- T2 problem test
    print("\n== T2 problem test: best momentum beta vs batch size (method=%s, final val loss, lower is better)" % a.t2_method)
    sub = [r for r in rs if r["method"] == a.t2_method]
    betas = sorted({r["beta"] for r in sub})
    print("%6s | " % "B" + " ".join("%18s" % ("beta=%g" % b) for b in betas) + " | best")
    for B in sorted({r["B"] for r in sub}):
        row, means = [], {}
        for b in betas:
            vals = {}
            for r in sub:
                if r["B"] == B and abs(r["beta"] - b) < 1e-9:
                    vals.setdefault(r["lr"], []).append(r["final_val"])
            if vals:
                lr = min(vals, key=lambda k: np.mean(vals[k]))
                m, se = mean_se(vals[lr])
                means[b] = m
                row.append("%18s" % fmt(m, se).replace(" +/- n/a", ""))
            else:
                row.append("%18s" % "-")
        print("%6d | " % B + " ".join(row) + " | beta=%g" % min(means, key=means.get) if means else "")
    # --- oracle comparison
    print("\n== Oracle test (T1): muon_gain vs best-tuned Muon, beta_ref=%g" % a.beta_ref)
    mkeys = ["adamw", "muon", "muon_svd", "muon_gain", "nsgd"] + sorted({r["mkey"] for r in rs if r["method"] == "muon_sw"})
    for B in sorted({r["B"] for r in rs}):
        line = {}
        for meth in mkeys:
            t = best_lr_table(rs, meth, B, a.beta_ref)
            if t:
                lr, vals, grid = t
                m, se = mean_se(list(vals.values()))
                edge = "  (!lr at grid edge)" if lr in (grid[0], grid[-1]) and len(grid) > 1 else ""
                print("  B=%-4d %-10s best lr=%-7g n=%d  %s%s" % (B, meth, lr, len(vals), fmt(m, se), edge))
                line[meth] = (lr, vals)
        base = [mm for mm in ("muon", "muon_svd") if mm in line]
        if base:
            bm = min(base, key=lambda mm: np.mean(list(line[mm][1].values())))
            for ch in [k for k in line if k not in ("adamw", "muon", "muon_svd")]:
                common = sorted(set(line[bm][1]) & set(line[ch][1]))
                diffs = [line[bm][1][s_] - line[ch][1][s_] for s_ in common]  # >0 means challenger better
                if not diffs:
                    continue
                m, se = mean_se(diffs)
                if len(diffs) < 3:
                    verdict = "n<3 seeds: inconclusive"
                else:
                    verdict = "BETTER (>2 SE)" if m > 2 * se else ("WORSE (>2 SE)" if m < -2 * se else "no clear difference")
                print("  B=%-4d %s better than %s by %s (n=%d)  -> %s" % (B, ch, bm, fmt(m, se), len(diffs), verdict))
        print()


# ----------------------------------------------------------------------------- T3
def cmd_t3(a):
    from torch.func import functional_call
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Data(load_text(a.data_dir), a.ctx, device)
    cks = sorted(Path(a.ckpt_dir).glob("*.pt"))
    assert len(cks) >= 4, "need >=4 checkpoints in --ckpt_dir"
    sds = [torch.load(p, map_location=device) for p in cks]
    names, K = list(sds[0].keys()), len(sds)
    stack = {n: torch.stack([sd[n] for sd in sds]) for n in names}
    model = GPT(data.vocab, a.ctx, a.d, a.layers, a.heads).to(device).eval()
    gen = torch.Generator().manual_seed(123)
    fitb = [data.batch(data.valA, 64, gen) for _ in range(20)]
    evb = [data.batch(data.valB, 64, gen) for _ in range(60)]

    def losses(w, batches):
        params = {n: torch.tensordot(w, stack[n], dims=([0], [0])) for n in names}
        return torch.stack([F.cross_entropy(functional_call(model, params, (x,)).reshape(-1, data.vocab), y.reshape(-1))
                            for x, y in batches])

    def ema_w(dec):
        w = torch.tensor([dec ** (K - 1 - j) for j in range(K)], device=device)
        return w / w.sum()

    def unif_w(k):
        w = torch.zeros(K, device=device)
        w[-k:] = 1.0 / k
        return w

    cand = {"last": unif_w(1)}
    for dec in (0.5, 0.7, 0.8, 0.9, 0.95, 0.98):
        cand["ema%.2f" % dec] = ema_w(dec)
    for k in (2, 4, 8, K):
        if k <= K:
            cand["unif_last%d" % k] = unif_w(k)
    with torch.no_grad():
        fit_scores = {n: losses(w, fitb).mean().item() for n, w in cand.items()}
    best_ema = min((n for n in cand if n.startswith("ema")), key=fit_scores.get)
    logits = torch.zeros(K, device=device, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=0.1)
    for _ in range(a.fit_iters):
        opt.zero_grad()
        losses(torch.softmax(logits, 0), fitb).mean().backward()
        opt.step()
    w_or = torch.softmax(logits, 0).detach()
    with torch.no_grad():
        ev = {n: losses(w, evb) for n, w in cand.items()}
        ev["ORACLE_simplex"] = losses(w_or, evb)
    print("checkpoints: %d   (held-out split B, %d batches)" % (K, len(evb)))
    for n, v in ev.items():
        print("  %-16s val loss %.4f" % (n, v.mean().item()))
    d = (ev[best_ema] - ev["ORACLE_simplex"]).cpu().numpy()
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    print("\nbest EMA on split A: %s;  oracle improvement over it on split B: %.4f +/- %.4f -> %s" % (
        best_ema, m, se, "PASS (>2 SE)" if m > 2 * se else "no clear win"))
    d2 = (ev["last"] - ev["ORACLE_simplex"]).cpu().numpy()
    print("oracle improvement over last iterate: %.4f +/- %.4f" % (d2.mean(), d2.std(ddof=1) / np.sqrt(len(d2))))
    print("oracle weights (oldest -> newest):", np.round(w_or.cpu().numpy(), 3).tolist())
    print("note: pass T3 only if it also holds for a second LR schedule (rerun the sweep with --sched const).")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)

    def model_args(p):
        p.add_argument("--d", type=int, default=192)
        p.add_argument("--layers", type=int, default=4)
        p.add_argument("--heads", type=int, default=4)
        p.add_argument("--ctx", type=int, default=128)
        p.add_argument("--data_dir", default="data")

    s = sp.add_parser("sweep")
    model_args(s)
    s.add_argument("--methods", default="muon")
    s.add_argument("--Bs", default="8,32,128")
    s.add_argument("--lrs", default="0.02")
    s.add_argument("--betas", default="0.95")
    s.add_argument("--seeds", default="0")
    s.add_argument("--steps", type=int, default=1500)
    s.add_argument("--warm", type=int, default=50)
    s.add_argument("--sched", default="cosine", choices=["cosine", "const"])
    s.add_argument("--adam_lr", type=float, default=3e-3)
    s.add_argument("--wd", type=float, default=0.01)
    s.add_argument("--diag_every", type=int, default=50)
    s.add_argument("--eval_every", type=int, default=250)
    s.add_argument("--save_ckpt_every", type=int, default=0)
    s.add_argument("--switches", default="-1", help="muon_sw: step(s) at which Muon switches to the linear step (0 = linear from the start)")
    s.add_argument("--kappa_every", type=int, default=10)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_sweep)

    u = sp.add_parser("summarize")
    u.add_argument("dirs", nargs="+")
    u.add_argument("--beta_ref", type=float, default=0.95)
    u.add_argument("--t2_method", default="muon")
    u.set_defaults(fn=cmd_summarize)

    t = sp.add_parser("t3")
    model_args(t)
    t.add_argument("--ckpt_dir", required=True)
    t.add_argument("--fit_iters", type=int, default=150)
    t.set_defaults(fn=cmd_t3)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
