#!/usr/bin/env python3
"""
reladam_gate.py - gate test for RelAdam (relative-epsilon Adam) on MinAtar DQN.

  Adam   : step = lr * m / (sqrt(v) + eps)
  RelAdam: step = lr * (1+kappa) * m / (sqrt(v) + kappa*rho),  rho = sqrt(mean_tensor(v))

Usage (Kaggle: `pip install minatar` first; CPU workers, no GPU needed):
  python reladam_gate.py quick                      # 1-2 min smoke test + measured time projection
  python reladam_gate.py run --phase sel            # lr selection runs (seeds 0,1)
  python reladam_gate.py run --phase eval           # evaluation runs at selected lr (seeds 2-6) + invariance runs
  python reladam_gate.py report                     # tables + G1/G2 verdicts
Runs are cached in <out>/results.jsonl, so `run` resumes after a session cut.

Gates (fixed before looking at results):
  G1: plain Adam, best eps in {1e-4,1e-3,1e-2} beats eps=1e-8 by > 2 SE on >= 2 games.
  G2: ONE kappa shared across games (a) is within 1 SE of each game's best eps on >= 2 games and
      (b) beats Adam eps=1e-8 AND eps=1e-5 by > 2 SE on >= 2 games.
  lr and hyperparameters are chosen on seeds 0,1; all verdicts use seeds 2-6 only.
"""
import argparse, json, math, os, sys, time
import numpy as np

try:
    import torch, torch.nn as nn, torch.nn.functional as F
except ImportError:
    torch = None

GAMES = ["breakout", "space_invaders", "asterix"]
EPS = [1e-8, 1e-5, 1e-4, 1e-3, 1e-2]
KAPPA = [0.3, 1.0, 3.0, 10.0]
LRS = [1e-4, 3e-4, 1e-3]
SEL, EVAL = [0, 1], [2, 3, 4, 5, 6]
CONFIGS = [("adam", e) for e in EPS] + [("rel", k) for k in KAPPA]
INV_CONFIGS, INV_SCALES, INV_LR = [("rel", 1.0), ("adam", 1e-4)], [1e-3, 1.0, 1e3], 3e-4


# ---------------------------------------------------------------- envs
class Dummy:
    """Tiny 'catch' game for smoke tests without MinAtar."""
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed); self.C, self.nA = 2, 3; self.reset()
    def reset(self):
        self.x, self.bx, self.by = 5, int(self.rng.integers(0, 10)), 0
    def state(self):
        o = np.zeros((10, 10, 2), bool); o[9, self.x, 0] = 1; o[self.by, self.bx, 1] = 1; return o
    def act(self, a):
        self.x = min(9, max(0, self.x + a - 1)); self.by += 1
        if self.by == 9: return (1.0 if self.bx == self.x else 0.0), True
        return 0.0, False


def make_env(game, seed):
    if game == "dummy":
        e = Dummy(seed); return e, e.C, e.nA
    from minatar import Environment
    e = Environment(game, sticky_action_prob=0.1, random_seed=seed)
    return e, e.state_shape()[2], e.num_actions()


# ---------------------------------------------------------------- optimiser
if torch is not None:
    class RelAdam(torch.optim.Optimizer):
        """method='adam': hp is eps (identical formula to torch Adam). method='rel': hp is kappa."""
        def __init__(self, params, lr, method, hp, betas=(0.9, 0.999)):
            super().__init__(params, dict(lr=lr))
            self.method, self.hp, self.b1, self.b2, self.t = method, float(hp), betas[0], betas[1], 0

        @torch.no_grad()
        def step(self):
            self.t += 1
            c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
            for g in self.param_groups:
                for p in g["params"]:
                    if p.grad is None: continue
                    st = self.state[p]
                    if "m" not in st:
                        st["m"], st["v"] = torch.zeros_like(p), torch.zeros_like(p)
                    st["m"].mul_(self.b1).add_(p.grad, alpha=1 - self.b1)
                    st["v"].mul_(self.b2).addcmul_(p.grad, p.grad, value=1 - self.b2)
                    mh, sv = st["m"] / c1, (st["v"] / c2).sqrt()
                    if self.method == "adam":
                        upd = mh / (sv + self.hp)
                    else:
                        rho = (st["v"].mean() / c2).sqrt()
                        upd = (1 + self.hp) * mh / (sv + self.hp * rho + 1e-30)
                    p.add_(upd, alpha=-g["lr"])

        def sqrt_v_quantiles(self):
            c2 = 1 - self.b2 ** max(self.t, 1)
            vs = [(self.state[p]["v"] / c2).sqrt().flatten() for g in self.param_groups for p in g["params"] if "v" in self.state[p]]
            if not vs: return []
            x = torch.cat(vs).double()
            return [float(v) for v in torch.quantile(x, torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99], dtype=torch.double))]


# ---------------------------------------------------------------- one DQN run
def key_of(job):
    game, method, hp, lr, scale, seed, steps = job
    return f"{game}|{method}|{hp:g}|{lr:g}|{scale:g}|{seed}|{steps}"


def run_one(job):
    game, method, hp, lr, scale, seed, steps = job
    torch.set_num_threads(1); torch.manual_seed(seed); rng = np.random.default_rng(seed)
    env, C, nA = make_env(game, seed); env.reset()
    def net():
        return nn.Sequential(nn.Conv2d(C, 16, 3), nn.ReLU(), nn.Flatten(), nn.Linear(1024, 128), nn.ReLU(), nn.Linear(128, nA))
    q, qt = net(), net(); qt.load_state_dict(q.state_dict())
    opt = RelAdam(q.parameters(), lr, method, hp)
    N = min(100_000, steps)
    O, O2 = np.zeros((N, C, 10, 10), bool), np.zeros((N, C, 10, 10), bool)
    Aa, Rr, Dd = np.zeros(N, np.int64), np.zeros(N, np.float32), np.zeros(N, np.float32)
    eps_steps, start = max(1, min(100_000, steps // 3)), min(5000, steps // 10)
    obs = lambda: np.ascontiguousarray(env.state().transpose(2, 0, 1))
    o, ep_ret, ep_len, rets, diverged, t0 = obs(), 0.0, 0, [], False, time.time()
    for t in range(steps):
        if rng.random() < max(0.1, 1 - 0.9 * t / eps_steps): a = int(rng.integers(nA))
        else:
            with torch.no_grad(): a = int(q(torch.from_numpy(o[None]).float()).argmax())
        r, done = env.act(a); r = float(r); o2 = obs(); ep_ret += r; ep_len += 1
        i = t % N; O[i], O2[i], Aa[i], Rr[i], Dd[i] = o, o2, a, r, float(done)
        o = o2
        if done or ep_len >= 5000:
            rets.append((t, ep_ret)); env.reset(); o, ep_ret, ep_len = obs(), 0.0, 0
        if t >= start and t % 4 == 0:
            idx = rng.integers(0, min(t + 1, N), 32)
            ob, o2b = torch.from_numpy(O[idx]).float(), torch.from_numpy(O2[idx]).float()
            ab, rb, db = torch.from_numpy(Aa[idx]), torch.from_numpy(Rr[idx]), torch.from_numpy(Dd[idx])
            with torch.no_grad(): y = rb + 0.99 * (1 - db) * qt(o2b).max(1)[0]
            loss = scale * F.smooth_l1_loss(q(ob).gather(1, ab[:, None])[:, 0], y)
            if not torch.isfinite(loss): diverged = True; break
            opt.zero_grad(); loss.backward(); opt.step()
        if t % 1000 == 0: qt.load_state_dict(q.state_dict())
    tail = [x for (tt, x) in rets if tt >= 0.8 * steps]
    if len(tail) < 5: tail = [x for (_, x) in rets[-10:]]
    score = float(np.mean(tail)) if tail and not diverged else 0.0
    return dict(key=key_of(job), game=game, method=method, hp=hp, lr=lr, scale=scale, seed=seed, steps=steps,
                score=score, episodes=len(rets), diverged=diverged, secs=round(time.time() - t0, 1),
                sqrt_v_q=opt.sqrt_v_quantiles())


# ---------------------------------------------------------------- bookkeeping
def load(out):
    res, path = {}, os.path.join(out, "results.jsonl")
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line:
                d = json.loads(line); res[d["key"]] = d
    return res


def mean_se(xs):
    xs = np.asarray(xs, float)
    if len(xs) == 0: return float("nan"), float("nan")
    return float(xs.mean()), float(xs.std(ddof=1) / math.sqrt(len(xs))) if len(xs) > 1 else 0.0


def scores(res, game, method, hp, lr, seeds, steps, scale=1.0):
    out = []
    for s in seeds:
        d = res.get(key_of((game, method, hp, lr, scale, s, steps)))
        if d is not None: out.append(d["score"])
    return out


def pick_lr(res, game, method, hp, steps):
    best = None
    for lr in LRS:
        xs = scores(res, game, method, hp, lr, SEL, steps)
        if len(xs) < len(SEL): return None
        m = float(np.mean(xs))
        if best is None or m > best[0]: best = (m, lr)
    return best  # (sel mean, lr)


def sel_jobs(games, steps):
    return [(g, m, hp, lr, 1.0, s, steps) for g in games for (m, hp) in CONFIGS for lr in LRS for s in SEL]


def eval_jobs(res, games, steps):
    jobs, missing = [], 0
    for g in games:
        for (m, hp) in CONFIGS:
            b = pick_lr(res, g, m, hp, steps)
            if b is None: missing += 1; continue
            jobs += [(g, m, hp, b[1], 1.0, s, steps) for s in EVAL]
    jobs += [(games[0], m, hp, INV_LR, c, s, steps) for (m, hp) in INV_CONFIGS for c in INV_SCALES for s in EVAL[:3]]
    return jobs, missing


def execute(jobs, out, workers):
    os.makedirs(out, exist_ok=True)
    res = load(out); seen, todo = set(), []
    for j in jobs:
        k = key_of(j)
        if k not in res and k not in seen: seen.add(k); todo.append(j)
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} cached or duplicate, {len(todo)} to run, {workers} workers"); sys.stdout.flush()
    if not todo: return
    import multiprocessing as mp
    t0, n = time.time(), 0
    with open(os.path.join(out, "results.jsonl"), "a") as f, mp.get_context("spawn").Pool(workers) as pool:
        for d in pool.imap_unordered(run_one, todo):
            f.write(json.dumps(d) + "\n"); f.flush(); n += 1
            el = time.time() - t0
            print(f"[{n}/{len(todo)}] {d['key']} score={d['score']:.2f} {d['secs']}s  eta {el / n * (len(todo) - n) / 60:.0f} min"); sys.stdout.flush()


# ---------------------------------------------------------------- commands
def cmd_quick(a):
    try:
        import minatar  # noqa: F401
        game = a.games[0]
    except ImportError:
        game = "dummy"; print("minatar not installed -> using built-in dummy game (pip install minatar for the real test)")
    steps, sps = 20000, []
    for (m, hp) in [("adam", 1e-8), ("rel", 1.0)]:
        d = run_one((game, m, hp, 3e-4, 1.0, 0, steps))
        sps.append(steps / d["secs"])
        print(f"{m} hp={hp:g}: score={d['score']:.3f} episodes={d['episodes']} {d['secs']}s ({sps[-1]:.0f} steps/s) sqrt_v quantiles(1,10,50,90,99%)={['%.1e' % x for x in d['sqrt_v_q']]}")
    for c in (1e-3, 1e3):
        d2 = run_one((game, "rel", 1.0, 3e-4, c, 0, steps))
        print(f"rel kappa=1 loss_scale={c:g}: score={d2['score']:.3f} (scale=1 gave {d['score']:.3f}; equal up to float32 rounding drift)")
    n_sel = len(sel_jobs(a.games, a.steps)); n_eval = len(a.games) * len(CONFIGS) * len(EVAL) + len(INV_CONFIGS) * len(INV_SCALES) * 3
    hrs = lambda n: n * a.steps / np.mean(sps) / a.workers / 3600
    print(f"\nProjection at {np.mean(sps):.0f} steps/s, {a.workers} workers, {a.steps} steps/run: sel {n_sel} runs ~{hrs(n_sel):.1f} h, eval <= {n_eval} runs ~{hrs(n_eval):.1f} h")
    print("(projection assumes the 20k-step speed holds; later steps are slower once learning starts dominating)")


def cmd_run(a):
    if a.phase in ("sel", "all"): execute(sel_jobs(a.games, a.steps), a.out, a.workers)
    if a.phase in ("eval", "all"):
        jobs, missing = eval_jobs(load(a.out), a.games, a.steps)
        if missing: print(f"WARNING: {missing} (game, config) pairs lack selection runs; run --phase sel first")
        execute(jobs, a.out, a.workers)


def cmd_report(a):
    res, steps = load(a.out), a.steps
    print(f"{len(res)} results in {a.out}. Scores = mean return of episodes ending in the last 20% of steps. Verdicts use seeds {EVAL} only.\n")
    table = {}
    for g in a.games:
        print(f"== {g} ==")
        print(f"  {'config':<14}{'lr*':>8}{'sel mean':>10}{'eval mean':>11}{'SE':>8}{'n':>3}")
        for (m, hp) in CONFIGS:
            b = pick_lr(res, g, m, hp, steps)
            if b is None: print(f"  {m + ' ' + format(hp, 'g'):<14} (selection runs missing)"); continue
            xs = scores(res, g, m, hp, b[1], EVAL, steps); mu, se = mean_se(xs)
            table[(g, m, hp)] = dict(lr=b[1], sel=b[0], mu=mu, se=se, n=len(xs))
            edge = " lr at grid edge" if b[1] in (LRS[0], LRS[-1]) else ""
            print(f"  {m + ' ' + format(hp, 'g'):<14}{b[1]:>8g}{b[0]:>10.2f}{mu:>11.2f}{se:>8.2f}{len(xs):>3}{edge}")
        qs = [res[k]["sqrt_v_q"] for k in res if res[k]["game"] == g and res[k]["method"] == "adam" and res[k]["hp"] == 1e-8 and res[k]["steps"] == steps and res[k]["sqrt_v_q"]]
        if qs: print("  sqrt(v) quantiles at end, Adam eps=1e-8 (median over runs; 1,10,50,90,99%): " + " ".join(f"{x:.1e}" for x in np.median(np.array(qs), 0)))
        print()

    def beats(x, y, k):  # x better than y by more than k SE of the difference
        return x["n"] >= 2 and y["n"] >= 2 and (x["mu"] - y["mu"]) > k * math.hypot(x["se"], y["se"])
    def within(x, y, k):  # x not worse than y by more than k SE
        return x["n"] >= 2 and y["n"] >= 2 and (y["mu"] - x["mu"]) <= k * math.hypot(x["se"], y["se"])

    # G1
    g1 = []
    for g in a.games:
        big = [(table[(g, "adam", e)]["sel"], e) for e in EPS if e >= 1e-4 and (g, "adam", e) in table]
        if not big or (g, "adam", 1e-8) not in table: print(f"G1 {g}: incomplete"); continue
        e = max(big)[1]; ok = beats(table[(g, "adam", e)], table[(g, "adam", 1e-8)], 2)
        g1.append(ok); print(f"G1 {g}: best large eps (picked on sel seeds) = {e:g}: {table[(g, 'adam', e)]['mu']:.2f} vs eps=1e-8 {table[(g, 'adam', 1e-8)]['mu']:.2f} -> {'beats by >2 SE' if ok else 'no >2 SE win'}")
    print(f"G1 verdict: {'PASS' if sum(g1) >= 2 else 'FAIL'} ({sum(g1)}/{len(a.games)} games; need 2)\n")

    # G2: one shared kappa, chosen on selection seeds by mean of per-game normalised score
    cand = []
    for k in KAPPA:
        if all((g, "rel", k) in table for g in a.games):
            norm = [table[(g, "rel", k)]["sel"] / max(1e-9, max(abs(table[(g, m, hp)]["sel"]) for (m, hp) in CONFIGS if (g, m, hp) in table)) for g in a.games]
            cand.append((float(np.mean(norm)), k))
    if not cand: print("G2: incomplete"); return
    ks = max(cand)[1]; ga, gb = [], []
    for g in a.games:
        x = table[(g, "rel", ks)]
        alle = [(table[(g, "adam", e)]["sel"], e) for e in EPS if (g, "adam", e) in table]
        if not alle or (g, "adam", 1e-8) not in table or (g, "adam", 1e-5) not in table: print(f"G2 {g}: incomplete"); continue
        eb = max(alle)[1]; a_ok = within(x, table[(g, "adam", eb)], 1)
        b_ok = beats(x, table[(g, "adam", 1e-8)], 2) and beats(x, table[(g, "adam", 1e-5)], 2)
        ga.append(a_ok); gb.append(b_ok)
        print(f"G2 {g}: rel kappa={ks:g} {x['mu']:.2f}+-{x['se']:.2f} | best eps={eb:g} {table[(g, 'adam', eb)]['mu']:.2f} ({'within 1 SE' if a_ok else 'worse'}) | eps=1e-8 {table[(g, 'adam', 1e-8)]['mu']:.2f}, eps=1e-5 {table[(g, 'adam', 1e-5)]['mu']:.2f} ({'beats both >2 SE' if b_ok else 'no'})")
    print(f"G2 verdict: {'PASS' if sum(ga) >= 2 and sum(gb) >= 2 else 'FAIL'} (a: {sum(ga)}/{len(a.games)}, b: {sum(gb)}/{len(a.games)}; need 2 each)\n")

    print(f"Loss-scale invariance on {a.games[0]} (lr={INV_LR:g}, seeds {EVAL[:3]}):")
    for (m, hp) in INV_CONFIGS:
        cells = []
        for c in INV_SCALES:
            mu, se = mean_se(scores(res, a.games[0], m, hp, INV_LR, EVAL[:3], steps, scale=c)); cells.append(f"c={c:g}: {mu:.2f}+-{se:.2f}")
        print(f"  {m} {hp:g}: " + "   ".join(cells))
    print("  (RelAdam should be flat across c up to rounding drift; Adam with fixed eps should not be.)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["quick", "run", "report"])
    p.add_argument("--phase", default="all", choices=["sel", "eval", "all"])
    p.add_argument("--out", default="out_rel")
    p.add_argument("--steps", type=int, default=400_000)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    p.add_argument("--games", nargs="+", default=GAMES)
    a = p.parse_args()
    if a.cmd != "report" and torch is None: sys.exit("PyTorch is required for quick/run")
    {"quick": cmd_quick, "run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
