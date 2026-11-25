"""Numerical check of Section III's stochastic counterexample and its generalization.

Part 1: reproduce App C (E d51, E mu51 d51) from the stated warm-up history.
Part 2: for several brake-type gains mu(|g|), search two-point unbiased distributions
        where plain Adam's expected step is > 0 and the gated step is < 0.
Python 3, NumPy only.
"""
import math
import numpy as np

b1, b2 = 0.9, 0.999
eps_a = eps_v = 1e-8


# ---------------- Part 1: reproduce Appendix C ----------------
def appendix_c():
    x = math.sqrt((1 - b2**50) / ((1 - b2) * (b2 + b1**2)))
    hist = [0.0] * 48 + [x, -b1 * x]          # g1..g50
    m = v = 0.0
    for t, g in enumerate(hist, start=1):
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
    mh50, vh50 = m / (1 - b1**50), v / (1 - b2**50)
    t = 51
    out = {}
    for g, p in ((10.0, 0.2), (-1.0, 0.8)):
        m1 = b1 * m + (1 - b1) * g
        v1 = b2 * v + (1 - b2) * g * g
        d = (m1 / (1 - b1**t)) / (math.sqrt(v1 / (1 - b2**t)) + eps_a)
        rho = g * g / (vh50 + eps_v)
        z = 0.9 * 1.0 + 0.1 * rho                      # frozen z=1 before step 51
        mu = min(2.0, max(0.1, z ** -0.5))             # P law, kp=1/2, no I/D
        out[g] = (p, d, mu)
    Ed = sum(p * d for p, d, mu in out.values())
    Emd = sum(p * mu * d for p, d, mu in out.values())
    print(f"x = {x:.7f} (paper 5.1935651); mhat50 = {mh50:.2e}, vhat50 = {vh50:.7f}")
    print(f"E d51      = {Ed:.7f} (paper 0.0358267)")
    print(f"E mu*d51   = {Emd:.7f} (paper -0.0451770)")
    print("multipliers:", {g: round(mu, 4) for g, (p, d, mu) in out.items()})


# ---------------- Part 2: generalization ----------------
def d_adam(g, h=1.0, t=51):
    """One-step Adam direction from a state with mhat_{t-1}=0, vhat_{t-1}=h (bias-corrected)."""
    bt = (1 - b2) / (1 - b2**t)
    gam = (1 - b1) / (1 - b1**t)
    return gam * g / math.sqrt((1 - bt) * h + bt * g * g)


def gate_governor(a, h=1.0, bz=0.9):
    z = bz * 1.0 + (1 - bz) * a * a / h
    return min(2.0, max(0.1, z ** -0.5))


GATES = {
    "governor P (smoothed, c=.1)": gate_governor,
    "lagged clip (no smoothing)": lambda a: min(1.0, (a * a) ** -0.5) if a > 0 else 1.0,
    "sqrt-brake mu=(1+g^2)^-0.25": lambda a: (1 + a * a) ** -0.25,
    "very mild brake mu=(1+g^2)^-0.01": lambda a: (1 + a * a) ** -0.01,
    "constant mu=0.5 (control)": lambda a: 0.5,
}


def search(gate, n=400000, seed=0):
    rng = np.random.default_rng(seed)
    hits, tested = 0, 0
    best = None
    for _ in range(n):
        x = math.exp(rng.uniform(math.log(0.05), math.log(100)))   # positive sample
        y = math.exp(rng.uniform(math.log(0.05), math.log(100)))   # negative sample magnitude
        r = math.exp(rng.uniform(0, math.log(20)))                 # r = p x / ((1-p) y) > 1  => mean > 0
        w = r * y / x                                              # odds p/(1-p)
        p = w / (1 + w)
        dx, dy = d_adam(x), d_adam(y)
        mx, my = gate(x), gate(y)
        Ed = p * dx - (1 - p) * dy
        Emd = p * mx * dx - (1 - p) * my * dy
        tested += 1
        if Ed > 0 and Emd < 0:
            hits += 1
            if best is None or Emd / Ed < best[0]:
                best = (Emd / Ed, x, y, p)
    return hits, tested, best


if __name__ == "__main__":
    print("=== Part 1: Appendix C ===")
    appendix_c()
    print("\n=== Part 2: two-point unbiased distributions, mean>0, Adam E d>0, gated E<0 ===")
    for name, gate in GATES.items():
        hits, tested, best = search(gate)
        line = f"{name:36s} hits {hits:6d}/{tested}"
        if best:
            ratio, x, y, p = best
            line += f"   example: x={x:.2f} y={y:.2f} p={p:.3f} gatedE/AdamE={ratio:.2f}"
        print(line)
