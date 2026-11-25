"""Toy: can the 'D channel' (gradient difference) measure curvature under minibatch noise?

Model: f = 0.5 theta' H theta, H diagonal (log-uniform 0.1..10), n coords. Two independent half-batch gradients
g^(i) = H theta + z^(i), z ~ N(0, s2) per coordinate. Plain SGD, lr eta.
Target: curvature along the actual steps  R = sum <H dth, dth> / sum <dth, dth>  (known exactly in the toy).

Sensors (all use only gradients, no Hessian):
  naive  : full-batch  <gbar_t - gbar_{t-1}, dth_t> / <dth_t, dth_t>      (what a D term / stochastic BB sees)
  cross  : half-batch roles alternate each step; update uses half A only, sensor uses half B:
           <gB_t - gB_{t-1}, dth_t> / <dth_t, dth_t>
numpy only.
"""
import numpy as np

rng = np.random.RandomState(1)
n, s2, T, burn = 2000, 1.0, 6000, 500
h = np.exp(rng.uniform(np.log(0.1), np.log(10), n))


def run(eta, mode):
    th = rng.randn(n) * 0.1
    g_prev = {0: h * th, 1: h * th}    # previous half-batch gradients per half (index = which half)
    gbar_prev = h * th
    num = den = true_num = 0.0
    prev_role = 0
    gp = [h * th + rng.randn(n) * np.sqrt(s2), h * th + rng.randn(n) * np.sqrt(s2)]
    # state: theta_{t-1}, gradients at theta_{t-1} for both halves
    for t in range(T):
        upd = t % 2                     # half used for the update at this step (roles alternate)
        sen = 1 - upd
        if mode == "naive":
            step = -eta * 0.5 * (gp[0] + gp[1])    # update from full batch gradient at theta_{t-1}
        else:
            step = -eta * gp[upd]                  # update from ONE half only
        th_new = th + step
        gn = [h * th_new + rng.randn(n) * np.sqrt(s2), h * th_new + rng.randn(n) * np.sqrt(s2)]
        if t >= burn:
            if mode == "naive":
                dg = 0.5 * (gn[0] + gn[1]) - 0.5 * (gp[0] + gp[1])
            else:
                dg = gn[sen] - gp[sen]             # sensor half: independent of the step
            num += dg @ step
            den += step @ step
            true_num += (h * step) @ step
        th, gp = th_new, gn
    return num / den, true_num / den


for eta in (0.01, 0.003):
    for mode in ("naive", "cross"):
        est, true = run(eta, mode)
        print("eta=%.3f  %-5s sensor: estimate %8.3f   true R %6.3f   1/eta %7.1f" % (eta, mode, est, true, 1 / eta))
