# btp_PID_optimiser_experiments

Experiment scripts for the BTP on a PID-style "governor" that scales the Adam/AdamW step size
(multiplier mu from a smoothed gradient-to-second-moment ratio, with P and I terms).

All scripts are plain Python. Most are numpy-only; the notebook generators and a few experiments need
torch / sklearn / scipy. Keep all files in the same folder (`neural_shift_exp.py` imports `digits_numpy.py`).

## Core quadratic benchmark (numpy only)
50 parameters, curvature 1..sqrt(1000), 2000 steps. eta is tuned on one regime, then evaluated on fresh seeds.

| Script | Question it answers |
|---|---|
| `robust_shift_exp.py` | Tune once on "jump x5", then face 9 noise-shift regimes. Governor vs AdamW / clip / StableAdamW |
| `unlagged_pi_test.py` | Same, with the unlagged (current-gradient) sensor, independent seeds |
| `ablate_pi.py` | Where does the win come from? Lag, smoothing, P vs PI, brake-only |
| `pi_vs_tunedP.py` | Does the integral term beat a *tuned* P governor? |
| `pi_vs_gain_check.py` | Does I beat simply raising the P gain? |
| `gov_variants_check.py` | Predictable (lagged) and leaky-integral variants |
| `early_shift_probe.py` | Why I helps when the noise jump comes early |
| `rho_star_check.py`, `rho_star_ema.py` | Different targets rho* for the controller (fixed, warm-up mean, slow EMA) |
| `noiseless_cap_check.py`, `noiseless_peaks.py` | Noiseless case, effect of the speed-up cap M and peaks |
| `gov_lag_check.py` | Closed-form quasi-stationary prediction vs simulation |
| `section3_reversal_check.py` | Reproduces the Section III / Appendix C counterexample |
| `pid_adam_identity_check.py` | AdaPID identities (D = innovation, I = momentum, Kd=0 equals Adam) |

## Small neural nets
| Script | Needs | What |
|---|---|---|
| `digits_numpy.py`, `digits_brake.py` | numpy, scipy, sklearn | Digits MLP, clean / spikes / jump conditions |
| `neural_shift_exp.py` | same (imports `digits_numpy`) | Batch-size shift test on the Digits MLP |
| `make_fmnist_notebook.py` | writes a Colab notebook | Fashion-MNIST CNN, batch size 256 -> 8 |
| `make_transformer_notebook.py` | writes a Colab notebook | Small GPT, learning-rate sweep, spikes and divergence |

Generate a notebook with: `python make_fmnist_notebook.py fmnist.ipynb`

## Side explorations
`gate0.py` (Muon noise calibration, torch), `gate_drift.py` (optimizers under drift; try `python gate_drift.py selftest`),
`reladam_gate.py` (relative-epsilon Adam on MinAtar DQN), `secant_sensor_toy.py`, `polar_vs_linear_toy.py`, `rl_endog_toy.py`.

## Install
```
pip install numpy scipy scikit-learn            # core + digits experiments
pip install torch torchvision pandas matplotlib # notebooks, gate0, reladam_gate
```
