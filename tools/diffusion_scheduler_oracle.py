#!/usr/bin/env python3
"""diffusion_scheduler_oracle.py

Generate a float64 numpy ORACLE for neuraldiffusion.pas / TestNeuralDiffusion.

Reproduces, in plain numpy float64, the exact conventions of
TNNetDiffusionScheduler:
  * 1-based timesteps 1..T, index 0 = clean anchor (alpha_bar_0 = 1).
  * linear beta schedule beta_t = beta1 + (betaT-beta1)*(t-1)/(T-1).
  * alpha_t = 1-beta_t, alpha_bar_t = prod_{s<=t} alpha_s.
  * a DETERMINISTIC DDIM (eta=0) reverse trajectory driven by a FIXED, analytic
    "model" eps(x_t, t) = sin(0.01*t) * x_t (a deterministic stand-in for a real
    network so the trajectory is reproducible without training). Same callback is
    used on the Pascal side.

The DDIM step matches the unit (eta=0):
    x0 = (x_t - sqrt(1-ab_t)*eps)/sqrt(ab_t)
    x_{tPrev} = sqrt(ab_prev)*x0 + sqrt(1-ab_prev)*eps

Emits a small JSON fixture committed at tests/fixtures/diffusion_oracle.json.
"""
import json
import os
import numpy as np

T = 200
BETA1 = 1.0e-4
BETAT = 0.02
NUM_STEPS = 10       # DDIM strided subsequence
N = 8                # vector length of the toy "image"


def build_linear(T, beta1, betaT):
    beta = np.zeros(T + 1, dtype=np.float64)
    alpha = np.ones(T + 1, dtype=np.float64)
    abar = np.ones(T + 1, dtype=np.float64)
    prod = 1.0
    for t in range(1, T + 1):
        b = beta1 + (betaT - beta1) * (t - 1) / (T - 1)
        beta[t] = b
        alpha[t] = 1.0 - b
        prod *= (1.0 - b)
        abar[t] = prod
    return beta, alpha, abar


def model_eps(x, t):
    # Deterministic analytic stand-in for a trained eps predictor.
    return np.sin(0.01 * t) * x


def ddim_trajectory(abar, x0_noise, num_steps):
    x = x0_noise.copy()
    traj = [x.copy()]
    for k in range(num_steps - 1, -1, -1):
        t = 1 + round(k * (T - 1) / (num_steps - 1))
        tprev = 1 + round((k - 1) * (T - 1) / (num_steps - 1)) if k > 0 else 0
        eps = model_eps(x, t)
        ab_t = abar[t]
        ab_prev = abar[tprev]
        x0 = (x - np.sqrt(1.0 - ab_t) * eps) / np.sqrt(ab_t)
        x = np.sqrt(ab_prev) * x0 + np.sqrt(1.0 - ab_prev) * eps
        traj.append(x.copy())
    return x, traj


def lam(abar, t):
    return 0.5 * (np.log(abar[t]) - np.log(1.0 - abar[t]))


def dpmpp_2m_trajectory(abar, x_start, num_steps):
    # DPM-Solver++(2M), data-prediction form (matches neuraldiffusion.pas).
    x = x_start.copy()
    prevx0 = None
    prevlam = None
    for k in range(num_steps - 1, -1, -1):
        t = 1 + round(k * (T - 1) / (num_steps - 1))
        tprev = 1 + round((k - 1) * (T - 1) / (num_steps - 1)) if k > 0 else 0
        eps = model_eps(x, t)
        abt = abar[t]
        x0 = (x - np.sqrt(1.0 - abt) * eps) / np.sqrt(abt)
        lamt = lam(abar, t)
        if tprev == 0:
            if prevx0 is not None:
                hlast = lamt - prevlam
                h = -hlast
                r0 = hlast / h
                x = (1 + 1 / (2 * r0)) * x0 - (1 / (2 * r0)) * prevx0
            else:
                x = x0
            break
        abprev = abar[tprev]
        lamprev = lam(abar, tprev)
        h = lamprev - lamt
        if prevx0 is None:
            x = (np.sqrt(abprev) / np.sqrt(abt)) * x \
                - np.sqrt(1 - abprev) * (np.exp(-h) - 1) * x0
        else:
            hlast = lamt - prevlam
            r0 = hlast / h
            D = (1 + 1 / (2 * r0)) * x0 - (1 / (2 * r0)) * prevx0
            x = (np.sqrt(abprev) / np.sqrt(abt)) * x \
                - np.sqrt(1 - abprev) * (np.exp(-h) - 1) * D
        prevx0 = x0
        prevlam = lamt
    return x


def main():
    beta, alpha, abar = build_linear(T, BETA1, BETAT)
    # Fixed deterministic start "noise" (no RNG dependence across languages).
    x_start = np.array([(i - N / 2) * 0.3 for i in range(N)], dtype=np.float64)
    final, traj = ddim_trajectory(abar, x_start, NUM_STEPS)
    dpm_final = dpmpp_2m_trajectory(abar, x_start, NUM_STEPS)

    # A few probe timesteps for the schedule math check.
    probes = [1, 50, 100, 150, 200]
    fixture = {
        "T": T,
        "beta1": BETA1,
        "betaT": BETAT,
        "num_steps": NUM_STEPS,
        "N": N,
        "x_start": x_start.tolist(),
        "probe_t": probes,
        "beta": [beta[t] for t in probes],
        "alpha": [alpha[t] for t in probes],
        "alpha_bar": [abar[t] for t in probes],
        "ddim_final": final.tolist(),
        "dpmpp_2m_final": dpm_final.tolist(),
    }
    out = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures",
                       "diffusion_oracle.json")
    out = os.path.abspath(out)
    with open(out, "w") as f:
        json.dump(fixture, f, indent=2)
    print("wrote", out)
    print("alpha_bar @ probes:", fixture["alpha_bar"])
    print("ddim_final:", final)


if __name__ == "__main__":
    main()
