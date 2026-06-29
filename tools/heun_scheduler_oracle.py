#!/usr/bin/env python3
"""heun_scheduler_oracle.py

Generate a float64 numpy ORACLE for the Heun 2nd-order sampler added to
neuraldiffusion.pas / TestNeuralDiffusion (smHeun).

This reproduces k-diffusion's deterministic `sample_heun` / Karras EDM
"Algorithm 2" in the SAME VP -> VE bridge the Pascal scheduler uses:

  * 1-based timesteps 1..T, index 0 = clean anchor (alpha_bar_0 = 1).
  * linear beta schedule beta_t = beta1 + (betaT-beta1)*(t-1)/(T-1).
  * alpha_t = 1-beta_t, alpha_bar_t = prod_{s<=t} alpha_s.
  * sigma_t = sqrt((1-ab_t)/ab_t)  (VE noise level), sigma_0 = 0.
  * The VE sample is  y = x_t / sqrt(ab_t).  The denoiser D(y;sigma) returns the
    predicted clean image x0:  x0 = (x_t - sqrt(1-ab_t)*eps)/sqrt(ab_t)  with the
    fixed analytic stand-in model eps(x_t,t) = sin(0.01*t)*x_t (same as the
    existing oracle).
  * Heun update over the VE sample y from sigma -> sigma_next:
      d   = (y - x0) / sigma
      y_e = y + d*(sigma_next - sigma)              # Euler predictor
      if sigma_next == 0:  y = y_e                  # skip corrector on last step
      else:
          x0_2 = D at the predicted point/timestep
          d2   = (y_e - x0_2) / sigma_next
          y    = y + (sigma_next - sigma)*0.5*(d + d2)
    Then re-noise back to VP at the target timestep: x_{tPrev} = sqrt(ab_prev)*y.

The timestep grid uses the Karras rho=7 spacing (the natural pairing), snapped
to the nearest schedule timestep, exactly as BuildTimestepSchedule(tsKarras).

Emits tests/fixtures/heun_oracle.json.
"""
import json
import os
import numpy as np

T = 200
BETA1 = 1.0e-4
BETAT = 0.02
NUM_STEPS = 10
N = 8
RHO = 7.0


def build_linear(T, beta1, betaT):
    abar = np.ones(T + 1, dtype=np.float64)
    prod = 1.0
    for t in range(1, T + 1):
        b = beta1 + (betaT - beta1) * (t - 1) / (T - 1)
        prod *= (1.0 - b)
        abar[t] = prod
    return abar


def sigma_of(abar, t):
    ab = abar[t]
    if ab >= 1.0:
        return 0.0
    return np.sqrt((1.0 - ab) / ab)


def sigma_to_t(abar, target):
    best, bestd = 1, abs(sigma_of(abar, 1) - target)
    for t in range(2, T + 1):
        d = abs(sigma_of(abar, t) - target)
        if d < bestd:
            bestd, best = d, t
    return best


def karras_schedule(abar, num_steps):
    sig_min = sigma_of(abar, 1)
    sig_max = sigma_of(abar, T)
    inv = 1.0 / RHO
    sched = []
    for k in range(num_steps):
        frac = 0.0 if num_steps == 1 else k / (num_steps - 1)
        tgt = (sig_max ** inv + frac * (sig_min ** inv - sig_max ** inv)) ** RHO
        sched.append(sigma_to_t(abar, tgt))
    sched.append(0)
    return sched


def model_eps(x_t, t):
    return np.sin(0.01 * t) * x_t


def predict_x0(abar, x_t, t):
    abt = abar[t]
    eps = model_eps(x_t, t)
    return (x_t - np.sqrt(1.0 - abt) * eps) / np.sqrt(abt)


def heun_trajectory(abar, x_start, num_steps):
    sched = karras_schedule(abar, num_steps)
    x = x_start.copy()  # VP sample x_t
    for k in range(num_steps):
        t = sched[k]
        tprev = sched[k + 1]
        abt = abar[t]
        sigma = sigma_of(abar, t)
        sigma_next = sigma_of(abar, tprev)  # 0 if tprev == 0
        # VE sample at current step.
        y = x / np.sqrt(abt)
        x0 = predict_x0(abar, x, t)
        d = (y - x0) / sigma
        y_e = y + d * (sigma_next - sigma)
        if tprev == 0:
            y_new = y_e
            # final hop: y is already the clean image (sigma_next == 0).
            x = y_new  # sqrt(ab_prev)=1
        else:
            ab_prev = abar[tprev]
            # The predicted VP sample for the 2nd denoiser eval.
            x_e = np.sqrt(ab_prev) * y_e
            x0_2 = predict_x0(abar, x_e, tprev)
            d2 = (y_e - x0_2) / sigma_next
            y_new = y + (sigma_next - sigma) * 0.5 * (d + d2)
            x = np.sqrt(ab_prev) * y_new
    return x


def main():
    abar = build_linear(T, BETA1, BETAT)
    x_start = np.array([(i - N / 2) * 0.3 for i in range(N)], dtype=np.float64)
    heun_final = heun_trajectory(abar, x_start, NUM_STEPS)
    fixture = {
        "T": T,
        "beta1": BETA1,
        "betaT": BETAT,
        "num_steps": NUM_STEPS,
        "N": N,
        "rho": RHO,
        "x_start": x_start.tolist(),
        "heun_karras_final": heun_final.tolist(),
    }
    out = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures",
                       "heun_oracle.json")
    out = os.path.abspath(out)
    with open(out, "w") as f:
        json.dump(fixture, f, indent=2)
    print("wrote", out)
    print("heun_karras_final:", heun_final)


if __name__ == "__main__":
    main()
