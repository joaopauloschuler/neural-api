#!/usr/bin/env python3
"""min_snr_oracle.py

Generate a float64 numpy ORACLE for the Min-SNR-gamma loss weighting added to
neuraldiffusion.pas (TNNetDiffusionScheduler.SNRWeight).

Min-SNR-gamma (Hang et al. 2023; diffusers snr_gamma):
  SNR(t) = ab_t / (1 - ab_t)  (= exp(2*lambda(t))).
  eps-prediction weight = min(SNR, gamma) / SNR
  v-prediction   weight = min(SNR, gamma) / (SNR + 1)

gamma = 5.0 is the paper default. gamma = +inf reproduces the eps base case
(weight = 1) and the standard v base case (SNR/(SNR+1)).

Emits tests/fixtures/min_snr_oracle.json.
"""
import json
import os
import math
import numpy as np

T = 200
BETA1 = 1.0e-4
BETAT = 0.02


def build_abar(T, beta1, betaT):
    abar = np.ones(T + 1, dtype=np.float64)
    prod = 1.0
    for t in range(1, T + 1):
        b = beta1 + (betaT - beta1) * (t - 1) / (T - 1)
        prod *= (1.0 - b)
        abar[t] = prod
    return abar


def snr(abar, t):
    return abar[t] / (1.0 - abar[t])


def weight(abar, t, gamma, pred):
    s = snr(abar, t)
    clamped = min(s, gamma)
    if pred == "v":
        return clamped / (s + 1.0)
    return clamped / s  # eps


def main():
    abar = build_abar(T, BETA1, BETAT)
    probes = [1, 25, 50, 100, 150, 200]
    inf = float("inf")
    fixture = {
        "T": T,
        "beta1": BETA1,
        "betaT": BETAT,
        "probe_t": probes,
        "snr": [snr(abar, t) for t in probes],
        "eps_gamma5": [weight(abar, t, 5.0, "eps") for t in probes],
        "v_gamma5": [weight(abar, t, 5.0, "v") for t in probes],
        "eps_gammainf": [weight(abar, t, inf, "eps") for t in probes],
        "v_gammainf": [weight(abar, t, inf, "v") for t in probes],
    }
    out = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures",
                       "min_snr_oracle.json")
    out = os.path.abspath(out)
    with open(out, "w") as f:
        json.dump(fixture, f, indent=2)
    print("wrote", out)
    print("eps_gamma5:", fixture["eps_gamma5"])
    print("v_gamma5:", fixture["v_gamma5"])
    print("eps_gammainf:", fixture["eps_gammainf"])
    print("v_gammainf:", fixture["v_gammainf"])


if __name__ == "__main__":
    main()
