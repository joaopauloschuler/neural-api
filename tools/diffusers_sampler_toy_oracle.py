#!/usr/bin/env python3
"""diffusers_sampler_toy_oracle.py

Float64 diffusers oracle for the PNDM / Euler / Euler-ancestral cases of
TestNeuralDiffusion. Same toy setup as tools/diffusion_scheduler_oracle.py:
T = 200 linear betas 1e-4..0.02, an 8-vector start x[i] = (i - 4)*0.3 and the
analytic "model" out(x, t) = sin(0.01*(t+1))*x (t+1 = the 1-based Pascal Tt).
The model sees the scheduler's scale_model_input, so for Euler it is fed the
VP sample y/sqrt(sigma^2+1), as TNNetDiffusionScheduler keeps it.

Prints Pascal constants to paste into tests/TestNeuralDiffusion.pas:
  PNDM (skip_prk_steps, linspace, set_alpha_to_one False), eps and v-pred;
  Euler (trailing spacing, so init_noise_sigma = sigma_max);
  Euler-ancestral (leading, steps_offset 0) with the noise it drew per step.

Coded by Claude (AI).

Usage (inside the `x` venv):  python tools/diffusers_sampler_toy_oracle.py
"""
import numpy as np
import torch
import diffusers.schedulers.scheduling_euler_ancestral_discrete as euler_a_module
from diffusers import (EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       PNDMScheduler)

T = 200
N = 8
COMMON = dict(num_train_timesteps=T, beta_start=1e-4, beta_end=0.02,
              beta_schedule="linear")


def start():
    return torch.tensor([(i - N / 2) * 0.3 for i in range(N)], dtype=torch.float64)


def toy(x, t):
    return np.sin(0.01 * (float(t) + 1)) * x


def run(scheduler, steps):
    scheduler.set_timesteps(steps)
    x = start() * scheduler.init_noise_sigma
    for t in scheduler.timesteps:
        out = toy(scheduler.scale_model_input(x, t), t)
        x = scheduler.step(out, t, x).prev_sample
    return scheduler.timesteps.tolist(), x


def pascal(name, values, comment):
    body = ", ".join(repr(float(v)) for v in values)
    print("  // " + comment)
    print("  %s: array[0..%d] of double =\n    (%s);" % (name, len(values) - 1, body))


def main():
    for pred, name in (("epsilon", "cOraclePNDM"), ("v_prediction", "cOraclePNDMV")):
        ts, x = run(PNDMScheduler(skip_prk_steps=True, set_alpha_to_one=False,
                                  timestep_spacing="linspace", prediction_type=pred,
                                  **COMMON), 10)
        pascal(name, x.tolist(), "PNDM %s, timesteps %s" % (pred, ts))
    ts, x = run(EulerDiscreteScheduler(timestep_spacing="trailing", **COMMON), 10)
    pascal("cOracleEuler", x.tolist(), "Euler trailing, timesteps %s" % ts)

    noises = []
    original = euler_a_module.randn_tensor

    def recording_randn(*args, **kwargs):
        value = original(*args, **kwargs)
        noises.append(value.clone())
        return value
    euler_a_module.randn_tensor = recording_randn
    scheduler = EulerAncestralDiscreteScheduler(timestep_spacing="leading",
                                                steps_offset=0, **COMMON)
    scheduler.set_timesteps(4)
    x = start() * scheduler.init_noise_sigma
    generator = torch.Generator().manual_seed(7)
    for t in scheduler.timesteps:
        out = toy(scheduler.scale_model_input(x, t), t)
        x = scheduler.step(out, t, x, generator=generator).prev_sample
    euler_a_module.randn_tensor = original
    flat = [v for noise in noises[:-1] for v in noise.tolist()]
    pascal("cEulerANoise", flat, "Euler-a step noises (last step's is unused)")
    pascal("cOracleEulerA", x.tolist(), "Euler-a leading, timesteps %s"
           % scheduler.timesteps.tolist())


if __name__ == "__main__":
    main()
