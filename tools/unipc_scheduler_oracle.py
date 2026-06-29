#!/usr/bin/env python3
"""unipc_scheduler_oracle.py

Generate a float64 numpy ORACLE for the UniPC (UniPCMultistepScheduler) sampler
added to neuraldiffusion.pas as TNNetSamplerMethod.smUniPC.

diffusers is NOT installed in this environment, so this is a SELF-CONTAINED
float64 reimplementation of the exact diffusers UniPCMultistepScheduler algorithm
(multistep_uni_p_bh_update predictor + multistep_uni_c_bh_update corrector),
specialized to the conventions of TNNetDiffusionScheduler:

  * 1-based timesteps 1..T, index 0 = clean anchor (alpha_bar_0 = 1).
  * linear beta schedule beta_t = beta1 + (betaT-beta1)*(t-1)/(T-1).
  * alpha_t = 1-beta_t, alpha_bar_t = prod_{s<=t} alpha_s.
  * the SAME deterministic analytic stand-in "model" used by the other oracle:
        eps(x_t,t) = sin(0.01*t)*x_t
  * uniform DDIM-style strided timestep subsequence (same indices the Pascal
    Sample() driver builds for tsUniform).

UniPC config reproduced here (the spec defaults):
    solver_order = 2, predict_x0 = True, thresholding = False,
    lower_order_final = True, solver_type = 'bh2',
    no first-order correction at the very first step.

Notation (diffusers):
    alpha_t = sqrt(alpha_bar_t),  sigma_t = sqrt(1-alpha_bar_t),
    lambda_t = log(alpha_t) - log(sigma_t)  (half-log-SNR; increases as t falls).

Emits tests/fixtures/unipc_oracle.json.
"""
import json
import os
import numpy as np

T = 200
BETA1 = 1.0e-4
BETAT = 0.02
NUM_STEPS = 10
N = 8
ORDER = 2  # solver_order


def build_linear(T, beta1, betaT):
    abar = np.ones(T + 1, dtype=np.float64)
    prod = 1.0
    for t in range(1, T + 1):
        b = beta1 + (betaT - beta1) * (t - 1) / (T - 1)
        prod *= (1.0 - b)
        abar[t] = prod
    return abar


def model_eps(x, t):
    return np.sin(0.01 * t) * x


def to_x0(x, eps, ab):
    return (x - np.sqrt(1.0 - ab) * eps) / np.sqrt(ab)


class UniPC:
    """Faithful float64 port of diffusers UniPCMultistepScheduler core, bh2,
    predict_x0=True, for the 1-based VP schedule above."""

    def __init__(self, abar, order=2, solver_type="bh2"):
        self.abar = abar
        self.order = order
        self.solver_type = solver_type
        self.predict_x0 = True
        self.lower_order_final = True
        # alpha_t, sigma_t, lambda_t tables (continuous in alpha_bar).
        self.alpha_t = np.sqrt(abar)
        self.sigma_t = np.sqrt(1.0 - abar)
        self.lambda_t = np.log(self.alpha_t) - np.log(self.sigma_t)
        # multistep history (most-recent last)
        self.model_outputs = []   # x0 predictions
        self.timestep_list = []   # the integer t for each stored output
        self.last_sample = None

    def lam(self, t):
        return self.lambda_t[t]

    # ---- predictor: multistep_uni_p_bh_update ----------------------------
    def uni_p_bh_update(self, m0, x, t, s0, order):
        # m0: current x0 prediction (model_output at current step s0)
        # x : current sample x_{s0}
        # t : target timestep, s0: current timestep
        lambda_t = self.lam(t)
        lambda_s0 = self.lam(s0)
        alpha_t = self.alpha_t[t]
        sigma_t = self.sigma_t[t]
        sigma_s0 = self.sigma_t[s0]
        h = lambda_t - lambda_s0

        rks = []
        D1s = []
        # previous outputs (excluding current m0 which is model_outputs[-1])
        for i in range(1, order):
            ti = self.timestep_list[-(i + 1)]
            mi = self.model_outputs[-(i + 1)]
            lambda_si = self.lam(ti)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)
        rks.append(1.0)
        rks = np.array(rks, dtype=np.float64)

        R = []
        b = []
        hh = -h if self.predict_x0 else h
        h_phi_1 = np.expm1(hh)          # e^{hh} - 1
        h_phi_k = h_phi_1 / hh - 1.0

        factorial_i = 1.0
        if self.solver_type == "bh1":
            B_h = hh
        elif self.solver_type == "bh2":
            B_h = np.expm1(hh)
        else:
            raise ValueError(self.solver_type)

        for i in range(1, order + 1):
            R.append(np.power(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= (i + 1)
            h_phi_k = h_phi_k / hh - 1.0 / factorial_i

        R = np.array(R, dtype=np.float64)
        b = np.array(b, dtype=np.float64)

        if len(D1s) > 0:
            D1s = np.stack(D1s, axis=0)   # (order-1, N)
            if order == 2:
                rhos_p = np.array([0.5], dtype=np.float64)
            else:
                rhos_p = np.linalg.solve(R[:-1, :-1], b[:-1])
        else:
            D1s = None

        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            if D1s is not None:
                pred_res = np.einsum("k,k...->...", rhos_p, D1s)
            else:
                pred_res = 0.0
            x_t = x_t_ - alpha_t * B_h * pred_res
        else:
            raise NotImplementedError
        return x_t

    # ---- corrector: multistep_uni_c_bh_update ----------------------------
    def uni_c_bh_update(self, m_t, last_sample, x, t, s0, order):
        # m_t: x0 prediction at the JUST-COMPUTED step t (model output at t)
        # last_sample: x_{s0} (sample BEFORE the predictor produced x)
        # x : the predictor output x_t (this_sample)
        lambda_t = self.lam(t)
        lambda_s0 = self.lam(s0)
        alpha_t = self.alpha_t[t]
        sigma_t = self.sigma_t[t]
        sigma_s0 = self.sigma_t[s0]
        h = lambda_t - lambda_s0

        m0 = self.model_outputs[-1]      # x0 pred at s0 (current latest)
        rks = []
        D1s = []
        for i in range(1, order):
            ti = self.timestep_list[-(i + 1)]
            mi = self.model_outputs[-(i + 1)]
            lambda_si = self.lam(ti)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)
        rks.append(1.0)
        rks = np.array(rks, dtype=np.float64)

        R = []
        b = []
        hh = -h if self.predict_x0 else h
        h_phi_1 = np.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1.0

        factorial_i = 1.0
        if self.solver_type == "bh1":
            B_h = hh
        elif self.solver_type == "bh2":
            B_h = np.expm1(hh)

        for i in range(1, order + 1):
            R.append(np.power(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= (i + 1)
            h_phi_k = h_phi_k / hh - 1.0 / factorial_i

        R = np.array(R, dtype=np.float64)
        b = np.array(b, dtype=np.float64)

        if len(D1s) > 0:
            D1s = np.stack(D1s, axis=0)
        else:
            D1s = None

        # corrector solves the FULL R (order x order) for rhos_c.
        if order == 1:
            rhos_c = np.array([0.5], dtype=np.float64)
        else:
            rhos_c = np.linalg.solve(R, b)

        D1_t = m_t - m0
        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * last_sample - alpha_t * h_phi_1 * m0
            if D1s is not None:
                corr_res = np.einsum("k,k...->...", rhos_c[:-1], D1s)
            else:
                corr_res = 0.0
            x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        else:
            raise NotImplementedError
        return x_t


def unipc_trajectory(abar, x_start, num_steps, order=2):
    """Mirror diffusers UniPCMultistepScheduler.step() loop exactly.

    Per step (current timestep t_cur, target t_next):
      1. m_t = x0 prediction from the INCOMING sample at t_cur.
      2. if step>0: corrector adjusts the sample using m_t, the previous
         step's stored output/sample/timestep. The model is NOT re-run; the
         stored output stays the pre-correction m_t.
      3. push m_t (pre-correction) + t_cur into history.
      4. predictor produces the next sample x_{t_next}.
      5. last_sample := the (corrected) current sample.
    """
    uni = UniPC(abar, order=order, solver_type="bh2")
    schedule = []
    for k in range(num_steps):
        kk = num_steps - 1 - k
        t = 1 + round(kk * (T - 1) / (num_steps - 1))
        schedule.append(t)
    schedule.append(0)

    x = x_start.copy()
    lower_order_nums = 0
    this_order = 1   # predictor order chosen in the PREVIOUS step (= corrector order)
    for step in range(num_steps):
        t_cur = schedule[step]
        t_next = schedule[step + 1]

        # 1. model output (x0 prediction) at current step from incoming sample.
        eps = model_eps(x, t_cur)
        m_t = to_x0(x, eps, abar[t_cur])

        # 2. corrector (uses PREVIOUS step's stored output/timestep + last_sample).
        #    The corrector order is the order the PREVIOUS step's predictor used.
        use_corrector = (step > 0) and (uni.last_sample is not None)
        if use_corrector:
            c_order = this_order
            x = uni.uni_c_bh_update(
                m_t=m_t,
                last_sample=uni.last_sample,
                x=x,
                t=t_cur,
                s0=uni.timestep_list[-1],
                order=c_order,
            )

        # 3. push current (pre-correction) output into history.
        if len(uni.model_outputs) >= order:
            uni.model_outputs.pop(0)
            uni.timestep_list.pop(0)
        uni.model_outputs.append(m_t)
        uni.timestep_list.append(t_cur)

        # 4. predictor order with lower_order_final.
        if uni.lower_order_final:
            ord_cap = min(order, num_steps - step)
        else:
            ord_cap = order
        p_order = min(ord_cap, lower_order_nums + 1)
        # carry the actually-used predictor order to the next step's corrector.
        this_order = p_order
        uni.last_sample = x.copy()
        x = uni.uni_p_bh_update(
            m0=m_t,
            x=x,
            t=t_next,
            s0=t_cur,
            order=p_order,
        )
        if lower_order_nums < order:
            lower_order_nums += 1
    return x


def main():
    abar = build_linear(T, BETA1, BETAT)
    x_start = np.array([(i - N / 2) * 0.3 for i in range(N)], dtype=np.float64)
    final = unipc_trajectory(abar, x_start, NUM_STEPS, ORDER)

    fixture = {
        "T": T, "beta1": BETA1, "betaT": BETAT,
        "num_steps": NUM_STEPS, "N": N, "order": ORDER,
        "solver_type": "bh2",
        "x_start": x_start.tolist(),
        "unipc_final": final.tolist(),
    }
    out = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures",
                       "unipc_oracle.json")
    out = os.path.abspath(out)
    with open(out, "w") as f:
        json.dump(fixture, f, indent=2)
    print("wrote", out)
    print("unipc_final:", final)


if __name__ == "__main__":
    main()
