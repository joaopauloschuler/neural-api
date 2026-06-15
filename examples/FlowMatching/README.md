# Flow Matching / Rectified Flow on MNIST

A minimal **Flow Matching / Rectified Flow** generative model that learns to
turn Gaussian noise into 28×28 MNIST digits by transporting samples along
**straight lines** in pixel space. It is the modern ODE / optimal-transport
alternative to the DDPM diffusion example in
[`examples/DiffusionMNIST`](../DiffusionMNIST): the **same tiny time-conditioned
U-Net**, but a much simpler objective and sampler.

References: Lipman et al., *Flow Matching for Generative Modeling* (2023,
[arXiv:2210.02747](https://arxiv.org/abs/2210.02747)); Liu et al., *Rectified
Flow* (2023, [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)).

## The idea

Take a noise sample `x0 ~ N(0, I)` and a real data sample `x1` (an MNIST digit).
Define the **linear interpolant** — a straight line between them — at a
continuous time `t ~ U(0,1)`:

```
x_t = (1 - t) * x0 + t * x1
```

A point moving along that line has the **constant velocity**

```
dx_t/dt = x1 - x0
```

So we train a network `v_theta(x_t, t)` to **regress that velocity** with a
plain mean-squared-error loss (the conditional flow-matching / rectified-flow
objective):

```
L = E_{t, x0, x1}  || v_theta(x_t, t) - (x1 - x0) ||^2
```

There is **no noise schedule**, no `alpha_bar` tables, and no score / noise
parameterisation — just a velocity field regressed along straight paths.

## Sampling: a forward ODE

Generation is deterministic forward-ODE integration. Start from pure noise
`x = x0 ~ N(0, I)` at `t = 0` and take a handful of explicit **Euler** steps up
to `t = 1`:

```
x_{t+dt} = x_t + dt * v_theta(x_t, t),    dt = 1 / NumSteps
```

After `NumSteps` steps we land at `t = 1`: a generated digit. Because the
learned transport paths are (near) straight, **10–50 Euler steps already give
good samples** — far fewer than the ancestral DDPM loop. This example uses 25.

## Contrast with DDPM (the score / noise-prediction view)

| | DDPM (`DiffusionMNIST`) | Flow Matching (here) |
|---|---|---|
| Forward path | corrupt `x1` on a fixed beta schedule `x_t = √ᾱ_t·x1 + √(1−ᾱ_t)·eps` | straight line `x_t = (1−t)·x0 + t·x1` |
| Training target | predict the **noise** `eps` (a score-like target) | regress the **velocity** `x1 − x0` |
| Schedule | linear beta schedule + `alpha_bar` tables | none — a single straight line, `t ~ U(0,1)` |
| Sampling | stochastic ancestral reverse loop (or deterministic DDIM) | deterministic **forward Euler ODE** |
| Steps | hundreds (ancestral) / 10–50 (DDIM) | 10–50 Euler |

Same U-Net, much simpler maths: the schedule collapses to a line, noise
prediction becomes velocity regression, and the reverse diffusion loop becomes a
forward ODE.

## Time conditioning (continuous `t`)

The network needs to know how far along the path it is. We reuse
`TNNetSinusoidalTimeEmbedding` — the very layer DDPM uses. That embedding maps a
scalar to a sinusoidal vector with `angle = t * freq[i]` and was designed for
the **integer** timestep range of diffusion (`t` up to a few hundred). Flow
matching uses a **continuous** `t ∈ [0,1]`, which would barely move the
embedding angles. To keep it well-conditioned (i.e. to use the numeric range it
was built for) we feed the embedding `t * 1000`. The training target and the ODE
step still use the true continuous `t ∈ [0,1]`; only the value handed to the
embedding is rescaled. (`cTimeScale = 1000` in the source.)

## The network

A small two-input U-Net (image + time), identical in structure to the DDPM
example minus the class-label branch:

```
image (28,28,1) -- TNNetInput
t     (1,1,1)   -- TNNetInput -> TNNetSinusoidalTimeEmbedding(64)
                                -> FullConnect MLP (shared cond vector)

enc1: Conv(16) -> GroupNorm -> FiLM(t) -> Conv(16)  ... skip A;  MaxPool -> 14x14
enc2: Conv(32) -> GroupNorm -> FiLM(t) -> Conv(32)  ... skip B;  MaxPool ->  7x7
mid : Conv(48) -> GroupNorm -> FiLM(t) -> Conv(48)
dec2: Upsample 14x14 ; DeepConcat(skip B) ; Conv(32) -> GroupNorm -> FiLM(t) -> Conv(32)
dec1: Upsample 28x28 ; DeepConcat(skip A) ; Conv(16) -> GroupNorm -> FiLM(t) -> Conv(16)
head: ConvLinear(1) -> predicted velocity v_hat (28,28,1)
```

The time embedding is injected into every block as a per-channel scale/shift via
`TNNet.AddFiLMConditioned` (FiLM); skips reuse `TNNetDeepConcat` and the decoder
upsamples with `TNNetUpsample`. All layers already existed — the only new
content versus DDPM is the linear-interpolant velocity target and the Euler ODE
sampler.

## Run

The MNIST idx-ubyte files are **reused** from the sibling DiffusionMNIST example
(loaded via `../DiffusionMNIST/...`); nothing is copied here. Run from this
directory so the relative path resolves:

```bash
lazbuild FlowMatching.lpi
cd examples/FlowMatching
../../bin/x86_64-linux/bin/FlowMatching          # SMOKE: a few hundred steps, ~2 min CPU
../../bin/x86_64-linux/bin/FlowMatching --full   # longer training, sharper digits
```

- **SMOKE** (default): 500 steps, batch 16, an 8×8 Euler-sampled grid. The
  velocity MSE falls from ~2.8 to ~0.35 within a few hundred steps; output is a
  noisy-but-recognisable `flowmatching_samples.png`. Good enough to confirm the
  pipeline trains, integrates the ODE, and writes a PNG without NaN.
- **`--full`**: 6000 steps, batch 32, a 10×10 grid — sharp digits.

Output is `flowmatching_samples.png` in the working directory (pixels are
`[-1,1]` internally, mapped to `0..255` grayscale). Generated PNGs are
git-ignored.
