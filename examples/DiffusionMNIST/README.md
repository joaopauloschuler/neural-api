# DiffusionMNIST

The repository's first **generative-by-diffusion** example: a Denoising
Diffusion Probabilistic Model (DDPM, [Ho et al. 2020](https://arxiv.org/abs/2006.11239))
that learns to synthesise 28x28 MNIST digits from pure Gaussian noise. It trains
a small time-conditioned U-Net to predict the noise added to an image, then runs
the ancestral (reverse) sampling loop to draw fresh digits and writes them as a
PNG grid.

It also supports **class-conditional generation** (pick the digit you want) via
**classifier-free guidance (CFG)** and a **deterministic DDIM fast sampler**
(10-50 steps instead of the full ancestral loop). The default run now writes a
10x10 grid where **each row is a requested digit 0..9**, sampled with DDIM+CFG.

This example also demonstrates `TNNetSinusoidalTimeEmbedding` (the layer coded
specifically for diffusion timestep conditioning) and `TNNet.AddFiLMConditioned`
(`TNNetFiLM`) for injecting the timestep into every U-Net block.

## What a DDPM is

A fixed **forward process** gradually corrupts a clean image `x_0` into pure
noise `x_T` over `T` steps by adding a little Gaussian noise on a **linear beta
schedule** (`beta_1 = 1e-4 .. beta_T = 0.02`). With `alpha_t = 1 - beta_t` and
`alpha_bar_t = prod_{s<=t} alpha_s`, the corruption has a closed form that jumps
straight to any timestep:

```
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps,   eps ~ N(0, I)
```

A network `eps_theta(x_t, t)` is trained to **predict the noise** `eps` that was
added (the simple epsilon-prediction MSE objective). Once it can denoise, the
**reverse (ancestral) process** starts from `x_T ~ N(0, I)` and walks back down
to `x_0`, at each step subtracting the predicted noise:

```
x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1 - alpha_bar_t) * eps_hat)
          + sqrt(beta_t) * z       (z ~ N(0,I), and z = 0 at t = 1)
```

Here `T = 200` (modest, so CPU sampling is fast).

## Time conditioning (TNNetSinusoidalTimeEmbedding + FiLM)

The network must know how much noise to expect at the current step, so the
integer timestep `t` is fed as a second input. `TNNetSinusoidalTimeEmbedding`
maps the scalar `t` to a sinusoidal embedding vector; a small MLP refines it
into a shared conditioning vector. Each U-Net block then calls
`TNNet.AddFiLMConditioned(featureLayer, timeEmbedding)`, which produces a
per-channel scale/shift (`gamma`, `beta`) from the embedding and applies
`out = gamma * feature + beta` via `TNNetFiLM`. This is the standard DDPM
time-embedding -> FiLM recipe.

## Class conditioning + classifier-free guidance (CFG)

To ask for a **specific digit** the network also takes the class label `y`
(0..9). We embed `y` with a `TNNetEmbedding` and **add** that vector onto the
sinusoidal time embedding *before* the shared cond MLP, so the single FiLM cond
vector carries both *how much noise* (`t`) and *which digit* (`y`).

To enable CFG we reserve **label index 10 as a special null/unconditional
token** and embed classes `0..10`. During training, with probability ~0.1, the
real label is replaced by the null token (**label dropout**). The one network
therefore learns both branches: the conditional `eps(x_t, t, y)` and the
unconditional `eps(x_t, t, null)`. At sampling time we run the net **twice** per
step and extrapolate away from the unconditional prediction:

```
eps = eps_uncond + s * (eps_cond - eps_uncond)        (guidance scale s, here 3.0)
```

`s = 0` recovers the plain unconditional model; `s > 1` sharpens the requested
class (at some cost to diversity).

## DDIM deterministic fast sampler

The ancestral DDPM loop above takes one network pass per timestep (`T` of them,
plus a second pass for CFG). The **DDIM** sampler (Song et al. 2020) is
deterministic (`eta = 0`) and runs over a short **strided subsequence** of
timesteps — here `cDDIMSteps = 25` instead of `T = 200`. At each step with
`t_prev` the previous timestep in the subsequence (`0` at the end, where
`alpha_bar = 1`):

```
x0_pred    = (x_t - sqrt(1 - abar_t) * eps) / sqrt(abar_t)
x_{t_prev} = sqrt(abar_{t_prev}) * x0_pred + sqrt(1 - abar_{t_prev}) * eps
```

`eps` here is the **CFG-guided** prediction, so DDIM and CFG compose directly.
This is what produces a clean class-conditional grid in a handful of steps; the
default run uses DDIM+CFG, and the slower ancestral loop is still available in
code (`SampleOne`).

## The network (small three-input U-Net)

```
image (28,28,1) -- TNNetInput
t     (1,1,1)   -- TNNetInput -> TNNetSinusoidalTimeEmbedding(64)  --\
y     (1,1,1)   -- TNNetInput -> TNNetEmbedding(11 -> 64)  ----------> Sum
                              -> FullConnectReLU x2  (shared cond vector)

enc1: ConvReLU(16,28x28) -> GroupNorm -> FiLM(t) -> ConvReLU(16) ...... skip A
      MaxPool /2 -> 14x14
enc2: ConvReLU(32,14x14) -> GroupNorm -> FiLM(t) -> ConvReLU(32) ...... skip B
      MaxPool /2 -> 7x7
mid : ConvReLU(48,7x7)   -> GroupNorm -> FiLM(t) -> ConvReLU(48)
dec2: Upsample x2 -> 14x14 ; DeepConcat(skip B) ; ConvReLU(32) -> GroupNorm
      -> FiLM(t) -> ConvReLU(32)
dec1: Upsample x2 -> 28x28 ; DeepConcat(skip A) ; ConvReLU(16) -> GroupNorm
      -> FiLM(t) -> ConvReLU(16)
head: ConvolutionLinear(1, 3x3)  -> predicted noise eps_hat (28,28,1)
```

Skip connections reuse `TNNetDeepConcat` (depth-axis concat); the decoder
upsamples with the parameter-free `TNNetUpsample`. Every layer already existed
in the library -- the new content is the noise schedule, the epsilon-prediction
training loop and the ancestral sampling loop.

## Data

Standard MNIST `idx-ubyte` files in the working directory (the same files every
MNIST example here uses: `train-images.idx3-ubyte`, `train-labels.idx1-ubyte`,
`t10k-images.idx3-ubyte`, `t10k-labels.idx1-ubyte`). If they are not present the
program prints the download hint and exits cleanly. Download from
[the MNIST page](http://yann.lecun.com/exdb/mnist/) (or any standard mirror) and
gunzip the four files into this directory. Pixels are rescaled to `[-1, 1]` for
diffusion.

## Build & run

```
cd examples/DiffusionMNIST
lazbuild DiffusionMNIST.lpi --build-mode=Release
../../bin/x86_64-linux/bin/DiffusionMNIST
```

Two run modes:

- **default (smoke)**: a few hundred training steps then a 10x10
  class-conditional sample grid (DDIM+CFG); finishes in a couple of minutes on
  one CPU. Enough to see the loss fall and confirm a PNG grid is written without
  NaN.
- **full**: pass `--full` for many more steps (sharper digits); still a 10x10
  class-conditional grid.

The generated grid is written to `diffusion_samples.png` in the working
directory (grayscale, `GridN*28` square).

## Sample output (smoke mode)

```
Training: 500 steps, batch 16, schedule T=200 (beta 1.0E-004..0.02).
  step    1 /  500   eps-MSE = 1.16370   elapsed = 0.2s
  step   25 /  500   eps-MSE = 0.41816   elapsed = 5.3s
  step  100 /  500   eps-MSE = 0.25064   elapsed = 21.3s
  ...
  step  500 /  500   eps-MSE = 0.11035   elapsed = 106.2s

Sampling a 10x10 class-conditional grid via 25-step DDIM + CFG (s=3.0)...
Wrote sample grid: diffusion_samples.png
No NaN/Inf in generated samples.
Done. View diffusion_samples.png (row r = requested digit r mod 10).
```

## Reading the result

The epsilon-prediction MSE starts at ~1.0 (the trivial baseline of predicting
`eps ~ N(0,1)` as zero, whose MSE equals the noise variance 1.0) and falls well
below it within a hundred steps, confirming the U-Net is genuinely learning to
denoise rather than emitting zeros. A from-scratch diffusion net produces large
gradient spikes before the schedule statistics settle, so the training loop
clips the per-step update magnitude (`ForceMaxAbsoluteDelta`).

Smoke-mode samples are recognisably digit-shaped blobs but noisy; `--full`
training sharpens them considerably. This is a deliberately tiny CPU-friendly
configuration, not a state-of-the-art generator.

## Follow-ups

- **CIFAR-10 / colour**: swap the loader for `loadCifar10Dataset`, widen the
  channels and use a 3-channel head -- the schedule and sampling loop are
  unchanged.
- **DDIM / fewer steps**: *done* — deterministic DDIM sampler (`SampleOneDDIM`,
  `cDDIMSteps`) for fast sampling with far fewer than `T` reverse steps.
- **Class conditioning + CFG**: *done* — label embedding added to the time
  embedding, trained with label dropout, sampled with classifier-free guidance
  (`PredictEpsCFG`, `cGuidance`).
- **Latent diffusion**: run the diffusion in the latent space of an
  autoencoder (see `examples/AnomalyAutoencoder` / the imported SD VAE) instead
  of pixel space.
- **v-prediction / cosine schedule**: alternative objectives/schedules that
  often train more stably than linear-beta epsilon-prediction.
```

