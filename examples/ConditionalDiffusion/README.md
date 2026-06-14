# ConditionalDiffusion

A **class-conditional** image-generation example on MNIST. The earlier
generative image examples — [VisualGAN](../VisualGAN),
[DiffusionMNIST](../DiffusionMNIST) (in its plain form) and
[FlowMatching](../FlowMatching) — all draw **unconditional** samples: you get a
random digit, you cannot ask for a specific one. This example fills that gap and
is built around a single question — *what does the classifier-free-guidance (CFG)
weight actually do?* — which it answers **quantitatively**: the CFG weight `w`
trades sample **diversity** for class **fidelity**.

Ho & Salimans 2022, *Classifier-Free Diffusion Guidance*
(https://arxiv.org/abs/2207.12598), on top of the DDPM of Ho et al. 2020
(https://arxiv.org/abs/2006.11239).

## Conditioning mechanism

The denoiser is a small time-conditioned U-Net (mirroring DiffusionMNIST). The
class label `y ∈ 0..9` is mapped through a learned `TNNetEmbedding` to a vector
the same width as the sinusoidal **time** embedding, and the two vectors are
**added** before a shared cond MLP. A single FiLM (`TNNetFiLM`) cond vector
therefore carries both *how much noise* (`t`) and *which digit* (`y`) and
modulates every conv block.

## Label dropout → classifier-free guidance

Label index `10` is reserved as a dedicated **NULL / unconditional** token.
During training, with probability `cLabelDrop` the real label is replaced with
this null token (**label dropout**). The one network therefore learns *both* the
conditional score `eps(x_t, t, y)` and the unconditional score
`eps(x_t, t, null)`. At sampling time the net is run twice per step and the
predictions are extrapolated by the reusable scheduler's CFG mixer:

```
eps = eps_uncond + w * (eps_cond - eps_uncond)        (guidance weight w)
```

- `w = 0` → purely unconditional (most diverse, least faithful)
- `w = 1` → the plain conditional model
- `w > 1` → over-emphasises the requested class (most faithful, least diverse)

## What it measures

A tiny side **MNIST classifier** is trained purely to score class fidelity. The
denoiser is then sampled at a sweep of CFG weights `w ∈ {0, 1, 2, 4, 8}` and, per
weight, the run reports:

- **class-fidelity** — fraction of generated digits the classifier assigns to the
  *requested* class (should **rise** with `w`), and
- **diversity proxy** — mean per-pixel standard deviation across the samples of
  one requested digit (should **fall** with `w` as the model collapses onto the
  class mode).

The clean trade-off (fidelity up, diversity down as `w` grows) emerges once the
denoiser is **well trained** — run `--full`. In the short default **SMOKE** run
the denoiser is barely trained (like the sibling [DiffusionMNIST](../DiffusionMNIST)
smoke, whose grid is also noisy), so the samples are noisy and the printed numbers
mainly **validate the pipeline** (metrics computed, no NaN/Inf, PNG written); the
program self-reports the observed direction of each metric rather than asserting
the textbook one.

## Reuse

The forward noising (`AddNoise` / q_sample), the reverse DDIM trajectory
(`Sample`) and the CFG mix (`ApplyCFG`) all come from the reusable,
model-agnostic `neural/neuraldiffusion.pas` (`TNNetDiffusionScheduler`). No
noising or sampling loop is hand-rolled here.

## Run modes

- **default (SMOKE)**: a few hundred denoiser steps + a short classifier, then
  the CFG sweep on a few samples per weight; finishes in ~2–3 minutes on one CPU.
  Enough to watch the loss fall and see the fidelity/diversity trend.
- **`--full`**: many more training steps and more samples per weight for a sharp,
  statistically cleaner trade-off and a nicer PNG grid.

## Output

A PNG grid `conditional_samples.png` whose **rows** are increasing CFG weights
and whose **columns** are independent samples of one chosen digit, so you can
*see* diversity shrink as `w` grows. A per-`w` fidelity/diversity table is printed
to stdout. The run asserts there are no NaN/Inf pixels.

## Data

Standard MNIST `idx-ubyte` files in the working directory (symlinked from
[DiffusionMNIST](../DiffusionMNIST)). If absent the program prints a hint and
exits cleanly.

## Build / run

```
cd examples/ConditionalDiffusion
LAZUTILS_PATH=$(find /usr/lib/lazarus /usr/share/lazarus -name "utf8process.ppu" -printf "%h\n" 2>/dev/null | head -1)
ulimit -v 3000000
fpc -O3 -Mobjfpc -Sh -Fu../../neural -Fu"$LAZUTILS_PATH" ConditionalDiffusion.lpr
./ConditionalDiffusion          # SMOKE
./ConditionalDiffusion --full   # sharper, cleaner trend
```

Pure CPU.
