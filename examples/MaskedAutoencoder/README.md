# Masked Autoencoder (MAE) — self-supervised visual pretraining

A small, pure-CPU implementation of the **Masked Autoencoder** of
He et al. 2022, *Masked Autoencoders Are Scalable Vision Learners*
([arXiv:2111.06377](https://arxiv.org/abs/2111.06377)), with a **linear-probe**
payoff that shows the self-supervised encoder is a genuinely better feature
extractor than a randomly-initialised one — learned from **unlabelled pixels
alone**.

This is the self-supervised-vision example in the repo: the existing
autoencoders (`VisualAutoencoder`, `SparseAutoencoder`, `AnomalyAutoencoder`,
`GumbelAnnealingAutoencoder`) are all *full-image* reconstructors. MAE is
different: it **drops most of the image** and reconstructs the missing patches
from the few that remain, which is what forces the encoder to learn structure.

## What MAE does

1. Cut each image into a grid of non-overlapping patches (here 3×3 → 16 tokens).
2. Randomly **mask** a large fraction of the patches each step (here 50%).
3. Embed and encode the token sequence with a ViT-style transformer encoder
   (`AddTransformerEncoderBlock` ×3) plus **learned positional embeddings**, so
   every token knows where it sits in the grid.
4. A lighter transformer decoder (×1) + a per-token linear head reconstructs the
   raw pixels of **every** patch.
5. The reconstruction MSE is applied **only to the masked patches** — the model
   must hallucinate the missing content from the visible context, with no labels.

## The payoff (the whole point)

After self-supervised pretraining we **freeze** the encoder and train a single
linear+softmax classifier on its mean-pooled token features (a *linear probe*),
on a small labelled set, and compare against the **same** probe on a
randomly-initialised encoder of the **same architecture**. A typical run:

```
Linear probe (frozen encoder, single linear classifier):
  MAE-pretrained encoder  test-acc = 0.768
  random-init   encoder  test-acc = 0.656
  chance level                     = 0.250
=> the frozen-encoder linear probe is 11.2 points more accurate than
   the same architecture at random init (0.656 -> 0.768)
```

The probe accuracy is averaged over several probe initialisations so the reported
gap is stable and reproducible (the RNG is seeded, so the numbers above reproduce
exactly).

## The task — frequency, not shape

The downstream classes are binary **stripe textures** whose label is the stripe
**period** (spatial frequency) ∈ {2,3,4,5}, with the stripe **orientation
randomised as a nuisance** the classifier must ignore. All classes share the same
first-order statistics (duty cycle / mean intensity), so they differ *only* in
spatial frequency — a **global** property no single patch reveals. Two design
choices make the MAE advantage clear at tiny scale:

* **Reconstruction is solvable.** A discrete phase and a regular period mean the
  few visible patches pin the global pattern, so masked patches are genuinely
  predictable and the MAE has real structure to learn.
* **The encoder is a narrow bottleneck** (`cDModel = 8`). A random projection of
  that width has little capacity and lands well below ceiling, whereas the MAE
  objective shapes the same bottleneck into frequency-sensitive, class-relevant
  features. (With a wide encoder, random nonlinear features are a famously strong
  linear-probe baseline and the gap disappears — narrowing the bottleneck is what
  exposes the value of the learned representation.)

## How this differs from canonical MAE (documented honestly)

Canonical MAE makes the **encoder** cheaper by feeding it only the
(variable-length) *visible* tokens and inserting mask tokens just before the
**decoder**. A static computation graph cannot express a per-sample variable
sequence length, so here the encoder runs over the **full** token grid with masked
positions occupied by a shared **(fixed, zero) mask token**; the asymmetric
"encode-visible-only" compute speedup is therefore dropped.

Everything else is faithful MAE:

* random **per-image** masking of a configurable fraction,
* **learned positional** embeddings,
* a transformer **encoder + lighter decoder** asymmetry,
* a reconstruction loss applied **only to the masked patches**.

The masked-only loss is realised exactly without any custom layer: the regression
target is built so that **visible** patch entries equal the current prediction
(zero error there) while **masked** entries hold the true pixels. For a linear
output head, `FOutputError = prediction − target` is precisely the per-element MSE
gradient, so backprop only ever sees gradient from the masked patches.

The new code versus a plain ViT is exactly: the random patch **mask** (gather of
visible vs scatter of mask tokens, here done in-place at the input), the
**masked-only loss target** construction, and the **encoder-end** feature tap that
the linear probe reads.

## Build & run

```
cd examples/MaskedAutoencoder
LAZUTILS_PATH=$(find /usr/lib/lazarus /usr/share/lazarus -name "utf8process.ppu" -printf "%h\n" 2>/dev/null | head -1)
ulimit -v 3000000
fpc -B -Fu../../neural -Fu"$LAZUTILS_PATH" -Mobjfpc -Sh -O2 MaskedAutoencoder.lpr
./MaskedAutoencoder
```

Runs in well under a minute on CPU (~10 s on a modern laptop) and stays under the
3 GB virtual-memory cap.

## Things to try

* Raise `cMaskN` toward 75% (canonical MAE ratio) and add pretraining steps.
* Widen `cDModel` and watch the random baseline catch up — a direct illustration
  of why random nonlinear features are a strong baseline and why the bottleneck
  matters here.
* Shrink `cProbeTrain` (fewer labels) — the MAE advantage typically grows in the
  few-label regime.
