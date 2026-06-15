# VQVAE

A **full discrete-image VQ-VAE pipeline** on MNIST — the *generative* payoff of
`TNNetVectorQuantizer`. The two diagnostic siblings
([VQCodebookUsage](../VQCodebookUsage) / [VQCodebookCollapse](../VQCodebookCollapse))
only *probe codebook health* on synthetic blobs; here the same vector-quantization
bottleneck is wired into a real convolutional encoder/decoder that reconstructs
28×28 digits, and then a small autoregressive transformer **prior** is fitted over
the discrete code grids so we can **sample brand-new digits**.

Van den Oord et al. 2017, *Neural Discrete Representation Learning*
(https://arxiv.org/abs/1711.00937).

## The two stages

**Stage 1 — representation.** Train `encoder -> TNNetVectorQuantizer -> decoder`
to reconstruct MNIST (reconstruction MSE). The gradient flows through the
non-differentiable `argmin` via the layer's built-in **straight-through
estimator**, and the two auxiliary terms (commitment loss + codebook loss) keep
the encoder outputs and codebook vectors close. All of that quantization math
already lives in `TNNetVectorQuantizer`; this example just builds an encoder and
decoder around it. Writes a reconstruction grid PNG (top rows = originals, bottom
rows = reconstructions) and reports reconstruction MSE + codebook usage.

**Stage 2 — prior + generation.** FREEZE the encoder, run it over a batch of
digits to harvest their 7×7 grids of discrete code indices, fit a tiny **causal
transformer** LM over those 49-token sequences (vocab = number of codes), then
AUTOREGRESSIVELY sample new 49-token grids, look each token up in the codebook to
rebuild a `7×7×cEmb` `z_q` tensor, and DECODE it through the frozen decoder to
synthesise a digit. Writes a generated-samples grid PNG. Mirrors the classic
two-stage VQ-VAE recipe (a discrete autoencoder first, an autoregressive prior
over its codes second — PixelCNN in the paper, a small GPT-style decoder here,
reusing `AddTransformerEncoderBlock` with `CausalMask=true` exactly like
[TinyGPT](../TinyGPT)).

## The accessor that makes it possible

Turning the continuous latent into a *discrete token grid* needs one public
accessor on `TNNetVectorQuantizer` (`neural/neuralnetwork.pas`):

```pascal
Codes[x, y] := LVQ.ChosenCodeIndex(X, Y);  // discrete argmin token at (X,Y)
                                           // cached on the last Compute()
```

This reads back the `argmin` code chosen at each grid cell on the most recent
forward pass — exactly the integer sequence the autoregressive prior is fitted
over. Out-of-range coordinates return `0`. Covered by
`TestVectorQuantizerChosenCodeIndex`.

## Run modes

- **default (SMOKE)**: a short representation run + a short prior run; finishes
  in ~2 minutes on one CPU. Enough to watch the reconstruction MSE fall, confirm
  the codebook does not collapse, and write both PNGs without NaN.
- **`--full`**: more epochs / a bigger harvest for noticeably sharper output.

## Data

Standard MNIST `idx-ubyte` files in the working directory (the same files every
MNIST example here uses). If absent the program prints a hint and exits cleanly.

## Build / run

```
cd examples/VQVAE
fpc -O3 -Mobjfpc -Sh -dRelease -dAVX2 -Fu../../neural -Fi../../neural VQVAE.lpr
./VQVAE          # SMOKE
./VQVAE --full   # sharper
```

Pure CPU.
