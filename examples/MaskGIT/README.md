# MaskGIT: non-autoregressive masked-token image generation

A **non-autoregressive (parallel iterative)** image generator over discrete VQ
codebook tokens (Chang et al. 2022, *MaskGIT: Masked Generative Image
Transformer*, [arXiv:2202.04200](https://arxiv.org/abs/2202.04200)). It is the
masked-token counterpart of [VQVAE](../VQVAE): it reuses the SAME discrete
VQ-VAE front end (a tiny conv encoder → `TNNetVectorQuantizer` → decoder over
MNIST that turns every 28×28 digit into a `7×7` grid of 49 discrete code
indices), but replaces VQVAE's causal autoregressive prior with:

1. a **bidirectional** transformer (`AddTransformerEncoderBlock` with
   `CausalMask=FALSE` — the exact opposite of the VQVAE/TinyGPT causal prior),
2. a **masked-token** training objective: a random fraction of the 49 tokens is
   replaced by a dedicated `[MASK]` token and the model predicts the original ids
   at the masked positions only, and
3. **parallel iterative decoding**: generation starts with the whole grid masked
   and, over only ~10 steps, predicts every position at once, keeps the
   most-confident fraction per a **cosine** mask schedule, re-masks the rest, and
   repeats — ~10 forward passes for the whole 49-token image instead of VQVAE's
   49 autoregressive passes.

The new content is the `[MASK]`-token corruption (`FillMaskedInput`) and the
confidence-based cosine unmasking scheduler (`GenerateGrid`); the transformer
block builder, the VQ encode/decode, and the PNG I/O are reused (no new layer
class).

## Two stages

* **Stage 1 (representation)**: train encoder → `TNNetVectorQuantizer` → decoder
  to reconstruct MNIST; report reconstruction MSE + codebook usage; revive dead
  codes to avoid collapse; write a reconstruction grid PNG.
* **Stage 2 (masked transformer + parallel generation)**: freeze the encoder,
  harvest `7×7` code grids, train the bidirectional masked transformer (report
  masked-token prediction accuracy rising), then generate a grid of digits by
  parallel cosine-schedule decoding through the frozen VQ decoder; report
  generated-grid codebook usage + a NaN/Inf check; write a samples PNG.

## Build / run

This example ships **without** a `.lpi`; build it directly with `fpc`:

```
cd examples/MaskGIT
LAZUTILS_PATH=$(find /usr/lib/lazarus /usr/share/lazarus -name "utf8process.ppu" -printf "%h\n" 2>/dev/null | head -1)
fpc -O3 -Mobjfpc -Sh -Fu../../neural -Fu"$LAZUTILS_PATH" MaskGIT.lpr
./MaskGIT          # SMOKE
./MaskGIT --full   # more epochs / bigger harvest, sharper digits
```

## Run modes

- **default (SMOKE)**: short representation run + short masked-transformer run;
  well under five minutes on one CPU (the committed `time.txt` shows ~4 min). The
  point is to watch the masked-token accuracy rise and confirm the parallel
  decode fills a coherent 49-token grid in ~10 steps without NaN — the smoke
  output is deliberately rough/undertrained.
- **`--full`**: more epochs / bigger harvest for sharper digits.

## Data

Standard MNIST idx-ubyte files, looked up first in the working directory then in
`../VQVAE` and `../DiffusionMNIST` (the files every MNIST example here uses; not
copied/committed). If absent, the program falls back to a **synthetic** dataset
(random bright bars) so the demo still runs offline in CI; the pipeline and
metrics are unchanged.

## Output

`maskgit_reconstructions.png` (top = original, bottom = reconstruction) and
`maskgit_generated.png` (the parallel-decoded samples grid), plus the per-epoch
metrics printed to stdout. Pure CPU.
