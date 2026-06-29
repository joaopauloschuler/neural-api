# VQGAN — sharpening a discrete autoencoder with perceptual + adversarial loss

Trains the **same** discrete autoencoder (conv encoder → `TNNetVectorQuantizer`
codebook → conv decoder) two ways on identical data and compares them: a plain
**L2 VQ-VAE** (reconstruction-only) versus a **VQ-GAN** (Esser et al. 2021,
*"Taming Transformers for High-Resolution Image Synthesis"*,
[arXiv:2012.09841](https://arxiv.org/abs/2012.09841)) that adds a perceptual
(feature-matching) term and a PatchGAN **adversarial** term switched on after a
warm-up. Adversarial + perceptual pressure is what turns blurry VQ-VAE
reconstructions into sharper ones with crisper edges — the whole point of VQ-GAN
over plain-L2 VQ-VAE. Fully **synthetic** (generated in-code; no dataset
download), pure CPU.

## Data (synthetic shapes)

Grayscale `16×16` images in `[-1,1]`: filled circles and rectangles on a dark
background (the same family used by `examples/Pix2Pix`). Both autoencoders see
the identical data and identical weight init for a fair comparison.

## Model

The discrete autoencoder (built once per run, mode = adversarial on/off):

```
Input(16,16,1)
  -> ConvolutionReLU(16) -> GroupNorm -> MaxPool   # 16 -> 8
  -> ConvolutionReLU(32) -> GroupNorm -> MaxPool   # 8  -> 4
  -> ConvolutionReLU(32) -> PointwiseConvLinear(16)  # z_e (4,4,16)
  -> TNNetVectorQuantizer(K=32, beta=0.05)         # codebook, straight-through
  -> PointwiseConvReLU(32)
  -> Upsample -> ConvolutionReLU(32) -> GroupNorm  # 4 -> 8
  -> Upsample -> ConvolutionReLU(16)               # 8 -> 16
  -> ConvolutionLinear(1)                          # reconstruction
```

The PatchGAN discriminator (used only by the VQ-GAN arm) is three strided
`TNNetConvolutionLinear` + `TNNetLeakyReLU` blocks; the hidden map after the
first LeakyReLU is the feature-matching tap.

## The VQ-GAN loss schedule

```
total = recon_L1 + lambda_perc * perceptual + (commitment from the VQ layer)
        + [ lambda_adv * adversarial ]   <- only after the warm-up epochs
```

- **Codebook** commitment/codebook gradients live inside `TNNetVectorQuantizer`;
  the code prints `ActiveCodeCount` and a perplexity (`exp` of the code-usage
  entropy) so you can watch the codebook stay healthy.
- **Perceptual** term reuses the LPIPS primitives from `neuralpretrained.pas`
  (`LPIPSUnitNormalize` + `LPIPSStageDistance`). Full LPIPS needs an imported VGG
  checkpoint (avoided here), so the term is a feature-matching distance over the
  discriminator's own hidden feature map (Salimans/Wang-style).
- **Adversarial** term uses the LSGAN loop and the gradient-surgery trick from
  `examples/Pix2Pix` / `examples/CycleGAN`: freeze D, read the pixel gradient off
  D's input-layer `OutputError`. The discriminator trains every step so it is
  ready when the adversarial term flips on after warm-up.

Both perceptual and adversarial pixel gradients are accumulated into the
generator's last-layer error and backpropagated by hand (see the manual-backprop
convention: `IncDepartingBranchesCnt` + `ResetBackpropCallCurrCnt`).

## Build & run

```
lazbuild examples/VQGAN/VQGAN.lpi
./bin/x86_64-linux/bin/VQGAN          # smoke run (default; well under 5 min on CPU)
./bin/x86_64-linux/bin/VQGAN --full   # longer training for a sharper result
```

## Output

- Per-epoch: recon-L1, perceptual term, active codes / K + perplexity,
  discriminator loss, and an edge-sharpness metric (mean absolute first-difference
  gradient; blurry reconstructions have low edge energy).
- A final held-out comparison: original edge sharpness vs L2-VQVAE vs VQ-GAN
  (recon-L1 + edge), an ASCII panel (`original | L2-VQVAE | VQ-GAN`), and a PPM
  strip `vqgan_sample.ppm`.

The edge-sharpness gain is modest at this toy scale and not guaranteed on every
seed (the smoke run may report no gain and suggest `--full`); the example
demonstrates the loss schedule and codebook bookkeeping, not a headline result.
