# Inpainting: free-form rectangular-hole completion (context encoder)

Fill a hole in an image from its surroundings — the **context-encoder** task
(Pathak et al. 2016, *Context Encoders: Feature Learning by Inpainting*). Input
and target share the same image: a random rectangular region is zeroed out and the
network must hallucinate the missing pixels from the visible context. Synthetic
data, pure CPU; the default SMOKE run is fast.

## What it does

Each sample is a synthetic colored-shapes scene (a red circle and/or green
rectangle on a dark blue gradient, in `[-1,1]`). For every sample a random
rectangle is masked out; the **4-channel input** stacks the masked RGB with the
binary mask on the depth axis (`[maskedR | maskedG | maskedB | mask]`), telling the
net exactly which pixels are missing.

The generator is a **U-Net built by one `TNNet.AddUNet` call** (same builder as
`../Pix2Pix` / `../UNetSegmentation`), whose skip connections carry the surrounding
context across the bottleneck:

```pascal
G.AddLayer(TNNetInput.Create(16, 16, 4, 1));   // [maskedRGB | mask]
G.AddUNet(Depth, BaseFeatures, 3, Taps, {UseNorm=}false);
G.AddLayer(TNNetHyperbolicTangent.Create());   // RGB in [-1,1]
```

The loss is a **masked-region-weighted L1 + (1 − SSIM)** (hole pixels weighted
`6.0`, context `1.0`) via `neuralimagemetrics.ComputeSSIMLossAndGradient`, injected
through `TNNet.Backpropagate` with the pseudo-target identity
`Desired = Output − GradOut`. An optional `--adv` PatchGAN discriminator (LSGAN)
adds an adversarial gradient on the completed image, reusing the `../Pix2Pix`
adversarial wiring.

## Running

```
cd examples/Inpainting
lazbuild Inpainting.lpi --build-mode=Release   # or: fpc -Fu../../neural Inpainting.lpr
./Inpainting            # smoke run (reconstruction-only, default)
./Inpainting --adv      # add the optional PatchGAN adversarial term
./Inpainting --full     # longer training for a sharper result
```

## Output

Held-out L1/SSIM for the whole image and for the hole interior separately, an
ASCII panel (masked | reconstructed | original) and `inpainting_sample.ppm`.

## Notes

- The SSIM window is 11×11, so the grid is fixed at 16×16; the U-Net grid must be
  divisible by `2^Depth`.
- All data is generated in code — no download, fully offline.

Coded by Claude (AI).
