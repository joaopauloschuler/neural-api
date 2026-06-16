# AutoAugment (RandAugment / TrivialAugment)

Demonstrates the automatic single-image augmentation policy added to
`neuraldatasets`: a fixed op bank plus the **RandAugment** and
**TrivialAugment** selection policies and **RandomErasing / Cutout**, all
operating in place on a `TNNetVolume` image in the library's neuronal
`[-2..2]` domain.

## Op bank (`TNeuralAugOp`)

`csaIdentity`, `csaAutoContrast`, `csaEqualize`, `csaRotate`, `csaShearX`,
`csaShearY`, `csaTranslateX`, `csaTranslateY`, `csaPosterize`, `csaSolarize`,
`csaColor`, `csaContrast`, `csaBrightness`, `csaSharpness`.

Magnitudes follow the torchvision transforms-v2 `_AUGMENTATION_SPACE` ranges
on a `0..NeuralAugMaxMagnitude` (default `0..30`) integer scale; `M = 0` is
(close to) identity for every op, and `rotate(0)`, `shear(0)`, `translate(0)`,
`posterize(8 bits)` are bit-identity.

## Policies

* `NeuralRandAugment(V, N, M)` — apply `N` ops drawn uniformly from the bank at
  the same fixed magnitude `M` (Cubuk et al. 2020).
* `NeuralTrivialAugment(V)` — apply exactly one op with magnitude drawn
  uniformly; parameter-free (Müller & Hutter 2021).
* `NeuralRandomErasing(V, prob, ...)` — erase a random rectangle of the image
  (Zhong et al. 2020 / Cutout, DeVries & Taylor 2017).

All three are deterministic for a fixed `RandSeed`.

## Wiring into training

`TNeuralAugmentationPolicy` bundles a chosen policy plus an optional
RandomErasing pass into a `procedure(pInput; ThreadId) of object` matching the
new opt-in `TNeuralImageFit.ImageAugmentationFn` hook. The hook runs **after**
the built-in flip + pad-crop pipeline, so existing CIFAR examples can opt in
without disturbing the default:

```pascal
Pol := TNeuralAugmentationPolicy.Create(napTrivialAugment, 2, 9, 0.25);
Fit.ImageAugmentationFn := @Pol.Augment;
```

On real CIFAR-10 (see `examples/SimpleImageClassifier`) enabling RandAugment or
TrivialAugment + RandomErasing on top of flip+crop is expected to give a small
but consistent top-1 lift, matching the torchvision originals.

This example runs the WITHOUT vs WITH path on a tiny synthetic dataset on pure
CPU in a few seconds, exercising the hook end to end.
