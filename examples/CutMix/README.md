# CutMix data augmentation

This example demonstrates the CutMix data-augmentation helper added to
`neuralvolume.pas`.

[CutMix (Yun et al. 2019, *CutMix: Regularization Strategy to Train Strong
Classifiers with Localizable Features*)](https://arxiv.org/abs/1905.04899)
pastes a random rectangle from a second image into the first and mixes the
targets by the pasted-area fraction:

```
r            = sqrt(1 - lambda)          # cut ratio
box          = (r*W) x (r*H), random center, clamped to the image
x[box]       = x_partner[box]            # paste partner pixels (all channels)
lambda_adj   = 1 - box_area / (W*H)      # TRUE area fraction after clamping
y_mix        = lambda_adj*y_self + (1-lambda_adj)*y_partner
```

with `lambda ~ Beta(alpha, alpha)`.

## API

In `neuralvolume.pas`:

- `function CreateCutMixVolumePairList(Original: TNNetVolumePairList; Alpha: TNeuralFloat = 1.0; FixedLambda: TNeuralFloat = -1.0): TNNetVolumePairList;`
  Returns a NEW owning list where each input has a random rectangle of a
  randomly-permuted partner's input pasted in (across the full depth), and the
  target is mixed by the true pasted-area fraction. The input list is not
  mutated. Pass `FixedLambda >= 0` to force a deterministic lambda (e.g. tests).
- `procedure ComputeCutMixBox(W, H: integer; Lambda, CenterFracX, CenterFracY: TNeuralFloat; out X0, Y0, BoxW, BoxH: integer);`
  The standard CutMix `rand_bbox` geometry: cut ratio `sqrt(1-Lambda)`, box
  centered at `(CenterFracX*W, CenterFracY*H)`, clamped to the image bounds.
  Exposed separately so the geometry is deterministic / testable.
- `function RandomBetaValue(Alpha: TNeuralFloat): TNeuralFloat;`
  Beta(Alpha,Alpha) sampler shared with Mixup. `Beta(1,1) == Uniform(0,1)`
  (fast path); general alpha uses two Gamma draws (Marsaglia & Tsang 2000).

## Running

```
lazbuild CutMix.lpi
# or:
fpc -B -Funeural -Mobjfpc -Sh -O2 examples/CutMix/CutMix.lpr
./examples/CutMix/CutMix
```

The program trains the same tiny convolutional classifier on a synthetic
2-class image toy (solid-color 8x8 patches plus noise) with and without CutMix
and prints a validation-accuracy comparison. It is pure CPU and finishes in a
few seconds.
