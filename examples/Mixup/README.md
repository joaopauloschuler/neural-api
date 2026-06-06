# Mixup data augmentation

This example demonstrates the Mixup data-augmentation helper added to
`neuralvolume.pas`.

[Mixup (Zhang et al. 2018, *mixup: Empirical Risk Minimization*)](https://arxiv.org/abs/1710.09412)
forms synthetic training pairs by convex-combining two real pairs:

```
x_mix = lambda*x_i + (1-lambda)*x_j
y_mix = lambda*y_i + (1-lambda)*y_j
```

with `lambda ~ Beta(alpha, alpha)`.

## API

In `neuralvolume.pas`:

- `function CreateMixedVolumePairList(Original: TNNetVolumePairList; Alpha: TNeuralFloat = 1.0; FixedLambda: TNeuralFloat = -1.0): TNNetVolumePairList;`
  Returns a NEW owning list where each pair is mixed with a randomly-permuted
  partner pair (standard minibatch mixup). The input list is not mutated.
  Pass `FixedLambda >= 0` to force a deterministic lambda (e.g. tests).
- `procedure MixVolumes(Output, A, B: TNNetVolume; Lambda: TNeuralFloat);`
  `Output := Lambda*A + (1-Lambda)*B`, reusing the AVX-backed volume ops.
- `function RandomBetaValue(Alpha: TNeuralFloat): TNeuralFloat;`
  Beta(Alpha,Alpha) sampler. `Beta(1,1) == Uniform(0,1)` (fast path); general
  alpha uses two Gamma draws (Marsaglia & Tsang 2000).

## Running

```
lazbuild Mixup.lpi
# or:
fpc -B -Funeural -Mobjfpc -Sh -O2 examples/Mixup/Mixup.lpr
./examples/Mixup/Mixup
```

The program trains the same tiny classifier on a synthetic 2-class toy with
and without mixup and prints a validation-accuracy comparison. It is pure CPU
and finishes in a few seconds.
