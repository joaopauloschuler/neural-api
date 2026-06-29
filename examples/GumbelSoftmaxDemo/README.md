# GumbelSoftmaxDemo — differentiable categorical sampling

Demonstrates **`TNNetGumbelSoftmax`**, a differentiable categorical sampling
head that computes

```
y = softmax((logits + g) / tau),   g ~ Gumbel(0,1)
```

The Gumbel-Softmax (a.k.a. Concrete) trick replaces a non-differentiable
`argmax`/categorical sample with a smooth, temperature-controlled relaxation, so
gradients can flow through a "sampled" discrete choice. As the temperature `tau`
shrinks, the output sharpens from a soft distribution towards a one-hot vector.

## The layer

`TNNetGumbelSoftmax` is constructed with two arguments — the temperature `tau`
and a `hard` flag:

```pascal
NN.AddLayer(TNNetGumbelSoftmax.Create(tau, hard));
```

The layer operates on the depth axis of its input volume (here a length-`5`
logit vector) and produces an output of the same shape:

- **Soft mode (`hard = 0`)** — emits the relaxed distribution
  `softmax((logits + g) / tau)`. The layer is disabled by default, so on the
  deterministic inference path used by `NN.Compute` the Gumbel noise `g` is
  dropped and the output is simply `softmax(logits / tau)`.
- **Hard straight-through mode (`hard = 1`)** — emits a one-hot vector at the
  argmax (the forward value is discretized while gradients still use the soft
  relaxation).

## What the demo does

The program builds tiny one-layer networks (`TNNetInput` →
`TNNetGumbelSoftmax`) over a fixed 5-class logit vector
`(2.5, 1.0, 0.3, -0.5, 0.8)` and runs two parts:

**(a) Soft-mode temperature sweep.** On the deterministic inference path it
sweeps `tau` across `{2.0, 1.0, 0.5, 0.1}`, printing the resulting softmax
distribution and its Shannon entropy at each temperature. As `tau` shrinks the
distribution sharpens towards one-hot and the entropy falls.

**(b) Hard straight-through mode.** With `hard = 1` it shows that the layer
emits a one-hot vector (mass on the argmax logit, class 0).

A small helper computes the Shannon entropy `-Σ p·ln(p)` of each output
distribution for the printout.

## Running

```
cd examples/GumbelSoftmaxDemo
fpc -O3 -Mobjfpc -Sh -Fu../../neural GumbelSoftmaxDemo.lpr
./GumbelSoftmaxDemo
```

(or open `GumbelSoftmaxDemo.lpi` in Lazarus). Pure CPU, single thread, finishes
instantly.

## Expected output

The program prints its own headline,

```
TNNetGumbelSoftmax demo: differentiable categorical sampling.
y = softmax((logits + g) / tau),  g ~ Gumbel(0,1).
```

then part (a) — a row per temperature showing the logits, each `tau`'s softmax
distribution `y`, and its `entropy`, where the entropy decreases monotonically as
`tau` falls from `2.0` to `0.1` (the distribution sharpens towards one-hot) —
followed by part (b), a single `hard y = ...` line that is a one-hot vector at
the argmax logit (class 0), and finally `Done.`.

Coded by Claude (AI).
