# Information-plane trajectory (Information Bottleneck)

This example reproduces the **information-plane trajectory** of the Information
Bottleneck story (Tishby & Zaslavsky 2015; Shwartz-Ziv & Tishby 2017, *"Opening
the Black Box of Deep Neural Networks via Information"*). For a tiny
fully-connected classifier on a small synthetic binary task it tracks the
mutual-information pair `(I(X;T), I(T;Y))` of **each** hidden layer `T` across
training epochs and prints every layer's path through the 2-D information plane
as an ASCII scatter.

The narrative target is the two reported phases:

* a fast **fitting / ERM phase** where both `I(X;T)` and `I(T;Y)` rise (the
  layer becomes both more informative about the input *and* about the label);
* a slow **compression phase** where `I(X;T)` **drops** while `I(T;Y)` stays
  high — the layer forgets input detail irrelevant to the label.

No new layer and no new gradient machinery: the only addition over ordinary
training is a **forward pass** (`Net.Compute`) over the full dataset at each
logged epoch, reading each hidden layer's `Output` volume.

## The MI estimator (original binning)

For each hidden layer with `width` neurons, at each logged epoch:

1. Each neuron's activation is mapped to one of `B` equal-width bins over the
   activation's range. The per-sample bin-tuple is that sample's discrete
   **code** for layer `T`.
2. `H(T)` is the plug-in (empirical-histogram) entropy of the distribution of
   distinct codes over all `N` samples.
3. `I(X;T) = H(T) - H(T|X)`. The net is deterministic and every input is unique
   (Gaussian jitter on the bits), so `H(T|X) = 0` and **`I(X;T) = H(T)`** — the
   standard binning identity for a deterministic net.
4. `I(T;Y) = H(T) - H(T|Y)`, with `H(T|Y) = Σ_y p(y) · H(T | Y=y)` computed from
   per-class code histograms.

All quantities are reported in **bits** (log base 2).

## The honest headline: tanh vs ReLU

The binning estimator **requires a saturating activation** to show compression.
The example ships **two arms** so it demonstrates the controversy rather than
overclaiming:

* **Arm A (headline) — tanh trunk** (`TNNetFullConnect`, bounded to `[-1,1]`).
  The fixed-width bins are meaningful, and the compression bend appears: after
  the label is fit (`I(T;Y) → 1` bit), `I(X;T)` falls over the rest of training,
  most strongly in the deepest hidden layer.
* **Arm B (contrast) — ReLU trunk** (`TNNetFullConnectReLU`, unbounded). With no
  fixed range, the bins are rescaled to the observed `[0, max]` every epoch;
  this very ill-definedness washes out the clean compression bend. This is the
  point of Saxe et al. 2018, *"On the Information Bottleneck Theory of Deep
  Learning"*, which showed the compression phase is largely an artifact of
  double-saturating nonlinearities + binning.

## Sample output

The tanh arm (Arm A) shows the bend clearly. `I(T;Y)` saturates at 1 bit by the
end of the fitting phase, then `I(X;T)` drops for the deeper layers:

```
 epoch  meanNLL | layer  I(X;T)   I(T;Y)   (bits)
    10  0.0100 | L1    10.708    1.000
                | L2     9.003    1.000
                | L3     5.609    1.000     <- end of fitting
   200  0.0017 | L1    10.562    1.000
                | L2     8.093    1.000
                | L3     3.592    1.000     <- compression: I(X;T) fell 5.61->3.59
```

The deepest layer `L3` loses ~2 bits of `I(X;T)` while keeping `I(T;Y)=1`; `L2`
loses ~0.9 bits. On the ASCII plane their paths bend **left** along the top
edge — the compression bend. The first hidden layer `L1` barely compresses
(it sits near the `log2(N)` ceiling), which is also the expected pattern
(compression strengthens with depth).

The ReLU arm (Arm B), by contrast, fits the label just as well (`I(T;Y)=1`) but
`I(X;T)` stays essentially **flat** after fitting (e.g. `L3` ~5.20 → 5.14 over
190 epochs) — no clean compression bend.

## Pitfalls (read before changing the constants)

* `I(X;T)` is upper-bounded by `log2(#samples)` (here `log2(2048) ≈ 11` bits)
  and by `log2(B^width) = width·log2(B)`. Keep `width`, `B`, and the sample
  count balanced — the defaults (`width=4`, `B=30`, `N=2048`) are chosen so
  neither ceiling dominates.
* `I(T;Y)` is upper-bounded by `H(Y)`; for the balanced binary label here that
  is 1 bit.
* **Absolute MI values are estimator-dependent.** The robust, reproducible
  signal is the **shape** of the trajectory and the **tanh-vs-ReLU difference**,
  not the nats/bits. Do not read the absolute numbers as ground-truth mutual
  information.

## Building and running

```
lazbuild InformationPlane.lpi
../../bin/x86_64-linux/bin/InformationPlane
```

Pure CPU, tiny MLP, deterministic (fixed `RandSeed`); the whole run finishes in
well under a minute and well under the 5-minute budget.
