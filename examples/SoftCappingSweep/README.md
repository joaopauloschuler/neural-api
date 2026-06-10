# SoftCappingSweep — causal-mask + logit SoftCapping interaction study

Modern decoder language models (e.g. Gemma) "soft-cap" their output logits
before the softmax / cross-entropy via

```
capped = c * tanh(logit / c)
```

which smoothly squashes any logit into the open interval `(-c, +c)`. The cap
`c` trades two forces against each other:

* a **tight** `c` keeps the value entering the softmax bounded and
  well-conditioned, but limits how confident the model can become, and
* a **loose** (or absent) cap lets logits grow without bound — maximally
  expressive but prone to large, ill-conditioned activations.

This example trains the **same tiny causal next-token model** once per cap
setting

```
c in {5, 10, 20, 30, infinity}
```

where the `infinity` arm inserts **no** `TNNetSoftCapping` layer at all (the
uncapped baseline). Every arm shares the RNG seed, architecture, data,
learning rate and epoch count, so the only thing that varies is the soft-cap
applied to the logits immediately before the softmax.

## Task

A tiny char-level **copy-the-previous-token** stream over a small synthetic
vocabulary: `target[i] = input[i-1]` (begin token at the first position). It
is causal (the answer is always strictly to the left) and learnable enough
that an unconstrained model wants to drive its logits large to express high
confidence — exactly the regime where soft-capping bites.

## Architecture

A single-head **causal** attention block; only the cap changes between arms:

```
TNNetInput(SeqLen, 1, 1)
-> TNNetEmbedding(Vocab, d_model)
-> packed Q|K|V via TNNetPointwiseConvLinear + three TNNetSplitChannels
-> scores = TNNetDotProducts(Q, K) / sqrt(d_k)
-> reshape -> TNNetMaskedFill   { causal upper triangle }
-> reshape -> ReLUL -> PointwiseSoftMax(depth)
-> TNNetDotProducts(ValueT, W)
-> TNNetPointwiseConvLinear(Vocab)   { raw logits  <- probed }
-> [ TNNetSoftCapping(c) ]           { THE SWEPT KNOB; omitted for inf }
-> TNNetPointwiseSoftMax(1)
```

The attention wiring mirrors `examples/ALiBiSlopeSweep`.

## Metrics reported per arm

* **train-CE** — final-step mean training cross-entropy.
* **probe-CE** — mean cross-entropy on a held-out probe stream.
* **eff-norm** — max `|value fed to the softmax|` (post-cap; the *effective*
  max-logit-norm).
* **raw-norm** — max `|pre-cap logit|` the network produced.

## Build & run

```
lazbuild --build-mode=Release SoftCappingSweep.lpi
../../bin/x86_64-linux/bin/SoftCappingSweep
```

Pure CPU, single thread, no dataset download. Runs in ~10 seconds.

## Sample result

```
   cap c     train-CE     probe-CE     eff-norm      raw-norm
  -------    ---------    ---------    ---------    ----------
       5      1.55809      1.51806       5.0000       51.2716
      10      1.59840      1.54552       9.9991       49.8464
      20      1.77714      1.74167       5.8501        6.0261
      30      1.77871      1.74431       5.2137        5.2671
     inf      1.77957      1.74587       4.8460        4.8460
```

## Takeaway

The **effective** logit-norm fed to the softmax is strictly bounded by the
cap `c` (`eff-norm` never exceeds `c`) — that is precisely what SoftCapping
buys, and it is what conditions the softmax input. But `tanh` *saturates*: to
stay confident under a tight cap, the network simply **inflates its raw
pre-cap logits** (`raw-norm` balloons to ~50 at `c=5`, versus ~5 when the cap
is loose or absent). So a tight cap conditions the value entering the softmax
without truly taming the underlying logits.

On this easy task the tight-cap arms even reach a *lower* cross-entropy — the
bounded softmax input keeps optimization stable enough that the network learns
the copy rule faster within the fixed step budget; the loose / uncapped arms
plateau slightly higher. The headline lesson is that "logit norm" depends on
*where you measure*: SoftCapping caps the softmax input, not the raw projection.
```
