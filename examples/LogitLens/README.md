# LogitLens

Tiny example for `TNNet.LogitLensReport`, the **logit-lens** diagnostic
(nostalgebraist 2020; cf. "Tuned Lens", Belrose et al. 2023) that answers
**"if we read out the prediction at THIS layer using the network's OWN trained
output head, what would it already say?"** — the model's running, self-decoded
belief at each depth, fitting **zero** parameters.

The program builds a small **constant-width** softmax classifier
(`6 -> FC10+ReLU x4 -> FC4 -> SoftMax`) on the same synthetic 4-class
**XOR-quadrant + concentric-rings** task used by the LinearProbeReport example,
then prints `TNNet.LogitLensReport(NN, Probes)` on the **same UNLABELLED** probe
batch for three configurations:

1. **RUN 1**: a freshly-initialised network (no training). The untrained body
   carries no usable signal, so the lens confidence stays near `1/NumClasses`
   with high, flat entropy — the model has no early "belief" to decode.
2. **RUN 2**: the same architecture after a short training run. The lens
   confidence rises and the lens entropy falls steadily toward the head as the
   readout commits to an answer.
3. **RUN 3**: forcing `HeadStartIdx` to the **last** layer => a single-layer
   head; the lens degenerates to the trivial "everything resolves at the last
   layer" profile (a built-in sanity case).

The classifier **body is kept at a constant width** so every hidden layer feeds
the head at the same size and is therefore lens-compatible; the raw input layer
(width `6 != 10`) shows up explicitly as a **SKIPPED** layer — the honest
width-compatibility constraint of the classic lens.

## How the lens works

The report locates the trailing **readout head** (default `HeadStartIdx` = the
last trainable layer plus its activation/softmax tail). For every earlier layer
`L` whose flat activation is shape-compatible with the head's expected input, it
**splices** that layer's activation into the head's input slot and recomputes
**only** the head layers to obtain a per-layer **lens distribution** `p_L` (the
softmax-family head output is read as an already-normalised probability vector).
No probe is fitted — the lens **reuses the model's own trained head**.

## What it reports

Per compatible layer:

- **(a)** the **agreement** rate `mean_x[argmax(p_L) == argmax(p_final)]` as an
  ASCII bar chart across depth;
- **(b)** the **crystallization depth** — the shallowest layer after which the
  lens argmax matches the final argmax and never flips again — as a per-batch
  mean **and** a 10-bin per-sample histogram;
- **(c)** the per-layer mean top-1 **confidence** and lens **entropy** (the
  readout sharpens with depth);
- **(d)** the per-layer **`KL(p_L || p_final)`** curve (a monotone decrease means
  the residual stream is incrementally refining toward the final answer — the
  headline picture).

Width-incompatible layers are listed explicitly as **SKIPPED**.

## Built-in correctness checks (printed as PASS/FAIL)

- Applying the lens **at the head input** (no substitution) reproduces
  `p_final` **exactly** — agreement `1.0`, KL `0`.
- A **single-layer head** degenerates to the trivial "everything resolves at the
  last layer" profile (RUN 3).

## Distinct from the neighbours

- `LinearProbeReport` **fits** a fresh ridge probe per layer; the logit lens
  fits **nothing** and reuses the model's **own** trained head.
- `ActivationPatchingReport` does **causal** cross-input activation swaps.
- `FeatureSeparabilityReport` measures cluster **geometry** with no readout.

Pure forward-only — `NN.Compute` plus a per-head-layer recompute; weights are
never stepped, there is no backward pass, and the net's live state is restored
on exit.

## Build & run

```
cd examples/LogitLens
lazbuild LogitLens.lpi
../../bin/x86_64-linux/bin/LogitLens
```

Pure CPU, no dataset download, total runtime well under a minute.
