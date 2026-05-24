# EarlyExitNetwork

Self-contained **BranchyNet** (Teerapittayanon, McDanel & Kung 2016,
[*BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks*](https://arxiv.org/abs/1709.01686))
demo of **anytime / adaptive inference**: one trunk of stacked `FC+ReLU` blocks
carries an **auxiliary softmax classifier head** branching off after **each**
intermediate block as well as the final block. All heads are trained **jointly**
(deep supervision), then a **confidence-gated dynamic-compute policy** at
inference lets **easy** inputs leave at a shallow head while **hard** inputs run
the full depth — saving compute at essentially the full-depth accuracy.

## The difficulty-graded task

A synthetic 4-class 2-D problem mixing two regimes (seeded, reproducible):

- **EASY** samples — tight, well-separated Gaussian blobs at the four corners.
  One block decides them with near-certainty.
- **HARD** samples — points in a small central annulus whose class is a **finely
  striped angular sector** (8 wedges interleaved onto the 4 classes), plus light
  label noise. The boundary is curved and chopped into many pieces, so a shallow
  head cannot be confident about them.

## Architecture (K blocks, K heads)

```
Input(2)
  -> Block1 (FC+ReLU) --branch--> head1 = FC(4) -> SoftMax
  -> Block2 (FC+ReLU) --branch--> head2 = FC(4) -> SoftMax
  -> Block3 (FC+ReLU) --branch--> head3 = FC(4) -> SoftMax
  -> Block4 (FC+ReLU) --branch--> head4 = FC(4) -> SoftMax
  Concat([head1..head4]) -> single packed output of width K*NumClasses = 16
```

`NN.Compute(x)` yields **all K heads' probabilities at once**; head `h` occupies
output channels `[h*NumClasses .. (h+1)*NumClasses-1]`. The trunk is re-anchored
after each branch (a passthrough `TNNetIdentity` off the block) so the next block
continues the **shared trunk**, not the head — this is the standard
branch-off-an-intermediate-layer wiring (`AddLayerAfter`, as in
`examples/DomainAdversarial`).

## Joint training (deep supervision, manual loss loop)

The framework's automatic `Fit` seeds the gradient only at the **last** layer, so
for deep supervision we seed it ourselves — but cleanly, without hand-editing
`OutputError`:

- Each head ends in a `TNNetSoftMax`, so the concat output holds the per-head
  softmax **probabilities** `p`.
- We pass a **packed target** = the same one-hot label repeated once per head.
  The framework's `ComputeOutputErrorWith` then forms, per head, `(p - onehot)` —
  exactly the **softmax-cross-entropy gradient**.
- A **single** `Backpropagate` splits that packed error through the `Concat` into
  every branch and **accumulates into the shared trunk**. Minimising the summed
  gradient minimises the **sum of the heads' cross-entropies** (deep supervision).

**Manual-gradient gotcha:** multi-head accumulation needs `SetBatchUpdate(True)` —
the per-sample default zeroes `Neurons[].Delta` between samples and would lose the
accumulation. We use the batch idiom: `ClearDeltas` at batch start, accumulate
over the minibatch, `UpdateWeights`, repeat (mirrors `examples/GradientNoiseScale`).

## Inference: the confidence gate

For each test sample, `Compute` once, then walk heads **shallow → deep**; **exit**
at the first head whose softmax max-probability **exceeds** `tau` (the prediction
is that head's argmax). If no early head exceeds `tau`, fall through to the
deepest head. The exit depth (number of blocks executed) is the **compute proxy**
for the trade-off axis.

> The FLOPs helper `TNNet.CountFLOPsPerLayer` returns a **formatted string**
> report (printed once here for context), not a parseable number — so exit depth
> is used as the compute axis.

We sweep `tau` over `{0.0, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0}` and print, per `tau`,
the accuracy and the average exit depth **split by easy vs hard samples**, then
render an ASCII scatter of **accuracy (y) vs average exit depth (x)**.

## The two built-in invariants (asserted & printed)

1. **`tau = 1.0` ⇒ full-depth accuracy.** Because the gate uses strict
   "exceeds" and softmax never exceeds 1.0, no early head can fire at `tau = 1.0`,
   so every sample runs to the deepest head. The gated accuracy then equals the
   plain full-depth net's accuracy (read off the deepest head) **exactly**, and
   the average exit depth is **exactly K**.
2. **Average exit depth is monotone non-decreasing in `tau`.** Raising the
   confidence bar can only push samples deeper, never shallower.

Both print `PASS`.

## How to read the trade-off plot

The ASCII chart plots accuracy (y) against average exit depth (x). Points to the
**left** are cheap (most samples exited at the first block); points to the
**right** are full-depth. The headline result is in the sweep table's
`easy-depth` vs `hard-depth` columns: at every gating `tau` the **easy** samples
exit near depth 1 while the **hard** striped/margin samples climb toward depth K —
so average compute sits far below K at essentially the full-depth accuracy. That
is the BranchyNet anytime/adaptive-inference win: **spend depth only on the inputs
that need it.**

## Contrast with `examples/PredictionDepth`

`examples/PredictionDepth` is a **post-hoc, forward-only k-NN probe** on a
**fixed, single-head** network: it measures *at what layer the net makes up its
mind* about each example, without any extra trained parameters and **without
changing the compute**. It is a *diagnostic of difficulty*.

`EarlyExitNetwork` is the opposite direction: the early heads are **trained**
(jointly, deep supervision) and they **actually gate compute** at inference — easy
inputs are answered by a shallow head and the deeper blocks are never executed for
them. PredictionDepth *observes* where decisions happen; EarlyExitNetwork *acts*
on confidence to *save real work*.

## Build & run

```
cd examples/EarlyExitNetwork
lazbuild EarlyExitNetwork.lpi
../../bin/x86_64-linux/bin/EarlyExitNetwork
```

Pure CPU, no dataset download, deterministic (seeded). Total runtime ~10 s and a
few MB of RAM.
