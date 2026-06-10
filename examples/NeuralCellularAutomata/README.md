# Growing Neural Cellular Automata (tiny, pure-CPU)

A from-scratch, dependency-free reproduction of Mordvintsev, Randazzo, Niklasson &
Levin (2020), **"Growing Neural Cellular Automata"**
([distill.pub/2020/growing-ca](https://distill.pub/2020/growing-ca/)), shrunk to a
`16×16` grid so it trains in about a minute on two CPU cores. It uses **no new layer
classes** — the whole model is composed from existing layers, with
`TNNetConvolutionSharedWeights` as the key enabler. Frames are printed as **ASCII**,
so there is no image-library dependency and no binary output.

## The model

The world is a `16×16` grid of cells. Each cell carries `Ch = 12` channels: the
first four are the **visible** state `(R, G, B, alpha)`, the remaining eight are
**hidden** scratch channels the rule may use freely. One CA **rule step** is a
shared-weight residual conv stack applied to every cell in parallel, in place:

```
perceive : learned 3×3 conv (padding 1, no bias)   -> PDim perception channels
1×1 ReLU : TNNetPointwiseConvReLU                   (Hid hidden units)
1×1 lin  : TNNetPointwiseConvLinear (no bias)       (Ch update channels)
update   : grid := clamp( grid + dgrid )            (residual + bounded leaky-ReLUL)
```

The rule is applied `T = 32` times. **Every step after the first reuses the SAME
weights** via `TNNetConvolutionSharedWeights`. This weight tying is the whole point:
without it each step would learn its own filters and the result would not be a single
"growth rule"; with it the trainable parameter count is **independent of `T`** (here
~4.3k weights regardless of how many steps are unrolled).

```pascal
PerL  := NN.AddLayer(TNNetConvolutionLinear.Create(cPDim, 3, 1, 1, 1));
ReluL := NN.AddLayer(TNNetPointwiseConvReLU.Create(cHid));
LinL  := NN.AddLayer(TNNetPointwiseConvLinear.Create(cCh, {SuppressBias=}1));
// ... step 1 residual + bound ...
for step := 2 to cSteps do
begin
  NN.AddLayerAfter(TNNetConvolutionSharedWeights.Create(PerL), Grid);  // reuse perceive
  NN.AddLayer(TNNetConvolutionSharedWeights.Create(ReluL));            // reuse 1×1 ReLU
  LinL := NN.AddLayer(TNNetConvolutionSharedWeights.Create(LinL));     // reuse 1×1 lin
  Grid := NN.AddLayer(TNNetSum.Create([Grid, LinL]));                  // residual update
  Grid := NN.AddLayer(TNNetReLUL.Create(-10, 10, 1));                  // bound state
end;
```

## Training (backprop-through-time over the unrolled rule)

Because the `T` steps are ordinary layers wired in sequence, `TNNet.Backpropagate`
already walks the whole chain end to end and **accumulates the shared rule's gradient
across all `T` applications** — that is exact BPTT through the recurrence, for free,
once the graph is built. We use the batch-update idiom: `SetBatchUpdate(True)` so that
per-sample backprop does not zero the shared `Delta`, then `ClearDeltas → Compute →
Backpropagate → clip → UpdateWeights`. The L2 loss compares the **final** grid's RGBA
to a fixed target glyph (a chunky letter **"A"**); the eight hidden channels get
`desired = current output`, so they receive no gradient pull and stay free scratch.

The seed is a **single live pixel** at the centre (alpha = 1). The update head is
**zero-initialised** so the CA begins as an identity map (the standard Growing-NCA
trick), and we flush that zero head into every shared layer's weight cache with one
no-op `ClearDeltas/UpdateWeights` before training so the very first forward pass sees
it. From there the net learns to **grow the target out of the seed**.

## Headline result

Trained for 600 iterations (~61 s on two cores), the L2 loss falls from ~0.9 to
**~0.002**, and the final grid is a clearly recognisable "A" grown from one seed pixel.
Rendering the alpha channel as ASCII over the unrolled steps shows the growth:

```
step 04:                step 16:                step 32 (final):
  (diffuse noise)         |    -%%%%@#:. . |       | .   %@%%@@   . |
                          |   =@@@@@@@%-:  |       |    %@@@@@@@ .  |
                          |  :@@#..  @@@:  |       |   @@%    @@@   |
                          |  @@@      %@%  |       |  @@@      @@%  |
                          |  @@@@%@@%%@@%  |       |  @@@@@@@@@@@@  |
                          |  @@@ :    #@@  |       |  @@@ .    @@@  |
                          |  @@%-.:. :@@@  |       |  @@@      @@@  |
```

(`@%#*+=-:.` is a density ramp on the alpha/alive channel; blank = below the
`alpha > 0.1` "alive" threshold.) Step 4 is undifferentiated activity, by step 16 the
"A" has emerged, and step 32 is a clean glyph.

## What fit the CPU/memory budget (and what did not)

Honest notes, in the spirit of the other examples:

* **Full BPTT through ALL `T = 32` unrolled shared steps fit comfortably** — it is
  stable and fast (~61 s for 600 iterations on two cores, well under the five-minute
  budget, with no memory blow-up). We did **not** need truncated BPTT or a shorter `T`.
* **Stability required three guards, all standard for NCA.** A 32-deep residual
  recurrence with a tied weight overflows to `NaN` within a single update otherwise:
  1. **zero-init the linear update head** (identity-map start);
  2. a **bounded leaky `TNNetReLUL(-10, 10)`** clamping the state after each step;
  3. **gradient-norm clipping** (`NormalizeMaxAbsoluteDelta`) with a modest learning
     rate.
  These were established with a small feasibility probe before scaling up.
* **Dropped from the paper, to keep the demo tiny and deterministic:** the *stochastic
  per-cell update mask* and the *sample-replacement pool*. The demo trains a single
  seed→target sample; the `alpha > 0.1` "alive" masking is approximated by the learned
  alpha channel at render time rather than gating the update.
* **Regeneration after damage (stretch goal): not included.** A persistence/pool
  training regime is what makes NCA robust to damage; reproducing that faithfully
  needs the pool and many more iterations, which would push past the budget, so it is
  noted here rather than half-implemented.

## Running

```
lazbuild NeuralCellularAutomata.lpi
stdbuf -oL -eL ../../bin/x86_64-linux/bin/NeuralCellularAutomata
```

(`stdbuf -oL` keeps the per-iteration logs line-buffered so they are not lost.)
