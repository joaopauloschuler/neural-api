# Mixture of Depths

A tiny pure-CPU demonstration of the **Mixture-of-Depths** conditional-compute
idea from Raposo et al. 2024,
[*Mixture-of-Depths: Dynamically Allocating Compute in Transformer-Based Language
Models*](https://arxiv.org/abs/2404.02258), built with the
`TNNet.AddMixtureOfDepths` builder.

## What Mixture-of-Depths does

MoD routes along the **sequence axis**. A per-token learned scalar **router**
decides *whether* each sequence position is **processed** by a wrapped block or
**skips** it via the residual / identity path, under a fixed per-block
**capacity**:

- only the top-`Capacity` tokens (by router score) enter the block;
- the remaining `SeqLen - Capacity` tokens bypass it unchanged.

Because the capacity is *fixed*, every tensor shape stays **static** (the paper's
key trick versus dynamic-shape sparse routing) while the per-block FLOPs drop by
`(SeqLen - Capacity) / SeqLen`. The router stays on the gradient path because its
sigmoid weight **multiplies** the block output for the chosen tokens (without
that multiply the discrete top-k choice would have no gradient).

```
TNNetInput(8, 1, 1)                       # 8 token ids along X (scalar, normalised)
  -> TNNetPointwiseConvLinear(d_model=16)  # linear lift (NO nonlinearity)
  -> AddMixtureOfDepths(                    # <-- the MoD block
        [ TNNetPointwiseConvReLU(24)        #   wrapped shape-preserving FFN
          TNNetPointwiseConvLinear(16) ],   #   (the net's ONLY nonlinearity)
        Capacity )                          #   top-Capacity of 8 tokens processed
  -> TNNetPointwiseConvLinear(vocab=5)      # per-token logits (linear)
  -> TNNetPointwiseSoftMax(1)               # softmax across depth
```

Internally `AddMixtureOfDepths` wires:

1. a **per-token router logit** via `TNNetPointwiseConvLinear(1)` →
   `(SeqLen,1,1)` (a per-token projection **must** be pointwise, not
   `FullConnect`, which would flatten the sequence and zero the input gradient);
2. `TNNetSigmoid` → a per-token weight in `(0,1)`;
3. **top-`Capacity` selection** by transposing X↔Depth to `(1,1,SeqLen)`,
   applying `TNNetTopK(Capacity)`, and transposing back — non-selected tokens
   become `0`, selected tokens keep their sigmoid weight;
4. broadcasting the masked weight across `d_model` and **cell-multiplying** it
   into the wrapped block output (`TNNetCellMulByCell`);
5. the residual sum `y = x + weight * Block(x)`, leaving **skipped positions at
   their input value**.

With `Capacity = SeqLen` the top-k keeps every token and the wrapper reduces
exactly to a per-token scalar-gated residual block (no token is ever skipped) —
this degenerate equality is pinned by a unit test.

## The synthetic task

A position-dependent next-token target over an 8-long sequence and a 5-char
vocabulary, generated in-code (no dataset download):

```
target[t] = S[t]                  for EVEN t   # identity copy   (easy)
target[t] = (2*S[t] + 1) mod V    for ODD  t   # periodic remap  (hard)
```

The token id is fed as a **normalised scalar**, *not* a learned embedding. This
is deliberate: an embedding + linear read-out could memorise any per-token map
and would make the wrapped block redundant. With a scalar input, the **only
nonlinearity in the entire net is the ReLU inside the MoD block**, so:

- the **even** (identity) positions are solved by the linear skip path alone;
- the **odd** (non-monotone periodic remap) positions can *only* be produced by
  tokens the router chooses to send through the block.

So the block is a genuinely scarce resource, and shrinking its capacity must cost
accuracy on exactly the positions that need it — the MoD trade-off made visible.

## The sweep

The same architecture is trained with `Capacity ∈ {SeqLen, SeqLen/2, SeqLen/4}`
(= `{8, 4, 2}`). Every arm reseeds `RandSeed` to the same value before building
and before training, so data and initialisation are identical across arms — only
the capacity differs. The program then:

- charts **final cross-entropy vs the processed-token fraction**
  (`Capacity/SeqLen`), the FLOP/accuracy trade; and
- prints a **router-by-position histogram**: how often each token position is
  among the processed (selected) tokens, an interpretability view of where the
  fixed compute budget is spent.

## Build & run

```
lazbuild examples/MixtureOfDepths/MixtureOfDepths.lpi
bin/x86_64-linux/bin/MixtureOfDepths
```

Pure CPU, single-threaded (manual `Compute`/`Backpropagate`), no external data,
finishes in a few seconds. The compiled binary lands in `bin/x86_64-linux/bin/`
(shared with the other examples), not inside this directory. The run is
non-interactive (no trailing `ReadLn`).

## What it shows

- A **loss-vs-fraction** chart, one row per capacity: processed fraction, final
  cross-entropy, accuracy, parameter count, and a text bar. Loss should rise (and
  accuracy fall) as the processed fraction drops.
- A **router-by-position histogram**: with `Capacity = SeqLen` every position is
  processed (100%); as capacity drops the router must triage which positions keep
  the block.
- Four NaN/Inf-guarded sanity checks printed as `PASS`/`WARN`/`FAIL`:
  1. all arms produced a finite final loss,
  2. the full-capacity arm beats the uniform baseline `ln(vocab)`,
  3. full capacity reaches the lowest loss of the sweep, and
  4. loss is non-decreasing as the processed fraction falls (the trade-off).

Per-arm numbers are **seed-dependent**; the graceful-degradation *trend* and the
position histogram are the point, not exact values.

## Sample output

Real output from a recent run (`RandSeed = 424242`):

```
Training Capacity=8 (100% processed) ... done. final_CE=0.6019 acc=0.592 1.57s
Training Capacity=4 (50% processed) ... done. final_CE=0.7002 acc=0.534 1.49s
Training Capacity=2 (25% processed) ... done. final_CE=0.8233 acc=0.439 1.83s

=== Final cross-entropy vs processed-token fraction ===
capacity  frac_proc  final_CE      acc   params   loss
8             1.000    0.6019    0.592      880   #############################
4             0.500    0.7002    0.534      880   ##################################
2             0.250    0.8233    0.439      880   ########################################

=== Router selection by POSITION (how often each token is processed) ===
position      0    1    2    3    4    5    6    7   (H=hard/odd, .=easy/even)
kind          .    H    .    H    .    H    .    H
cap=8       100  100  100  100  100  100  100  100  % of probes processed
cap=4        59   60   53   48   48   48   43   42  % of probes processed
cap=2        34   34   29   27   23   22   15   15  % of probes processed

=== Sanity checks ===
[PASS] all arms produced a finite (no NaN/Inf) final loss.
[PASS] full-capacity arm beats the uniform baseline (1.609).
[PASS] full capacity reaches the lowest loss of the sweep.
[PASS] loss is non-decreasing as the processed fraction falls (the trade).
```

The headline trade-off is clean: as the processed fraction falls `1.0 → 0.5 →
0.25`, final cross-entropy rises monotonically `0.60 → 0.70 → 0.82` and accuracy
degrades gracefully `0.59 → 0.53 → 0.44` — exactly the Mixture-of-Depths selling
point of buying FLOPs with a controlled accuracy cost at a static, known shape.

The router-by-position histogram shows the fixed budget being spent and shrinking
with capacity. On this seed the router learned a mostly **positional** allocation
(earlier positions favoured) rather than a sharp hard-vs-easy (odd-vs-even)
content split — an honest, seed-dependent outcome: the demo reliably exposes the
*capacity trade-off*, while the precise content of the learned triage varies with
the seed.
