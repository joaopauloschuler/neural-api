# HopfieldAssociativeMemory — one-shot associative recall

A **one-shot associative recall** demo built on `TNNetModernHopfield`, the
continuous (modern) Hopfield network of Ramsauer et al. 2020
([arXiv:2008.02217](https://arxiv.org/abs/2008.02217)). **No training** happens:
four binary 8×8 patterns are written **directly** into the layer's pattern bank,
and the layer recalls a clean memory from a corrupted query. It contrasts a
single retrieval step (K=1, ordinary softmax attention → blurry blend) with
iterated retrieval (K=3 → sharpens toward the exact nearest stored memory),
showing the energy-based associative-memory behavior.

## What it uses

- `TNNet` with `TNNetInput` + `AddModernHopfieldRetrieval(NPAT=4, KSteps, BETA=0.08)`,
  one net with K=1 and one with K=3.
- Patterns are stored as {-1,+1} values written straight into
  `L.Neurons[0].Weights[...]` (the learnable pattern bank) — no `Fit`.
- Recall via `NN.Compute`; quality measured by Hamming distance (sign-thresholded)
  and continuous L2 distance to the true pattern.

## Running

No arguments, no dataset, no download. The patterns are hardcoded and the seed is
fixed (`RandSeed := 42`). Pure CPU, well under a minute.

```
cd examples/HopfieldAssociativeMemory
# build with lazbuild HopfieldAssociativeMemory.lpi (or fpc), then:
./HopfieldAssociativeMemory
```

For each of the 4 stored patterns it builds a corrupted query (bottom half masked
+ 6 random pixel flips), prints the query and the K=1 and K=3 retrievals as ASCII
grids with their Hamming/L2 distances, and finishes with a summary table plus a
verdict on whether the iterated K=3 retrieval beat the single-pass K=1 attention.

Coded by Claude (AI).
