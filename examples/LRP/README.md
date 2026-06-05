# LRP

Tiny example for `TNNet.LRPReport`, a **Layer-wise Relevance Propagation**
diagnostic (Bach et al. 2015, *On Pixel-Wise Explanations for Non-Linear
Classifier Decisions by Layer-Wise Relevance Propagation*).

Unlike `SaliencyReport` and `GradCAMReport` (which are **gradient** methods),
LRP is **not** a gradient method: it back-**distributes** the chosen output
logit's *relevance* through the net under a **conservation rule**, so the total
relevance is preserved at every layer boundary. Relevance is *redistributed*,
not differentiated.

Given a trained classifier and a single probe sample (a `TNNetVolume` input),
the report:

1. forwards the probe once (reusing the existing forward machinery — no forward
   activation is re-derived), picks the explained class `c = argmax(f(x))` (or
   `ForcedClass`), and seeds the last layer's relevance with `R_c = logit_c`;
2. walks the layer list from output toward input applying, on each
   linear/dense (`TNNetFullConnect` family) layer, the **epsilon-rule**

   ```
   R_i = sum_j (a_i * w_ij) / (sum_k a_k * w_kj + eps * sign(z_j)) * R_j
   ```

   where `a_i` is the i-th input activation, `w_ij` the weight from input `i` to
   output neuron `j`, `z_j` the pre-activation, and `eps` (default `1e-2`) a
   stabiliser whose sign follows `z_j`. ReLU / activation / Input / Identity
   layers pass relevance through 1:1.

It prints:

- **(a) the per-layer-boundary relevance-conservation residual**
  `|sum(R_in) - sum(R_out)|` — the headline LRP sanity check. The epsilon-rule
  trades exact conservation for numerical stability, so the residual is `O(eps)`
  and shrinks to zero as `eps -> 0`. A residual that stays bounded by `eps` is
  healthy; one that blows up signals a layer whose rule is undefined.
- **(b) the top-K most-relevant input positions** `(channel, x, y)` with their
  signed relevance;
- **(c) a per-channel ASCII relevance heatmap** over the input plane
  (`~10` buckets ``" .:-=+*#%@"``, brightest = highest `|R|`).

### Honest skipping

Layers whose backward relevance rule is **undefined** under the epsilon-rule
(attention, normalisation, softmax, spatial convolution, pooling, …) are
**skipped honestly**: relevance is passed through unchanged and the layer is
listed as `SKIPPED (no epsilon rule)` rather than faking a value.

Pure CPU; reuses the existing forward path. No optimiser step is applied, so the
inspected network's weights are left untouched (the live state is restored by a
final clean recompute before returning).

## What this demo shows

On a synthetic `6x6x1` two-class image task (class 0 = bright `2x2` blob in the
top-left; class 1 = bright blob in the bottom-right, plus background noise) it
trains a small **dense** classifier
(`FullConnectReLU -> FullConnectReLU -> FullConnectLinear -> SoftMax`). A dense
stack is used on purpose: the epsilon-rule has an exact closed form there, so
the conservation residual is meaningful and bounded by `eps`. It then runs the
report on:

1. a **clean, correctly-classified** class-0 probe — the top-relevant input
   positions cluster on the discriminative top-left blob, and the conservation
   residual stays `O(eps)`;
2. a **noisier** class-1 probe — relevance spreads out a little more.

The `SoftMax` head is reported as skipped/pass-through, demonstrating the honest
handling of layers without an epsilon-rule.

## Build & run

```
cd examples/LRP
lazbuild LRP.lpi
../../bin/x86_64-linux/bin/LRP
```

Pure CPU, no dataset download. Total runtime is well under a minute.
