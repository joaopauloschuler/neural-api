# ActivationPatching

Tiny example for `TNNet.ActivationPatchingReport`, the forward-only **causal**
activation-patching / causal-tracing diagnostic (Vig et al. 2020, Meng et al.
ROME 2022, Wang et al. IOI 2022) that answers **"which layer's activations carry
the information that decides this prediction?"** — measured by intervention, not
correlation.

The program builds a small **branched** softmax classifier — a main ReLU
feature-extractor branch plus a **raw-input skip** fused by a `Concat` just
before the head:

```
Input(6) --> FC12+ReLU --> FC12+ReLU --> FC12+ReLU  (main branch)
      \                                        \
       \------------------ skip ----------- Concat --> FC8+ReLU --> FC2 --> SoftMax
```

on a synthetic 2-class **XOR-of-signs** task: the class is decided *entirely* by
whether `sign(x0)` and `sign(x1)` agree (the rest of the input coordinates are
pure noise distractors). XOR is **not** linearly separable, so the deciding
feature must be **computed** — and by construction it is computed inside the
main branch. After a short training run we take a `(CleanInput, CorruptInput)`
pair that the trained net maps to **different** argmax classes (the corrupt
input is the clean one with the sign of `x1` flipped, which flips the XOR label)
and run the causal trace.

## What it reports

`TNNet.ActivationPatchingReport(NN, CleanInput, CorruptInput [, TargetIdx])`:

1. runs a **clean** forward pass and caches every layer's `Output`;
2. runs a **corrupt** forward pass;
3. for each layer `L` in turn, restores **only** layer `L`'s cached clean
   activation into the corrupt run (`FLayers[L].Output.CopyNoChecks(...)`),
   recomputes layers `L+1..last`, and reads off the **recovery**

   ```
   r_L = (logit_c(patch_L) - logit_c(corrupt)) / (logit_c(clean) - logit_c(corrupt))
   ```

   where `c` is the clean argmax class (`TargetIdx`, default = clean argmax).
   `r_L ≈ 1` ⇒ patching layer `L` alone restores the clean decision (the causal
   information lives there); `r_L ≈ 0` ⇒ that layer carries nothing causal for
   this contrast.

It prints:

- **(a)** a per-layer `r_L` ASCII bar chart across depth — the causal-trace
  curve (the headline plot), with a per-layer argmax and un-normalised
  `delta_logit = logit_c(patch_L) - logit_c(corrupt)` column (so a near-zero
  normalisation denominator is visible);
- **(b)** the **argmax-flip layer** (shallowest `L` whose patch flips the
  corrupt argmax back to `c`);
- **(c)** the **peak-recovery layer** and its `r_L`;
- **(d)** an **early / late / distributed** localisation verdict;
- **(e)** the two built-in **faithfulness checks**: `r_0 == 1` (patching the
  input reconstructs the full clean run) and `r_last == 1` (the last layer's
  `Output` *is* the logits).

### Ground-truth localisation check

Patching a **single main-branch layer** leaves the **corrupt** raw input still
flowing on the skip path, so those layers recover only a little (`r_L ≈ 0`).
Recovery **jumps to `~1` at the `Concat` fusion layer**, where the patched clean
activation overwrites the whole fused representation (skip included) and the
head sees an entirely clean input. That known-by-construction jump is the
localisation this example demonstrates. Patching the input (`L=0`) fixes **both**
the branch and the skip at once, so `r_0 == 1` exactly.

### Denominator-collapse path

The example also calls the report with `CorruptInput := CleanInput`. The clean
and corrupt runs then agree on class `c`, the recovery denominator collapses,
and the report emits a clear **WARNING** instead of dividing by zero (the
un-normalised delta-logits are still reported).

## Forward-only

No backward pass and no weight steps are run inside the report. The **only**
mutation is the transient activation overwrite, reverted by a final clean
recompute before the report returns — the trained weights are never touched.

This is **distinct from** `SaliencyReport` (input-space gradient attribution),
`LayerSensitivityReport` (random *weight* jitter — never swaps real activations
between two inputs) and `LinearProbeReport` / `FeatureSeparabilityReport` (what
is *present* in a layer, not what the rest of the net causally *uses*).

## Build & run

```
cd examples/ActivationPatching
lazbuild ActivationPatching.lpi
../../bin/x86_64-linux/bin/ActivationPatching
```

Pure CPU, no dataset download, total runtime well under a minute.
