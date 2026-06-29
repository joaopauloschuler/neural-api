# TitansMemory — test-time neural long-term memory for long-context recall

Demonstrates **`TNNetTitansMemory`**, the test-time neural long-term memory
layer (Behrouz et al. 2024, *"Titans: Learning to Memorize at Test Time"*, the
Memory-as-Context leaf-layer variant), and contrasts it head-to-head with a
param-matched fixed-decay linear-attention baseline, **`TNNetRetention`**, on a
**long-context associative recall** task.

## The layer

A plain decay / linear-attention recurrence accumulates `k(x)·v` with a
**constant** decay it cannot adapt, so a stored association unavoidably bleeds
toward zero over a long irrelevant span. Titans instead carries an inner memory
`M_t` — a small MLP whose weights are **gradient-descended at inference** on the
per-token associative loss `‖M_t(k_t) − v_t‖²`, with two mechanisms the
fixed-decay baseline lacks:

```
(a) MOMENTUM / "surprise"   S_t = eta*S_{t-1} - theta*grad
        a surprising STORE token keeps writing for several steps (deep encoding)
(b) data-dependent FORGET   M_t = (1 - alpha_t)*M_{t-1} + S_t
        alpha_t small for bland tokens, so stored memories persist across the gap
```

So the forget gate can leave a stored association untouched while the distractor
tokens stream past, exactly where a constant-rate accumulator decays its memories
into noise.

## The task — long-context associative recall

Each sequence has three phases over a `(SeqLen, 1, cInDim)` volume, where each
input token is `[ key one-hot | value vector | store flag | query flag ]`:

```
STORE      : cNumPairs distinct (key -> value) WRITE tokens up front (store_flag=1)
DISTRACTOR : a long span of cDistractor random noise tokens (both flags 0)
QUERY      : re-present one stored key (query_flag=1); target = ITS stored value
```

The long distractor span between the writes and the query is what makes this a
**long-context** problem: the stored associations must survive many irrelevant
intervening steps. The value vocabulary is a fixed bank of value vectors
(`InitValueBank`), and recall is scored both as **mean squared recall error** at
the query position and as **exact-recall accuracy** via nearest-neighbour decode
over that bank.

## The bake-off

Two arms share the **same I/O contract** and a **matched parameter budget**,
trained on the same recall stream for the same number of steps:

| arm | mixer | front-end → readout |
|-----|-------|---------------------|
| Titans | `TNNetTitansMemory(cHidden)` | `PointwiseConvLinear(cModelDim)` → memory → `PointwiseConvLinear(cValueDim)` |
| Retention | `TNNetRetention(cModelDim, 0.9, LearnGamma=false)` | `PointwiseConvLinear(3*cModelDim)` (Q\|K\|V) → retention → `PointwiseConvLinear(cValueDim)` |

Both arms start from `TNNetInput.Create(cSeqLen, 1, cInDim)`. The Retention arm
uses a **plain FIXED, non-learnable decay** (gamma `0.9`), so it is the "plain
linear-attention" baseline. Per-sample SGD (`lr=0.01`, momentum `0.9`),
`cTrainSteps` steps per arm, then evaluation over `cEvalSeqs` held-out
sequences. Pure CPU, tiny dims (`model_dim=12`, hidden=16), finishes in a few
minutes.

## Running

```
cd examples/TitansMemory
fpc -O3 -Mobjfpc -Sh -Fu../../neural TitansMemory.lpr
./TitansMemory
```

(or open `TitansMemory.lpi` in Lazarus). Pure CPU.

## Expected output

The program prints its configuration and per-arm weight counts, trains both
arms, then reports recall MSE and exact-recall accuracy for each:

```
=== Titans neural long-term memory: long-context recall ===
keys=5  value_dim=4  pairs/seq=4  distractor=24  seq_len=29  model_dim=12

Titans    params = ...
Retention params = ...

training both arms on the SAME recall stream (6000 steps each)...

eval over 400 held-out long-context recall sequences:
  Titans (neural memory) : recall MSE = ...   exact-recall acc = ...%
  Retention (fixed decay): recall MSE = ...   exact-recall acc = ...%
```

The headline self-check: when the Titans arm reaches lower recall MSE and at
least matching accuracy, it prints

```
OK: the test-time neural memory recalls stored values better than the fixed-decay baseline across the long distractor span.
```

otherwise it prints a `WARNING:` that Titans did not beat the Retention
baseline. Exact numbers are seed-dependent; the **contrast** is the point.

Coded by Claude (AI).
