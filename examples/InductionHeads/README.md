# InductionHeads

A pure-CPU, single-threaded reproduction of the headline result of
**Olsson et al. 2022, "In-context Learning and Induction Heads."** A *tiny*
two-layer causal attention transformer spontaneously forms an **induction
head** that performs in-context copying.

## What it does

The phenomenon: on a sequence containing a repeated prefix

```
... [A][B] ... [A] -> ?
```

the model predicts `[B]` by finding the **earlier occurrence** of the current
token `[A]`, looking at the token that **followed** it, and copying that token
forward.

The toy task: draw a random prefix of length `L = 10` over a vocabulary of
`V = 12`, and **concatenate it with itself** to form a sequence of length
`2L = 20`. The model is trained on next-token prediction under a **strict
causal mask**. Every position `t` in the second half (`t >= L`) has token
`S[t] = S[t-L]`, and its true next token `S[t+1] = S[t+1-L]` is deterministically
recoverable *by induction*: find the earlier copy of `S[t]` at `t-L`, look one
position past it (`t-L+1`), and copy that token. The first half is unseen random
noise — nothing in the strictly-causal past determines `S[t+1]`, so first-half
next-token accuracy can only be near chance (`1/V`). The gap between
near-perfect second-half accuracy and chance first-half accuracy **is**
in-context learning.

## Why it differs from the other attention examples

- **`examples/AttentionCopyTask/`** — a *single non-causal* head doing a
  position-based identity copy of a **non-repeated** sequence
  (`output[i] = input[i]`). The winning strategy is "route query `i` to key `i`":
  trivial, one head, no composition. Induction is fundamentally harder.
- **`examples/CausalMaskSanity/`** — demonstrates *why* a causal mask matters
  (an unmasked head cheats by reading the answer that sits ahead of it), but it
  uses a single head and a fixed arithmetic recurrence, not content-based
  prefix matching.
- **`examples/AttentionEntropyReport/`** and other single-net diagnostic
  reports — measure attention *entropy*; they do not train a composed two-layer
  circuit nor demonstrate in-context learning.

Induction requires three ingredients those examples lack:

1. a **causal mask** (query `i` sees only keys `<= i`) so the model cannot peek
   at the answer;
2. **repeated random sequences**, so position is useless and the only winning
   strategy is content-based prefix matching;
3. **two-layer composition**: a layer-1 "previous-token head" that writes into
   each position information about the token *before* it, feeding a layer-2
   "prefix-matching / induction head" that matches the current token against
   earlier positions and copies the following token.

## Architecture (Option A)

This example uses **Option A**: two hand-wired single-head causal
`TNNetScaledDotProductAttention` blocks, each with a residual connection, so
each block's `.AttentionWeights` map can be read back directly.

```
Input(2L)                                   token ids on X
 -> Embedding(V, d_model, EncodeZero=1)
 -> SinusoidalPositionalEmbedding           parameter-free positions
 == BLOCK 1 (previous-token head) =========================================
 -> PointwiseConvLinear(3*d_k)              per-token Q|K|V  (NOT FullConnect!)
 -> ScaledDotProductAttention(d_k, Causal)  layer-1 attention
 -> PointwiseConvLinear(d_model)            project head output back to d_model
 -> Sum([resid_in, block1_out])             residual
 == BLOCK 2 (induction / prefix-matching head) ============================
 -> PointwiseConvLinear(3*d_k)
 -> ScaledDotProductAttention(d_k, Causal)  layer-2 attention  <- READ THIS MAP
 -> PointwiseConvLinear(d_model)
 -> Sum([resid_in, block2_out])             residual
 == readout ===============================================================
 -> PointwiseConvLinear(V)                  per-position vocab logits
 -> PointwiseSoftMax(1)                     softmax across depth
```

Dimensions: `V=12`, `L=10` (`SeqLen=20`), `d_model=24`, `d_k=24`, 240 epochs of
batch 48, per-sample SGD (`lr=0.005`, momentum 0.9). Per-token projections
**must** be `TNNetPointwiseConvLinear`; `TNNetFullConnect` would flatten/mix the
whole sequence and destroy the per-position structure.

## How to run

```
cd examples/InductionHeads
fpc -O3 -Mobjfpc -Sc -Sh -veiq -Fu../../neural -Fu../../neural/pas-core-math InductionHeads.lpr
./InductionHeads
```

Trains and self-checks in well under a minute on a single core (~17 s observed).

## The previous-token head (layer-1 attention heatmap)

The two-head *composition* only works if layer 1 actually behaves like a
previous-token head, so the example now reads layer 1's `.AttentionWeights`
too and renders it. Rows are queries `i`, columns are keys `j`:

```
        key: 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9
  q 0 tok 7 : @
  q 1 tok 9 : = +
  q 2 tok11 : . - +
  q 3 tok 8 :   . - +
  q 4 tok 1 :     . - +
  q 5 tok10 :       . - +
  q 6 tok 3 :         . - =
  q 7 tok 6 :           . - =
  q 8 tok 0 :             . - +
  q 9 tok11 :               . - +
  q10 tok 7 :                 . - +
  q11 tok 9 :                   : = :
  q12 tok11 :                   . : : .
  q13 tok 8 : .   .           . . . . . .
  ...
```

**Honest reading.** This trained layer-1 head is a *local-recency window*
dominated by self-attention: each query `i` keeps the brightest cell on itself
(key `i`, the `+`/`=` on the diagonal) and spreads the rest over the nearest
earlier tokens. So the **raw** mass on the previous token `i-1` is only modest
(~`0.32`). What is unambiguous is that among the *strictly-earlier* tokens
(`j < i`), the immediately-previous token `i-1` is the clear winner — it is the
**argmax of the past on ~100% of first-half rows** and carries **~70% of the
past mass** (the `-` cell that sits just left of the diagonal). That
backward-looking "what came right before me" signal is exactly what the
layer-2 induction head reads back, so the example asserts on those two
past-relative measures rather than pretending the head is a razor-sharp `t-1`
stripe (it is not — see the budget notes). In the deep repeated tail
(`q13`..`q19`) the layer-1 map diffuses; the composition does not need those
rows sharp, so the readout is taken over the first-half rows where the window
is clean.

## The induction stripe (layer-2 attention heatmap)

A glyph-shaded ASCII heatmap of the layer-2 head for one probe. Rows are queries
`i`, columns are keys `j`; the mask makes the upper triangle empty. The
**induction stripe** is the bright diagonal in the lower block: a second-half
query at row `i` puts almost all its mass on key `i-L+1` — one position past the
earlier copy of its own token.

```
        key: 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9
  q 0 tok 3 : @
  q 1 tok 6 :   @
  q 2 tok 7 :     @
  q 3 tok 1 :       @
  q 4 tok 3 :         @
  q 5 tok 0 :           @
  q 6 tok 0 :             @
  q 7 tok11 :               @
  q 8 tok10 :                 @
  q 9 tok 2 : @
  q10 tok 3 :   @
  q11 tok 6 :   . %
  q12 tok 7 :       @
  q13 tok 1 :         @
  q14 tok 3 :           @
  q15 tok 0 :             @
  q16 tok 0 :               @
  q17 tok11 :                 @
  q18 tok10 :                   @
  q19 tok 2 :                     @
```

In the second half (`q10`..`q19`) every query attends to the key one past the
earlier occurrence of its token — that diagonal stripe is the induction head.

## Sample output (results + gates)

```
Chance accuracy (1/vocab)                :   8.33%
First-half  next-token accuracy (unseen) :   8.47%   (near chance)
Second-half next-token accuracy (repeat) :  99.68%   (in-context copy)

First-half  mean cross-entropy           : 2.6417
Second-half mean cross-entropy           : 0.0780
In-context learning score (CE2 - CE1)    : -2.5637   (want << 0)

Prev-token (layer-1) t-1 stripe mass     : 0.3200
Uniform-causal baseline mass (layer-1)   : 0.2143
Prev-token t-1 share of strictly-past    : 0.7003
Prev-token t-1 argmax-of-past fraction   : 1.0000
Prefix-match (induction) stripe mass     : 0.9783
Uniform-causal baseline mass             : 0.0688

GATE 1 (in-context copy)   : PASS  (2nd-half 99.7% >> 1st-half 8.5%)
GATE 2 (ICL score < 0)     : PASS  (CE2-CE1 = -2.5637)
GATE 3 (induction stripe)  : PASS  (stripe 0.9783 > 2x uniform 0.0688)
GATE 4 (prev-token head)   : PASS  (t-1 is argmax-of-past on 100.0% of rows, 70% of past mass)
```

## What the built-in checks assert

1. **In-context copy (mandatory, `Halt(1)` on failure):** second-half
   next-token accuracy `> 0.8` **and** clearly above first-half accuracy
   (`> first-half + 0.4`). The model copies the repeated half but cannot predict
   the unseen first half above chance — it is *copying, not memorizing*.
2. **In-context learning score (mandatory, `Halt(1)` on failure):** mean CE at
   the late repeated positions minus mean CE at the early positions is strongly
   negative (`< 0`). Seeing a token earlier in the same sequence makes
   predicting its successor far cheaper — the definition of in-context learning.
3. **Induction stripe (rendered + scored; soft gate):** mean attention mass that
   second-half queries place on the induction-target key `i-L+1`, compared to
   the uniform-causal baseline. Asserts `stripe > 2x baseline` and `stripe > 0.25`.
4. **Previous-token head (rendered + scored; soft gate):** the matching *layer-1*
   readout. Over the first-half query rows, the immediately-previous token `i-1`
   is the **argmax over the strictly-earlier keys** (self excluded) and carries
   the bulk of the past mass. Asserts `t-1 argmax-of-past fraction > 0.9`
   **and** `t-1 share of strictly-past mass > 0.5`. Together with gate 3 this
   pins **both** halves of the two-head composition the paper identifies — the
   layer-1 previous-token head feeding the layer-2 prefix-matching head — not
   just the layer-2 head.

## Budget notes (honesty)

- Everything fit the budget comfortably: full training **and** self-checks
  finish in ~17 s on one CPU core, far under the 5-minute limit, so no claim had
  to be reduced. The headline near-100% second-half accuracy is the *full*
  result, not a reduced one.
- Gates 3 and 4 (the attention-readout checks) are implemented as **soft** gates
  per the task spec: both heatmaps are always rendered and the scores always
  printed, but only gates 1 and 2 hard-`Halt(1)`. In practice both pass strongly.
- **The layer-1 head is NOT a razor-sharp `t-1` stripe**, and the example does
  not pretend it is. The trained head is a self-dominated local-recency window:
  the raw mass it puts on the previous token `i-1` is only ~`0.32` (vs a ~`0.21`
  uniform baseline). The honest, robust signal is *relative to the past*: among
  the strictly-earlier keys, `i-1` is the argmax on ~100% of first-half rows and
  holds ~70% of the past mass. Gate 4 asserts those two past-relative numbers
  (`> 0.9` and `> 0.5`), comfortably below the observed `1.00`/`0.70`, instead of
  asserting a clean `t-1`-mass threshold the model does not actually achieve.
  This is the layer-1 analogue of the layer-2 stripe, measured the way the data
  actually behaves.
- A single repeat (prefix concatenated once) is enough to make the second half
  fully determined; more repeats were unnecessary and would only enlarge the
  sequence and slow training.
```
