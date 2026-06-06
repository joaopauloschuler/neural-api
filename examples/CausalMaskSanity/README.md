# CausalMaskSanity — why a next-token attention model needs a causal mask

A controlled, self-checking demonstration that a self-attention next-token
model trained **without** a causal mask *cheats*: it peeks at **future** tokens
to drive its **train** loss to near-zero, yet is useless for honest
left-to-right generation. The identical model trained **with** a strictly
causal mask is forced to predict token *t* using only tokens ≤ *t*, so its
train loss is honestly higher but it actually **generalizes** to autoregressive
generation.

## The idea

A decoder-style language model predicts, at every position *t*, the next token
`tok[t+1]`. During training the whole sequence is on the input at once
("teacher forcing"). If attention is allowed to look **anywhere**, the model
discovers a trivial shortcut: the correct answer for position *t* is literally
sitting one slot ahead at input position *t+1*. It just routes query *t* to key
*t+1* and copies it — one hop, no reasoning. Train loss collapses to ≈ 0. But
at **generation** time `tok[t+1]` does not exist yet, so the shortcut
evaporates. This is exactly the bug the **causal mask** exists to prevent.

## The task (rigged so peeking is easier than reasoning)

A sequence over a small vocabulary `V` follows a 2nd-order recurrence:

```
tok[0], tok[1] ~ uniform(0..V-1)
tok[t]         = (tok[t-1] + 2*tok[t-2] + 1) mod V        (t >= 2)
```

The true next token at position *t* depends **only** on positions ≤ *t*
(specifically `tok[t]` and `tok[t-1]`), so a causal model **can** learn it — but
it must *combine two past tokens*, a genuinely non-trivial computation. The
non-causal model never bothers: it copies the literal answer one slot ahead.
That asymmetry is the whole point — cheating is *cheaper* than reasoning here,
so the unmasked arm reaches a strictly lower train loss while learning nothing
that generalizes.

## The two arms (identical except the mask)

```
Input(SeqLen) -> Embedding(V, d) -> SinusoidalPositionalEmbedding
  -> PointwiseConvLinear(3*d_k)               { pack Q | K | V along depth }
  -> ScaledDotProductAttention(d_k, CAUSAL?)  { the ONLY difference }
  -> PointwiseConvLinear(V)                    { per-position vocab logits }
  -> PointwiseSoftMax(1)                       { softmax across depth }
```

Same architecture, seed, learning rate, epochs, batch and data; only the
`CausalMask` flag on `TNNetScaledDotProductAttention` differs.

### About the mask

`TNNetScaledDotProductAttention`'s `CausalMask` flag applies, *before* the
row-softmax, exactly the strictly-upper-triangular additive fill
`score[j>i] := -1e9` that the standalone layers **`TNNetMaskedFill`** /
**`TNNetTriangularCausalMask`** apply to an explicit *(key, query)* score
matrix — same upper triangle, same `-1e9` constant. We use the built-in flag
because SDPA computes its scores internally (there is no exposed score tensor to
slot a separate `TNNetMaskedFill` in front of); the masking *operation* is
identical.

## What it proves (three self-checking gates, `Halt(1)` on failure)

1. **Cheating.** The unmasked arm reaches a strictly **lower** train
   cross-entropy than the masked arm (it exploits the future-token shortcut).
2. **Generalization.** Under **true** autoregressive generation — every future
   position is blanked so the model literally cannot peek — the masked arm's
   next-token accuracy is strictly **higher** than the unmasked arm's. The cheat
   does not survive honest generation.
3. **Evidence (where the cheat lives).** Averaged over probe sequences, the
   unmasked head places substantial attention mass on **future** keys (`j > i`),
   while the masked head places ≈ 0 there — read straight off the read-only
   `AttentionWeights` map.

## Build & run

With Lazarus:

```
lazbuild CausalMaskSanity.lpi
./CausalMaskSanity      # binary lands under ../../bin/<cpu>-<os>/bin/
```

Or directly with the Free Pascal compiler (pure CPU, single-threaded,
deterministic, runs in well under two minutes):

```
fpc -B -Fu../../neural -Mobjfpc -Sh -O2 CausalMaskSanity.lpr && ./CausalMaskSanity
```

## Sample output

```
CausalMaskSanity: masked vs unmasked next-token self-attention.
Task: 2nd-order recurrence  tok[t] = (tok[t-1]+2*tok[t-2]+1) mod 6  (vocab 6, seqlen 7).
...
======================================================================
RESULTS
======================================================================
Final train cross-entropy (lower = "fits" the train set better):
  UNMASKED : 0.000245   <-- cheats, drives loss lower
  MASKED   : 0.924347   <-- honestly higher

Teacher-forced next-token accuracy (FULL sequence visible -- the
metric the unmasked arm games by reading tok[t+1] off the input):
  UNMASKED : 100.00%
  MASKED   :  69.75%

TRUE autoregressive generation accuracy (future positions blanked,
model sees only the past) -- the honest test:
  UNMASKED :  26.33%   <-- the cheat evaporates
  MASKED   :  69.75%   <-- generalizes to generation

Mean attention mass on FUTURE keys (j>i) over supervised query rows:
  UNMASKED : 0.8407   <-- the head literally looks ahead
  MASKED   : 0.0000   <-- masked out (~0)
======================================================================
GATE: PASS
```

The headline contradiction is the lesson: the unmasked model is "perfect"
(100%) when it can read the future, but craters to 26% the moment it has to
generate honestly — while the masked model behaves the same (70%) in both
settings because it never depended on the future in the first place.
