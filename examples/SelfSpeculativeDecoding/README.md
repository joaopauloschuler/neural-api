# Self-Speculative Decoding (Multi-Token-Prediction heads as their own draft)

[`examples/SpeculativeDecoding`](../SpeculativeDecoding) needs **two** networks:
a small fast DRAFT proposes a block of tokens and the big TARGET verifies them
in one pass. This example drops the second net entirely. A model trained with
`TNNet.AddMultiTokenPrediction` (Gloeckle et al. 2024, *Better & Faster LLMs via
Multi-token Prediction*; deployed at scale by DeepSeek-V3) already emits, at
every position `t`, `NumFuture` parallel softmax heads forecasting the tokens at
`t+1, t+2, ..., t+NumFuture`. At inference, heads `1..NumFuture-1` **are the
draft** — the model speculates about its own future in the *same* forward pass
that commits the next token.

## The loop (one model forward per pass; commits 1..NumFuture tokens)

Each pass forwards the committed prefix **plus** the pending draft tokens.
Reading row `r`, head `h` gives the model's forecast for position `r+1+h`.

1. **Verify** the pending drafts left-to-right: draft `j` (at position `L+j`,
   proposed by head `j+1` last pass) is accepted iff it equals head-0's greedy
   argmax at row `L+j-1` — which is *exactly* the token plain greedy decoding
   would emit there, because every row left of the first mismatch sees only
   committed-or-accepted (i.e. greedy-identical) causal context.
2. On the **first mismatch**, head-0's argmax at that row is the *correct*
   greedy token, so commit it (the standard speculative-decoding "bonus": even
   a rejection yields one token) and discard the remaining drafts. If **all**
   drafts are accepted, head-0 at the last draft row yields a bonus token
   beyond the block.
3. **New drafts** for the next pass come from heads `1..NumFuture-1` at the
   same row that produced the last committed token — a row whose context is
   fully committed, so the proposals are well-defined.

**Why this is exact.** Every committed token is head-0's argmax at a row whose
causal context consists only of already-committed tokens; the verification pass
recomputes that argmax bit-for-bit identically to plain one-token-at-a-time
greedy decoding. Greedy self-speculative decoding therefore reproduces plain
greedy decoding **exactly** — the program asserts sequence equality on every
benchmark run and `Halt(1)`s on any divergence (the headline gate).

**Why it is faster.** Plain greedy spends one full forward per token; the
self-speculative loop commits `1 + #accepted drafts` tokens per forward. The
input window here is fixed-size, so every forward costs the same and the
wall-clock ratio tracks the forwards-per-token ratio directly.

## The toy

A tiny char-level corpus (a few sentences with repeated clauses, vocab 25), a
causal-attention trunk (`Embedding → SinPos → Q|K|V slab →
AddMultiHeadSelfAttention(causal) → pointwise MLP`), and
`AddMultiTokenPrediction(NumFuture=4)`: head 0 is the ordinary next-char head,
heads 1..3 forecast 2..4 chars ahead and double as the draft. The model is
heavily overfit **on purpose** — the accept-rate signal (and the speedup) is
the point, not generalization.

## Sample output (measured, 2-core CPU, ~75 s total)

```
GATE  Exactness: self-speculative greedy == plain greedy, per run
  run  0 : EXACT   forwards plain=48  spec=20
  ...                                              (all 12 runs EXACT)

ACCEPT RATES per head distance (head h forecasts t+1+h; h>=1 drafts)
  head  dist   verified   accepted   accept-rate
    1    +2        213        146        68.5%
    2    +3        146        116        79.5%
    3    +4        116        104        89.7%
  drafts discarded unverified (past first mismatch): 164

  tokens generated (truncated)      : 576
  plain forward passes              : 576
  speculative forward passes        : 225
  mean tokens committed per pass    : 2.63
  forward passes saved              : 60.9%

WALL CLOCK (4 reps x 12 runs x 48 tokens)
  plain greedy       :   8544.1 ms  (3.708 ms/token)
  self-speculative   :   4192.8 ms  (1.820 ms/token)
  wall-clock speedup : 2.04x   (forward-pass ratio 2.56x)

GATE : PASS  (all 12 speculative continuations identical to plain greedy)
```

Notes on reading the numbers:

* Head-2/3 accept rates are **conditional** — those drafts are only verified
  after all earlier drafts in the block were accepted, which selects for
  "easy" (memorized) stretches of the corpus, so the conditional rate *rises*
  with distance even though the unconditional forecast gets harder.
* The wall-clock speedup (2.04×) trails the forward-pass ratio (2.56×)
  slightly: per-pass bookkeeping (argmax scans, input refills) and OS timing
  noise add a small constant per pass. On a real model where the forward
  dominates, the two converge.
* This is the **greedy** (deterministic) variant, so exactness is bit-for-bit
  equality. The *sampling* variant with the accept/residual-resample rule —
  and the proof it preserves the target distribution — lives in
  [`examples/SpeculativeDecoding`](../SpeculativeDecoding).
* Forward-only; each pass recomputes the window from scratch. Composing this
  with the new SDPA KV-cache incremental-decode path
  ([`examples/IncrementalDecode`](../IncrementalDecode)) is a separate open
  task (the two are orthogonal: speculation cuts the *number* of passes, the
  cache flattens the *cost per pass*).

A deterministic smoke test of the accept/verify loop
(`TestSelfSpeculativeDecodeGreedyExactness`) lives in
`tests/TestNeuralNumerical.pas`: on an untrained fixed-seed MTP net (exactness
holds for *any* model), speculative greedy must equal plain greedy and must
not take more passes than plain takes forwards.

## Build & run

```
lazbuild examples/SelfSpeculativeDecoding/SelfSpeculativeDecoding.lpi
./bin/x86_64-linux/bin/SelfSpeculativeDecoding
```
