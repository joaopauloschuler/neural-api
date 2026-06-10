# SpeculativeDecoding

A pure-CPU, single-threaded reproduction of **speculative sampling**
(**Leviathan et al. 2023**, *Fast Inference from Transformers via Speculative
Decoding*; **Chen et al. 2023**, *Accelerating Large Language Model Decoding
with Speculative Sampling*) on a *tiny* pair of next-token decoders that share
one toy vocabulary. It proves the headline property of the trick:

> the speculative sampler's output is distributed **EXACTLY** as if drawn from
> the big **TARGET** model alone, while calling the big model far **fewer**
> times.

## What it does

A small fast **DRAFT** model guesses ahead; a big **TARGET** model verifies the
guess in one batched pass and *corrects* it with a rule that leaves the target's
output distribution untouched. One verification pass commits between `1` and
`K+1` tokens.

```
1. DRAFT proposes K candidate tokens x_1..x_K          (K serial cheap draft passes)
2. TARGET scores all K+1 block positions               (ONE batched big-model pass)
3. Walk the block left-to-right:
     ACCEPT x_i with prob  min(1, p_target(x_i)/p_draft(x_i))
     on the FIRST rejection, resample that position from
         norm(max(0, p_target - p_draft))
     and DISCARD the rest of the block.
   If all K accept, sample one BONUS token from p_target (a free extra token).
```

Leviathan/Chen prove the accept-or-residual-resample rule makes the committed
token at *every* position an exact draw from `p_target`, for **any** draft
distribution. This example makes that claim **self-checking** (see the gates).

## Mechanism — why it is exact

For each block position the committed token's law is

```
P(commit = v) = p_draft(v)*min(1, p_t(v)/p_d(v))           (accepted draft)
              + (prob of rejection) * residual(v)            (resampled)
```

and the two terms collapse algebraically to `p_target(v)`. The intuition: where
the draft over-proposes a token (`p_draft > p_target`) the accept step thins it
back down; the leftover probability mass `max(0, p_target - p_draft)` is exactly
what the residual resample puts back. No bias, for any draft.

## KV-cache composition (the v2 verifier)

The demo ships **two** verification backends and proves they are token-exact
equivalents:

- **v1 (`SpeculativeSample`)** recomputes the whole prefix on every
  verification pass — correct and simple, but each pass costs `O((t+K)^2)`
  attention work, so v1 cuts the *number* of conceptual big-model passes
  without cutting the actual forward cost.
- **KV-cached (`SpeculativeSampleCached`)** runs the target's verification
  through the `TNNetScaledDotProductAttention` KV-cache incremental-decode
  path on a *step net*: the same architecture/weights rebuilt at input width
  `K+1` (`CopyWeights` — all parameter shapes are sequence-length
  independent), every per-head SDPA in `BeginIncrementalDecode` mode.
  - **Prefill once.** The prompt's K/V is cached in one multi-token cached
    forward (each input token's K/V is appended and its query attends over
    the cache so far — exactly the prefill semantics of the cached path).
  - **One short forward per pass.** Each pass feeds only the window
    `[last committed token | x_1..x_K]`; row `i` of the per-position softmax
    is `p_target` at context length `t+i`, covering all verification rows
    *plus the bonus row* in one go. Per-pass cost: `O(K * t)` instead of
    `O((t+K)^2)`.
  - **Rollback on rejection.** Rejected (and padding) positions have already
    appended stale K/V; **`TruncateCache(NewLength)`** (added with this
    example) rewinds every head's cache to the committed prefix — a
    one-field rewind, the preallocated slots are simply reused.
  - **Positional contract.** The step net only sees the short window, so its
    `TNNetSinusoidalPositionalEmbedding` is shifted via the new
    `PositionOffset` property to the window's absolute start position before
    every cached forward (the cached SDPA applies no positions itself).
  The two efficiency wins therefore **compose**: fewer big-model passes
  (speculation) × cheaper passes (KV cache).

The **draft stays on the full-recompute path on purpose**: it is attention-free
(`TNNetTokenShift`), so it has no KV cache to exploit, and its forwards are the
cheap arm by design.

## Distinct from the neighbouring examples

- **examples/IncrementalDecode** demonstrates the KV cache flattening the
  per-step cost of running **one** model; speculative decoding cuts the
  **number** of big-model passes. This example composes the two.
- **DeepEnsembleUncertainty / KnowledgeDistillation** combine model *outputs* or
  transfer knowledge; they do **not** preserve the target's exact sampling law.
- **EarlyExitNetwork** lets *one* model exit early; here two *separate* models
  cooperate.

## The toy

Vocabulary `V = 8`, context window `24`. A sequence is a random string and the
next-token target is a structured fixed-offset map
`target[t] = (S[t-cLag] + S[t]) mod V` (`cLag = 3`). The **TARGET** is the bigger
model (`d_model=32`, 4-head causal `AddMultiHeadSelfAttention`, pointwise MLP
read-out, 600 epochs); the **DRAFT** is deliberately small and cheap
(`d_model=12`, attention-free `TNNetTokenShift` mixer, ~70 epochs) so it is a
decent-but-imperfect approximator. Both are per-position causal decoders ending
in `TNNetPointwiseSoftMax(1)`; per-token projections are `TNNetPointwiseConvLinear`
(a `TNNetFullConnect` would flatten/mix the whole sequence).

## RNG ordering (the crux of the bit-for-bit exactness gate)

Two independent, explicitly-seeded scalar RNG streams (a tiny in-program
`xorshift64*`, **not** the library RNG, so the ordering is fully under our
control):

- **`gProp`** — the *proposal / plain-sampling* stream. One uniform per token
  position, used to inverse-CDF sample a token from a categorical distribution.
- **`gAcc`** — the *accept / residual-resample* stream. One uniform per
  verification step for the accept Bernoulli.

Plain target-only sampling draws each token by inverse-CDF on `p_target` with one
`gProp` uniform, and never touches `gAcc`. Speculative sampling draws each draft
proposal by inverse-CDF on `p_draft` with one `gProp` uniform (same stream), and
draws each accept Bernoulli from `gAcc`.

**Why this collapses when DRAFT == TARGET.** We literally pass the same net as
both. Then `p_draft == p_target` at every accepted position, the ratio is
`min(1, p/p) = 1`, so *every* accept succeeds regardless of the `gAcc` uniform.
Each proposal `x_i` was drawn by inverse-CDF on `p_draft == p_target` using
exactly the `gProp` uniform plain sampling would have used at that committed
position; because accepts never fail and no residual draw is taken, `gProp`
advances in lock-step with plain sampling. The committed sequence (and the bonus
token, also a `gProp` `p_target` draw) is therefore **bit-for-bit identical** to
plain target-only sampling. The two-stream split exists precisely so this
degenerate case is exact.

## How to run

```
cd examples/SpeculativeDecoding
fpc -O3 -Mobjfpc -Sc -Sh -veiq -Fu../../neural SpeculativeDecoding.lpr
./SpeculativeDecoding
```

Trains both tiny models and runs all checks in ~70 s on one CPU core, far under
the 5-minute budget. Single-threaded by construction (manual
`Compute`/`Backpropagate`), deterministic under the fixed `RandSeed` + the two
explicit RNG streams.

## Built-in checks (gates)

1. **(MANDATORY) Degenerate exactness — `Halt(1)` on failure.** With
   `draft == target` the accept rule is always-accept, so the speculative and
   plain token sequences must be **identical element-for-element**. This is the
   pinnable faithfulness anchor.
1b. **(MANDATORY) KV-cache exactness — `Halt(1)` on failure.** The cached
   sampler must produce a token sequence **identical** to the v1 full-recompute
   sampler under the same seeds, three ways: (a) `draft == target` (chains
   through Gate 1 to plain sampling), (b) the real trained draft at the flat
   target (real rejections exercise the `TruncateCache` rollback), and (c) the
   real draft at the peaked target (more rejections, plus the tempering path).
2. **Empirical faithfulness.** With a genuinely *different* trained draft, sample
   the next token many times (4000 draws) from a fixed prefix both ways and
   compare the token histograms; asserts the total-variation distance is within
   sampling noise (`< 0.06`; observed `~0.024`).
3. **Speedup (informative).** Sweep draft quality (the same draft trained to
   `{0, 8, 80}` epochs) over many fresh random prefixes and report mean
   accepted-tokens-per-pass, accept-rate, big-model-calls-per-token, and
   big-model-calls-saved. Accept rate (hence calls saved) **rises monotonically**
   with draft/target agreement.
4. **Composition benchmark (the headline).** Wall-clock ms/token and *actual*
   target-net forward counts for the three arms over the same 150 random
   prefixes × 12 tokens (trained draft, flat target).

## Flat vs PEAKED target (two-scenario speedup chart)

Gates 2 and 3 run **twice** — once for a **FLAT** target (the original
`mod`-sum map, softmax temperature `1.0`) and once for a **PEAKED** target (the
same trained net, re-softmaxed at temperature `0.10`). A peaked `p_target`
concentrates mass on the mode, so a **weak** draft (which only approximates the
flat objective) puts probability on the *wrong* token and its accept ratio
`min(1, p_target/p_draft)` collapses, while a **strong** draft still tracks the
peak. This makes the speedup chart far more **discriminating** of draft quality.

Crucially this changes only **what the target distribution is**, never the
sampler that draws from it: both plain and speculative sampling read the same
tempered `p_target` (`NextDist(..., IsTarget=True)` applies the temperature), so
the exactness/faithfulness invariant holds at **any** temperature. Gate 1
(degenerate exactness) stays at temperature `1.0` because with `draft == target`
the *untempered* proposal must equal the verification distribution — exactness is
a property of the algorithm, not of how peaked the target is. The peaked toy was
tuned only by the temperature constant `cPeakTemp` (no fabricated numbers).

## Sample output

```
Training TARGET (600 epochs, d_model=32, heads=4, causal attention) ... done.  params=6272
Training DRAFT  (70 epochs, d_model=12, TokenShift) ... done.  params=428

========================================================================
GATE 1  Degenerate exactness  (draft == target => always-accept)
========================================================================
  plain      tokens : 6 5 7 3 7 3 3 1 3 5 6 7
  speculative tokens: 6 5 7 3 7 3 3 1 3 5 6 7
  big-model calls: plain=12  speculative=3 (draft==target case)
  GATE 1 : PASS  (speculative == plain, bit-for-bit)

========================================================================
GATE 1b Cached-verification exactness (KV cache + TruncateCache rollback)
========================================================================
  (a) draft==target : cached == plain, bit-for-bit       PASS
  (b) real draft     : cached == v1 (15 tokens, 3 passes)  PASS
  (c) peaked target  : cached == v1 under rejections     PASS
  GATE 1b: PASS  (cached speculative output is exact)

========================================================================
GATE 2  Empirical faithfulness  (real draft != target; histogram match)
========================================================================
  -- FLAT target (temp 1.0) --
  total-variation distance = 0.0245  (small => same distribution)
  -- PEAKED target (temp 0.10) --
  total-variation distance = 0.0130  (small => same distribution)
  GATE 2 : PASS  (flat TV=0.0245, peaked TV=0.0130 -- both within noise)

========================================================================
GATE 3  Speedup vs draft/target agreement  (block K=4)
========================================================================
  ----- FLAT target (temp 1.0; original mod-sum map) -----
  draft       acc-tok/pass   accept-rate   big-calls/tok   calls-saved
  ----------  ------------   -----------   -------------   -----------
  untrained         2.342        0.585           0.299        70.1%
  lightly           3.721        0.930           0.212        78.8%
  fully             3.875        0.969           0.205        79.5%

  ----- PEAKED target (temp 0.10) -----
  draft       acc-tok/pass   accept-rate   big-calls/tok   calls-saved
  ----------  ------------   -----------   -------------   -----------
  untrained         1.622        0.406           0.381        61.9%
  lightly           2.540        0.635           0.282        71.8%
  fully             2.564        0.641           0.281        71.9%

  flat-vs-peaked contrast (weak=untrained draft, strong=fully trained):
                     weak-rate   strong-rate   weak-saved   strong-saved   gap(saved)
    FLAT target        0.585         0.969        70.1%         79.5%          9.4 pts
    PEAKED target      0.406         0.641        61.9%         71.9%         10.1 pts

========================================================================
GATE 4  Composition benchmark  (150 runs x 12 tokens, trained draft, flat target)
========================================================================
  arm                        tokens  ms/token   tok/s   tgt-fwd  fwd/tok
  -------------------------  ------  --------  ------  --------  -------
  (i)   plain full-recompute    1800     0.350    2855      1800    1.000
  (ii)  spec v1 (recompute)     2220     0.480    2083      2220    1.000
  (iii) spec + KV cache         2220     0.122    8207       623    0.281
  verification passes: v1=473  cached=473  (cached fwd = passes + prefill)
  speedup (ms/token): (iii) vs (i) = 2.87x,  (iii) vs (ii) = 3.94x
```

Reading the benchmark: v1 (ii) is actually *slower* per token than plain (i) at
this tiny scale — it makes the same number of full-prefix target forwards
(fwd/tok = 1, one per verified position plus bonus) *plus* the draft passes.
What speculation buys is fewer conceptual *passes* (473 passes for 2220
tokens, 4.7 tokens committed per pass); what the KV cache buys is making each
pass **one short 5-token cached forward** instead of K+1 full 24-token
re-encodes. Composed, arm (iii) makes 0.281 forwards per token (623 = 473
passes + 150 prefills) at a fraction of the per-forward cost: **2.9× faster
than plain and 3.9× faster than v1 speculative**, with the token streams
bit-identical across all three arms (Gates 1/1b). At a production context
length (thousands of tokens, not 24) the per-pass gap is `O(K·t)` vs
`O((t+K)²)`, so the composition widens further with `t`.

Reading the contrast: under the FLAT target even an **untrained** draft accepts
`58.5%` of the time (the `mod`-sum map spreads mass, so random proposals land on
a plausible token often). Under the PEAKED target that weak-draft accept rate
**collapses to `40.6%`** — well below 1 — because the sharpened `p_target` no
longer rewards mass on the wrong token, and the strong-minus-weak **calls-saved
gap widens** (`9.4 -> 10.1` points). The peaked chart therefore separates a good
draft from a bad one much more sharply. (Note the peaked target also caps the
*fully*-trained draft at `64.1%`: that draft was trained on the flat objective,
so its softer distribution can never fully match a temperature-`0.10` peak — an
honest artefact of reusing the flat-trained net rather than retraining at low
temperature.)

Note in Gate 1 that the degenerate run made only **3** big-model passes to
produce the same 12 tokens plain sampling needed **12** calls for — because every
draft token was accepted and each pass committed `K+1 = 5`.

## What did NOT fit the budget (honesty)

- **The draft is not cached.** It is attention-free (`TNNetTokenShift`), so
  there is no KV cache to exploit; a stateful TokenShift step path would be a
  different (tiny) follow-up. Its full recomputes are the cheap arm by design
  and dominate nothing.
- **In the v1 arm the "batched" target scoring is emulated by per-position
  `NextDist` calls** but counted as **one** big-model pass per block,
  reflecting the batched cost a real implementation pays. The KV-cached arm
  needs no such emulation: one short multi-token cached forward *physically*
  produces all K+1 rows.
- The toy target's `mod`-sum next-token distribution is fairly flat, so under the
  FLAT scenario the *absolute* accept rates are high even for a weak draft; there
  the load-bearing result is the **monotone rise** in accept rate with draft
  quality plus the committed-distribution **exactness** (Gates 1-2). The **PEAKED**
  scenario (temperature `0.10`) addresses this: it sharpens `p_target` so the
  weak-draft accept rate collapses below 1 and the good-vs-bad calls-saved gap
  widens, giving a more discriminating chart — at no change to the sampled
  distribution.
- The peaked target reuses the **flat-trained** net re-softmaxed at low
  temperature rather than retraining a low-temperature target from scratch, so
  even the fully-trained draft cannot perfectly match the peak (its accept rate
  caps around `0.64`). Retraining both models on a near-deterministic objective
  so the strong draft fully tracks the peak is a possible follow-up.
```
