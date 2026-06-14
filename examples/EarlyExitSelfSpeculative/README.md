# Early-exit / self-speculative decoding (LayerSkip / CALM)

Single-model self-speculative decoding where the model is its **own** draft
model — **no** second checkpoint and **no** separate prediction head. "Easy"
tokens are decoded from an **intermediate layer** through the model's **own LM
head**, falling back to full depth when the draft disagrees.

## The idea

For every decode step:

1. **Full forward** gives the mature distribution `p_final` and its argmax (the
   verifier).
2. **Early-exit draft** via the frozen-body **LogitLens splice** (the same
   "logits at layer k" idiom `TNNet.LogitLensReport` and `DecodeDoLa` use):
   snapshot the intermediate **exit layer**'s activation — already computed by
   the full forward — copy it into the LM-head input slot, and recompute **only
   the head sub-stack** to read `p_exit`, the early draft distribution.
3. **Static-gate accept/verify** (LayerSkip / CALM): if the early exit is
   confident (`max p_exit ≥ Confidence`) it drafts `argmax(p_exit)`; that draft
   is **accepted** iff it equals the full-depth argmax (**exact-greedy** verify,
   exactly as `examples/SpeculativeDecoding` does for a *separate* draft net).

The **emitted token is always the full-depth argmax**, so the accepted sequence
is **bit-identical** to plain greedy decoding. The early exit only changes the
accept/reject counters — i.e. how much tail-layer work a *cached* decoder could
skip — never the output.

## How this differs from the two existing examples

- `examples/SelfSpeculativeDecoding` drafts from **MTP prediction heads**
  (separate per-token prediction heads), not an intermediate-layer exit.
- `examples/EarlyExitNetwork` is a **BranchyNet classification** demo (auxiliary
  softmax heads per block, trained jointly).

Here the draft is the model's own **intermediate-layer readout** through the
single shared LM head — no extra parameters, no second net.

## Library routine

`neural/neuraldecode.pas`:

```pascal
function DecodeEarlyExitSelfSpeculative(NN: TNNet; const Prompt: string;
  MaxLen: integer; out Stats: TNNetEarlyExitStats;
  const StopStrings: array of string;
  ExitLayer: integer = -1; Confidence: TNeuralFloat = 0.0;
  HeadStartIdx: integer = -1): TNNetDecodeResult; overload;
```

- `ExitLayer` — intermediate exit layer (its `Output` is spliced into the head
  input); `-1` auto-picks the midpoint below the head input.
- `Confidence` — early-exit gate in `[0,1]`; draft only when
  `max p_exit ≥ Confidence`. `Confidence > 1` disables drafting (pure greedy,
  still bit-identical); `0` always drafts.
- `Stats` — `TNNetEarlyExitStats` (steps, draft proposals, accepted, rejected,
  acceptance rate).

A convenience overload drops the `Stats` out-param.

## What the program does (pure CPU, ~1 min)

1. builds a small constant-width char-level LM and trains it briefly on a
   deterministic corpus so the intermediate-layer readout becomes a usable
   draft;
2. decodes a fixed prompt with plain greedy AND with
   `DecodeEarlyExitSelfSpeculative`, and **asserts the outputs are identical**
   (`Halt(1)` on mismatch — this must never happen);
3. prints the early-exit accept/reject counters + acceptance rate, and a
   **tokens/sec** figure for both paths at matched output.

## Sample output

```
Prompt              : "abc"
Greedy continuation : "iabaiiaaacaaaaaa"
Self-spec continuation: "iabaiiaaacaaaaaa"

CORRECTNESS CHECK: PASS - self-speculative output is BIT-IDENTICAL to plain greedy.
Early-exit stats : steps=16  drafts=16  accepted=2  rejected=14  acceptance=12.5%

Timing 80 tokens per path...
  plain greedy     : ~1500 tokens/sec
  self-speculative : ~1330 tokens/sec
  acceptance rate this run: 12.5%
```

**On the tokens/sec figure:** v1 has **no cached tail-skip** yet, so the
self-speculative path here does the full forward **plus** a head-only splice and
is intentionally a touch *slower* per token. The headline is the **acceptance
rate**: with the cached tail-skip (the open follow-up) each accepted token would
let the verifier reuse the draft's prefix and **skip the tail layers**, turning
the acceptance rate directly into a speedup — at bit-identical output.

## Correctness guarantee (tested)

The accepted greedy sequence equals plain greedy `Compute`-loop decoding
**bit-for-bit**, because every emitted token is verified by the full model.
Regression tests in `tests/TestNeuralDecode.pas`:

- `TestEarlyExitMatchesGreedyBitIdentical` — output identical to `DecodeGreedy`
  with the draft consulted every step;
- `TestEarlyExitHighConfidenceMatchesGreedy` — `Confidence > 1` disables drafts
  yet stays bit-identical;
- `TestEarlyExitAcceptCountsAreConsistent` — `accepted + rejected == proposals`,
  proposals ≤ steps, and acceptance rate consistency.

## Open follow-up

Per-token-adaptive exit (choose the exit layer per step from a confidence/early-
exit classifier) plus the cached tail-skip that turns accepted drafts into saved
compute.

Coded by Claude (AI).
