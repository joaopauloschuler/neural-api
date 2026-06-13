# Constrained (Structured) Decoding

Demonstrates the `TNNetTokenConstraint` hook from `neural/neuraldecode.pas`: a
caller-supplied "allowed next tokens" filter that the streamed generation loop
(`GenerateTokensStreamed` / `GenerateStringStreamed`, and `DecodeGreedy`)
applies to the post-softmax probability row **after** the repetition penalties
and **before** the sampler — zeroing every disallowed token and renormalizing
the survivors. Passing `nil` keeps generation bit-for-bit unchanged.

The same tiny **untrained** char-level network is decoded three ways:

1. **Free** (top-p sampling): character soup — the net is untrained.
2. **JSON mode** (`TNNetJSONConstraint`): a character-level JSON pushdown
   automaton (`TNNetJSONStateMachine`) tracks the brace/bracket stack and the
   object/array/string/number/literal context. A token is allowed exactly when
   feeding its characters through a clone of the automaton accepts all of them
   (multi-character BPE tokens validate transitively), and EOS only becomes
   legal once a complete top-level value stands. Even an untrained model can
   therefore **only emit valid JSON** — every sample is checked back through a
   fresh automaton and parses.
3. **Forced sequence** (`TNNetForcedSequenceConstraint`): multiple-choice
   answering — generation is forced down one of the candidates
   `yes` / `no` / `maybe` (a trie over candidate token sequences), characters
   the free model would never line up.

A small hand-set bias toward structural JSON characters keeps the sampled
values short (an untrained uniform model closes a string with probability
~1/96 per step); it is a stand-in for a trained model's preferences and plays
no part in correctness — validity comes from the automaton alone.

Typical output:

```
=== FREE decoding (untrained net, top-p 0.9) ===
  sample 1: "bvjF:B<9xx M4/Cz 0"
=== JSON-constrained decoding (same net, same sampler) ===
  sample 2: {}
     -> legal JSON prefix: yes; complete value: TRUE
=== Forced-sequence decoding (multiple choice yes/no/maybe) ===
  the untrained net was forced to answer: "maybe"
```

Documented JSON-grammar deviations (stricter than the spec, never looser) and
the all-tokens-masked fallback policy are in the `neuraldecode` unit header.
Runs in seconds on CPU.
