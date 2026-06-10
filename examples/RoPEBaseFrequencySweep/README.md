# RoPEBaseFrequencySweep

Empirically interrogate the "magic 10000" baked into almost every RoPE
implementation. Train the **same** tiny single-head causal attention model
**once per base frequency** on a position-sensitive next-token task, then
print a results table so a reader can see whether the cargo-culted constant
is actually optimal.

## What is a RoPE base frequency?

RoPE (Rotary Position Embedding) injects absolute position by **rotating**
consecutive channel pairs of the embedding by an angle that grows linearly
with the token's position. For channel-pair index `i` (`0 .. d/2-1`) the
rotation frequency is

```
theta_i = base^(-2*i / d)
```

so the first pair rotates almost a full turn per token (high frequency) and
the last pair barely turns over the whole sequence (low frequency). The
attention dot product between a query at position `m` and a key at position
`n` then depends only on the **relative** offset `m - n`, which is exactly
the signal a self-attention layer needs to reason about distance.

The single scalar `base` controls how fast the per-pair frequencies decay:

- a **small** base (e.g. `1e2`) packs every pair into high frequencies: even
  the slow channels turn quickly, so far-apart positions alias onto similar
  rotations and long-range distance is hard to read off, but nearby offsets
  are resolved very sharply.
- a **large** base (e.g. `1e5`) stretches the low-frequency tail flat: the
  slow channels barely move across the sequence, giving smooth long-range
  position signal but coarser short-range resolution.

The constant `base = 10000` comes straight from the RoPE / original
Transformer sinusoidal paper and has been copied into countless
reimplementations verbatim. It is an empirical choice, not a derived
constant.

In the library the rotation is provided by `TNNetRotaryEmbedding`. The base
is the constructor argument: `TNNetRotaryEmbedding.Create(Base)` (the default
`TNNetRotaryEmbedding.Create()` keeps the standard `10000.0`). The layer
requires an **even** `Depth` and `SizeY = 1`. This example sweeps `base` over
`{1e2, 1e3, 1e4, 1e5}`.

## What the sweep tests

Whether `10000` is special on a task that genuinely rewards short-range
positional resolution. A flat task cannot separate the bases, so the task is
chosen to make position matter:

- **copy-from-k-steps-back**: the target at position `i` is the input token
  at position `i - cOffset` (a fixed begin token where `i < cOffset`). With
  `cOffset = 3` over a length-24 sequence, the model must use the position
  encoding to pick out the key exactly three steps back. Self-attention is
  permutation-invariant over the keys, so without a usable relative-offset
  signal the task is unsolvable; with RoPE the relative rotation between
  query `i` and key `i-3` is a fixed offset, and how sharply that offset
  reads out of the dot product depends on the frequency spread set by `base`.

Vocab 8, sequence length 24, offset 3, embedding dim 32 (even, required by
RoPE), single head. Every arm shares the RNG seed, steps, learning rate and
data, so the only thing that varies is the scalar passed to
`TNNetRotaryEmbedding.Create`.

## The shared stack

```
TNNetInput(SeqLen, 1, 1)                  # token IDs along X
  -> TNNetEmbedding(Vocab, d_model)       # learned token vectors
  -> TNNetRotaryEmbedding.Create(Base)    # <-- THE SWEPT KNOB
  -> single-head CAUSAL attention (same wiring as PositionEncodingBakeoff):
        Q | K | V via three TNNetSplitChannels on a packed projection
        ValueT = TransposeXD(V)
        scores = DotProducts(Q, K) / sqrt(d_k)   # (key, 1, query)
        reshape -> (key, query, 1)
        -> TNNetMaskedFill                        # causal upper-triangle
        reshape -> (key, 1, query) -> ReLUL -> softmax (depth axis)
        -> DotProducts(ValueT, W)                 # weighted sum of V
  -> TNNetPointwiseConvLinear(Vocab)      # per-position logits
  -> TNNetPointwiseSoftMax(1)             # softmax across depth
```

## Build & run

```
cd examples/RoPEBaseFrequencySweep
lazbuild RoPEBaseFrequencySweep.lpi --build-mode=Default
../../bin/x86_64-linux/bin/RoPEBaseFrequencySweep
```

Or compile directly with fpc (point `-Fu` at the LazUtils `lib` dir that
holds `utf8process.ppu`, exactly as `tests/RunAll.sh` discovers it):

```
fpc -B -Mobjfpc -Sh -O2 -Fu../../neural -Fu<lazutils-lib-dir> RoPEBaseFrequencySweep.lpr
```

Pure CPU, no dataset download. All four arms combined finish in well under a
minute.

## Sample output

Actual run on a single CPU thread:

```
========================================================================
RESULTS (lower cross-entropy / higher accuracy is better):
========================================================================
     base      train-CE      val-CE      val-acc
  --------     --------     --------     -------
       100     0.00063      0.00059      100.0%
      1000     0.00069      0.00063      100.0%
     10000     0.00077      0.00071      100.0%
    100000     0.00082      0.00076      100.0%

========================================================================
Best base by validation CE: 100  (val-CE=0.00059).
The cargo-culted 10000 is NOT the winner on this task -- the base is empirical, not sacred.
Total runtime for all 4 arms: 48.7s.
```

## Reading the result

Every arm **solves** the task (100% validation accuracy) — with only three
positions to look back and a generous training budget, all four bases learn a
usable relative-offset signal. The interesting part is the loss, which is
**monotone in base**: the smaller the base, the lower the final train and
validation cross-entropy. `base = 100` is the cleanest fit and the standard
`10000` is beaten by `100`, `1000` (and ties roughly with `100000` on the
wrong side). The differences are small but consistent across train CE and
val CE, and the same ordering shows up early in training (at step 40 the
larger-base arms have already broken away from the initial plateau while
`base = 100` lags, but it overtakes by convergence).

The intuition matches the frequency story above. This task only needs to
resolve a **short** offset of three positions, so the high-frequency channels
do all the work and the flat low-frequency tail that a large base buys is
wasted capacity — a smaller base, which spends more channels on sharp
short-range resolution, fits marginally better.

The takeaway is not "always use 100". It is that the RoPE base is a
**tunable inductive-bias knob, not a sacred constant**. The optimal value
depends on the range of offsets the task actually rewards: a short-range
retrieval like this one prefers a smaller base; long-context models — the
setting `10000` (and the even larger bases used in modern long-context
fine-tunes) was tuned for — need the flat low-frequency tail to keep distant
positions distinguishable. Whenever you can afford a short sweep, treating
`10000` as a default rather than a law is worthwhile — that is exactly the
empirical check this example performs.
