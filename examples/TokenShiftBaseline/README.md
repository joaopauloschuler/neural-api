# TokenShift Baseline

A head-to-head bake-off of two **token-mixing primitives** on the *same*
tiny char-level sequence task. The two arms share an identical embedding
front-end, an identical per-token MLP read-out, the same data and the
same weight initialisation; the **only** thing that differs is *how
information moves between sequence positions*.

Arms:

- **Arm 1 — TokenShift:** `Embedding -> TNNetTokenShift -> MLP -> softmax`.
  `TNNetTokenShift` is the RWKV-style, **attention-free** time-mixing
  primitive
  `y[t,c] = mix[c]*x[t,c] + (1-mix[c])*x[t-1,c]`.
  It is O(n), parameter-cheap (one `Depth`-long `mix` vector) and mixes
  information from **exactly one step back (t-1)**. It can capture
  first-order / lag-1 context but *structurally cannot* reach an
  arbitrary earlier position.

- **Arm 2 — Attention:** `Embedding -> [QKV slab] ->
  AddMultiHeadSelfAttention(causal) -> MLP -> softmax`. Self-attention is
  O(n²) and far heavier in parameters, but with the parameter-free
  sinusoidal positions every query can route to *any* earlier key, so it
  can recover a long-range, fixed-offset dependency that TokenShift
  cannot. `AddMultiHeadSelfAttention` consumes a packed Q|K|V slab of
  depth `3*d_model` and out-projects back to `d_model`, so the arm adds a
  `TNNetPointwiseConvLinear(3*d_model)` in front of it — exactly as
  `AddTransformerEncoderBlock` does.

```
TNNetInput(16, 1, 1)                      # 16 token ids along X
  -> TNNetEmbedding(vocab=6, d_model=24)
  -> TNNetSinusoidalPositionalEmbedding   # parameter-free positions
  -> [ MIXER ]                            # TokenShift  |  QKV -> MHSA(causal)
  -> TNNetPointwiseConvLinear(32)         # per-token MLP hidden (d_ff)
  -> TNNetReLU
  -> TNNetPointwiseConvLinear(6)          # per-token vocab logits
  -> TNNetPointwiseSoftMax(1)             # softmax across depth
```

`TNNetPointwiseConvLinear` (not `TNNetFullConnect`) is used throughout
the sequence body: a pointwise (1×1) projection is applied independently
at every position, preserving the `(SeqLen, 1, d_model)` sequence axis
that both `TNNetTokenShift` (which requires `SizeY = 1`) and the
attention builder expect.

## The synthetic task

A fixed-offset **copy** task with two regimes. Each sample is a random
length-16 string over a 6-char vocabulary; the target is a copy of an
earlier source token:

```
target[t] = S[t-1]      for t <  4    # lag-1 copy   (S[-1] := 0)
target[t] = S[t-4]      for t >= 4    # long-range fixed-offset copy
```

Both regimes are deterministic copies of a **single** source token — the
only difference is *how far back* that source sits.

- The **lag-1** region is exactly what TokenShift's t-1 mixing exposes,
  so TokenShift solves it and clears the uniform baseline `ln(vocab)`.
- The **lag-4** region sits too far back for a one-step shift to reach,
  but causal self-attention can route a query to a fixed earlier offset
  (via the sinusoidal positions), so it copies that token too.

Everything is generated in-code (no dataset download). `RandSeed` is
reset to the same value before each arm's data generation and before
building/initialising its net, so both arms see identical inputs and
identical embedding init; only the mixing layer differs.

## Build & run

```
lazbuild examples/TokenShiftBaseline/TokenShiftBaseline.lpi
bin/x86_64-linux/bin/TokenShiftBaseline
```

Pure CPU, single-threaded (manual `Compute`/`Backpropagate`), no external
data, finishes in well under a minute (~8 s on a typical machine). The
compiled binary lands in `bin/x86_64-linux/bin/` (shared with the other
examples), not inside this directory. The run is non-interactive (no
trailing `ReadLn`).

## What it shows

- A comparison table with one row per arm: parameter count, initial
  cross-entropy, final cross-entropy, overall next-token accuracy,
  **acc_lag1** (early positions `t < 4`, source `S[t-1]` — TokenShift's
  turf), **acc_long** (positions `t >= 4`, source `S[t-4]` — attention's
  turf), and wall-clock seconds.
- Three NaN/Inf-guarded sanity checks printed as `PASS`/`FAIL`/`WARN`:
  1. both arms produced a finite final loss,
  2. both arms beat the uniform-guess baseline `ln(vocab) = 1.79`, and
  3. (informative) attention clearly leads on the long-range region.

On this task the two arms split exactly along the mechanistic line they
were designed to expose: **both** clear the uniform baseline (TokenShift
on the strength of the lag-1 region alone), but **attention wins
decisively** because it *also* solves the long-range copy that a fixed
one-step shift simply cannot see. TokenShift nails `acc_lag1 = 1.000`
yet sits at chance (`~1/6 = 0.167`) on `acc_long`; attention reaches
`1.000` on both. The per-arm numbers are **seed-dependent**, but the
qualitative split (TokenShift blind beyond t-1, attention not) is the
structural point and is not seed-dependent.

## Sample output

Real output from a recent run (`RandSeed = 424242`):

```
TokenShiftBaseline: attention-free TokenShift vs causal self-attention.
Task: fixed-offset copy. target[t]=S[t-1] for t<4, else S[t-4]   (seqlen 16, vocab 6).
A lag-1 region (TokenShift can reach the source) and a lag-4 region
(only attention can reach the source). Both arms can beat the uniform
baseline; attention should win by also solving the long-range region.
Shared front-end: Embedding(24) -> SinPos -> [MIXER] -> PWConv(32) -> ReLU -> PWConv(6) -> softmax.
Uniform-guess baseline loss = ln(vocab) = 1.7918, chance acc = 0.167.
Same data, same init, same read-out; only the token-mixing layer changes.

Training Arm 1 (TNNetTokenShift) ... done.  final_CE=1.3449  1.83s
Training Arm 2 (AddMultiHeadSelfAttention) ... done.  final_CE=0.0001  5.94s

=== Comparison (lower CE is better; accuracy is argmax over vocab) ===
arm            params  init_CE  final_CE      acc  acc_lag1 acc_long     sec
TokenShift       1128    1.895     1.345    0.372     1.000    0.163    1.83
Attention        3408    1.873     0.000    1.000     1.000    1.000    5.94

acc_lag1 = early positions t<4 (source is S[t-1], TokenShift turf).
acc_long = positions t>=4 (source is S[t-4], attention turf).

=== Sanity checks ===
[PASS] both arms produced a finite (no NaN/Inf) final loss.
[PASS] both arms beat the uniform baseline (1.792).
[PASS] attention wins the long-range signal: acc_long 1.000 > 0.163.
```

TokenShift uses **~3×** fewer weights (1128 vs 3408) and trains faster,
and it fully solves the part of the task within its reach — but the
long-range copy is out of reach by construction, which is precisely the
trade-off this harness is built to make visible.
