# Sequence Mixer Bakeoff

A head-to-head bake-off of **four token-mixing primitives** on the *same*
tiny char-level next-token task. Every arm shares an identical embedding
front-end, an identical per-token MLP read-out, the same data and the same
weight initialisation; the **only** thing that differs is *how information
moves between sequence positions*.

| Arm | Layer | Mixing | Cost | Reach |
|-----|-------|--------|------|-------|
| 1 | `TNNetTokenShift` | RWKV-style lag-1 time-mix | O(n) | exactly t-1 |
| 2 | `TNNetCausalConv1D` | depthwise causal conv, kernel K | O(n·K) | K-1 steps back |
| 3 | `TNNetDiagonalSSM` | diagonal linear recurrence (SSM-lite) | O(n) | recurrent state (unbounded in principle) |
| 4 | `AddMultiHeadSelfAttention` | causal scaled dot-product attention | O(n²) | any earlier position |

The four arms differ **only** in the middle (the `[MIXER]`); everything
else is shared:

```
TNNetInput(16, 1, 1)                      # 16 token ids along X
  -> TNNetEmbedding(vocab=6, d_model=24)
  -> TNNetSinusoidalPositionalEmbedding   # parameter-free positions
  -> [ MIXER ]                            # one of the four primitives
  -> TNNetPointwiseConvLinear(32)         # per-token MLP hidden (d_ff)
  -> TNNetReLU
  -> TNNetPointwiseConvLinear(6)          # per-token vocab logits
  -> TNNetPointwiseSoftMax(1)             # softmax across depth
```

`TNNetPointwiseConvLinear` (not `TNNetFullConnect`) is used throughout the
sequence body: a pointwise (1×1) projection is applied independently at
every position, preserving the `(SeqLen, 1, d_model)` sequence axis that
`TNNetTokenShift`, `TNNetCausalConv1D`, `TNNetDiagonalSSM` (all of which
require `SizeY = 1`) and the attention builder expect.

How each arm is wired into the `[MIXER]` slot:

- **TokenShift:** a single `TNNetTokenShift` (depth-preserving, learns a
  `Depth`-long `mix` vector).
- **CausalConv1D:** a single `TNNetCausalConv1D(d_model, K)` — `NumFeatures
  = d_model` so the conv preserves the stream width, `K = 5` (dense,
  `Dilation = 1`), giving a receptive field of `K-1 = 4` steps, exactly
  enough to cover the lag-4 source.
- **DiagonalSSM:** a single `TNNetDiagonalSSM` (depth-preserving, learns
  per-channel `a, b, c, e`).
- **Attention:** a `TNNetPointwiseConvLinear(3*d_model)` Q|K|V slab feeding
  `AddMultiHeadSelfAttention(d_model, heads=4, causal=True)`, which
  out-projects back to `d_model` — exactly as `AddTransformerEncoderBlock`
  does. (Per-token projection over a `(SeqLen,1,d_model)` tensor must be a
  pointwise conv, not `TNNetFullConnect`, or the input gradient is zeroed.)

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

- The **lag-1** region (`t < 4`, `acc_lag1`) is reachable by **all four**
  primitives.
- The **lag-4** region (`t >= 4`, `acc_long`) is the **discriminator**: a
  one-step shift cannot see it, a causal conv can iff its kernel `K > 4`,
  an SSM can iff its recurrence learns to pin that exact offset, and
  attention can route to it via the sinusoidal positions.

Everything is generated in-code (no dataset download). `RandSeed` is reset
to the same value before each arm's data generation and before
building/initialising its net, so all arms see identical inputs and
identical embedding init; only the mixing layer differs.

## Build & run

```
lazbuild examples/SequenceMixerBakeoff/SequenceMixerBakeoff.lpi
bin/x86_64-linux/bin/SequenceMixerBakeoff
```

Pure CPU, single-threaded (manual `Compute`/`Backpropagate`), no external
data, finishes in a few seconds (~12 s total for all four arms on a typical
machine). The compiled binary lands in `bin/x86_64-linux/bin/` (shared with
the other examples), not inside this directory. The run is non-interactive
(no trailing `ReadLn`).

## What it shows

- A comparison table with one row per arm: trainable parameter count,
  initial cross-entropy, final cross-entropy, overall next-token accuracy,
  **acc_lag1** (early positions `t < 4`, source `S[t-1]` — reachable by
  all), **acc_long** (positions `t >= 4`, source `S[t-4]` — the
  discriminating region), and wall-clock seconds.
- Three NaN/Inf-guarded sanity checks printed as `PASS`/`FAIL`/`WARN`:
  1. all arms produced a finite final loss,
  2. all arms beat the uniform-guess baseline `ln(vocab) = 1.79`, and
  3. (informative) some longer-reach mixer clearly beats TokenShift on the
     long-range region.

The four arms sort exactly along their **reach**: all clear the uniform
baseline and all nail the lag-1 region (`acc_lag1 ≈ 1.0`), but they split
on `acc_long`. TokenShift sits at chance (`~1/6 = 0.167`) on the long-range
copy because a fixed one-step shift structurally cannot see `S[t-4]`.
CausalConv1D and Attention both reach `1.000` — the conv because its `K=5`
window covers the offset, attention because it can route to any earlier
position. DiagonalSSM lands in between: its recurrent state does carry some
long-range signal (well above chance) but a leaky diagonal recurrence
smears rather than sharply pinning a *single* fixed offset, so it does not
fully solve the copy in this budget. The per-arm numbers are
**seed-dependent**, but the reach-vs-cost ordering is the structural point.

## Sample output

Real output from a recent run (`RandSeed = 424242`):

```
SequenceMixerBakeoff: four token-mixing primitives on ONE next-token task.
Arms: TokenShift (RWKV lag-1, O(n)), CausalConv1D (depthwise causal conv,
O(n*K)), DiagonalSSM (linear recurrence / SSM, O(n)), Attention (causal
SDPA, O(n^2)). Identical embedding front-end, MLP read-out, data and init;
only the sequence-mixing layer differs.
Task: fixed-offset copy. target[t]=S[t-1] for t<4, else S[t-4]   (seqlen 16, vocab 6).
Lag-1 region is reachable by all; the lag-4 region is the discriminator.
Shared front-end: Embedding(24) -> SinPos -> [MIXER] -> PWConv(32) -> ReLU -> PWConv(6) -> softmax.
CausalConv kernel K=5 (reaches K-1=4 back). Uniform-guess baseline = ln(vocab) = 1.7918, chance acc = 0.167.

Training Arm 1 (TNNetTokenShift) ... done.  final_CE=1.3449  1.92s
Training Arm 2 (TNNetCausalConv1D) ... done.  final_CE=0.0001  3.07s
Training Arm 3 (TNNetDiagonalSSM) ... done.  final_CE=0.9882  1.34s
Training Arm 4 (AddMultiHeadSelfAttention) ... done.  final_CE=0.0001  5.64s

=== Comparison (lower CE is better; accuracy is argmax over vocab) ===
arm            params  init_CE  final_CE      acc  acc_lag1 acc_long     sec
TokenShift       1128    1.895     1.345    0.372     1.000    0.163    1.92
CausalConv1D     3984    1.880     0.000    1.000     1.000    1.000    3.07
DiagonalSSM      1200    2.417     0.988    0.546     0.952    0.410    1.34
Attention        3408    1.873     0.000    1.000     1.000    1.000    5.64

acc_lag1 = early positions t<4 (source is S[t-1], reachable by all).
acc_long = positions t>=4 (source is S[t-4], the discriminating region).

=== Sanity checks ===
[PASS] all arms produced a finite (no NaN/Inf) final loss.
[PASS] all arms beat the uniform baseline (1.792).
[PASS] a longer-reach mixer beats TokenShift on acc_long: CausalConv1D 1.000 > 0.163.

Interpretation: TokenShift only reaches t-1, so it solves the lag-1 region
but is near chance on the long-range copy. CausalConv1D reaches K-1 steps
back and so can cover the lag-4 source with a wide-enough kernel.
DiagonalSSM carries a recurrent state (unbounded reach in principle) and
Attention can route to any earlier position. Per-arm numbers are
seed-dependent; the reach-vs-cost trade-off is the structural point.
```

## Reading the result

The headline is the `acc_long` column. All four arms ace `acc_lag1`
(lag-1 is in everyone's reach), so that column carries no signal; the
discriminator is whether the mixer can reach **four** steps back:

- **TokenShift** (`acc_long ≈ 0.16`, chance) — a fixed lag-1 shift is
  structurally blind beyond `t-1`. Cheapest arm (1128 weights).
- **CausalConv1D** (`acc_long = 1.000`) — a `K=5` causal window covers the
  lag-4 source; it fully solves the task at moderate cost (3984 weights),
  faster than attention.
- **DiagonalSSM** (`acc_long ≈ 0.41`) — its recurrent state lifts it well
  above chance, but a leaky diagonal recurrence smears rather than sharply
  pinning a single fixed offset, so it does not fully copy in this budget.
  Nearly the cheapest arm (1200 weights).
- **Attention** (`acc_long = 1.000`) — routes to any earlier position and
  solves the task, at the highest compute cost (O(n²)) and 3408 weights.

This folds together the tasklist entries *"Causal-conv vs token-shift vs
SDPA on the same toy next-token task"* (plus the `TNNetCausalConv1D` and
`TNNetDiagonalSSM` "fourth contender" follow-ups) into one harness. The
per-arm numbers shift with the seed, but the reach-driven ordering
(TokenShift < SSM < {Conv, Attention}) is the structural point and is not
seed-dependent.
