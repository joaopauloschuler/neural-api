# Hopfield Retrieval

A **modern Hopfield network** run as a single step of softmax attention,
following Ramsauer et al. 2020, *"Hopfield Networks is All You Need"*
(https://arxiv.org/abs/2008.02217).

## The idea: modern Hopfield retrieval IS attention

Store `K` patterns as the rows of a matrix `X` (shape `K x d`). Given a
(possibly corrupted or partial) query vector `q` in `R^d`, **one**
Hopfield retrieval step is:

```
retrieved = X^T softmax(beta * X q)
```

This is *exactly* scaled dot-product attention with the query `q`, with
keys = values = the stored patterns `X`, and an inverse-temperature
`beta` that plays the role of the `1 / sqrt(d)` scaling in a transformer.
The softmax turns the `K` similarities `beta * <X[k], q>` into a
distribution over the stored patterns, and the retrieved vector is the
corresponding convex combination of those patterns.

The whole behaviour is governed by `beta`:

- **Low `beta`** — the softmax is near-uniform, so the retrieved vector
  is a blurry **average** of all stored patterns. Useless as a memory.
- **High `beta`** — the softmax saturates onto the single nearest stored
  pattern, so one step **snaps** the corrupted query cleanly to that
  pattern. This is the associative-memory / pattern-completion regime.

No training, no iteration, no backprop: retrieval is one forward step.

This demo drives the **shipped library layer** for exactly this rule:
`TNNetModernHopfield`, built with `TNNet.AddModernHopfieldRetrieval`. The
stored patterns are loaded straight into the layer's bank
(`Neurons[0].Weights`, a `(K,1,d)` volume), the corrupted query is the
`(1,1,d)` input, `beta` is the layer's inverse temperature and `KSteps`
is the number of iterated update steps. Here `KSteps = 1` — a single
softmax-attention step — and the retrieved vector is read straight from
the layer output. No retrieval math is re-implemented in the example.

## The retrieval task

Everything is generated in-code (no dataset download), with a fixed
`RandSeed` for reproducibility:

1. Build `K = 6` random **bipolar** (`+/-1`) patterns of dimension
   `d = 32`. Bipolar random vectors in high `d` are near-orthogonal, so
   the bank is well separated (the program prints the worst-case
   `|cosine|` between distinct stored patterns).
2. For each stored pattern, **corrupt** it by flipping the sign of a
   fraction of its dimensions, then run one Hopfield step.
3. Report, per pattern: the input cosine to the true pattern, the
   retrieved cosine, which stored pattern the retrieved vector is
   **nearest** to (argmax cosine over the bank) and that cosine.

## The beta sweep

The headline payoff is the **blurry-average -> clean-snap** transition.
The program sweeps `beta` at a fixed corruption level and sweeps the
corruption level at a fixed high `beta`, printing mean retrieval cosine
for each. At low `beta` the retrieved vector is a blur of the whole
bank; as `beta` grows it collapses onto the single nearest pattern.

## Build & run

```
lazbuild examples/HopfieldRetrieval/HopfieldRetrieval.lpi
bin/x86_64-linux/bin/HopfieldRetrieval
```

Pure CPU, no external data, **forward-only** — finishes in about a
second. The compiled binary lands in `bin/x86_64-linux/bin/` (shared
with the other examples), not inside this directory. The run is
non-interactive (no trailing `ReadLn`).

## What it shows

- A per-pattern retrieval table at the sanity-check operating point
  (`beta = 8`, flip fraction `0.15`): every corrupted query is completed
  back to its true stored pattern (`cos(out) = 1.000`) and the retrieved
  vector is nearest the correct stored pattern.
- The **same** queries replayed at low `beta = 0.1`: now the retrieved
  vectors are blurry averages (lower `cos(out)`, attention mass spread
  across several patterns) — a vivid contrast against the high-`beta`
  clean snap.
- A `beta` sweep and a corruption sweep of mean retrieval cosine.
- Two NaN/Inf-guarded sanity checks printed as `PASS`/`FAIL`:
  1. at high `beta` and low corruption, all `K` patterns recover with
     cosine `> 0.95` **and** every query's retrieval lands nearest the
     correct stored pattern, and
  2. the blurry-average -> clean-snap transition is real: low `beta`
     produces a measurably blurry average while high `beta` snaps
     essentially perfectly.

## Expected output sketch

Real output from a recent run (`RandSeed = 42`):

```
Worst-case |cosine| between distinct stored patterns: 0.313
(near 0 => well separated => clean single-step retrieval).

=== Per-pattern retrieval (single Hopfield step) ===
  beta=8.00  flip=0.15
    pattern     cos(in)   cos(out)  nearest    cos_n     ok
    p0            0.375      1.000        0    1.000    yes
    p1            0.625      1.000        1    1.000    yes
    p2            0.750      1.000        2    1.000    yes
    p3            0.750      1.000        3    1.000    yes
    p4            0.813      1.000        4    1.000    yes
    p5            0.688      1.000        5    1.000    yes

=== Same queries, LOW beta (blurry average, NOT a memory) ===
  beta=0.10  flip=0.15
    pattern     cos(in)   cos(out)  nearest    cos_n     ok
    p0            0.875      0.980        0    0.980    yes
    p1            0.375      0.853        1    0.853     no
    ...

=== Beta sweep (mean retrieval cosine, flip=0.15) ===
  blurry average <----------------------------> clean snap
    beta              0.1      0.5      1.0      4.0     16.0
    mean_cos        0.944    1.000    1.000    1.000    1.000

=== Corruption sweep (mean retrieval cosine, beta=8.0) ===
    flip_frac        0.00     0.10     0.20     0.30     0.40
    mean_cos        1.000    1.000    1.000    0.826    0.556

=== Sanity checks ===
[PASS] beta=8.0 flip=0.15: all 6 patterns recovered (cos>0.95) and the retrieval landed nearest the correct stored pattern.
[PASS] blurry->snap: low beta=0.1 is a blurry average (mean_cos=0.952) while high beta=8.0 snaps cleanly (mean_cos=1.000).
```

The corruption sweep also shows the capacity edge: past a critical flip
fraction (here around `0.3`) the corrupted query falls outside the basin
of attraction of its source pattern and a single step can snap it to the
wrong stored pattern, so mean retrieval cosine drops.
