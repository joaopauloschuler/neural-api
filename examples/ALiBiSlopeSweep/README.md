# ALiBiSlopeSweep

Empirically interrogate the "magic 8" baked into almost every ALiBi
implementation. Train the **same** tiny single-head causal attention model
**once per slope base** on a locality-sensitive next-token task, then print
a results table so a reader can see whether the cargo-culted constant is
actually optimal.

## What is an ALiBi slope?

ALiBi (Attention with Linear Biases) adds a per-head, distance-dependent
bias directly to the raw attention scores, instead of feeding positional
content into the token stream:

```
score(query Y, key X) += slope[h] * (X - Y)
```

The per-head slope is a geometric sequence:

```
slope[h] = 2^(-Base * (h + 1) / H)
```

where `H` is the number of heads and `Base` is a scalar exponent. In the
original ALiBi paper `Base = 8`, and that value has been copied into
countless reimplementations verbatim. A larger `Base` makes the bias decay
faster with distance (a steeper recency prior that hugs the query's own
position); a smaller `Base` makes it flatter (attention reaches further
back). The `8` is an empirical choice, not a derived constant.

In the library the bias is provided by `TNNetALiBi`. The slope base is the
constructor argument: `TNNetALiBi.Create(Base)` (the default
`TNNetALiBi.Create()` keeps the standard `8.0`). This example sweeps
`Base` over `{4, 6, 8, 12}`.

## What the sweep tests

Whether `8` is special on a task that genuinely rewards a recency prior.
A flat task cannot separate the bases, so the task is chosen to make
locality matter:

- **copy-the-most-recent-vowel**: the alphabet is a few consonants and a
  few vowels; the target at every position is the **most recently seen
  vowel** at or before that position (the begin token if none yet). The
  answer is almost always a nearby token, so the slope that points
  attention at the right distance back wins. A slope that is too steep
  (large `Base`) collapses onto the query's own position and can miss a
  vowel a step or two back; one that is too flat (small `Base`) spreads
  attention too far.

Vocab 7 (3 vowels), sequence length 16, embedding dim 32, single head
(`H = 1`). Every arm shares the RNG seed, steps, learning rate and data,
so the only thing that varies is the scalar passed to `TNNetALiBi.Create`.

## The shared stack

```
TNNetInput(SeqLen, 1, 1)                  # token IDs along X
  -> TNNetEmbedding(Vocab, d_model)       # learned token vectors
  -> single-head CAUSAL attention (same wiring as PositionEncodingBakeoff):
        Q | K | V via three TNNetSplitChannels on a packed projection
        ValueT = TransposeXD(V)
        scores = DotProducts(Q, K) / sqrt(d_k)   # (key, 1, query)
        reshape -> (key, query, 1)
        -> TNNetALiBi.Create(Base)                # <-- THE SWEPT KNOB
        -> TNNetMaskedFill                        # causal upper-triangle
        reshape -> (key, 1, query) -> ReLUL -> softmax (depth axis)
        -> DotProducts(ValueT, W)                 # weighted sum of V
  -> TNNetPointwiseConvLinear(Vocab)      # per-position logits
  -> TNNetPointwiseSoftMax(1)             # softmax across depth
```

## Build & run

```
cd examples/ALiBiSlopeSweep
lazbuild ALiBiSlopeSweep.lpi --build-mode=Default
../../bin/x86_64-linux/bin/ALiBiSlopeSweep
```

Or compile directly with fpc (point `-Fu` at the LazUtils `lib` dir that
holds `utf8process.ppu`, exactly as `tests/RunAll.sh` discovers it):

```
fpc -B -Mobjfpc -Sh -O2 -Fu../../neural -Fu<lazutils-lib-dir> ALiBiSlopeSweep.lpr
```

Pure CPU, no dataset download. All four arms combined finish in well under
half a minute.

## Sample output

Actual run on a single CPU thread:

```
========================================================================
RESULTS (lower cross-entropy / higher accuracy is better):
========================================================================
  Base k    slope=2^-k     train-CE     val-CE      val-acc
  ------    ----------    ---------    ---------    -------
      4     0.062500      0.27856      0.25000       88.8%
      6     0.015625      0.31894      0.28095       87.3%
      8     0.003906      0.32895      0.28866       87.2%
     12     0.000244      0.33206      0.29106       87.2%

========================================================================
Best base by validation CE: k = 4  (val-CE=0.25000).
The cargo-culted 8 is NOT the winner on this task -- the constant is empirical, not sacred.
Total runtime for all 4 arms: 16.7s.
```

## Reading the result

On this recency task the trend is **monotone**: smaller `Base` (flatter
slope, attention reaches further back) trains and validates better, and
the standard `8` is beaten by both `4` and `6`. The differences are modest
but consistent across train CE, val CE and accuracy.

The takeaway is not "always use 4". It is that the slope base is a
**tunable inductive-bias knob, not a sacred constant**. The optimal value
depends on how far back the relevant signal sits: this single-head task
needs to look several positions back to find the most recent vowel, so a
gentler decay helps; the paper's `8` was tuned for deep multi-head
language models where the geometric spread of slopes across many heads
covers a range of distances. Whenever you can afford a short sweep,
treating `8` as a default rather than a law is worthwhile - that is
exactly the empirical check this example performs.
