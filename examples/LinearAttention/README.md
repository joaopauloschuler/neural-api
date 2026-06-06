# LinearAttention

Wall-clock **scaling probe** for `TNNetLinearAttention`, the first
sub-quadratic (softmax-free / kernelized) attention layer in this repo
(Katharopoulos et al. 2020, *Transformers are RNNs*,
<https://arxiv.org/abs/2006.16236>).

## The idea

Scaled dot-product attention forms an `SeqLen x SeqLen` score matrix,
softmaxes it, and weights `V` — an `O(SeqLen^2)` core. Linear attention
replaces the softmax with a positive feature map `phi(x) = elu(x)+1`
applied to `Q` and `K`, then exploits associativity:

```
S = sum_s phi(K_s) (x) V_s     # a d_k x d_v matrix, accumulated ONCE
Z = sum_s phi(K_s)             # a d_k normaliser vector
Out_t = (phi(Q_t) . S) / (phi(Q_t) . Z)
```

The per-forward cost is `O(SeqLen * d_k * d_v)` — **linear** in sequence
length — and the `SeqLen x SeqLen` matrix is never materialised. This is
the **non-causal** (full-prefix) variant: every query reads the same
`S`/`Z`. (A causal variant needs a running prefix-sum of `S`/`Z` — the
"attention is an RNN" identity — and is a separate follow-up.)

## What this probe does

Builds a 1-layer `TNNetLinearAttention` net, runs a forward pass at
`SeqLen` in `{16, 32, 64, 128, 256}` with `d_k` held fixed, times each,
and prints a table. The **ratio** column (time vs. the previous, half-as-
long row) should sit near **2x** — linear growth. Quadratic softmax
attention would roughly **4x** each time `SeqLen` doubles. The
**us / token** column should stay roughly flat.

```
TNNetInput(SeqLen, 1, 3*d_k)        # depth packs Q | K | V
  -> TNNetLinearAttention(d_k)      # non-causal, no softmax, no NxN matrix
```

## Build & run

```
lazbuild LinearAttention.lpi
../../bin/x86_64-linux/bin/LinearAttention
```

Pure CPU, runs in a few seconds on a single thread; modest memory.

## Expected output sketch

```
--------------------------------------------------------------------------
  SeqLen       total ms     ms / forward     us / token      ratio
--------------------------------------------------------------------------
      16         132.00           0.6600         41.250          -
      32         237.00           1.1850         37.031      1.80x
      64         417.00           2.0850         32.578      1.76x
     128         938.00           4.6900         36.641      2.25x
     256        1983.00           9.9150         38.730      2.11x
--------------------------------------------------------------------------
```

Ratios cluster around `2x` (linear), not `4x` (quadratic), and the
per-token cost stays roughly constant — the hallmark of sub-quadratic
attention. Absolute numbers vary by machine.
