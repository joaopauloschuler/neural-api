# Retention Dual Form (RetNet)

Trains the **parallel** form of *Retention* (RetNet, Sun et al. 2023,
[*Retentive Network: A Successor to Transformer for Large Language Models*](https://arxiv.org/abs/2307.08621))
on a tiny char-level next-token task, then runs the **same trained weights**
through a hand-rolled **recurrent** loop and asserts the two forward passes
agree token-for-token. This proves Retention's headline property: the parallel
training form and the O(1)-state recurrent inference form are *mathematically
identical*.

## What Retention is

Retention replaces softmax attention `softmax(Q·Kᵀ/√d)·V` with a **softmax-free**
mixer whose only score weighting is a **fixed exponential-decay causal mask**.

### Parallel form (training — what `TNNetRetention.Compute` runs)

```
D[n,m] = γ^(n-m)   for n ≥ m, else 0        (lower-triangular decay mask)
out[n] = Σ_{m≤n} (Q[n] · K[m]) · D[n,m] · V[m]
```

There is **no softmax**. `D` is a constant multiplicative weight: older tokens
are geometrically down-weighted by their relative distance `n−m`, and future
tokens are hard-masked to zero (causality). `γ ∈ (0,1)` is a per-head constant.

### Recurrent form (inference — hand-rolled in the demo)

```
S_n   = γ · S_{n-1} + K_nᵀ V_n     (S is a d_k × d_k state matrix)
out_n = Q_n · S_n
```

Each step needs only the previous state `S_{n-1}` — O(d_k²) memory, **O(1) in
sequence length** — yet produces the *identical* `out_n`. That equivalence is
the dual form, and is what makes RetNet train like a transformer but generate
like an RNN.

## The decay mask

`D[n,m] = γ^(n-m)` for `m ≤ n`. The library builds it incrementally (start at
`1` on the diagonal, multiply by `γ` each step back) so no `Power()` is needed
per cell. It is a **fixed elementwise multiplier** on the raw `Q·Kᵀ` score, so
the backward pass is the scaled-dot-product-attention structure with the softmax
Jacobian replaced by the constant mask `D` — pinned by a finite-difference
gradient check in `tests/TestNeuralNumerical.pas`
(`TestRetentionGradientCheck`, max error ≈ 3·10⁻⁴).

## The equality gate (mandatory)

After training, for every probe sequence and every position the demo computes
the recurrent `out_n` from the exact Q|K|V slab the layer saw and compares it to
the parallel `TNNetRetention.Output`. If the max absolute difference exceeds the
fp tolerance (`1e-4`, single precision) the program `Halt(1)`s — in the style of
`examples/SpeculativeDecoding`'s draft==target gate.

## Library API

```pascal
// Single layer (one head), gamma fixed:
NN.AddLayer(TNNetRetention.Create(d_k, {gamma=}0.90));

// Multi-head builder over a (SeqLen,1,3*d_model) Q|K|V slab:
//   per-head split -> per-head TNNetRetention -> concat -> pointwise out-proj.
// One gamma per head, following the paper's geometric schedule
//   gamma_h = 1 - 2^(-GammaMinExp - h).
NN.AddRetention(d_model, Heads {, GammaMinExp});
```

`TNNetRetention` reuses the same `Q|K|V` depth-slab convention as
`TNNetScaledDotProductAttention` (input depth `3*d_k`, `SizeY=1`). It has no
trainable parameters; `d_k` is stored in `FStruct[0]` and `γ` in `FFloatSt[0]`
so it round-trips through save/load.

## Honest v1 scope

- **γ is a fixed constant**, not learned (the paper's geometric per-head
  schedule). Learning γ via direct gradient is a logged follow-up.
- **Only the parallel + naive-recurrent forms ship.** The chunkwise-recurrent
  hybrid (a throughput optimisation, not a new capability) is skipped.
- The demo uses a **single head** so the hand-rolled recurrent replay is simple;
  `AddRetention` supports `H` heads (one γ each).
- The toy copy task is there to give the trained weights *something* non-trivial
  to mix; the point of the program is the dual-form equality gate, **not** a high
  accuracy. Fixed-γ single-head retention has a limited, decaying reach, so its
  next-token accuracy on an exact fixed-offset copy is only modestly above
  chance — that is expected and honest.

## Build & run

```bash
cd examples/RetentionDualForm
fpc -B -Fu../../neural -Fu/usr/share/lazarus/4.4.0/components/lazutils/lib/x86_64-linux \
    -Mobjfpc -Sh -O2 -dUseCThreads RetentionDualForm.lpr
./RetentionDualForm
```

Pure CPU, single-threaded, no external dataset; finishes in a few seconds.

## Sample output

```
RetentionDualForm: train PARALLEL Retention, then verify the
RECURRENT form reproduces it token-for-token (the RetNet dual form).
Task: fixed-offset copy target[t]=S[t-3] (seqlen 14, vocab 6).
Single head, d_k=12, gamma=0.90 (FIXED). Decay mask D[n,m]=gamma^(n-m).

Built net: 12 layers, 1080 weights. Retention layer idx 5.
Retention gamma read back from layer: 0.900000 (requested 0.900000).

Training parallel form ... done in 2.57s.

Next-token accuracy (parallel form): 0.242  (chance 0.167).
Dual-form check over 400 probes x 14 positions x 12 dims:
  PARALLEL vs RECURRENT max abs diff = 9.54E-006  (tol 1.0E-004).

[PASS] recurrent form reproduces the parallel form within tolerance.
```

(Numbers are seed-dependent; the dual-form gate is the load-bearing result.)
