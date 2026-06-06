# SparseAutoencoder

A self-contained, pure-CPU reproduction of the headline result of Anthropic's
**Towards Monosemanticity: Decomposing Language Models With Dictionary
Learning** (Bricken et al. 2023,
<https://transformer-circuits.pub/2023/monosemantic-features/index.html>): a
**sparse autoencoder** (SAE) with an *overcomplete* dictionary, trained on dense
**polysemantic** activations, recovers the original sparse ground-truth features
as **monosemantic** dictionary atoms — and there is a sweet-spot sparsity weight,
with too little giving dense polysemantic atoms and too much killing atoms.

No new layer is needed; the demo uses only existing layers
(`TNNetFullConnectReLU`, `TNNetFullConnectLinear`).

## The setting (companion to `examples/Superposition/`)

`examples/Superposition/` shows the **problem**: a model packs `K` sparse
ground-truth features into `d < K` dimensions, storing them in *superposition* —
non-orthogonal, mutually-interfering directions. The resulting activation vector
is **polysemantic**: each coordinate responds to a blend of several underlying
features, so no single activation axis is one interpretable concept.

This example shows the **solution**. Given only the dense, polysemantic
activation vectors `a` (the ground-truth features are never seen during
training), can we **recover** the original sparse features by learning an
*overcomplete* dictionary — an SAE with many more hidden units than activation
dimensions, regularised so each hidden unit fires rarely? The Bricken et al.
headline: yes — the dictionary atoms become **monosemantic**, each aligning with
one ground-truth feature, and feature recovery is a *non-monotone* function of
the sparsity weight `lambda`.

## The toy

- `K = 16` sparse ground-truth features in a dense activation space of width
  `d = 8` (`d < K`: the superposition regime).
- Feature `k` is independently **active** with probability
  `cActiveProb · cProbDecay^k` (`0.10 · 0.94^k`, a mild geometric decay); when
  active its value is `~ U[0.5, 1.5]`, else `0`. Making features *unequally
  frequent* is what produces the classic non-monotone recovery curve: too-large
  `lambda` kills the **rare** features' atoms first (they pay the same `|h|`
  cost but rarely help reconstruction).
- A fixed random mixing matrix `Gtrue` (`d x K`, unit-norm columns = the true
  feature directions) mixes the sparse feature vector `f` down into a dense
  activation `a = Gtrue · f`. The SAE is trained **only** on `a` — it must
  unpack what `Gtrue` packed.

## The network

```
Input(d) -> TNNetFullConnectReLU(H)     {H >> d : the overcomplete dictionary code}
          -> TNNetFullConnectLinear(d)  {decoder D : atoms = columns of D}
```

with `d = 8`, `H = 22`. The **dictionary atom** of hidden unit `j` is the decoder
column `D[:,j]` (a `d`-vector): the activation pattern unit `j` writes back.
Monosemanticity = each atom `D[:,j]` points along one true feature column
`Gtrue[:,k]`. After every weight update the decoder atoms are renormalised to
**unit norm** (the Bricken et al. dictionary constraint) — without it the `L1`
code penalty is degenerate (the net would shrink `h` and grow the columns to
compensate, so `|h| -> 0` without ever zeroing an atom).

## The loss (reconstruction MSE + L1 on the hidden code)

Trained to minimise `L = ||out - a||^2 + lambda · sum_j |h_j|`, where `h` is the
hidden (ReLU) code. Both terms are delivered through the **stock backprop** by
manual gradient surgery in batch-update mode (`SetBatchUpdate(True)`, which makes
`Backpropagate` *sum* per-sample deltas into `Neurons[].Delta`):

- **Pass A — reconstruction MSE** via the pseudo-target trick (as in
  `examples/Superposition/`): the framework seeds the output error as
  `(output − target)`, so feeding
  `pseudo_i = out_i − (1/B)·(out_i − a_i)` makes the seeded error exactly the
  per-sample mean-MSE gradient `(1/B)·(out − a)`; full backprop.
- **Pass B — L1 sparsity** is the new piece. `d|h_j|/dh_j = sign(h_j)`
  (sub-gradient; `sign(0)=0`). After Pass A, the hidden layer's `OutputError` is
  **overwritten** with `(lambda/B)·sign(h)` and the hidden layer's
  `Backpropagate()` is called directly, so the `L1` sub-gradient flows
  hidden→input only (the decoder is untouched). The two passes accumulate into
  the same deltas.

The "`Too many backprop calls at TNNetFullConnectReLU`" messages printed during
the run are **expected** — they come from this intentional second backward pass.

## The sweep

`lambda ∈ {0, 1e-3, 1e-2, 1e-1, 3e-1, 1.0}` — spanning the
dense → sweet-spot → over-sparse (dead) regimes. `1e-1` is the recovery sweet
spot; `3e-1` and `1.0` are over-sparse (recovery drops, dead atoms grow).

## Built-in correctness signals (printed PASS / FAIL; `Halt(1)` on any failure)

1. **`lambda=0` is a plain over-parameterised AE:** reconstruction MSE `~ 0` but
   atoms are **polysemantic** — its monosemanticity score is low (near the raw
   activation-axis baseline) and strictly below the best swept `lambda`'s score.
2. **Recovered-feature count is non-monotone with an interior peak:** some
   intermediate `lambda` recovers more features than **both** `lambda=0` and the
   largest `lambda` (over-strong `L1` loses features by killing rare atoms).
3. **Dead-atom fraction grows at the largest `lambda`:** over-strong sparsity
   kills atoms outright (here from 27% dead at `lambda=0` to 95% at `lambda=1.0`).

## Running

```
lazbuild examples/SparseAutoencoder/SparseAutoencoder.lpi
./bin/x86_64-linux/bin/SparseAutoencoder
```

It must end with `ALL INVARIANTS PASS` and exit code 0.

Pure CPU, no external data, fully deterministic (fixed seed `424242`), finishes
in about **20 seconds**.

## Sample output

```
========================================================================
Towards Monosemanticity (Bricken et al. 2023) -- pure-CPU reproduction
========================================================================
Ground-truth features K=16, dense activation width d=8 (d<K: superposition).
Sparse AE: Input(8) -> ReLU(22){overcomplete dictionary} -> Linear(8){decoder}.
Per-feature activation prob = 0.10. Reconstruction MSE + L1 code penalty.
3000 steps x batch 128, lr=1.500. Atom j = decoder column D[:,j].
"Recovered" feature: some atom has cosine >= 0.80 to its true direction.

Raw activation-to-feature baseline (mean max-cos to an axis) =  0.567
```

The cross-`lambda` summary table — recovery rises to a peak at `lambda=0.1`
(16/16) then collapses as the dictionary is over-sparsified, while `meanL0`
falls monotonically and `dead%` climbs:

```
========================================================================
SWEEP SUMMARY
========================================================================
  lambda    mono   recovered   meanL0   reconMSE   dead%
  0.0000   0.700      1/16     10.67    0.00000     27
  0.0010   0.734      3/16     10.67    0.00005      9
  0.0100   0.788      8/16      4.41    0.00019     18
  0.1000   0.863     16/16      2.48    0.00599     23
  0.3000   0.826     14/16      1.41    0.02586     50
  1.0000   0.686      2/16      0.05    0.12934     95
```

The three built-in correctness signals:

```
(1) lambda=0 is a plain over-parameterised AE: low MSE but POLYSEMANTIC atoms.
    lambda=0 recon MSE =  0.00000 (should be ~0)
    lambda=0 mono = 0.700   best-lambda(0.1000) mono = 0.863
    plain-AE-is-polysemantic (low MSE, mono < best lambda) : PASS

(2) Recovered-feature COUNT is non-monotone with an INTERIOR peak.
    recovered by lambda: 1 3 8 16 14 2
    peak at lambda=0.1000 (recovered=16); ends: lambda0=1, lambdaMax=2
    interior peak (more recovery at an intermediate lambda) : PASS

(3) DEAD-ATOM fraction grows at the largest lambda (over-strong sparsity).
    dead% by lambda: 27 9 18 23 50 95
    dead-atom fraction at largest lambda > at lambda=0 : PASS

========================================================================
ALL INVARIANTS PASS
========================================================================
```

Per `lambda`, the program also prints a per-feature recovery block (best matching
atom and cosine, with a glyph-shaded `|cos|`) so "one atom == one interpretable
feature" recovery is visible directly.

## Expected reading

Feature recovery traces a clear **inverted-U** in `lambda`: at `lambda=0` the
overcomplete AE reconstructs almost perfectly (MSE `~0`) but spreads each feature
across many polysemantic atoms (only 1/16 recovered, `meanL0 ≈ 11`); as `lambda`
rises the code sparsifies and atoms align with single features, peaking at
`lambda=0.1` (16/16 recovered, `meanL0 ≈ 2.5`); push `lambda` further and the
`L1` starts killing atoms — the rare features go first — so recovery collapses
(14/16 → 2/16) and the dead-atom fraction explodes (23% → 50% → 95%). This is the
sweet-spot sparsity-weight phenomenon from the paper.

## References

- T. Bricken et al., *Towards Monosemanticity: Decomposing Language Models With
  Dictionary Learning*, Transformer Circuits Thread, 2023.
  <https://transformer-circuits.pub/2023/monosemantic-features/index.html>

## See also

This demo is the **solution** counterpart to `examples/Superposition/` (which
reads a trained model's own encoder geometry `G = D·W` to *measure* how features
are packed): here we train a *separate* overcomplete dictionary to *unpack* a
dense activation back into features and score recovery against known ground-truth.
It shares the manual-gradient-surgery idiom (pseudo-target reconstruction loss
plus a hand-delivered second backward pass) with `examples/Superposition/` and
`examples/SharpnessAwareMinimization/`.
