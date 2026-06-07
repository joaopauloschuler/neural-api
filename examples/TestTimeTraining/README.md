# Test-Time Training (TTT)

A tiny, pure-CPU demo of the **Test-Time Training** sequence-mixing layer
`TNNetTestTimeTraining` (Sun et al. 2024,
[arXiv:2407.04620](https://arxiv.org/abs/2407.04620)).

In TTT the recurrent hidden *state* is itself a small model `W` whose weights are
updated by **one explicit gradient-descent step** on a self-supervised
reconstruction loss at every timestep, so the forward scan literally *trains* an
inner net as it reads the sequence. Per token `t` (learnable view projections
`theta_K/theta_V/theta_Q` and inner learning-rate `eta = softplus(eta_raw)`):

```
k_t = theta_K x_t ;  v_t = theta_V x_t ;  q_t = theta_Q x_t
inner loss  ell_t(W) = 1/2 || W(k_t) - v_t ||^2          (self-supervised)
W_t = W_{t-1} - eta * grad_W ell_t(W_{t-1})              (the TTT update)
y_t = W_t(q_t)                                           (read-out)
```

Two inner-model variants behind one flag:

- **TTT-Linear** (`variant=0`): `W` is a single matrix; its MSE-gradient step is
  rank-1, reducing the scan to a delta-rule-like recurrence. (Closely related to
  the landed `TNNetDeltaNet`, but with a learnable per-layer inner LR `eta` and a
  raw key instead of DeltaNet's data-dependent sigmoid gate + L2-normalized key.)
- **TTT-MLP** (`variant=1`): `W = W2 * GeLU(W1 .)` is a two-layer net; its
  gradient step is a genuine **non-linear** fast-weight update that a single
  matrix cannot express. This non-linear inner state is the headline novelty.

The outer backward pass is **exact BPTT through the inner update** (a
Hessian-vector product for the MLP arm), verified by numerical-gradient tests in
`tests/TestNeuralNumerical.pas`.

## The demo

A **non-linear binding recall** task. Each write token carries two one-hot key
fields (a `row` and a `col`) plus the value `bank[row XOR col]` — a deliberately
non-linear (parity-like) binding of the two keys. A query re-presents one written
key; the target is its bank value. Three arms share the same I/O and a matched
memory width (`model_dim=16`):

- **TTT-MLP** — non-linear inner state.
- **TTT-Linear** — matrix inner state.
- **DeltaNet** — the landed matrix delta-rule recurrence.

**Headline:** the non-linear TTT-MLP inner state lifts recall on the parity
binding where the linear-state mixers plateau.

## Running

```bash
lazbuild TestTimeTraining.lpi
../../bin/x86_64-linux/bin/TestTimeTraining
```

Runs in a few seconds on 2 cores. Representative output:

```
eval over 400 held-out recall sequences:
  TTT-MLP    (non-linear inner state): recall MSE = 0.02313   exact-recall acc = 97.8%
  TTT-Linear (matrix inner state)    : recall MSE = 0.04111   exact-recall acc = 93.5%
  DeltaNet   (matrix delta rule)     : recall MSE = 0.22261   exact-recall acc = 28.8%
OK: the non-linear TTT-MLP inner state beats BOTH linear-state mixers on the parity binding task.
```
