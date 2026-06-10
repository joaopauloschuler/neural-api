# TopKMoE — hard top-k Mixture-of-Experts with a load-balancing auxiliary loss

This example demonstrates **sparse hard top-k Mixture-of-Experts (MoE) routing**
and shows that a **load-balancing auxiliary loss prevents expert collapse**.

It is the sparse-dispatch follow-up to the soft/dense `TNNet.AddMixtureOfExperts`
builder: instead of evaluating and blending *all* experts on every token, only
the `k` highest-gated experts contribute per token.

## What it shows

Two **identical** networks are trained on **identical** data with the
**identical** hand-rolled SGD loop, differing only in the load-balancing weight:

| model | aux-loss coeff | router behaviour |
|-------|----------------|------------------|
| (A) WITHOUT aux loss | `0` | free to collapse onto one expert |
| (B) WITH aux loss | `> 0` | pushed toward balanced load |

Both runs are deliberately **seeded collapsed** (the gate bias starts favouring
expert 0). At the end we feed a held-out batch through each model and print the
per-expert **token load** — how many tokens picked each expert into its top-k
set — plus the imbalance ratio `max/mean` (1.00 = perfectly uniform).

Representative output:

```
(A) WITHOUT load-balancing aux loss
  per-expert token load (held-out 200 samples):
  E0=1600 (100.0%)  E1=   0 (  0.0%)  E2=   0 (  0.0%)  E3=   0 (  0.0%)
    imbalance max/mean = 4.00  (1.00 = perfectly uniform)

(B) WITH    load-balancing aux loss
  per-expert token load (held-out 200 samples):
  E0= 444 ( 27.8%)  E1= 367 ( 22.9%)  E2= 418 ( 26.1%)  E3= 371 ( 23.2%)
    imbalance max/mean = 1.11  (1.00 = perfectly uniform)
```

**Headline:** without the aux loss the router stays fully collapsed (every token
routes to E0, `max/mean = 4.00`); with it the load is near-uniform across all
four experts (`max/mean = 1.11`). Runtime is ~20 s on one CPU core.

## The pieces

- **`TNNet.AddTopKMixtureOfExperts(InputLayer, NumExperts, ExpertHiddenDim,
  TopCnt, out AuxLossHead, AuxCoeff)`** — the sparse routing builder. It mirrors
  `AddMixtureOfExperts` but swaps the raw-SoftMax gate for a hard top-k gate and
  attaches a load-balancing loss head. It returns the block output (shape ==
  input) and yields the aux-loss head via the `out` parameter.

- **`TNNetTopKGate(TopCnt)`** — the hard-mask + renormalize layer. Per token it
  keeps the `TopCnt` largest gate weights, zeroes the rest, and renormalizes the
  survivors so they sum to 1 (`y_i = g_i / sum_{j in top} g_j`). The renorm is
  fused in so it backpropagates with an exact Jacobian.

- **`TNNetLoadBalanceLoss(TopCnt, Coeff)`** — the Switch-Transformer aux loss
  head (self-contained; reads the gate distribution directly, no external
  target):

  ```
  L_aux = coeff * E * sum_i ( f_i * P_i )
  ```

  where `E = NumExperts`, `f_i` = fraction of tokens whose top-k routing touches
  expert `i`, and `P_i` = mean gate probability for expert `i` over the sample's
  tokens. `f_i` is a non-differentiable hard count, so it is treated as a
  stop-gradient constant (standard Switch convention); the gradient flows only
  through `P_i`:  `dL_aux/dg_t[i] = coeff * E * f_i / T`.

## The synthetic task

A tiny per-token regression with `NumExperts` latent **groups**. Each token
carries its group id in channel 0 and its target is a *group-specific* linear
map of the token. The four groups need four different maps, so a healthy router
must dedicate roughly one expert per group — i.e. the natural solution is
balanced. A collapsed router (everything through one expert) can only fit one
group. This makes "did the load balance?" the visible difference between the two
runs.

## Two-leaf backward (implementation note)

The network is multi-output: the block output and the aux-loss head are two
leaves that **share** the gate sub-graph. The example hand-rolls the backward
pass so both leaves are visited within one `ResetBackpropCallCurrCnt` cycle (the
shared gate accumulates both gradients before propagating upstream). Because
neither leaf is the child of a later layer, each leaf's departing-branch counter
is bumped once at build time (otherwise `TestBackPropCallCurrCnt` mis-fires with
*"Too many backprop calls … Should be:0, Got:1"*).

## Build & run

```
lazbuild TopKMoE.lpi
../../bin/x86_64-linux/bin/TopKMoE
```

Pure CPU, single thread, tiny dims — finishes well under a minute.
