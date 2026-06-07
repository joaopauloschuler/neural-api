# Expert-Choice Mixture-of-Experts routing

This example contrasts **Expert-Choice routing** (Zhou et al. 2022,
*"Mixture-of-Experts with Expert Choice Routing"*) with the classic
**token-choice** top-k router, at matched per-expert capacity, to show that
expert-choice gives a **uniform per-expert load by construction** while
token-choice can be lopsided.

## The two dual routings

From the `(SeqLen x NumExperts)` gate-score matrix (each row a per-token softmax
over experts), there are two dual ways to pick the routing:

* **Token-choice** (Switch / Shazeer; `TNNet.AddTopKMixtureOfExperts`): each
  **token** keeps its top-k **experts** — argmax along the *expert* axis.
  Nothing constrains how many tokens land on a given expert, so load can
  collapse onto a few experts. The standard fix is a Switch-style
  load-balancing **auxiliary loss** (`TNNetLoadBalanceLoss`).

* **Expert-choice** (Zhou et al. 2022; `TNNet.AddExpertChoiceMixtureOfExperts`):
  the **transpose** — each **expert** keeps its top-`Capacity` **tokens** —
  argmax along the *token* (SizeX) axis. Every expert processes **exactly
  `Capacity` tokens**, so load balance is **structural** and **no aux loss is
  needed**. A token may be picked by 0, 1, or several experts.

## What this demo does

It builds one deliberately **skewed** gate matrix (most tokens prefer expert 0)
and applies both selection rules at matched capacity, then prints the per-expert
token counts. The expert-choice selection runs through the real
`TNNetExpertChoiceGate` layer (the same code path `AddExpertChoiceMixtureOfExperts`
uses), so the demo also exercises the layer end-to-end. Pure CPU, sub-second.

## Headline output

```
TOKEN-CHOICE top-1 (each token -> best expert):
    per-expert token count: [  7  3  2 ]   total=12  min=2  max=7  spread(max-min)=5
    -> load is UNCONSTRAINED and lopsided; needs a load-balance aux loss.

EXPERT-CHOICE Capacity (each expert -> best Capacity tokens):
    per-expert token count: [  4  4  4 ]   total=12  min=4  max=4  spread(max-min)=0
    -> every expert processes EXACTLY Capacity tokens, BY CONSTRUCTION.
       Uniform load with NO aux loss (spread = 0).
```

Token-choice piles 7 of 12 tokens onto expert 0 (spread 5); expert-choice gives
`[4, 4, 4]` (spread 0) for free.

## The builder

```pascal
Block := NN.AddExpertChoiceMixtureOfExperts(
  InputLayer,        // (SeqLen, 1, d_model); nil => GetLastLayer
  NumExperts,        // number of parallel expert MLPs
  ExpertHiddenDim,   // expert MLP hidden width
  Capacity);         // tokens each expert processes (clamped 1..SeqLen)
```

Same expert-MLP + gate-slice + broadcast + weighted-sum wiring as
`AddTopKMixtureOfExperts`, but the gate path uses `TNNetExpertChoiceGate`
(no survivor renormalization — the raw per-token softmax weight is the combine
weight) and there is **no** aux-loss head and **no** `out AuxLossHead` param.

## Build & run

```bash
lazbuild ExpertChoiceMoE.lpi
../../bin/x86_64-linux/bin/ExpertChoiceMoE
```
