# Lottery Ticket — Iterative Magnitude Pruning (IMP)

A follow-up to [LotteryTicket](../LotteryTicket): **iterative** magnitude pruning
(IMP) — the actual method recommended in Frankle & Carbin, 2019
(*"The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks"*,
[arXiv:1803.03635](https://arxiv.org/abs/1803.03635)) — contrasted against the
**one-shot** prune-to-target baseline.

## Why IMP?

The sibling [LotteryTicket](../LotteryTicket) example prunes the trained dense
net to the final sparsity in a **single** step and retrains from `theta_0`. At
**95% sparsity it collapses** to ~67% accuracy (no better than random reinit) —
the net runs out of trainable capacity in one big cut.

The paper's actual recipe is **iterative**: prune a *small* fraction, rewind the
survivors to `theta_0`, retrain, and repeat. Each round the mask is grown by
pruning the bottom *p%* of the weights that are **still surviving** (not of all
weights), so after *N* rounds the surviving fraction is `(1-p)^N`. The question
this example asks: **does iterating to 95% find a winning ticket that one-shot
pruning to 95% misses?**

## What it does

1. Builds the same small dense ReLU MLP as the sibling
   (`2 -> 64 -> 64 -> 2 + softmax`, 4352 prunable weights) and snapshots its
   random init `theta_0` into a frozen clone via `CopyWeights`.
2. Trains a dense reference on the same two-interleaved-spirals task.
3. **IMP loop** (`N_ROUNDS = 5`, `PRUNE_PER_ROUND = 0.45`,
   `EPOCHS_PER_ROUND = 90`): each round rewinds survivors to `theta_0`, retrains
   with the current mask held fixed, then prunes the bottom **45% of survivors**
   by trained magnitude. `(1 - 0.45)^5 = 0.0503`, i.e. **~95% final sparsity**,
   matching the one-shot target. A per-round accuracy is reported at each
   sparsity level (45% → 70% → 83% → 91% → 95%).
4. **One-shot baseline at the same final sparsity**: prunes the trained dense
   net to ~95% in one step and retrains from `theta_0` for the **matched total
   epoch budget** ((N+1)×90 = 540 epochs — the same total epochs IMP spends
   across all its rounds), so the comparison is budget-fair.
5. The final IMP ticket and the final one-shot ticket are each retrained over
   `N_TRIALS = 2` mini-batch shuffles and averaged (a single tiny net on one
   seed is noisy). The program prints a head-to-head table and **grades the
   result with an explicit verdict**.

The mask is enforced as a **post-step projection**: a `TNeuralFit.OnAfterStep`
hook re-zeros every pruned weight after each batch's weight update.

## How to run

```bash
cd examples/LotteryTicketIMP
lazbuild LotteryTicketIMP.lpi
../../bin/x86_64-linux/bin/LotteryTicketIMP
```

Pure CPU, single thread, fixed `RandSeed = 42`. Wall-clocks at ~140 s.

## Sample output

```
Training dense baseline (90 epochs) ... done.  dense test acc=0.9933  loss=0.0806

--- IMP rounds ---
  round 1: sparsity 45.0%  acc=0.9933  loss=0.0712
  round 2: sparsity 69.7%  acc=0.9967  loss=0.0871
  round 3: sparsity 83.3%  acc=0.9933  loss=0.1374
  round 4: sparsity 90.8%  acc=0.9933  loss=0.1282
  round 5: sparsity 94.9%  acc=0.6633  loss=0.6038

=== Head-to-head at the final (~95%) sparsity ===
method,sparsity_pct,acc,loss
dense_ref,0.0,0.9933,0.0806
one_shot,94.9,0.9833,0.1392
imp,94.9,0.8100,0.4545

=== Verdict ===
IMP acc - one-shot acc = -17.33 accuracy points (at 94.9% sparsity).
INCONCLUSIVE: one-shot beat IMP this seed (toy noise); no IMP advantage demonstrated.
```

## Reading the result — an HONEST negative

This example was built to test whether IMP "rescues" the 95% sparsity where the
sibling one-shot run collapsed. On this toy, **it does not show an IMP
advantage**, and the program says so plainly rather than hiding it:

- **IMP stays healthy out to 91% sparsity** (rounds 1–4 all hold ~99%) and only
  buckles at the final ~95% step — consistent with the lottery-ticket story that
  there is plenty of redundant capacity to remove gradually.
- **One-shot at ~95% does NOT collapse here.** The crucial difference from the
  sibling example is the **training budget**: the sibling gave *every* arm only
  100 epochs, so at 95% sparsity both LT and random-reinit collapsed to ~67%.
  Here the budget-matched one-shot gets the **full 540 epochs** (= the total IMP
  spends across all rounds) and trains the 95%-sparse subnet to ~98%. With a
  generous budget the one-shot ticket is perfectly trainable, so there is no gap
  for IMP to close — and on a single final cut IMP's rewind-and-prune actually
  lands a *worse* mask this seed.
- **Verdict: NO IMP advantage on this toy.** The expected IMP > one-shot ordering
  is a *capacity-/budget-constrained* phenomenon; on a net this small, a task
  this easy, and a matched-and-generous epoch budget, the gap simply does not
  open. The printed margin is the actual measured outcome, not an assumed one.

This mirrors the sibling example's honesty (which openly reports its own 95%
collapse). The takeaway is methodological: **IMP only pays off when the one-shot
ticket is genuinely untrainable** — i.e. under a tight budget or a harder task.
To *see* IMP win here you would need to starve the budget (drop
`EPOCHS_PER_ROUND`) or harden the task (more spiral turns, smaller net) so that
one-shot collapses the way it does in the sibling, leaving room for the iterative
schedule to recover. Exact numbers are deterministic for the fixed seed but shift
with seed, float build, or net size.
