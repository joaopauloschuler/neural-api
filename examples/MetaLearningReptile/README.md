# MetaLearningReptile

Reptile first-order meta-learning (Nichol, Achiam & Schulman 2018,
["On First-Order Meta-Learning Algorithms"](https://arxiv.org/abs/1803.02999))
on the paper's own sine-wave toy.

## What it does

Every other example in this repo trains **one** network to fit **one**
dataset. Reptile is different: it learns an *initialization* `theta` such that,
for a **freshly sampled** task, a handful of ordinary SGD steps adapt it to that
task fast (few-shot learning). It learns to *learn*.

The task distribution is tiny sine-regression tasks `y = A*sin(x + p)` with the
amplitude `A` and phase `p` drawn at random per task. The network is tiny:

```
TNNetInput(1)
  -> TNNetFullConnectReLU(16)
  -> TNNetFullConnectReLU(16)
  -> TNNetFullConnectLinear(1)
```

The outer Reptile loop (driven here through `TNNetReptileMetaTrainer`) is:

1. sample a task `T`;
2. `BeginTask` -> seed a worker net from the meta-weights (a `CopyWeights`
   clone, so the worker's layer refs stay live for inner-loop SGD);
3. run `k` ordinary SGD steps on `T`, adapting the worker to `phi`;
4. `MergeTask` -> move the meta-weights toward the adapted ones:
   `theta := theta + eps*(phi - theta)`.

The interpolation direction `(phi - theta)` is (Nichol et al. sec. 5) a
first-order approximation of the gradient that maximises inner-task
generalization — meta-learning, not weight averaging.

Two baselines are run for contrast, both starting from the **same** random init:

- **Random-init**: adapt the untrained net for the same `k` steps.
- **Joint-init**: a net pre-trained on the pooled *distribution* (the "average
  task", which is `~0` since `E[A*sin(x+p)] = 0`). It fits the mean well but has
  not learned to adapt.

`x` is normalised to `[-1,1]` and the inner learning rate is kept small so the
summed 20-point gradient on this tiny ReLU net stays numerically stable.

## How to run

```
lazbuild MetaLearningReptile.lpi
../../bin/x86_64-linux/bin/MetaLearningReptile
```

Pure CPU, single-threaded (it uses the manual
`Compute`/`Backpropagate`/`UpdateWeights` path, not `TNeuralFit`), and finishes
in about a minute — well under the 5-minute budget.

## Sample output

```
Reptile meta-training over 10000 sine tasks...
  meta iter 2000/10000
  meta iter 4000/10000
  meta iter 6000/10000
  meta iter 8000/10000
  meta iter 10000/10000
Joint-training baseline over 12000 steps...

Held-out adaptation (mean abs error over 20 tasks, lower is better):
  k     Reptile-init   Random-init    Joint-init
  1         0.6396         1.1614        0.9155
  2         0.4196         1.1599        0.8816
  4         0.2471         1.1568        0.8248
```

## Reading the result

Each row is the held-out mean-absolute error after adapting `k in {1,2,4}`
gradient steps from the given initialization (lower is better, same held-out
tasks for every column).

- From the **Reptile** meta-init the error drops sharply with each extra step
  (`0.64 -> 0.42 -> 0.25`): the init sits in a region from which a few steps
  snap onto any sine in the family.
- The **random** init barely moves (`~1.16` throughout): `k` steps are nowhere
  near enough to fit a fresh sine from scratch.
- The **joint** init is better than random but adapts slowly (`~0.92 -> 0.82`):
  it learned the average function, not how to adapt.

That gap is the whole point — Reptile learned to *learn sines*, not to fit the
mean sine. Exact numbers depend on the RNG seed but the ordering
(Reptile << Joint < Random) is stable.
