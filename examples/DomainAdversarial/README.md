# Domain-Adversarial Neural Network (DANN) demo

Smallest possible end-to-end demo of `TNNetGradientReversal`
(Ganin et al. 2015, "Unsupervised Domain Adaptation by Backpropagation",
https://arxiv.org/abs/1505.07818).

## Toy task

Two 2D-Gaussian-blob domains share the same binary class label, but
the class-conditional means are rotated 90 degrees between domain A
and domain B:

```
Domain A:  class 0 ~ N((-1,-1), 0.35I)    class 1 ~ N((+1,+1), 0.35I)
Domain B:  class 0 ~ N((-1,+1), 0.35I)    class 1 ~ N((+1,-1), 0.35I)
```

Because the per-class centroids differ between domains, a linear
classifier trained on (x, y) cannot solve both domains simultaneously
without exploiting some non-trivial structure of the trunk.

## Architecture

```
Input(x, y)
  -> Dense(16)+ReLU -> Dense(16)+ReLU              (shared trunk)
       +--> Dense(2) -> SoftMax                    (label head)
       \--> [TNNetGradientReversal(lambda)]?
            -> Dense(2) -> SoftMax                 (domain head)
  -> Concat(label_logits | domain_logits)
```

The two heads are concatenated on the depth axis so a single target
vector `(label_one_hot | domain_one_hot)` drives joint training. The
Gradient Reversal Layer (GRL) is the identity in the forward pass and
multiplies the upstream gradient by `-lambda` in the backward pass:
the domain-head loss continues to push the domain-head weights in the
"correct" descent direction, but the part of that gradient that
reaches the shared trunk is sign-flipped, so the trunk is steered
toward features the domain classifier cannot exploit.

## Toggle

Edit `cUseGRL` at the top of `DomainAdversarial.lpr`:

- `cUseGRL = True`  -> domain head collapses to ~0.500 (chance),
  label head reaches ~99% on both domains: the trunk has learned a
  domain-invariant feature.
- `cUseGRL = False` -> domain head becomes accurate; label head still
  trains, but the trunk is free to specialize per domain.

## Build & run

```
fpc -dRelease -dUseCThreads -O3 -Fu../../neural DomainAdversarial.lpr
./DomainAdversarial
```

Runs in ~2 seconds on CPU. No external data.
