# Knowledge distillation with KL divergence

This example demonstrates the `TNNetKLDivergence` loss head doing **knowledge
distillation** on a synthetic multi-class toy, and contrasts it with a
hard-label cross-entropy baseline of identical student capacity. It is pure
CPU, uses no external dataset, and finishes in well under a second.

Three classes of 2D Gaussian blobs (`K=3`), centered at the corners of a
triangle with **deliberate overlap** (`sigma=0.85`), are classified by a small
MLP. The overlap is intentional: it makes the teacher genuinely uncertain near
the class boundaries, so its softened output carries useful inter-class
structure ("dark knowledge") for the student to learn from.

## The recipe

1. **Teacher.** A comparatively wide `32-32` MLP ending in `SoftMax` is trained
   on hard one-hot labels until it reaches a decent accuracy.
2. **Soft targets.** For every training point the teacher's
   **temperature-softened** output distribution is recorded:
   `softmax(teacher_logits / T)` with `T=3`. We read the teacher's
   already-softmaxed probabilities, take `ln(.)` to recover the logits (up to an
   additive constant, which softmax ignores), divide by `T`, and re-softmax.
3. **Student, two ways** (same `6`-unit body in both):
   - **(a) distillation** — head = `SoftMax -> TNNetKLDivergence`, target = the
     teacher's soft distribution. Minimises `KL(p_teacher || q_student)`.
   - **(b) baseline** — head = `SoftMax` only, target = the hard one-hot label,
     trained with the framework's standard `(output - target)` cross-entropy
     gradient.
4. **Compare** final test accuracy: teacher vs distilled student vs hard-label
   student.

## How the KL head is wired

`TNNetKLDivergence` is a `TNNetIdentity` descendant. Its doc-comment in
`neural/neuralnetwork.pas` states that its input `q` must be a **probability
distribution** (e.g. the output of a `SoftMax`) and its target `p` is the
reference distribution. The per-position loss is the forward KL divergence

```
L = KL(p || q) = sum_i p_i * log(p_i / q_i)
```

The forward pass is an identity passthrough. The framework seeds the last
layer's `FOutputError` with `(output - target) = (q - p)`; `Backpropagate`
recovers `p = q - FOutputError` and replaces the residual with the analytic
gradient `dL/dq_i = -p_i / q_i` (with `q` clamped into `[1e-7, 1]` and
`target <= 1e-7` terms contributing a zero gradient). So to distil we:

```
Input(1, 1, 2)            // a single 2D point
FullConnectReLU(6)        // small shared student body
FullConnectLinear(K)
SoftMax                   // produces the probability head q
KLDivergence              // consumes q; target = teacher soft distribution
```

and feed the teacher's softened distribution as the **target** volume. The
`SoftMax` then backpropagates the KL gradient through its Jacobian. The
hard-label student is identical but drops the `KLDivergence` layer and is fed
one-hot targets.

> Because `TNNetKLDivergence` is an identity passthrough, `Net.Compute` still
> returns the `SoftMax` probabilities, so the argmax used for accuracy is read
> from the last layer for both students.

## Building and running

```
lazbuild KnowledgeDistillation.lpi
../../bin/x86_64-linux/bin/KnowledgeDistillation
```

The run is deterministic (`RandSeed` is fixed), pure CPU, and finishes in
well under a second.

## Sample output

```
KnowledgeDistillation: KL-divergence distillation vs hard-label baseline
Classes: 3  train/class: 120  test/class: 60  temperature: 3.0
(deliberately overlapping blobs, sigma=0.85, so soft targets carry inter-class structure)

[1] Training teacher (32-32 MLP) on hard labels for 60 epochs...
    epoch   1   train_CE=  0.5849
    epoch  15   train_CE=  0.3602
    epoch  30   train_CE=  0.3559
    epoch  45   train_CE=  0.3613
    epoch  60   train_CE=  0.3548
    teacher train_acc= 0.886   test_acc= 0.861

[2] Computing teacher temperature-softened soft targets (T=3.0)...
    example soft target for a train point: [0.749 0.085 0.166]

[3a] Training DISTILLATION student (6-unit MLP, KL head) for 80 epochs...
    epoch   1   mean_KL=  0.0415
    epoch  20   mean_KL=  0.0016
    epoch  40   mean_KL=  0.0015
    epoch  60   mean_KL=  0.0014
    epoch  80   mean_KL=  0.0013

[3b] Training HARD-LABEL student (same 6-unit MLP, CE head) for 80 epochs...
    epoch   1   train_CE=  0.7861
    epoch  20   train_CE=  0.3455
    epoch  40   train_CE=  0.3514
    epoch  60   train_CE=  0.3474
    epoch  80   train_CE=  0.3384

==================== TEST-SET ACCURACY ====================
  teacher (32-32)                :  0.861
  student, distilled (KL, 6)     :  0.861
  student, hard-label (CE, 6)    :  0.856
===========================================================
  -> On this run distillation BEAT the hard-label baseline.
  (Toy problem; reported numbers are exactly what this run produced.)
```

On this toy the distilled student (`0.861`) edges out the hard-label student
(`0.856`) and exactly matches the teacher's test accuracy at a fraction of the
capacity — the soft targets transfer the teacher's boundary structure. The
margin is small and the problem is a toy, so this is a modest, honest
illustration of the mechanism rather than a benchmark; the printed verdict
reflects whatever the run actually produces.
