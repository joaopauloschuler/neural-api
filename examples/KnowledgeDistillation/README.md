# KnowledgeDistillation example

End-to-end demo of **`TNeuralKDTrainer`** (`neural/neuralkd.pas`), classic
Hinton knowledge distillation (Hinton, Vinyals & Dean 2015):

> a small **student** net is trained to mimic a larger frozen **teacher** while
> still fitting the ordinary hard labels, via the blended loss
> `L = alpha * CE(label, softmax(z_s)) + (1-alpha) * T^2 * KL(softmax(z_t/T) || softmax(z_s/T))`.

## What it does

1. Builds a **larger char-level next-token teacher** (two hidden layers,
   hidden=96) and trains it with ordinary SGD on a structured synthetic token
   stream (a deterministic 2nd-order mixing recurrence over a 12-symbol
   vocabulary — genuinely learnable, not a trivial period).
2. Builds **two identical small students** (same RNG init, hidden=12) and
   trains them at **matched steps / examples-seen**:
   - **WITH KD** — `TNeuralKDTrainer.Create(Teacher, Student, alpha=0.3, T=3.0)`;
     the soft teacher targets supervise the student alongside the hard label.
   - **HARD-LABEL ONLY** — the *same* trainer with `alpha=1.0`, where the soft
     term vanishes and `Step()` becomes an ordinary cross-entropy SGD step
     (this equivalence is pinned by `TestAlphaOneMatchesPlainCE` in
     `tests/TestNeuralKD.pas`). Same data order, same learning rate, same
     number of `Step()` calls — the two runs differ **only** in the soft term.
3. Reports held-out **perplexity** for the teacher and both students with
   `TNNet.PerplexityReport` (`neural/neuralnetwork.pas`).

The teacher is run forward only inside the trainer, so it is frozen
(`TestTeacherWeightsUnchanged`); only the student moves.

## Representative output (pure CPU, ~25 s)

```
Teacher (hidden=96)            held-out perplexity : 14.06
Student WITH KD (alpha=0.3,T=3): held-out perplexity : 18.71
Student HARD-LABEL ONLY        : held-out perplexity : 40.91
```

The KD student more than halves the hard-label-only student's perplexity at
the **same** number of optimizer steps: the teacher's temperature-softened
distribution carries "dark knowledge" (the relative probabilities of the
*wrong* classes) that a single one-hot label cannot, and that extra signal
regularises the tiny student toward the teacher's generalising behaviour.
Numbers vary slightly with the build but the KD < hard-only gap is robust.

## Swapping in a real imported teacher (GPT-2 / TinyStories)

`TNeuralKDTrainer` only requires that teacher and student **share the
vocabulary width** and both end in `TNNetFullConnectLinear(Vocab) -> SoftMax`
(or `TNNetPointwiseSoftMax`); the architectures and input shapes are otherwise
unconstrained. To distill from a genuine pretrained LM, replace `BuildTeacher`
with one of the importers, e.g.

```pascal
uses neuralllmimport;   // BuildGPT2FromSafeTensors / .bin importers
...
Teacher := BuildGPT2FromSafeTensors('gpt2', {pInferenceOnly=}true);
```

and feed token windows in the teacher's tokenization. **Memory note:** a full
GPT-2 (124M) needs `pInferenceOnly` / `MakeInferenceOnly` to stay under a
3 GB cap (see the project memory note `gpt2-import-oom-and-slicer`). The tiny
synthetic teacher here keeps the demo inside that budget and under a minute,
while still showing the KD-vs-hard-label perplexity gap that is the whole
point of the trainer.

## Build

```
lazbuild KnowledgeDistillation.lpi
./KnowledgeDistillation
```

Run under a memory cap to stay safe:

```
ulimit -v 3000000
../../bin/x86_64-linux/bin/KnowledgeDistillation
```

Or, without Lazarus installed:

```
fpc -Mobjfpc -Sh -O3 -Fu../../neural KnowledgeDistillation.lpr
```
