# DiagonalSSM — diagonal-state linear-recurrence ("SSM-lite") sequence mixer

The smallest possible end-to-end demo of **`TNNetDiagonalSSM`**, the
diagonal-state linear-recurrence sequence mixer — an **O(n)** causal alternative
to an attention head and the first genuinely recurrent layer in the library. A
single `TNNetDiagonalSSM` layer learns to reproduce **fast- and slow-memory
running statistics** over a short sequence, and the example then prints the
learned per-channel decay spectrum so you can watch the channels specialise.

## The layer

`TNNetDiagonalSSM` is a **linear-time-invariant** (LTI) diagonal state-space
mixer: a per-channel one-pole recurrence whose decay, input gain, output gain and
feedthrough are *fixed* learned scalars (no input-conditioning — that is its
selective sibling `TNNetSelectiveSSM`, see `examples/SelectiveSSM`). Per channel
`d` it sweeps the sequence causally:

```
h_t = a[d]·h_{t-1} + b[d]·x_t        (one-pole linear recurrence)
y_t = c[d]·h_t      + e[d]·x_t        (e = S4D-style feedthrough)
```

The per-channel decay is constrained to `(0,1)` via `a[d] = sigmoid(a_raw[d])`,
so the four learnable per-channel tensors the example reads back are the raw decay
`a_raw`, the input gain `b`, the output gain `c` and the feedthrough `e`
(`Neurons[0..3].Weights.Raw[d]`). Because the same decay/gain is applied at every
step, an LTI recurrence can only build a smeared decaying average of the stream —
which is exactly the running-statistic task below.

## The task — fast vs slow running memory

Everything is generated on the fly; there is no external dataset. Each
`(input, target)` pair is a length-`cSeqLen` (12) sequence of random scalars in
`[-1, 1]` laid out along the X axis. The two target channels are the **exact
outputs of two one-pole teacher recurrences** with different decays:

```
target[t,0] = fast running statistic   (teacher decay 0.10 — forgets quickly)
target[t,1] = slow running statistic   (teacher decay 0.95 — long memory)
```

Reconstructing **both** a quickly-decaying and a long-memory statistic from the
same input forces the `cDepth = 4` recurrent channels to differentiate into
fast-forgetting (`a` near 0) and slow-remembering (`a` near 1) memory horizons.

## Pipeline

```
scalar seq -> TNNetInput(cSeqLen, 1, 1)
           -> TNNetPointwiseConvLinear(cDepth)   { lift the scalar to cDepth channels }
           -> TNNetDiagonalSSM                    { the recurrent mixer this example showcases }
           -> TNNetPointwiseConvLinear(cOut)      { per-position linear readout to the 2 targets }
```

Training is plain per-sample SGD (`TNNet.Compute` / `TNNet.Backpropagate`) over
`cSteps = 600` steps of batch `cBatch = 16`, learning rate `0.02`, momentum
`0.9`, with mean-squared-error reported. The decay spectrum is printed **before**
training (where `a_raw = 0` gives `a = 0.5` everywhere) and again **after**.

## Running

```
cd examples/DiagonalSSM
fpc -O3 -Mobjfpc -Sh -Fu../../neural DiagonalSSM.lpr
./DiagonalSSM
```

(or open `DiagonalSSM.lpi` in Lazarus; it builds to
`../../bin/$(TargetCPU)-$(TargetOS)/bin/DiagonalSSM`). Pure CPU, single thread —
runs in well under a minute.

## What to expect

The program prints its own headline:

> *DiagonalSSM demo: a single recurrent TNNetDiagonalSSM layer learns fast- and
> slow-memory running statistics over a length-12 sequence.*

It then prints the architecture (`NN.PrintSummary`), the **before-training** decay
spectrum (all `a = 0.5000`), a periodic training log
(`step … mean-MSE=… elapsed=…s` at step 1, every 50 steps and the last step), and
finally the **after-training** decay spectrum table:

```
Learned per-channel decay spectrum a[d] = sigmoid(a_raw[d]):
  channel    a_raw        a=sig(a_raw)      b         c         e
```

The closing message states the expectation: the channels differentiate into
**fast-forgetting (`a` near 0)** and **slow-remembering (`a` near 1)** memory
horizons, because the linear readout must reconstruct both a quickly-decaying and
a long-memory running statistic. Exact numbers are seed-dependent (`RandSeed`
fixed to 2026); the spread of the decay spectrum is the point.

Coded by Claude (AI).
