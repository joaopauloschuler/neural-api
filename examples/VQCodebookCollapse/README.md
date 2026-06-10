# VQCodebookCollapse

A **stress test** for the headline VQ-VAE failure mode -- **codebook collapse**,
where only a handful of codebook entries ever win the nearest-neighbour
`argmin` and the rest of the codebook is dead weight -- and a demonstration of
one **published mitigation** that reverses it.

Where the sibling [`examples/VQCodebookUsage`](../VQCodebookUsage) deliberately
**avoids** collapse (a healthy codebook converging to ~one code per cluster),
this example does the **opposite**: it deliberately **drives** collapse, charts
the active-code count **falling** over training, then shows the fix lifting it
back up.

## The probe API

Three runtime accessors on `TNNetVectorQuantizer`
(`neural/neuralnetwork.pas`):

```pascal
LVQ.ResetCodebookUsage();        // zero the per-code selection counters
... NN.Compute(probe sample) ... // each forward pass increments the winning code
LVQ.ActiveCodeCount();           // # distinct codes selected >= once since reset
LVQ.CodebookUsageCount(idx);     // win count for one code (histogram access)
```

The codebook vectors themselves are exposed like any other layer's trainable
weights, as a `TNNetVolume` per code:

```pascal
LVQ.Neurons[code].Weights.Raw[i]   // read/write code `code`, component `i`
```

This example is **example-only**: it never touches core source. The mitigation
WRITES codebook entries exclusively through that public `Neurons/Weights`
accessor.

## The two arms

Both arms train the same tiny encoder -> VQ -> decoder autoencoder on the same
synthetic data (`cClusters = 16` Gaussian blobs scattered in a bounded cube),
with `cK = 64` codes (far more than data modes). The codebook is **seeded from
data latents** so it starts well-spread and many codes are active at epoch 1 --
the codebook starts **healthy**.

* **COLLAPSE arm** -- no intervention. The collapse driver is an
  **over-regularized encoder**: after each epoch the encoder weights are
  multiplied by `cContract = 0.70`, so the latent cloud contracts toward a
  point. Peripheral codes lose all their points and die, so the active-code
  count **falls** sharply and stays pinned at a tiny number.

* **MITIGATED arm** -- identical training, but every `cReinitEvery = 3` epochs
  it runs **dead-code re-initialization** (the standard "random restart" /
  dead-code revival used in VQ-VAE-2, Jukebox and SoundStream): any code whose
  probe usage is `<= cDeadThresh` (i.e. dead) is re-seeded to a fresh live
  encoder output `z_e` plus small jitter, so it re-enters the `argmin`
  competition near where the data actually lives.

## Headline output

The two active-code trajectories are printed side by side, ending with a
`PASS/FAIL` **verdict** (`Halt(1)` on FAIL). A representative run:

```
epoch | COLLAPSE | MITIGATED
------+----------+----------
    1 |       23 |       23
    3 |       13 |       27
    6 |        7 |       22
    9 |        6 |       18
   12 |        5 |       18
   ...
   30 |        5 |       16

VERDICT: PASS - dead-code re-init lifted active codes from 5 to 16 (collapse arm stuck at 5).
```

The COLLAPSE arm falls **23 -> 5** active codes; dead-code re-init holds the
MITIGATED arm at **~16** (about the 16 true data modes).

## Build / run

```
cd examples/VQCodebookCollapse
fpc -O3 -Mobjfpc -Sh -dRelease -dAVX2 \
    -Fu../../neural -Fi../../neural \
    -Fu<lazutils>/lib/<arch> \
    VQCodebookCollapse.lpr
./VQCodebookCollapse
```

`<lazutils>` is the Lazarus `components/lazutils` dir (its compiled
`utf8process.ppu` is needed by `neuralthread`); `<arch>` is e.g.
`x86_64-linux`. Pure CPU, runs in well under two seconds.
