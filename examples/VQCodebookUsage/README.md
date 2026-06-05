# VQCodebookUsage

The demonstration companion to the new `TNNetVectorQuantizer` **codebook-usage
probe**, which exposes the headline VQ-VAE failure mode: **codebook collapse**,
where only a handful of codes ever win the nearest-neighbour `argmin` and the
rest of the codebook is dead weight.

## The probe API

Three runtime accessors were added to `TNNetVectorQuantizer`
(`neural/neuralnetwork.pas`):

```pascal
LVQ.ResetCodebookUsage();        // zero the per-code selection counters
... NN.Compute(probe sample) ... // each forward pass increments the winning code
LVQ.ActiveCodeCount();           // # distinct codes selected >= once since reset
LVQ.CodebookUsageCount(idx);     // win count for one code (histogram access)
```

The probe is **pure runtime bookkeeping**: it does **not** change the
quantization output, the cached `argmin`, any gradient (straight-through +
commitment + codebook), or serialization. The training forward/backward path is
byte-for-byte unchanged.

## The model

```
input (D dims)
  -> encoder: FullConnect... -> latent
  -> TNNetVectorQuantizer(K codes)   [nearest-code quantization]
  -> decoder: FullConnect... -> reconstruction (MSE)
```

The synthetic data is drawn from `cClusters` well-separated Gaussian blobs in
input space, so a **healthy** codebook should converge to use roughly one code
per cluster. We deliberately allocate **more** codes than clusters
(`K = 12` vs `5` clusters) so dead/unused codes are visible.

## Headline output

Each epoch the codebook is probed over a fixed batch: the example prints the
active-code count plus a per-code usage histogram, ending with a graded
**PASS/FAIL** verdict — PASS when the codebook stays healthy (uses
`>= #clusters` distinct codes), `Halt(1)` on collapse. A representative run
converges to **11 of 12 codes active** (well above the 5-cluster floor),
`VERDICT PASS`.

## Build / run

```
cd examples/VQCodebookUsage
fpc -O3 -Mobjfpc -Sh -dRelease -dAVX2 -Fu../../neural -Fi../../neural VQCodebookUsage.lpr
./VQCodebookUsage
```

Pure CPU, runs in under a second.
