# U-Net segmentation (`TNNet.AddUNet`)

Trains a real symmetric **U-Net** — built with the reusable `TNNet.AddUNet`
builder — on a self-contained synthetic-shapes segmentation task and reports
**Dice** and **IoU** on a held-out split. No external dataset, pure CPU; the
default SMOKE run finishes in well under five minutes.

## The task

Each sample is a single-channel `32x32` image holding one or two randomly
placed/sized **filled shapes** (circles and axis-aligned rectangles). The
ground-truth mask is `1` on shape pixels, `0` on background, and Gaussian noise
is added to the **input only**. Because the shapes are often small the
foreground is a minority of pixels — the regime where a region-overlap loss
(Dice) helps over a per-pixel loss.

## The network — one builder call

```pascal
NN.AddLayer(TNNetInput.Create(32, 32, 1, 1));
NN.AddUNet(Depth, BaseFeatures, 1, Taps, {UseNorm=}false); // 1 logit channel
NN.AddLayer(TNNetSigmoid.Create());   // per-pixel foreground probability
NN.AddLayer(TNNetDiceLoss.Create());  // analytic Dice gradient (reused head)
```

`AddUNet(Depth, BaseFeatures, OutputChannels, out EncoderTaps, UseNorm)` builds:

- **`Depth` encoder stages** — each `2x [Conv3x3(pad1) -> (Norm) -> ReLU]` then a
  `2x2` stride-2 `TNNetMaxPool` downsample; the feature count **doubles** per
  stage. The pre-pool feature map of every stage is recorded as a **skip tap**.
- a **bottleneck** (`2x [Conv3x3 -> (Norm) -> ReLU]`),
- **`Depth` decoder stages** — each nearest `x2` upsample (`TNNetDeMaxPool`) ->
  **`TNNetDeepConcat`** with the matching encoder tap (the skip connection) ->
  `2x` conv; the feature count **halves** per stage,
- a **`1x1` conv head** to `OutputChannels`.

The output spatial size **equals** the input size — the defining U-Net property
— so the 1-channel logit map is fed straight to a `Sigmoid` + `TNNetDiceLoss`
segmentation head (the **same** Dice loss as `../DiceSegmentation`). `AddUNet`
returns the encoder-tap layer **indices** in `EncoderTaps` (shallow→deep) so the
caller can inspect/re-wire the skips; the example prints them.

> **Constraint:** the input `SizeX`/`SizeY` must each be divisible by `2^Depth`
> so every downsample/upsample round-trips exactly (32 is fine up to Depth 5).

`UseNorm` toggles a `TNNetMovingStdNormalization` after each conv. This example
passes `false`: on this tiny task the plain `conv -> ReLU` U-Net optimizes more
stably with the global Dice gradient (the normalized variant tends to collapse
the shallow head to the trivial all-foreground prediction).

## Output

Per-eval Dice/IoU (threshold 0.5), one held-out sample as ASCII
(input / ground-truth / prediction), and a dependency-free **PPM** strip
`unet_sample.ppm` (input | ground-truth | prediction; convert with e.g.
`convert unet_sample.ppm unet_sample.png`). The console prints whatever the
numbers actually show.

## Build & run

```
cd examples/UNetSegmentation
LAZUTILS_PATH=$(find /usr/lib/lazarus /usr/share/lazarus -name "utf8process.ppu" -printf "%h\n" 2>/dev/null | head -1)
ulimit -v 3000000
fpc -O3 -Mobjfpc -Sh -Fu../../neural -Fu"$LAZUTILS_PATH" UNetSegmentation.lpr
./UNetSegmentation            # smoke run (fast, default)
./UNetSegmentation --full     # longer training for a sharper Dice/IoU
```

A representative **smoke** run (depth 3, base 8, 192 train / 64 test, 12 epochs,
~1 min on CPU) reaches roughly **Dice ≈ 0.97, IoU ≈ 0.95** on the held-out set.

Coded by Claude (AI).
