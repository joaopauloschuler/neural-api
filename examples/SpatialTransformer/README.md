# Spatial Transformer Network demo

End-to-end demo of `TNNetAffineGridSample`, the differentiable bilinear
grid-sampling core of a Spatial Transformer Network
(Jaderberg, Simonyan, Zisserman & Kavukcuoglu 2015,
"Spatial Transformer Networks", https://arxiv.org/abs/1506.02025).

## Toy task

A tiny, fully self-contained, offline dataset on a `20x20x1` canvas: 4 small
distinct glyphs (cross / box / horizontal bar / vertical bar), each ~6x6 drawn
in the centre. Every training and test sample is then **jittered** by a large
random translation (`+/- ~6 px`) and a light rotation (`+/- ~9 deg`), so the
small glyph roams across the large canvas.

Because the glyph is small and the canvas is large, a position-sensitive
fully-connected readout must memorise every (glyph x location) combination,
which it cannot do reliably — so the input jitter destroys its accuracy. This
is exactly the regime the Spatial Transformer is designed to rescue.

## Architecture

Both models share the SAME fully-connected classifier head
(`FullConnectReLU(16) -> FullConnectLinear(4) -> SoftMax`):

```
plain :  Input(20,20,1) -> [classifier]

STN   :  Input(20,20,1)
            |-- localization head: Conv3x3+ReLU(8) -> MaxPool -> Conv3x3+ReLU(8)
            |                      -> MaxPool -> FullConnectReLU(16)
            |                      -> FullConnectLinear(6)   (theta = 2x3 affine)
            +-- TNNetAffineGridSample(source = Input, theta = loc head)
                  -> [classifier]
```

The `TNNetAffineGridSample` layer warps the original input image by the affine
matrix predicted by the localization head, via bilinear interpolation of the 4
nearest source pixels. Its final `FullConnectLinear(6)` is **bias-initialised
to the identity affine `[1,0,0,0,1,0]`** (and its weights zeroed), exactly as
the paper prescribes, so the transformer starts as a no-op pass-through and
**learns** the canonicalising transform end-to-end purely from the
classification loss — no transform supervision. The conv localization head is
translation-tolerant, so it can estimate where the glyph is and the sampler
recentres it before the position-sensitive classifier sees it.

## Expected output

Both models see the SAME data stream (`RandSeed = 1234` before each build,
`9999` before each held-out evaluation). After 1500 minibatch steps:

```
plain CNN              accuracy ~ 0.88
CNN + Spatial Xformer  accuracy ~ 1.00   (recovers ~+12 accuracy points)
```

The headline: the STN front-end recovers the accuracy the plain head loses to
input jitter.

Total runtime: ~40 seconds on CPU. No external data is downloaded.

## Build & run

```
fpc -dRelease -dUseCThreads -O3 -Fu../../neural SpatialTransformer.lpr
./SpatialTransformer
```
