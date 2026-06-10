# Glimpse Downsampler — learned scale+translate "hard" attention glimpse

A `TNNetAffineGridSample`-based **glimpse** that reads a small canonical patch
from a larger, cluttered input by learning *where* and *how much* to zoom. This
is the **scale + translate-restricted** follow-up to
[`../SpatialTransformer`](../SpatialTransformer) (the full 6-DoF affine STN).

Coded by Claude (AI).

## The idea

A small bright glyph (~9×9, one of four content classes — cross / box /
horizontal bar / vertical bar) is dropped at a **random offset** onto a larger
`28×28` canvas sprinkled with bright clutter specks. A small,
position-**sensitive** classifier must read a `14×14` canonical patch. The
question is how that patch is produced:

* **(A) Learned glimpse downsampler.**
  ```
  input 28×28
    → localization head (conv/pool → FullConnectLinear(4))   emits (s_x,s_y,t_x,t_y)
    → TNNetScatterToAffine        scatter to θ = [ s_x 0 t_x ; 0 s_y t_y ]
    → TNNetAffineGridSample       learned crop/zoom over the 28×28 input
    → TNNetAvgPool(2)             28×28 → 14×14 canonical patch
    → shared classifier
  ```
  The localization head starts as a **wide** view (so the glyph is visible at
  cold-start and the localizer gets a learning signal) and then learns to
  **zoom in and translate** the glimpse onto the roaming glyph. The warp is
  restricted to **scale + translate only** (no rotation/shear).

* **(B) Fixed center-crop (baseline).** The same `14×14` patch is taken by a
  **fixed central** crop+resize of the `28×28` input, fed to the **same**
  classifier — blind to where the glyph actually landed.

**Headline:** on jittered/cluttered inputs the learned glimpse finds the glyph
and beats the fixed center-crop, which keeps staring at the (usually empty)
centre. A representative run:

```
  FIXED center-crop        accuracy = 0.906
  LEARNED glimpse          accuracy = 1.000
  Learned glimpse recovers +9.4 accuracy points the fixed crop misses.
```

## How the warp is restricted to scale + translate

The new parameter-free layer **`TNNetScatterToAffine`** takes a `Size=4` vector
`(s_x, s_y, t_x, t_y)` from the localization head and scatters it into the
`Size=6`, 2×3 affine matrix expected by `TNNetAffineGridSample`:

```
θ = [ s_x   0   t_x ]
    [  0   s_y  t_y ]
```

The two off-diagonal (rotation/shear) entries are held at a **hard zero** —
no parameter ever lands in those slots, so they can never drift away from zero.
Forward routes the 4 inputs to affine positions `{0,4,2,5}`; backward gathers
the gradients of exactly those positions back to the 4 inputs (the shear-slot
gradients are discarded). The layer has no trainable weights and round-trips
through `SaveToString` / `LoadFromString`.

The "downsampler" property — **output smaller than input** — is realised by the
`TNNetAvgPool(2)` after the sampler: the warped `28×28` glimpse is pooled to the
`14×14` canonical patch the classifier consumes.

## Running

```bash
lazbuild GlimpseDownsampler.lpi    # or: fpc -O3 -Mobjfpc -Sh -Fu../../neural GlimpseDownsampler.lpr
./GlimpseDownsampler
```

No data is downloaded; the cluttered dataset is synthesized on the fly. Runs in
about a minute on CPU.

## Tests

`tests/TestNeuralNumerical.pas` covers `TNNetScatterToAffine` with a
forward/shear-zero check, a finite-difference input-gradient check through the
full `ScatterToAffine → AffineGridSample` glimpse path, and a save/load
round-trip.

## Reference

Jaderberg, Simonyan, Zisserman & Kavukcuoglu (2015),
*Spatial Transformer Networks*, https://arxiv.org/abs/1506.02025.
