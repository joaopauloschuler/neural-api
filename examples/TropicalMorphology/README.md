# TropicalMorphology — a max-plus / min-plus dense layer with `TNNetTropicalLinear`

This example showcases **`TNNetTropicalLinear`**, a dense layer that computes in
the **tropical (max-plus / min-plus) semiring** instead of the usual
multiply-accumulate ring. Where every other dense layer in neural-api
(`TNNetFullConnect*`, `TNNetCirculantLinear`, `TNNetHouseholderLinear`,
`TNNetBitLinear`, `AddGroupedFullConnect`, …) computes `y_i = Σ_j W[i,j]·x_j`,
the tropical layer replaces the **product** with **addition** and the **sum**
with **max** (or **min**):

* **DILATION** (default): `y_i = max_j ( x_j + W[i,j] )`
* **ERODE** (constructor flag): `y_i = min_j ( x_j + W[i,j] )`

The weights `W` are learnable **additive thresholds** (a morphological
structuring element), and the combine op is `max` / `min`. As a result the layer
learns piecewise-linear **convex** (dilation) / **concave** (erosion) functions
and *tropical polynomials* — a **different hypothesis class** from any
linear layer, and from the *parameterless* max/min pooling layers.

Forward is `O(Din·Dout)`, like a dense layer. Backward is the same hard
subgradient that max-pooling uses: for each output `i` it caches the winning
index `j*` (arg-max for dilation, arg-min for erosion) and routes the upstream
error to exactly that one input and one weight:

```
dL/dx[j*] += dy_i      dL/dW[i,j*] += dy_i
```

This is non-differentiable at ties (same convention as `TNNetMaxPool` /
`TNNetMaxChannel`), so gradient tests use well-separated inputs away from the
kink. The mode flag round-trips through serialization via `FStruct[6]`.

## The task

We fit a **convex piecewise-linear target** — the upper envelope of three lines,

```
g(t) = max( -2t - 1 ,  0.3t + 0.2 ,  2.5t - 0.5 )
```

a convex fan/V shape. The scalar input `t` is lifted into a bank of `NFEAT`
learnable affine features (`b_j·t + c_j`) shared by both models; then:

| model | head | what it can represent |
|-------|------|-----------------------|
| **tropical** | `TNNetTropicalLinear` dilation: `max_j(feat_j + W_j)` | any **convex** piecewise-linear curve (a tropical polynomial) |
| baseline | same-width `TNNetFullConnectLinear` | a single **affine** function → one straight line |

A single linear layer over affine features is itself affine in `t` — *one
straight line, no matter how wide* — so it provably cannot bend to a convex fan.
The max-plus stack collapses the feature bank into `max_j(b_j·t + (c_j + W_j))`,
exactly the target's form.

We also train the **min-plus ERODE** sibling on the negated (concave) target to
show the dual.

## Running

```
cd examples/TropicalMorphology
lazbuild TropicalMorphology.lpi
../../bin/x86_64-linux/bin/TropicalMorphology
```

(or `fpc -Mobjfpc -Sh -O3 -Fu../../neural -dRelease -dAVX2 TropicalMorphology.lpr`,
or open `TropicalMorphology.lpi` in Lazarus). Pure CPU, finishes in a **few
seconds** — well under the 5-minute budget. No binaries are committed.

## Representative output

```
=== Final mean-squared error on the regression grid ===
  tropical dilation (convex target) : 0.000000
  linear baseline   (convex target) : 1.394320
  tropical erosion  (concave target): 0.000000
```

The **max-plus stack drives the convex-target error to ≈0**, while the
same-width linear layer plateaus at the best-fitting single straight line (≈1.39
MSE — it cannot bend). The **min-plus erode** sibling does the same for the
concave target. Different algebra, different hypothesis class.

---

# TropicalMorphologyConv — the SPATIAL conv `TNNetTropicalConv`

A second program in this directory, **`TropicalMorphologyConv.lpr`**, exercises
the **spatial** sibling **`TNNetTropicalConv`** — the convolutional max-plus /
min-plus layer (subclass of `TNNetConvolutionLinear`). Where the dense
`TNNetTropicalLinear` above mixes a flat vector, `TNNetTropicalConv` slides a
learnable **additive structuring element (SE)** over a 2-D feature map and takes
a per-cell max (dilation) or min (erosion) instead of a sum-of-products:

```
dilation:  y[o] = max_tap ( x[tap] + SE[tap] )
erosion :  y[o] = min_tap ( x[tap] - SE[tap] )   (erode flag = 1)
```

Constructor:
`TNNetTropicalConv.Create(NumFeatures, FeatureSize, InputPadding, Stride, Erode)`.
The whole SE lives in a single weight neuron; the erode flag round-trips via
`FStruct[6]`. Backward is the same hard one-hot arg-max/arg-min subgradient as
`TNNetMaxPool` (`FWinner` caches the winning tap per output cell).

## The morphological task

We build one `12×12` binary glyph (a width-4 cross) and, using a flat `3×3` SE,
compute its classical morphological **dilation** (thicker strokes) and
**erosion** (thinner strokes) as supervised targets. For each target we train:

* `TNNetTropicalConv` (1 feature, `3×3`, pad 1, stride 1, erode flag matched) —
  max-plus / min-plus;
* `TNNetConvolutionLinear` (same `1` feature, `3×3`, pad 1, stride 1) — the
  ordinary linear sum-of-products baseline.

Identical receptive-field size and weight count; only the **algebra** differs.

## Representative output

```
=== Per-pixel MSE / binary reconstruction accuracy ===
  DILATION  tropical conv : MSE 0.000000   acc 100.0%
  DILATION  linear   conv : MSE 0.070509   acc 100.0%
  EROSION   tropical conv : MSE 0.051980   acc  94.4%
  EROSION   linear   conv : MSE 0.074354   acc  97.2%
```

On the **trained objective (per-pixel MSE)** the tropical conv beats the
same-size linear conv on **both** targets. **Dilation** is a decisive, *exact*
win — the learnable additive SE reconstructs the thickened glyph to **MSE 0**
(100% binary accuracy) while the fixed linear filter cannot match the per-pixel
max and plateaus. **Erosion** is also won on MSE; it stops just short of MSE 0
because the **hard per-cell arg-min subgradient** (one-hot, like max-pool)
leaves a few boundary taps un-pressured — a property of the morphological
subgradient, not the linear baseline being a better fit. (The binary-accuracy
column can favor the blurrier linear output purely as a thresholding artifact;
MSE is the quantity actually optimized.) Runs in **~1 second**.

```
cd examples/TropicalMorphology
lazbuild TropicalMorphologyConv.lpi
../../bin/x86_64-linux/bin/TropicalMorphologyConv
```

Coded by Claude (AI).
