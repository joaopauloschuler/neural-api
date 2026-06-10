# KANConv

A small bake-off for [`TNNetKANConv`](../../neural/neuralnetwork.pas), the
**convolutional** Kolmogorov-Arnold layer — the spatial sibling of the dense
[`TNNetKANLayer`](../KANLayer) (KAN, Liu et al. 2024).

An ordinary convolution maps each `FeatureSize x FeatureSize x InputDepth`
receptive-field patch to one output per filter via a **linear dot product**
(sum of `weight * input`). `TNNetKANConv` replaces that linear dot product with a
sum of **learned univariate edge functions**, one per (kernel position, input
channel):

```
y_filter(x,y) = sum over the patch of  phi_p( input_p )
phi_p(t) = sum_{k=0..K} c_{p,k} * T_k(tanh(t))     (Chebyshev basis, degree K)
```

So per output filter there are `FeatureSize*FeatureSize*InputDepth*(K+1)`
trainable coefficients and **no output bias** (the `c_0` constant term plays that
role). The layer is initialized near-linear (degree-1 coefficient random, higher
orders ~0) so an untrained `TNNetKANConv` behaves like a tanh-squashed linear
convolution. It reuses the **exact** Chebyshev forward/backward math of
`TNNetKANLayer`, applied over a sliding receptive field with the usual
padding/stride convolution plumbing inherited from `TNNetConvolutionLinear`.

## The task

A 1-channel `8x8` image is mapped to a 1-channel image where each output pixel is
a **per-window-position nonlinear** function of its 3x3 neighbourhood:

```
y(x,y) = sum_p a_p * f_p( input_p )      (p sweeps the 3x3 window)
```

where each `f_p` is a **distinct** smooth nonlinearity (`sin`, `tanh`, `square`,
`cube`, `abs`, ...) and the `a_p` are fixed mixing weights. A plain linear
convolution can only fit the linearised part of this map; a KAN convolution can
fit each edge's nonlinear shape directly.

Two contenders map a 1-channel image -> 1-channel image (3x3 kernel, pad 1,
stride 1) at a comparable weight budget:

```
(A) TNNetKANConv(1, 3, 1, 1, K=4)                                  (per-edge Chebyshev)
(B) TNNetConvolutionLinear(4, 3, 1, 1) -> TNNetConvolutionLinear(1, 1, 0, 1)
```

Both arms are trained on the **same** data with the **same** schedule, and the
example prints each model's trainable weight count and final validation MSE.

## Sample output

```
KANConv: TNNetKANConv vs a linear conv on a per-edge NONLINEAR 3x3 map.
Task: 8x8 1-channel images. Target pixel = sum_p a_p * f_p(neighbour_p)
over the 3x3 window, where each f_p is a distinct smooth nonlinearity.
A KAN conv learns per-edge Chebyshev functions, matching the task structure.
Train=128  Val=32  Epochs=150  LR=0.010  Degree=4

Final results (lower val-MSE is better):
  TNNetKANConv(K=4)                         weights=  45   val-MSE=0.000583
  TNNetConvolutionLinear (4->1)             weights=  40   val-MSE=0.049819
```

The KAN convolution reaches a far lower error at a comparable parameter count
because its per-edge Chebyshev basis directly matches the per-window-position
nonlinearity of the task.

See also [KANLayer](../KANLayer) for the dense KAN counterpart.

## Build & run

```
cd examples/KANConv
lazbuild KANConv.lpi
../../bin/x86_64-linux/bin/KANConv
```

Pure CPU, no dataset download, synthetic data generated in-code, total runtime
well under a minute.
