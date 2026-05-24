# FiLMConditioning

End-to-end demo of **`TNNetFiLM`** — Feature-wise Linear Modulation (Perez et
al. 2018, *FiLM: Visual Reasoning with a General Conditioning Layer*,
<https://arxiv.org/abs/1709.07871>).

## The idea

FiLM applies a **per-channel affine modulation** to a feature map:

```
Out[x,y,c] = gamma[c] * input0[x,y,c] + beta[c]
```

The crucial point — and what stops this being a `TNNetChannelMul ->
TNNetChannelBias` duplicate — is that **`gamma` and `beta` are produced by a
SEPARATE conditioning branch, not by the layer's own trainable weights**.
`TNNetFiLM` is therefore a **parameter-free, two-input** layer:

```
input0 = feature map,         shape (SizeX, SizeY, Depth)
input1 = conditioning vector, shape (1, 1, 2*Depth)
         channels [0 .. Depth-1]       = gamma (per-channel scale)
         channels [Depth .. 2*Depth-1] = beta  (per-channel shift)
```

`gamma`/`beta` broadcast over the spatial dims. Backward is parameter-free and
routes error to **both** inputs:

```
dL/dinput0[x,y,c] = gamma[c] * dOut[x,y,c]
dL/dgamma[c]      = sum_{x,y} input0[x,y,c] * dOut[x,y,c]   -> input1[c]
dL/dbeta[c]       = sum_{x,y} dOut[x,y,c]                   -> input1[Depth+c]
```

Wiring is the usual multi-input style:

```pascal
NN.AddLayer(TNNetFiLM.Create([featureLayer, condLayer]));
```

## What this demo does

A tiny synthetic **conditional transform** task. A fixed feature vector of
depth 3 is modulated by one of `K = 4` per-class affine transforms; a class id
(one-hot) selects which:

```
target_k[c] = TrueGamma[k,c] * feature[c] + TrueBeta[k,c]
```

The one-hot id is fed through a `TNNetFullConnectLinear` conditioning branch
that must **learn** to emit the right `gamma|beta` vector per class; `TNNetFiLM`
then applies it. A single shared feature path thus produces `K` different
outputs purely via conditioning.

```
TNNetInput(1,1,Depth)   --------------------------\        (input0: feature)
TNNetInput(1,1,K)  -> FullConnectLinear(2*Depth)   -> Reshape(1,1,2*Depth)
                                                    \-> TNNetFiLM([feature, cond])
```

Only the conditioning FC trains (FiLM has no parameters). A falling MSE proves
error back-propagates through FiLM into the conditioning branch. The demo also
checks the **identity invariant**: feeding `gamma = 1, beta = 0` must reproduce
the feature unchanged.

## Build & run

```
lazbuild FiLMConditioning.lpi
../../bin/x86_64-linux/bin/FiLMConditioning
```

Pure CPU, single thread, runs in well under a second.

## Observed output

```
TNNetFiLM conditioning demo
  feature depth = 3, classes = 4
  feature vector = [1.00, -2.00, 0.50]

Training the conditioning FC (FiLM has no parameters of its own):
  epoch     1   mean MSE = 3.494104
  epoch  1000   mean MSE = 0.000000
  epoch  2000   mean MSE = 0.000000
  epoch  3000   mean MSE = 0.000000
  epoch  4000   mean MSE = 0.000000

Per-class result after training (target vs FiLM output):
------------------------------------------------------------
  class 0  target [ -1.000 -3.300 -1.600 ]   got [ -1.000 -3.300 -1.600 ]
  class 1  target [  0.500 -3.300 -0.350 ]   got [  0.500 -3.300 -0.350 ]
  class 2  target [  2.000 -3.300  0.900 ]   got [  2.000 -3.300  0.900 ]
  class 3  target [  3.500 -3.300  2.150 ]   got [  3.500 -3.300  2.150 ]
------------------------------------------------------------

Identity invariant (gamma=1, beta=0 -> output == feature):
  max |out - feature| = 0.000E+00 -> HELD
```

The conditioning FC learns each class's transform to numerical zero MSE, and
the identity invariant holds exactly — confirming both the forward modulation
and the gradient flow into the conditioning branch are correct.
