# Quantization-Aware Training (QAT)

A small, self-contained demonstration of the `TNNetFakeQuantize` layer and the
quantization-aware training (QAT) workflow.

## What it shows

Quantizing a trained network to int8 saves memory and speeds up inference, but
the quantization noise can cost accuracy. **Quantization-aware training**
inserts fake-quantization into the forward pass during a short fine-tune so the
weights learn to be robust to that noise and the accuracy is (mostly)
recovered.

The example prints three accuracy points on the same tiny task:

| Stage | What it is |
|-------|------------|
| **FLOAT** | a small conv/dense classifier trained in full precision |
| **PTQ**   | the same trained weights int8-quantized (`TNNet.QuantizeWeightsInt8`) with calibrated `TNNetFakeQuantize` activations, **no retraining** |
| **QAT**   | the quantized topology **fine-tuned** with the fake-quant rounding in the loop |

A typical run:

```
================ QAT demonstration results ================
Stage                         Test accuracy
-----------------------------------------------------------
FLOAT (full precision)         91.33 %
PTQ   (int8 weights, no QAT)   89.00 %
QAT   (fake-quant fine-tune)   90.67 %
===========================================================
PTQ drop vs float :  -2.33 pts
QAT vs PTQ        :   1.67 pts
```

So PTQ loses ~2.3 points to quantization and QAT recovers ~1.7 of them, landing
back near the float baseline. (Exact numbers depend on the build; the seed is
fixed so a given build reproduces.)

## The `TNNetFakeQuantize` layer

`TNNetFakeQuantize` is a per-tensor, symmetric, observer-driven fake
quantizer (a `TNNetIdentity` descendant):

* **Forward**: `out = dequant(quant(x))` with `scale = running_max_abs / qmax`
  (`qmax = 127` by default). While training and not frozen, an EMA observer
  updates `running_max_abs` from the activations.
* **Backward**: straight-through estimator — gradient passes unchanged inside
  the representable band `[-qmax*scale, qmax*scale]` and is zeroed outside it.
* **`Freeze` / `Unfreeze`**: stop / resume the observer. Freeze before
  inference so the calibrated scale is used.
* Diagnostics: `Scale` and `RunningMaxAbs` properties.

## How PTQ is made fair here

The PTQ baseline is *calibrated*, not naive: the fake-quant observers are
populated by running the training data through with **every trainable layer's
`LearningRate` set to 0** (the SGD update `delta = -lr*grad` is exactly zero, so
weights never move) for two epochs, then the observers are frozen and the
weights are int8-quantized. This is the standard "quantize after training, no
fine-tuning" baseline, so the gap to QAT is attributable to fine-tuning rather
than to a missing calibration.

## Dataset

Fully synthetic — a low signal-to-noise 4-class image task (each class is a
faint bright patch in a class-specific quadrant buried in strong additive
noise). Nothing is downloaded. The low SNR keeps the float net at a moderate
margin, which is exactly the regime where int8 quantization costs accuracy and
QAT can recover it.

## How to run

```bash
lazbuild --build-mode=Release examples/QATFineTune/QATFineTune.lpi
ulimit -v 3000000
./bin/x86_64-linux/bin/QATFineTune
```

Pure CPU, fixed `RandSeed`, runs in about 11 seconds and well under ~10 MB
resident.

Coded by Claude (AI).
