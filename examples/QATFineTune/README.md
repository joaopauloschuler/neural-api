# Quantization-Aware Training (QAT)

A small, self-contained demonstration of the `TNNetFakeQuantize` layer and the
quantization-aware training (QAT) workflow.

## What it shows

Quantizing a network to low bit-widths saves memory and speeds up inference,
but the quantization noise can cost accuracy. **Quantization-aware training**
puts fake-quantization in the forward pass during training so the weights learn
to be robust to that noise and the accuracy is (mostly) recovered.

The example prints three accuracy points on the same tiny task:

| Stage | What it is |
|-------|------------|
| **FLOAT** | a small conv/dense classifier trained in full precision |
| **PTQ**   | the same trained weights, int8-quantized (`TNNet.QuantizeWeightsInt8`) with calibrated low-bit `TNNetFakeQuantize` activations, **no retraining** |
| **QAT**   | the same quantized topology **trained with the fake-quant rounding in the loop** |

A typical (deterministic) run:

```
================ QAT demonstration results ================
Stage                         Test accuracy
-----------------------------------------------------------
FLOAT (full precision)          94.25 %
PTQ   (low-bit acts, no QAT)    88.00 %
QAT   (low-bit acts, fine-tune) 91.75 %
===========================================================
PTQ drop vs float :  -6.25 pts
QAT vs PTQ        :   3.75 pts
```

So low-bit activation PTQ loses ~6 points to quantization, and QAT recovers
most of that gap (~+3.75 pts), landing close to the float baseline. The run is
deterministic (`Fit.MaxThreadNum := 1`, fixed `RandSeed`), so a given build
reproduces these numbers exactly.

## The `TNNetFakeQuantize` layer

`TNNetFakeQuantize` is a per-tensor, symmetric, observer-driven fake
quantizer (a `TNNetIdentity` descendant):

* **Forward**: `out = dequant(quant(x))` with `scale = running_max_abs / qmax`.
  While training and not frozen, an EMA observer updates `running_max_abs` from
  the activations.
* **Backward**: straight-through estimator — gradient passes unchanged inside
  the representable band `[-qmax*scale, qmax*scale]` and is zeroed outside it.
* **`Freeze` / `Unfreeze`**: stop / resume the observer. Freeze before
  inference so the calibrated scale is used.
* Diagnostics: `Scale` and `RunningMaxAbs` properties.
* Constructor `Create(pQMax, pMomentum, pRunningMaxAbs, pFrozen)` selects the
  bit-width via `pQMax` (e.g. `127` = 8-bit, `5` = very low-bit).

The example deliberately uses an aggressive low-bit setting (`cActQMax = 5`):
at 8 bits the calibrated fake-quant is nearly lossless on this small net, so
there would be almost no PTQ gap for QAT to recover. Coarse quantization makes
the post-training drop clearly visible and gives QAT real headroom — the
textbook low-bit QAT story.

## How PTQ is made fair here

The PTQ baseline is *calibrated*, not naive: the fake-quant observers are
populated by running the training data through with **every trainable layer's
`LearningRate` set to 0** (the SGD update `delta = -lr*grad` is exactly zero, so
weights never move) for two epochs, then the observers are frozen and the
weights are int8-quantized. This is the standard "quantize after training, no
fine-tuning" baseline, so the gap to QAT is attributable to the QAT training
rather than to a missing calibration.

## Why QAT trains from scratch (not warm-started)

Warm-starting the QAT net from the float weights pins it to the float net's
sharp, quantization-sensitive minimum, and fine-tuning from there does not beat
PTQ. Training the QAT net from scratch with the fake-quant rounding in the loop
lets it converge to a genuinely quantization-robust optimum, which is what
recovers the accuracy.

## Dataset

Fully synthetic — a low signal-to-noise 4-class image task (each class is a
faint bright patch in a class-specific quadrant buried in strong additive
noise). Nothing is downloaded. The low SNR keeps the float net at a moderate
margin, the regime where quantization costs accuracy and QAT can recover it.

## How to run

```bash
lazbuild --build-mode=Release examples/QATFineTune/QATFineTune.lpi
ulimit -v 3000000
./bin/x86_64-linux/bin/QATFineTune
```

Pure CPU, single-threaded for determinism, fixed `RandSeed`, runs in a few
seconds and well under ~10 MB resident.

Coded by Claude (AI).
