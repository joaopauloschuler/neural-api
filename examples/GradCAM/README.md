# Grad-CAM class-discriminative localisation

This example demonstrates **`TNNet.GradCAMReport`**, an implementation of
Grad-CAM (Gradient-weighted Class Activation Mapping, Selvaraju et al., 2017).
Grad-CAM produces a *coarse, class-discriminative* heatmap by weighting a
convolution layer's feature maps with the gradient of the predicted logit and
keeping only the positive evidence:

```
alpha_k = mean_xy( d y_c / d A^k_xy )          (global-average-pooled gradient)
L_xy    = ReLU( sum_k alpha_k * A^k_xy )        (weighted, rectified feature sum)
```

It complements the fine, input-pixel attribution of
[`SaliencyReport`](../SaliencyReport/): saliency tells you *which pixels* the
network reacted to, Grad-CAM tells you *which region* of the conv feature map
drove the class decision. This is the resolution/locality trade-off: Grad-CAM
is coarse (it lives at a conv layer's feature-map grid, here 8x8) but
class-discriminative; saliency is fine (full input resolution) but noisier and
not inherently class-localised.

## The task

A tiny CNN (forked from the SaliencyReport demo) learns to separate two classes
of 8x8x2 images:

- **class 0:** a bright 3x3 blob in the **top-left** of channel 0
- **class 1:** a bright 3x3 blob in the **bottom-right** of channel 1

After a short training loop (well under a minute on CPU, no memory blow-up) the
program explains one **class-0** prediction two ways on the **same** sample:

1. `GradCAMReport`   — coarse Grad-CAM localisation at the deepest spatial conv layer
2. `SaliencyReport`  — fine pixel-space saliency / SmoothGrad / Integrated Gradients

Both are printed as ASCII heatmaps so the example is fully self-contained (no
image files).

## Built-in self-check

The example parses the coarse Grad-CAM peak cell out of the report, and asserts
it lands inside the class-0 (top-left) region of the input. It prints
`PASS` / `FAIL` and `Halt(1)`s on failure, so it doubles as a regression gate.
`GradCAMReport` is forward-only: it runs `Compute` + a one-hot `Backpropagate`
to read activations and gradients but never calls `UpdateWeights`, so the
trained weights are left untouched.

## Running

```
lazbuild examples/GradCAM/GradCAM.lpi
./bin/<arch>/bin/GradCAM
```

Expected tail of the output:

```
Self-check: Grad-CAM coarse peak cell = (0,1)
PASS: Grad-CAM peak falls inside the class-0 (top-left) region.
```

## API

```pascal
class function TNNet.GradCAMReport(
  NN: TNNet;                  // trained classifier
  Probe: TNNetVolume;         // input sample (already shaped for the net)
  ConvLayerIdx: integer = -1; // target conv layer (-1 = deepest spatial conv)
  ForcedClass: integer = -1   // class to attribute (-1 = predicted argmax)
): string;                    // ASCII report (coarse map + nearest-upsampled overlay)
```
