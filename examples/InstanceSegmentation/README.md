# Mask R-CNN instance segmentation (per-object mask)

The demo for `BuildMaskRCNNFromSafeTensors` (`neural/neuralpretrained.pas`), the
repo's first **instance-segmentation** importer (model_type `maskrcnn`:
torchvision `maskrcnn_resnet50_fpn`). Unlike [SegFormer](../SemanticSegmentation)
(one dense class map over the whole image), Mask R-CNN (He et al. 2017) emits a
**separate binary mask per object**: for each proposal box it RoIAligns the
chosen FPN pyramid level, runs 4 convs + a transposed-conv upsample, and emits a
per-class `H×W` mask.

**Scope v1** (matches the importer): the **RPN / anchor generator is skipped**.
The backbone FPN-input feature maps are supplied directly (the ResNet-50
backbone's C4/C5 taps in a real run) and ONE fixed proposal box is fed to
`RunMaskRCNN`, which returns the box-head class logits + box deltas and the mask
head's per-class mask logits. The demo runs that single proposal, picks the
best-scoring (non-background) class, sigmoids that class's mask, **overlays** it
(red channel) on a tiny synthetic image and writes the result.

## Build / run

```
cd examples/InstanceSegmentation
lazbuild InstanceSegmentation.lpi --build-mode=Release
# run from the repo root so the fixture path resolves:
../../bin/x86_64-linux/bin/InstanceSegmentation
```

## Input

The committed pico parity fixture `tests/fixtures/tiny_maskrcnn.safetensors`
(+ `..._config.json`) and the matching feature maps + proposal box pinned in
`tests/fixtures/tiny_maskrcnn_ref.json`, so it runs fully **offline** and
reproduces the test numbers. The fixture has **random** weights, so the mask is
not a meaningful object — the demo's job is to exercise the full FPN + RoIAlign +
box/mask-head pipeline end to end. A real run loads torchvision
`maskrcnn_resnet50_fpn` the same way (the math is identical, only the checkpoint
and feature maps differ).

## Output

Prints the config, the proposal box, the per-class logits and the best class,
then writes `instance_segmentation.ppm` and self-reports (asserts no NaN/Inf and
that the overlaid mask covers a sane fraction of pixels). Pure CPU, well under a
second on the fixture. Importer parity is asserted to max |diff| < 1e-4 vs a
numpy float64 oracle in `TestMaskRCNNParity`.
