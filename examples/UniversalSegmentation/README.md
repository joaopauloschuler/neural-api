# Mask2Former universal segmentation (mask-classification set prediction)

The demo for `BuildMask2FormerFromSafeTensors` + `RunMask2FormerSemantic` +
`DecodeMask2FormerSemantic` (`neural/neuralpretrained.pas`), the repo's first
**universal-segmentation** importer (model_type `mask2former`:
`facebook/mask2former-swin-tiny-*-semantic`). Distinct from both
[SegFormer](../SemanticSegmentation) (per-pixel argmax) and
[Mask R-CNN](../InstanceSegmentation) (RoIAlign on region proposals): Mask2Former
(Cheng et al. 2022, [arXiv:2112.01527](https://arxiv.org/abs/2112.01527)) does
**mask-classification set prediction** — a fixed set of learned object queries,
each predicting ONE binary mask + a class distribution, unifying
semantic/instance/panoptic in a single head, with no proposals and no per-pixel
classifier.

The conceptual core is **masked attention**: each decoder layer's
cross-attention is restricted to the foreground of the mask predicted by the
previous layer. Because that feedback loop is dynamic, the decoder is built as
one sub-net per layer and driven layer-by-layer by `RunMask2FormerSemantic`;
`DecodeMask2FormerSemantic` then folds the per-query class + mask logits into a
per-pixel semantic label map (softmax classes, drop the no-object slot, sigmoid
masks, class-weighted argmax).

**Scope v1**: semantic inference, decoder + heads only — the pixel-decoder
outputs (`mask_features` + the 3 multi-scale memory levels) are fed as
**precomputed inputs** (mirroring how Mask R-CNN v1 took FPN feature maps
directly).

## Build / run

```
cd examples/UniversalSegmentation
lazbuild UniversalSegmentation.lpi --build-mode=Release
# run from the repo root so the fixture path resolves:
../../bin/x86_64-linux/bin/UniversalSegmentation
```

## Input

The committed pico fixture `tests/fixtures/tiny_mask2former.safetensors`
(+ `..._config.json`) and its precomputed pixel-decoder feature maps in
`tests/fixtures/tiny_mask2former_io.json`, so it runs fully **offline**. Random
pico weights → wiring/throughput smoke (one class tends to win everywhere,
matching the HF reference map), not a trained segmenter. A real run loads
`facebook/mask2former-swin-tiny-*-semantic` the same way.

## Output

Prints the config and the semantic label map as a colored ASCII palette plus
per-class pixel counts, and writes `segmentation.ppm` (a colored label map). Pure
CPU, well under a second on the fixture. Importer parity is asserted to max
|diff| < 1e-4 vs the real `transformers` float64 forward in `TestMask2FormerParity`.
