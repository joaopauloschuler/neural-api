# Mask2Former universal segmentation (mask-classification set prediction)

The demo for `BuildMask2FormerFromSafeTensors` + `RunMask2FormerSemantic` +
`DecodeMask2FormerSemantic` / `DecodeMask2FormerInstance` /
`DecodeMask2FormerPanoptic` (`neural/neuralpretrained.pas`), the repo's first
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

`DecodeMask2FormerInstance` and `DecodeMask2FormerPanoptic` port
transformers' `Mask2FormerImageProcessor.post_process_instance_segmentation`
and `post_process_panoptic_segmentation`, including the bilinear resize of the
mask logits to the fixed 384x384 processed grid and the resize to a target
size (nearest for instance, bilinear for panoptic). Both return a segment-id
map plus a segment list (`Id`, `LabelId`, `WasFused`, `Score`):

- instance: top-`num_queries` (query, class) scores with the no-object class
  dropped; segment score = class score x mean sigmoid inside the `logit > 0`
  mask; map value `-1` = no instance, ids start at 0. The top-k runs in
  descending score order (transformers uses `topk(sorted=False)`, whose order is
  unspecified).
- panoptic: queries whose best class is the no-object class, or whose score is
  not above `threshold`, are removed; the rest are argmaxed per pixel by
  score x mask probability, and a query whose argmax area is not above
  `overlap_mask_area_threshold` x its own area (weighted probability >=
  `mask_threshold`) is dropped. Labels in the fuse list share one segment id.
  Map value `0` = no segment, ids start at 1; the whole map is `-1` when no
  query passes `threshold`.

**Scope v1**: decoder + heads only — the pixel-decoder
outputs (`mask_features` + the 3 multi-scale memory levels) are fed as
**precomputed inputs** (mirroring how Mask R-CNN v1 took FPN feature maps
directly).

## Build / run

```
cd examples/UniversalSegmentation
lazbuild UniversalSegmentation.lpi --build-mode=Release
# run from the repo root so the fixture path resolves:
../../bin/x86_64-linux/bin/UniversalSegmentation
../../bin/x86_64-linux/bin/UniversalSegmentation --task instance --threshold 0.1
../../bin/x86_64-linux/bin/UniversalSegmentation --task panoptic \
  --threshold 0.2 --mask-threshold 0.15 --overlap-threshold 0.3 --fuse 3
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--task semantic\|instance\|panoptic` | `semantic` | which post-processor runs |
| `--threshold X` | 0.5 | instance: minimum segment score; panoptic: minimum class score |
| `--mask-threshold X` | 0.5 | panoptic: binarizes the score-weighted mask probability |
| `--overlap-threshold X` | 0.8 | panoptic: minimum kept fraction of a query's mask |
| `--fuse a,b` | none | panoptic: label ids whose segments share one id |

The defaults are the transformers defaults. On the pico fixture the class
scores are about 0.34 and the segment scores about 0.29, so the defaults keep
nothing; the commands above show segments.

## Input

The committed pico fixture `tests/fixtures/tiny_mask2former.safetensors`
(+ `..._config.json`) and its precomputed pixel-decoder feature maps in
`tests/fixtures/tiny_mask2former_io.json`, so it runs fully **offline**. Random
pico weights → wiring/throughput smoke (one class tends to win everywhere,
matching the HF reference map), not a trained segmenter. A real run loads
`facebook/mask2former-swin-tiny-*-semantic` the same way.

## Output

Prints the config and the map as a colored ASCII palette, then per-class pixel
counts (semantic) or the segment list (instance/panoptic), and writes
`segmentation.ppm` (a colored map; for instance/panoptic the first palette
color is "no segment" and segment ids cycle over the other seven). Pure
CPU, well under a second on the fixture. Importer parity is asserted to max
|diff| < 1e-4 vs the real `transformers` float64 forward in `TestMask2FormerParity`;
the instance/panoptic decoders are checked against the transformers
post-processors in `TestMask2FormerInstancePanopticReference` and
`TestMask2FormerInstancePanopticFixture`.
