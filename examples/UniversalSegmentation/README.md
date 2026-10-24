# Mask2Former and EoMT universal segmentation (mask-classification set prediction)

Two routes: `--model mask2former` (the default, below) and `--model eomt`
([EoMT](#eomt-route---model-eomt)).

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
per-pixel semantic label map (bilinear resize of the mask logits to the
384x384 processed grid, softmax classes, drop the no-object slot, sigmoid
masks, class-weighted sum, bilinear resize to the target size, argmax), as
transformers' `post_process_semantic_segmentation`.

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
| `--fuse a,b` | none | panoptic: label ids whose segments share one id (EoMT: `stuff_classes`) |
| `--model mask2former\|eomt` | `mask2former` | which importer runs |
| `--model-dir DIR` | none | eomt: folder with `config.json`, `model.safetensors`, `preprocessor_config.json` |
| `--image FILE` | none | eomt: the image to segment (PNG, JPEG, BMP, ...) |

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

## EoMT route (`--model eomt`)

Status: **planned (coded)**. Only the pico fixtures in `tests/fixtures` have
been run; no real EoMT checkpoint has been loaded yet.

EoMT (Kerssies et al., CVPR 2025, [arXiv:2503.19108](https://arxiv.org/abs/2503.19108),
transformers `EomtForUniversalSegmentation`, model_type `eomt`) is an
encoder-only mask transformer: a DINOv2-with-registers ViT runs from pixels,
learned query tokens join the last blocks, and a class head plus a mask head
give the same per-query class + mask logits as Mask2Former. The route runs,
all in `neural/neuralpretrained.pas`:

1. `ReadEoMTProcessorConfigFromJSONFile` + `EoMTPreprocessImage`: a port of
   transformers' `EomtImageProcessorPil` (the PIL backend, as the SAM and DPT
   helpers). It resizes with PIL bilinear uint8 rounding
   (`get_size_with_aspect_ratio`, short side to `size.shortest_edge`, long
   side capped by `size.longest_edge`), then either pads bottom/right to the
   square input (`do_pad`, panoptic and instance checkpoints) or cuts the
   long side into overlapping square windows (`do_split_image`, semantic
   checkpoints), then rescales by `rescale_factor` and normalizes with the
   ImageNet mean/std. Padding happens before the normalization, so padded
   pixels hold `-mean/std`.
2. `BuildEoMTFromSafeTensors`, then one `NN.Compute` per window and
   `SplitEoMTOutput`.
3. The post-processors, at the original image size:
   - `DecodeEoMTSemantic` (`EoMTSemanticScores` gives the per-class scores):
     class-weighted sigmoid masks per window; split windows are averaged per
     pixel (`merge_image_patches`). transformers does not unpad the padded
     semantic map, and this port does not either.
   - `DecodeEoMTInstance`: no top-k. Every query whose best class (softmax,
     no-object dropped) x mean mask probability is at least `--threshold`
     (default 0.8) is kept, in query order. The masks are unpadded and resized
     to the original size first.
   - `DecodeEoMTPanoptic`: `compute_segments` with stuff classes (`--fuse`).
     Map value `-1` = no segment, ids start at 0.

The task defaults to the checkpoint's: semantic when
`preprocessor_config.json` sets `do_split_image` or the folder name contains
`semantic`, instance when it contains `instance`, otherwise panoptic.
`--task` overrides it; instance and panoptic need a `do_pad` checkpoint.

```
# a pico folder from the committed fixtures (run from the repo root):
mkdir -p /tmp/coco_panoptic_eomt_pico
cp tests/fixtures/tiny_eomt.safetensors /tmp/coco_panoptic_eomt_pico/model.safetensors
cp tests/fixtures/tiny_eomt_config.json /tmp/coco_panoptic_eomt_pico/config.json
cp tests/fixtures/tiny_eomt_preprocessor_config.json \
  /tmp/coco_panoptic_eomt_pico/preprocessor_config.json
bin/x86_64-linux/bin/UniversalSegmentation --model eomt \
  --model-dir /tmp/coco_panoptic_eomt_pico --image photo.png
# a real checkpoint folder:
bin/x86_64-linux/bin/UniversalSegmentation --model eomt \
  --model-dir coco_panoptic_eomt_large_640 --image photo.jpg
```

The output is the same as the Mask2Former route: the map as glyphs (only when
the image is at most 80 pixels wide), per-class counts or the segment list,
and `segmentation.ppm` at the original image size.

Checkpoints (the tue-mps organisation on the Hugging Face hub; the names were
not checked against the hub from this machine): `coco_panoptic_eomt_small_640`,
`coco_panoptic_eomt_base_640`, `coco_panoptic_eomt_large_640`,
`coco_panoptic_eomt_giant_640`, `coco_panoptic_eomt_large_1280`,
`coco_panoptic_eomt_giant_1280`, `ade20k_panoptic_eomt_large_640`,
`ade20k_panoptic_eomt_large_1280`, `ade20k_semantic_eomt_large_512`,
`cityscapes_semantic_eomt_large_1024`, `coco_instance_eomt_large_640`,
`coco_instance_eomt_large_1280`. The DINOv3-based `eomt_dinov3` model type is
not supported.

Parity: `TestEoMTParity` pins the forward (max |diff| about 2e-6 against
transformers in float64). `TestEoMTPreprocessPadParity` and
`TestEoMTPreprocessSplitParity` pin `pixel_values` against
`EomtImageProcessorPil` (max |diff| 4.8e-7), and
`TestEoMTPostprocessPadParity` / `TestEoMTPostprocessSplitParity` pin the
semantic scores (max |diff| 7.2e-7) and the semantic, instance and panoptic
maps and segments (0 mismatched pixels) on a 40x28 test image
(`tools/make_pico_eomt_fixture.py`).
