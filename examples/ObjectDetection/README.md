# ObjectDetection: DETR and RT-DETR end-to-end detection

The demo for the two DETR-family object-detection importers in
`neural/neuralpretrained.pas`:

- **DETR** (`BuildDetrFromSafeTensors`, `facebook/detr-resnet-50`), the default
  mode;
- **RT-DETR** (`BuildRtDetrFromSafeTensors`, `PekingU/rtdetr_r18vd` ..
  `rtdetr_r101vd`), with `--rtdetr`.

Each mode runs a full forward pass on one image, decodes the per-query
predictions, draws the surviving boxes and writes the annotated image.

## What it does

**DETR** is a ResNet backbone + a transformer **encoder-decoder** with learned
object queries, a 2-D sinusoidal spatial position embedding, a sigmoid-`cxcywh`
box head and a class head (**inference only, no Hungarian matcher**). After the
forward pass, `DecodeDetrDetections` softmaxes the class logits, drops the
"no-object" slot, thresholds on confidence, and converts each normalized `cxcywh`
box to pixel `xyxy`. The example writes `object_detection.ppm`.

**RT-DETR** is the real-time redesign of DETR:

- a **ResNet-vd** backbone (a deep stem of three 3x3 convs, and an average-pool
  before the 1x1 shortcut conv of each downsampling block);
- the **hybrid encoder**: one AIFI self-attention layer on the stride-32 map
  (2-D sine position embedding on queries and keys) and a CNN neck (top-down
  FPN then bottom-up PAN of CSP-RepVGG blocks, each RepVGG block folded into one
  3x3 conv at load time);
- **query selection**: anchors per feature-map cell (invalid ones masked), the
  encoder class and box heads, then the `num_queries` best cells
  (`TNNetTopKGatherTokens`) seed the decoder;
- a decoder of self-attention + **multi-scale deformable cross-attention**
  (`TNNetMultiScaleDeformableAttention`) with **iterative box refinement**.

`DecodeRtDetrDetections` ports transformers `post_process_object_detection`:
sigmoid of every (query, class) logit, the `num_queries` best pairs, score above
the threshold, best first, **no NMS**. The example tags each box with the class
name from `config.json` (`id2label`) and its score.

Both modes **self-report**: no NaN/Inf in the output and every decoded box
within the sigmoid `[0,1]` range before clamping to the canvas.

## Running

```
cd examples/ObjectDetection
lazbuild -B ObjectDetection.lpi
B=../../bin/x86_64-linux/bin/ObjectDetection
$B                                          # DETR, committed pico fixture
$B model.safetensors config.json [threshold]
$B --rtdetr                                 # RT-DETR, committed pico fixture
$B --rtdetr --image photo.jpg rtdetr_r18vd/model.safetensors [config.json] [threshold]
```

- `--image FILE` (RT-DETR) resizes the photo to the `N x N` net input as
  `RTDetrImageProcessor` does (PIL bilinear, then `/ 255`, no mean/std), maps the
  boxes back to the photo and writes `<name>_rtdetr.png`. `--size N` changes the
  input size (default 640, a multiple of 32).
- The RT-DETR config defaults to `config.json` beside the weights; the score
  threshold defaults to 0.5 (transformers' default).

Example: `PekingU/rtdetr_r18vd` on the COCO "two cats" photo
(`000000039769.jpg`, 640 x 480):

```
  5 detection(s) with score > 0.50:
    cat          score 0.9502  xyxy=(345,24,640,372)
    cat          score 0.9481  xyxy=(10,56,317,473)
    remote       score 0.9397  xyxy=(40,73,176,117)
    sofa         score 0.8090  xyxy=(0,1,640,476)
    remote       score 0.7905  xyxy=(333,77,370,188)
```

The forward pass took 4.2 s on one CPU core (AVX2 build); the process peaked at
about 1.9 GB of RAM at 640 x 640.

## Notes

- With no model argument each mode loads its committed **random** pico fixture
  (`tests/fixtures/tiny_detr.*`, `tests/fixtures/tiny_rtdetr.*`, used by the
  parity tests), fully offline, on the fixture's pinned image. Random weights
  give no real objects: the demo exercises the decode + draw pipeline. The DETR
  fixture threshold defaults to `0` so every query is drawn; the RT-DETR fixture
  threshold defaults to `0.75`.
- Importer parity: DETR < 1e-4 vs transformers float64
  (`TestDetrObjectDetectionParity`); RT-DETR logits, boxes and post-processed
  detections < 1e-4 vs transformers float64 on a basic-block (r18vd layout) and
  a bottleneck (r50vd layout, 4 feature levels) pico model
  (`TestRtDetrParity`, `TestRtDetrR50Parity`; generator
  `tools/make_pico_rtdetr_fixture.py`).
- RT-DETR status: planned (coded); not yet run by the user with a real
  checkpoint. The example output above is an agent run of `PekingU/rtdetr_r18vd`.
- Pure CPU, < 1 s on the fixtures.

Coded by Claude (AI).
