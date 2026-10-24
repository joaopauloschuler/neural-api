# ObjectDetection: DETR, RT-DETR and D-FINE end-to-end detection

The demo for the three DETR-family object-detection importers in
`neural/neuralpretrained.pas`:

- **DETR** (`BuildDetrFromSafeTensors`, `facebook/detr-resnet-50`), the default
  mode;
- **RT-DETR** (`BuildRtDetrFromSafeTensors`, `PekingU/rtdetr_r18vd` ..
  `rtdetr_r101vd`) and **RT-DETRv2** (the same entry point, model_type
  `rt_detr_v2`, `PekingU/rtdetr_v2_r18vd` .. `rtdetr_v2_r101vd`), with
  `--rtdetr`;
- **D-FINE** (`BuildDFineFromSafeTensors`, `ustc-community/dfine-nano-coco` ..
  `dfine-xlarge-coco` and the `-obj365` / `-obj2coco` variants), with
  `--dfine`.

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

**RT-DETRv2** changes only the decoder's deformable cross-attention: the
sampling offsets are scaled by `decoder_offset_scale` (RT-DETR fixes it at 0.5),
and `decoder_method` `"discrete"` reads the nearest feature-map cell instead of
the bilinear sample. `ReadRtDetrConfigFromJSONFile` reads both fields.

`DecodeRtDetrDetections` ports transformers `post_process_object_detection`:
sigmoid of every (query, class) logit, the `num_queries` best pairs, score above
the threshold, best first, **no NMS**. The example tags each box with the class
name from `config.json` (`id2label`) and its score.

**D-FINE** keeps the RT-DETR design and changes three parts:

- an **HGNetV2** backbone (`BuildHGNetV2Backbone`);
- a neck of `RepNCSPELAN4` fusion blocks and `SCDown` (pointwise + strided
  depthwise conv) downsampling;
- a decoder whose layers mix the self-attention output and the deformable
  cross-attention output through a learned sigmoid gate, and whose boxes come
  from **Fine-grained Distribution Refinement**: every layer predicts, for each
  of the four box sides, a distribution over `max_num_bins + 1` bins; the
  softmax of the bins times a fixed non-uniform weighting vector gives the side
  distance, and each layer adds its bin logits to the previous layer's. A small
  location-quality head (the top-k bin probabilities of each side and their
  mean) is added to the class logits. The net is built up to the `eval_idx`
  decoder layer; the layers after it do not change the inference outputs and
  are not built.

D-FINE uses the same post-process as RT-DETR (`DecodeRtDetrDetections`).

All modes **self-report**: no NaN/Inf in the output and every decoded box
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
$B --rtdetr --image photo.jpg rtdetr_v2_r18vd/model.safetensors   # RT-DETRv2
$B --dfine                                  # D-FINE, committed pico fixture
$B --dfine --image photo.jpg dfine-nano-coco/model.safetensors [config.json] [threshold]
```

- `--image FILE` (RT-DETR and D-FINE) resizes the photo to the `N x N` net
  input as `RTDetrImageProcessor` does (PIL bilinear, then `/ 255`, no
  mean/std), maps the boxes back to the photo and writes `<name>_rtdetr.png` or
  `<name>_dfine.png`. `--size N` changes the input size (default 640, a
  multiple of 32). A D-FINE config that sets `anchor_image_size` needs `--size`
  equal to it.
- The RT-DETR and D-FINE configs default to `config.json` beside the weights;
  the score threshold defaults to 0.5 (transformers' default).

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
  (`tests/fixtures/tiny_detr.*`, `tests/fixtures/tiny_rtdetr.*`,
  `tests/fixtures/tiny_dfine.*`, used by the parity tests), fully offline, on
  the fixture's pinned image. Random weights give no real objects: the demo
  exercises the decode + draw pipeline. The DETR fixture threshold defaults to
  `0` so every query is drawn; the RT-DETR and D-FINE fixture thresholds
  default to `0.75`.
- Importer parity: DETR < 1e-4 vs transformers float64
  (`TestDetrObjectDetectionParity`); RT-DETR logits, boxes and post-processed
  detections < 1e-4 vs transformers float64 on a basic-block (r18vd layout) and
  a bottleneck (r50vd layout, 4 feature levels) pico model
  (`TestRtDetrParity`, `TestRtDetrR50Parity`; generator
  `tools/make_pico_rtdetr_fixture.py`).
- RT-DETRv2 parity: logits, boxes and post-processed detections < 1e-4 vs
  transformers float64 on two pico models with `decoder_offset_scale` 0.7, one
  with bilinear and one with discrete sampling (`TestRtDetrV2Parity`,
  `TestRtDetrV2OpenCL`).
- D-FINE parity: logits, boxes and post-processed detections < 1e-4 vs
  transformers float64 on two pico models, one with the same number of
  sampling points on every level and one with per-level counts `[3, 6, 3]`,
  learnable affine blocks and `eval_idx` before the last decoder layer
  (`TestDFineParity`, `TestDFineOpenCL`).
- RT-DETR status: planned (coded); not yet run by the user with a real
  checkpoint. The example output above is an agent run of `PekingU/rtdetr_r18vd`.
- RT-DETRv2 status: planned (coded); no real `PekingU/rtdetr_v2_*` checkpoint
  has been run yet, by the user or by an agent.
- D-FINE status: planned (coded); the `ustc-community/dfine-*` checkpoints have
  not been run yet, by the user or by an agent. Both `decoder_method` values
  (`"default"` and `"discrete"`) are supported.
- DEIMv2 (transformers model_type `deimv2`) is not supported: its decoder uses
  RMSNorm and a SwiGLU FFN and its encoder differs from D-FINE's.
- Pure CPU, < 1 s on the fixtures.

Coded by Claude (AI).
