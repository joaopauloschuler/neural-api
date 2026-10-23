# YoloDetect — YOLOv8 and YOLO-World single-shot object detection

The demo for `BuildYoloFromSafeTensors` (`neural/neuralpretrained.pas`), the
repo's **YOLOv8** single-shot object-detection importer (ultralytics `yolov8n`: a
CSP/C2f backbone + SPPF + PANet feature-pyramid neck + a decoupled per-cell **DFL**
detect head over 3 strides, anchor-free, no transformer, inference only). It runs a
full YOLOv8 forward pass on **one** image, decodes the raw head, draws the surviving
boxes and writes an annotated image.

```
image (ImageSize x ImageSize x NumChannels)
  -> CSP/C2f backbone + SPPF + PANet neck
  -> decoupled per-cell DFL detect head (3 strides, anchor-free)
  -> DecodeYoloDetections: sigmoid class logits, DFL-decode each box side
     (softmax the reg_max bins -> expected ltrb distance -> xyxy pixels),
     greedy IoU NMS
  -> draw colored box outlines + "class score" labels -> PPM + (class, score, box) list
```

## The importer

`BuildYoloFromSafeTensors(ModelFile, Config, pTrainable, ConfigFile)` returns a
`TNNet` and fills `TYoloConfig` (printed by `YoloConfigToString`; `ImageSize`,
`NumChannels`, `NumClasses`, `RegMax`, `Strides[0..2]`).
`DecodeYoloDetections(Output, Config, ScoreThresh, IoUThresh)` turns the raw head
volume into a `TYoloDetectionArray` (per detection: `ClassId`, `Score`, `X1/Y1/X2/Y2`).
`YoloClassLabel(Config, ClassId)` gives the label of a class: its name from the
config, else the COCO-80 name when an 80-class yolov8 config has no names, else
the id as text.

## Build / run

With **no arguments** it loads the committed pico fixture
(`tests/fixtures/tiny_yolo.safetensors` + `tiny_yolo_config.json`), a tiny random
yolov8 used by the parity tests, so the demo runs with **no download, fully
offline**. If `tiny_yolo_io.json` is present alongside the model its pinned flat
`(c, y, x)` input is used (matching the parity test); otherwise a synthetic
gradient image is generated.

```
cd examples/YoloDetect
lazbuild -B YoloDetect.lpi
../../bin/x86_64-linux/bin/YoloDetect
../../bin/x86_64-linux/bin/YoloDetect [--image FILE] [model.safetensors] [config.json] [score_threshold] [iou_threshold]
../../bin/x86_64-linux/bin/YoloDetect --prompts "person,bus,dog" --clip CLIP_DIR --image FILE world.safetensors [config.json]
```

`--image FILE` letterboxes a real photo into the net input (`LetterboxYoloImage`,
ultralytics `LetterBox`), maps the boxes back to the photo
(`ScaleYoloBoxesToImage`) and writes `<name>_yolo.png`. Without `--image` the
demo runs on the pinned fixture input (or a synthetic gradient) and writes
`yolo_detect.ppm`. The config defaults to `config.json` next to the model.

Defaults: `score_threshold 0.5`, `iou_threshold 0.45`. The low score threshold on
the random fixture is deliberate so the draw path is exercised; on a **real**
ultralytics yolov8 checkpoint use ~0.25 (the ultralytics default).

## Pico fixture vs real checkpoint

The pico fixture has **random** weights, so its "detections" are not meaningful
objects — the demo's job is to exercise the full DFL decode + NMS + draw pipeline
end to end and **self-report**: it asserts no NaN/Inf in the network output and
that every decoded box has finite, canvas-clamped pixel coordinates. A real
pretrained checkpoint produces real detections through the exact same path.

## Output

Prints the config (image size, class count, reg_max, strides), the output-OK
check, the decoded `(class, score, xyxy)` list and the box self-check, then writes
the annotated image to `yolo_detect.ppm` (a P6 PPM, min-max stretched so boxes are
visible). Each box carries a tag in its colour with the class label and score
(e.g. `traffic light 0.53`), drawn with `DrawBitmapText`
(`neural/neuralimageresize.pas`).

## Class names

`ReadYoloConfigFromJSONFile` reads the class names into `TYoloConfig.ClassNames`
from `names` (a list, `["person", "bicycle", ...]`, or an ultralytics-style map,
`{"0": "person", ...}`), else from the HF `id2label` map; the count must equal
`num_classes`. With no names, an 80-class yolov8 config is labelled with the
built-in COCO-80 names (ultralytics `coco.yaml`), and any other config with the
class id. The pico fixture has no names and 3 classes, so its tags show the id
(`2 0.53`). The 5x7 font has `0-9`, `A-Z` (lower case is drawn as upper case),
space and `. , : - _ % # / ( ) ! ? + = '`; `& [ ] { } < > " ; |` are drawn
as look-alikes, and any other character, including each UTF-8 character, as one
`?`.

## Open-vocabulary detection: YOLO-World v2 (`--prompts`)

`--prompts "person,bus,dog"` switches to `BuildYoloWorldFromSafeTensors`, the
YOLO-World v2 importer (ultralytics `yolov8s-worldv2` / `yolov8m-worldv2`, Cheng
et al. 2024, [*YOLO-World*](https://arxiv.org/abs/2401.17270)). The class labels
are the prompt strings; there is no fixed class list.

1. `--clip DIR` is a Hugging Face CLIP folder (`openai/clip-vit-base-patch32`,
   the text encoder YOLO-World v2 was trained with): `model.safetensors` or
   `pytorch_model.bin`, `config.json`, and `tokenizer.json` (or `vocab.json` +
   `merges.txt`). The demo loads only the text tower (`BuildClipTextTower`) and
   encodes each prompt once with `ClipEncodeText`.
2. `BuildYoloWorld` folds the prompt embeddings into the detector's weights:
   - each neck C2fAttn's max-sigmoid attention becomes a 1x1 conv to
     (prompt, head) logits, a max over prompts (`TNNetMaxOut`) and a sigmoid
     gate;
   - the BatchNorm contrastive class head becomes one 1x1 conv with one output
     per prompt.
3. The net then has the layer layout of a YOLOv8 whose class count is the
   prompt count. `DecodeYoloDetections` decodes it unchanged.

The checkpoint must be the ultralytics state_dict as safetensors with the
`model.<i>....` key names. For `yolov8s-worldv2` the `config.json` is:

```
{"model_type": "yolo_world", "widths": [32, 64, 128, 256, 512], "depths": [1, 2],
 "num_classes": 80, "text_embed_dim": 512, "image_size": 640}
```

`num_classes` is the training vocabulary (80); it only sets the class-branch
width. The attention head counts default to the ultralytics rule (32 channels
per head); set `attn_heads` (4 entries) to override it. YOLO-World v1
(`ImagePoolingAttn`, L2 contrastive head) is rejected with an error.

Without `--clip` and without a model argument, `--prompts` runs the pico
fixture `tests/fixtures/tiny_yolo_world.*` with its pinned **random** text
rows (at most 3 prompts). Those prompts are labels only.

### Speed and accuracy vs OWL-ViT (examples/OpenVocabDetection)

| | YOLO-World v2 (`--prompts`) | OWL-ViT ([OpenVocabDetection](../OpenVocabDetection)) |
| --- | --- | --- |
| Image network | YOLOv8 conv net (CSP/C2f + PAN), 3 strides | CLIP ViT over every image patch |
| Text cost per image | none: prompts are folded into conv weights at load | text tower runs per query; every patch is scored against every query |
| Cost of more prompts | one 1x1 conv output per prompt in the class head + the 4 attention convs | one more text-tower pass + one more score per patch |
| Changing prompts | rebuild the detector (reload the checkpoint) | encode the new query only |
| Box quality | DFL boxes over 3 strides, NMS | one box per ViT patch |
| Vocabulary | short noun phrases (trained on detection + grounding data) | free-form phrases; also image-conditioned queries |

The YOLO-World paper reports 35.4 AP on LVIS with 52 FPS on a V100 for its
large model; that is the authors' measurement, not one taken with this
library. This library has not yet run a real `yolov8s-worldv2` checkpoint, so
the table compares structure and cost, not measured accuracy. Pick YOLO-World
for a fixed prompt list over many images or video frames, where YOLOv8-level
speed matters. Pick OWL-ViT when the queries change per image or need longer
phrases.

Importer parity: `TestYoloWorldParity` (random text rows) and
`TestYoloWorldClipTextRoute` (pico CLIP text tower → YOLO-World) match a
float64 oracle written from the ultralytics modules to max |diff| < 1e-4
(fixture `tools/yolo_tiny_fixture.py --world`).

Pure CPU, <1 s on the fixture.

Coded by Claude (AI).
