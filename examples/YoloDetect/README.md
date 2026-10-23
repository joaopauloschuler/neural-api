# YoloDetect — YOLOv8 single-shot object detection

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
../../bin/x86_64-linux/bin/YoloDetect [model.safetensors] [config.json] [score_threshold] [iou_threshold]
```

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

Pure CPU, <1 s on the fixture.

Coded by Claude (AI).
