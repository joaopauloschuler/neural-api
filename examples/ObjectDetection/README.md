# ObjectDetection: DETR end-to-end detection

The demo for the **DETR** object-detection importer
(`BuildDetrFromSafeTensors`, `neural/neuralpretrained.pas`;
`facebook/detr-resnet-50`). It runs a full DETR forward on one image, decodes the
per-query predictions, draws the surviving boxes and writes an annotated PPM.

## What it does

DETR is a ResNet backbone + a transformer **encoder-decoder** with learned object
queries, a 2-D sinusoidal spatial position embedding, a sigmoid-`cxcywh` box head
and a class head (**inference only, no Hungarian matcher**). After the forward
pass, `DecodeDetrDetections` softmaxes the class logits, drops the "no-object"
slot, thresholds on confidence, and converts each normalized `cxcywh` box to pixel
`xyxy`. The example draws each surviving box as a colored rectangle outline into
the image, prints the `(class, score, box)` list, and writes
`object_detection.ppm`.

It also **self-reports**: it asserts no NaN/Inf in the output and that every
decoded box lies within the sigmoid `[0,1]` range before clamping to the canvas.

## Running

```
cd examples/ObjectDetection
fpc -Fu../../neural ObjectDetection.lpr
./ObjectDetection                                  # committed pico fixture
./ObjectDetection model.safetensors config.json [threshold]
```

## Notes

- With no arguments it loads the committed **random** pico fixture
  `tests/fixtures/tiny_detr.*` (used by the parity tests) — fully offline. If a
  pinned `tiny_detr_io.json` is present its image is used; otherwise a synthetic
  gradient is generated.
- The pico fixture has random weights, so the "detections" are **not real
  objects** — the demo exercises the full decode + draw pipeline. A real
  `facebook/detr-resnet-50` checkpoint (e.g. with threshold ~0.7) yields real
  detections through the same path. The fixture threshold defaults to `0` so every
  query is drawn.
- Pure CPU, < 1 s on the fixture.

Coded by Claude (AI).
