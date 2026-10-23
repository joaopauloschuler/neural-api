# ObjectTracking: YOLOv8 + ByteTrack multi-object tracking

Follows every object through a video, frame by frame. A YOLOv8 detector
(`BuildYoloFromSafeTensors`, `DecodeYoloDetections` in `neuralpretrained.pas`)
finds the objects of each frame, and `TByteTracker` links the detections of
consecutive frames into tracks, so each object keeps one **track id** for as
long as it stays in view. The program draws every track box in the colour of
its id, with the id and the class written above the box (for example
`3 person`).

**Status: planned (coded).** The example has not been run with a real
`yolov8n` checkpoint on a real clip. It runs end to end on the random pico
fixture (see below), and the tracker is unit-tested
(`TestByteTrackKalmanPredictUpdate`, `TestByteTrackerSyntheticSequence`,
`TestByteTrackerLostBufferAndReset`), but it has not been compared with the
Python ByteTrack on the same detections.

## The tracker

`TByteTracker` is a port of the official ByteTrack `BYTETracker`
(Zhang et al., ECCV 2022, [*ByteTrack: Multi-Object Tracking by Associating
Every Detection Box*](https://arxiv.org/abs/2110.06864), `ifzhang/ByteTrack`).
Each track has a constant-velocity Kalman filter over the box centre, aspect
ratio and height. On every frame `Update`:

1. predicts every track one frame ahead with its Kalman filter;
2. matches the **high-score** detections (score > `--track-thresh`) to the
   tracks by IoU, solved exactly (`NeuralLinearSumAssignment`) with a cost
   limit of `--match-thresh`;
3. matches the **low-score** detections (0.1 < score < `--track-thresh`; as
   in the official code, a score equal to `--track-thresh` is dropped) to the
   tracks still unmatched, which recovers objects that are partly hidden;
4. starts a new track for each unmatched high-score detection with
   score >= `--track-thresh` + 0.1;
5. marks the unmatched tracks lost, and removes a lost track after
   `--track-buffer` frames (scaled by `--fps` / 30).

The tracker ignores the class: a person box can continue a car track. Use
`--classes` to track only some classes; the program removes the other
detections **before** the tracker sees them.

## Usage

```
cd examples/ObjectTracking
lazbuild -B ObjectTracking.lpi
../../bin/x86_64-linux/bin/ObjectTracking                    # pico fixture smoke
../../bin/x86_64-linux/bin/ObjectTracking --model yolov8n.safetensors \
  --input frames/ --output tracked/ --gif tracked.gif --classes 0
```

`--model` takes an ultralytics yolov8 checkpoint converted to safetensors, with
its `config.json` next to it (or `--config FILE`); see
[YoloDetect](../YoloDetect) for the format. The input folder holds the frames
of one clip as PNG / JPEG / PPM files, sorted by name with numbers compared by
value (`f2.png` before `f10.png`). For each frame the program writes
`<frame>_tracks.png` into the output folder and prints the detection count and
the tracks (id, class, score). The class is shown by name when the
`config.json` has `names` (a list or a `{"0": ...}` map) or `id2label`; an
80-class yolov8 config without names uses the built-in COCO names; otherwise
the class id is shown.

| Option | Meaning | Default |
| --- | --- | --- |
| `--model FILE` | yolov8 `.safetensors` | pico smoke when absent |
| `--config FILE` | its `config.json` | `config.json` next to the model |
| `--input DIR` | folder of consecutive frames | required with `--model` |
| `--output DIR` | where `<frame>_tracks.png` goes | required with `--model` |
| `--gif FILE` | also write the annotated frames as an animated GIF (frames of another size are left out) | none |
| `--classes LIST` | comma-separated class ids to track, e.g. `0,2` (COCO: 0 = person, 2 = car) | all |
| `--score S` | detector score threshold; keep it at or below 0.1 so the low-score association has boxes | 0.1 |
| `--iou T` | detector NMS IoU threshold | 0.7 |
| `--track-thresh T` | ByteTrack high-score threshold | 0.5 |
| `--match-thresh T` | IoU cost limit of the first association | 0.8 |
| `--track-buffer N` | frames a lost track is kept, at 30 fps | 30 |
| `--fps N` | frame rate of the clip; also the GIF frame rate | 30 |
| `--int8` | store the conv weights in int8 | off |
| `--gpu` | run the detector on OpenCL (build with `-dOpenCL`) | CPU |
| `--gpu-platform N`, `--gpu-device N` | OpenCL device selection | 0, 0 |

To turn a video into frames and the frames back into a video, use any external
tool, for example `ffmpeg -i clip.mp4 frames/f%04d.png` and
`ffmpeg -framerate 30 -i tracked/f%04d_tracks.png tracked.mp4`.

## Pico fixture smoke

With no arguments the program writes four synthetic 128x96 frames into
`tracking_smoke_frames/`, runs them through the committed random pico detector
(`tests/fixtures/tiny_yolo.safetensors`), and writes `tracking_smoke_out/` and
`tracking_smoke.gif`. The pico weights are random, so its boxes are not
objects; the smoke only shows that detection, class filter, tracking, drawing
and GIF writing run end to end and that the ids stay the same across frames.
It runs in well under a second on one CPU core.

Coded by Claude (AI).
