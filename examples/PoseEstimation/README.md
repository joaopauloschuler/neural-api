# PoseEstimation: human-pose keypoints with ViTPose

The demo for the **ViTPose** importer
(`BuildViTPoseFromSafeTensors`, `neural/neuralpretrained.pas`; Xu et al. 2022,
*ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation*). A
top-down keypoint estimator: it finds the joints of one person inside a person
box, so a whole photo needs one box per person.

## What it does

The example runs the top-down recipe of transformers' `VitPoseImageProcessor`
and `post_process_pose_estimation`:

1. `ViTPoseBoxToCenterScale` grows the person box to the model input aspect
   and pads it by 1.25.
2. `ViTPoseCropPerson` warps that region of the photo to the model input size
   (bilinear) and normalizes it with the ImageNet mean and std.
3. The ViTPose net turns the crop into one heatmap per joint.
4. `DecodeViTPoseKeypointsInImage` takes each heatmap peak, refines it with
   DARK (Gaussian smoothing + one Newton step on the log heatmap) and maps it
   back to photo pixels.

`EstimateViTPoseInBox` does steps 1-4 for one box. `EstimateMultiPersonPose`
runs a YOLOv8 detector first and calls `EstimateViTPoseInBox` once per person
box (class 0). The person boxes come from:

- `--box X,Y,W,H`: one box in photo pixels (top-left corner, width, height);
- `--yolo FILE`: a YOLOv8 checkpoint, one pose per detected person;
- neither: the whole photo is one box.

The output photo (`--output`, default `pose_out.png`) shows each box in green
with a `person <score>` tag above it (`DrawBitmapText`,
`neural/neuralimageresize.pas`) and each joint as a red square; the joints are
also printed.

## OpenPose image for ControlNet

`--openpose-out FILE` also writes the poses as an OpenPose control image at
the photo size: `DrawOpenPoseBodies` (`neural/neuralpretrained.pas`) maps the
17 COCO joints to the 18 OpenPose joints and draws limbs and dots as
controlnet_aux's DWpose detector does (`TestOpenPoseDrawParity` matches that
drawing code run with OpenCV 4.10 value for value).

DWpose draws at its `detect_resolution`, not at the photo size, so its
4-pixel sticks and radius-4 dots are relative to that size. The example does
the same by default (`DrawOpenPoseBodiesViaDetectResolution`):

1. `ControlNetAuxResizeImageSize` (`neural/neuralimageresize.pas`) computes
   controlnet_aux's `resize_image` size: the photo scaled so its short side is
   `--detect-resolution` (default 512), then each side rounded to a multiple
   of 64 with `np.round` (half to even). A 640 x 480 photo is drawn at
   704 x 512, a 1920 x 1080 photo at 896 x 512.
2. The joints are mapped to that size and drawn there.
3. The drawing is resized back to the photo size with bilinear
   interpolation (cv2 `INTER_LINEAR`, the filter DWpose uses for its own
   final resize).

DWpose itself returns the drawing at `resize_image(image_resolution)`
(also 512 by default, so the detect size) and leaves the resize to the
generation size to the pipeline; step 3 returns the photo size instead.
`--detect-resolution 0` draws at the photo size directly (thinner sticks on
large photos). That image is the input
of an openpose ControlNet such as `lllyasviel/sd-controlnet-openpose`;
`examples/ControlNetCanny --pose-photo` builds it from a whole photo and runs
the ControlNet in one go.

## Running

```
cd examples/PoseEstimation
lazbuild -B PoseEstimation.lpi
../../bin/x86_64-linux/bin/PoseEstimation          # pico fixture smoke
../../bin/x86_64-linux/bin/PoseEstimation --model vitpose/model.safetensors \
  --image photo.jpg --yolo yolov8n/model.safetensors --conf 0.25 \
  --openpose-out openpose.png
```

| Option | Meaning |
| --- | --- |
| `--model FILE` | ViTPose `model.safetensors` (`usyd-community/vitpose-base-simple`) |
| `--config FILE` | ViTPose `config.json` (default: next to `--model`) |
| `--image IMG` | the photo |
| `--box X,Y,W,H` | one person box |
| `--yolo FILE` | YOLOv8 safetensors, the same checkpoint `examples/YoloDetect` loads |
| `--yolo-config FILE` | YOLOv8 `config.json` (default: next to `--yolo`) |
| `--conf X` | person score threshold (default 0.25) |
| `--output FILE` | annotated photo (default `pose_out.png`) |
| `--openpose-out FILE` | also the OpenPose image of the poses (see [OpenPose image for ControlNet](#openpose-image-for-controlnet)) |
| `--detect-resolution N` | short side the OpenPose image is drawn at before the resize to the photo size (default 512, as DWpose; 0 = draw at the photo size) |

With no arguments the program loads the committed pico fixtures
(`tests/fixtures/tiny_vitpose.*` and `tests/fixtures/tiny_yolo.*`) from
`../../tests/fixtures/` or `tests/fixtures/`, runs the multi-person path on a
synthetic photo and prints the results.

## Notes

- The pico fixtures have random weights, so the smoke run prints boxes and
  joints that are not real people. `TestViTPoseTopDownParity` matches the
  keypoints of transformers to 2.3e-5 px on that model.
- A real run with `usyd-community/vitpose-base-simple` and `yolov8n` has not
  been made yet. Neither has an openpose ControlNet run on its
  `--openpose-out` image.
- transformers' processor has no flip test, so neither does this example.
- Inference only, CPU.

Coded by Joao Paulo Schwarz Schuler with Claude (AI).
