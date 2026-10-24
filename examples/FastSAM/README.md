# FastSAM: Real-Time Segment Anything with YOLOv8-seg

Segments every object in a photo, or only the one picked by a box, point or
text prompt, with the repo's **YOLOv8 instance-segmentation** importer
(`BuildYoloFromSafeTensors` with `"task": "segment"`,
`DecodeYoloSegmentationNative`, `FastSAMSelectByBox`, `FastSAMSelectByPoint`,
`FastSAMSelectByText` in `neuralpretrained.pas`).
FastSAM (Zhao et al. 2023, [*Fast Segment Anything*](https://arxiv.org/abs/2306.12156);
ultralytics `FastSAM-s` / `FastSAM-x`) is a yolov8s-seg / yolov8x-seg net
trained on SA-1B with a single class: a conv backbone instead of SAM's ViT
image encoder, so it runs at interactive speed on a CPU.

**Status: coded, not yet tested on a real checkpoint.** The importer is
parity-tested against a float64 reference of the ultralytics modules on a
random pico net (`TestYoloSegParity`: raw head and prototype map < 1e-4, boxes,
soft masks < 1e-4 and binary masks identical for both mask modes), the
image-size retina masks against a hand-computed `F.interpolate` oracle
(`TestYoloSegNativeMasks`), the prompt logic against FastSAM's
`prompt.py` rules (`TestFastSAMPromptSelection`), the text prompt on the pico
CLIP fixture (`TestFastSAMTextSelection`), and the OpenCL forward against the
CPU forward (`TestYoloSegOpenCLParity`). The text prompt has not been run with
a real CLIP checkpoint and tokenizer.

## How it works

1. `LoadImageFromFileIntoVolume` reads the image; the program letterboxes it
   to the net input (ultralytics `LetterBox`: keep the aspect ratio, bilinear
   resize, pad with gray 114, scale to 0..1).
2. The net is the YOLOv8 detector (C2f backbone, SPPF, PANet neck) plus the
   ultralytics `Segment` head: per stride a `cv4` branch adds `num_masks`
   mask coefficients to every cell of the raw head, and a `Proto` branch
   (conv, 2x transposed conv, conv, conv) on the stride-8 features gives a
   stride-4 map of `num_masks` prototype masks (`GetYoloProtoLayer`).
3. `DecodeYoloSegmentation` runs the box decode and NMS of
   `DecodeYoloDetections`, keeping at most `MaxDetections` boxes (ultralytics
   `max_det`, default 300); for a `fastsam` config it then applies FastSAM's
   box step (`FastSAMAdjustBoxes`: sides within 20 px of the border snap to
   it, a box with IoU > 0.9 against the whole image becomes the whole image).
   Each mask is sigmoid of coefficients . prototypes, bilinear upsample,
   > 0.5, with the box crop in one of two places:
   - retina masks (default, FastSAM's `retina_masks=True`; ultralytics
     `process_mask_native`, `DecodeYoloSegmentationNative`): the boxes are
     mapped to image pixels (`ScaleYoloBoxesToImage`, ultralytics
     `scale_boxes`), and `ComputeYoloMasksNative` removes the letterbox
     padding from the prototype-scale masks, upsamples them straight to the
     ORIGINAL image size and crops them to the image-pixel boxes.
   - `--no-retina` (ultralytics `process_mask`, `ComputeYoloMasks`): crop to
     the box at prototype resolution, then upsample to the net input, so the
     box edge of each mask follows the stride-4 prototype grid; step 5 maps
     these masks to the image.
4. A `--box` prompt keeps the mask with the highest IoU against the box
   (`FastSAMSelectByBox`). `--point` prompts walk the masks from largest to
   smallest: a mask under a label-1 point is added, a mask under a label-0
   point is removed (`FastSAMSelectByPoint`). A `--text` prompt follows the
   ultralytics `FastSAMPredictor`: masks of 100 pixels or fewer are skipped;
   for each other mask the program crops its box from the original image
   (no background fill), runs the CLIP image preprocessing and vision tower,
   and keeps the mask whose image embedding has the highest cosine with the
   CLIP text embedding of the prompt (`FastSAMSelectByText`). The text is
   tokenized by the CLIP folder's tokenizer and framed as
   `<|startoftext|> ... <|endoftext|>`. The original FastSAM repository's
   `prompt.py` differs: it crops the bounding rectangle of the mask pixels
   and pastes it on a white canvas of the full image size.
5. With `--no-retina` the program crops the letterbox padding off the masks
   and resizes them to the image. It writes a PNG with each mask blended in a
   palette colour.

## Run

```
./FastSAM                                                         # pico fixture smoke run
./FastSAM --model FastSAM-s/model.safetensors --image photo.jpg --output everything.png
./FastSAM --model FastSAM-s/model.safetensors --image photo.jpg --box 120,80,400,360 --output box.png
./FastSAM --model FastSAM-s/model.safetensors --image photo.jpg --point 250,200,1 --point 300,90,0 --gpu
./FastSAM --model FastSAM-s/model.safetensors --image photo.jpg --text "a black dog" --clip clip-vit-base-patch32 --output dog.png
```

Defaults follow FastSAM: `--conf 0.4`, `--iou 0.9`, retina masks on
(`--no-retina` switches them off), at most 300 detections. Box and point
coordinates are image pixels. Only one prompt kind is used per run: `--box`
first, then `--point`, then `--text`.

`--clip DIR` is a Hugging Face CLIP folder; FastSAM uses ViT-B/32, i.e.
[`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32):
`model.safetensors` (or `pytorch_model.bin`), `config.json`,
`preprocessor_config.json` and `tokenizer.json` (or `vocab.json` +
`merges.txt`).

## Checkpoint

The released `FastSAM-s.pt` / `FastSAM-x.pt` files are pickled ultralytics
models. Convert one on a machine with ultralytics installed: load it, take the
float `state_dict()` of its `model` and save it as `model.safetensors` (keys
`model.0.conv.weight` ... `model.22.proto.cv3.bn.running_var`). Next to it put
a `config.json`:

| checkpoint | config.json |
| --- | --- |
| FastSAM-s (yolov8s-seg) | `{"model_type": "fastsam", "widths": [32, 64, 128, 256, 512], "depths": [1, 2], "image_size": 1024}` |
| FastSAM-x (yolov8x-seg) | `{"model_type": "fastsam", "widths": [80, 160, 320, 640, 640], "depths": [3, 6], "image_size": 1024}` |

`model_type` `fastsam` implies `task` `segment`, one class and the FastSAM
box step; the head widths follow the ultralytics rules (`box_hidden`,
`cls_hidden`, `proto_hidden`, `mask_hidden`, `num_masks` can override them).
A plain ultralytics yolov8-seg checkpoint uses `"model_type": "yolov8",
"task": "segment"` plus its `num_classes`.

## Pico fixture

With no arguments the program loads `tests/fixtures/tiny_yolo_seg.safetensors`
(random weights, generated by `tools/yolo_tiny_fixture.py --seg`), segments a
synthetic 120x80 gradient, then runs one box prompt, one point prompt and one
text prompt (the pico CLIP fixture `tests/fixtures/tiny_clip.*` with fixed
token ids, since it has no tokenizer), and writes `fastsam_everything.png`,
`fastsam_box.png`, `fastsam_point.png` and `fastsam_text.png`.
The masks are not meaningful objects; the run exercises the whole pipeline in
well under a second.

Coded by Claude (AI).
