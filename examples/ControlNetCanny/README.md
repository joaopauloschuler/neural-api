# ControlNetCanny

Stable Diffusion 1.x text-to-image guided by a ControlNet, as diffusers'
`StableDiffusionControlNetPipeline`. The example drives `TStableDiffusionPipeline`
(`neural/neuralpretrained.pas`) with a `TStableDiffusionControlNet` conditioner:

```
prompt --CLIP tokenizer + text encoder--> text states (and the empty prompt's, for CFG)
for each sampler step, for the conditional and the unconditional pass:
  ControlNet(latents, text states, control image, t)  -> down + mid residuals (x scale)
  UNet(latents, text states, t, + residuals)            -> noise prediction
  CFG mix, scheduler step
VAE decoder -> PNG
```

The sampling loop (CFG, the sampler named by `scheduler_config.json`, the
steps) is the pipeline's own `Denoise`; the conditioner only adds the
ControlNet residuals to each UNet pass.

## Running

```sh
cd examples/ControlNetCanny
lazbuild -B ControlNetCanny.lpi
../../bin/x86_64-linux/bin/ControlNetCanny                       # pico fixtures
../../bin/x86_64-linux/bin/ControlNetCanny \
  --model /path/to/stable-diffusion-v1-5 \
  --controlnet /path/to/sd-controlnet-canny \
  --control edges.png --prompt "a futuristic city" --steps 25 --seed 1 \
  --out city.png --gpu
../../bin/x86_64-linux/bin/ControlNetCanny \
  --model /path/to/stable-diffusion-v1-5 \
  --controlnet /path/to/sd-controlnet-canny \
  --photo photo.jpg --edges-out edges.png --prompt "a futuristic city" --gpu
../../bin/x86_64-linux/bin/ControlNetCanny \
  --model /path/to/stable-diffusion-v1-5 \
  --controlnet /path/to/sd-controlnet-canny --control edges.png --scale 0.8 \
  --controlnet /path/to/sd-controlnet-depth --control depth.png --scale 0.5 \
  --control-start 0.2 --prompt "a futuristic city" --gpu
../../bin/x86_64-linux/bin/ControlNetCanny \
  --model /path/to/stable-diffusion-v1-5 \
  --controlnet /path/to/sd-controlnet-openpose \
  --pose-photo people.jpg --yolo /path/to/yolov8n/model.safetensors \
  --vitpose /path/to/vitpose-base-simple/model.safetensors \
  --pose-out pose.png --prompt "chef in the kitchen" --gpu
```

Each `--controlnet DIR` after the first adds another ControlNet (diffusers
`MultiControlNetModel`, through `TStableDiffusionMultiConditioner`).
`--control`, `--photo`, `--edges-out`, `--pose-photo`, `--pose-out`, `--scale`,
`--control-start` and `--control-end` apply to the latest `--controlnet` given before them; before
any `--controlnet` they apply to the first one. The canny thresholds, the
blur, the pose models and `--guess-mode` apply to all.

| Option | Meaning |
| --- | --- |
| `--model DIR` | diffusers Stable Diffusion 1.x folder (default `tests/fixtures/tiny_sd15`) |
| `--controlnet DIR` | diffusers `ControlNetModel` folder: `config.json` + `diffusion_pytorch_model.safetensors` (default `tests/fixtures/tiny_sd15_controlnet`); repeat it for several ControlNets |
| `--control FILE` | control image; for the canny ControlNet white edges on black (e.g. `cv2.Canny` output). Resized to the image size (Lanczos). Default: drawn circle and square outlines |
| `--photo FILE` | photo to take the edges from instead of `--control`: resized to the image size (PIL Lanczos, uint8 rounding), then `CannyEdges` |
| `--canny-low T`, `--canny-high T` | `cv2.Canny` thresholds (100, 200: the diffusers ControlNet recipe) |
| `--canny-blur R` | `GaussianBlurImage` radius (sigma) applied before `CannyEdges` (default 0: no blur, as `cv2.Canny`) |
| `--edges-out PNG` | also save the extracted edge map |
| `--pose-photo FILE` | photo to draw the OpenPose control image of, instead of `--control`: resized to the image size like `--photo`, then every person is found and drawn (see [Pose control image](#pose-control-image)); needs `--yolo` and `--vitpose` |
| `--yolo FILE`, `--yolo-config FILE` | YOLOv8 safetensors that finds the person boxes (class 0), e.g. `yolov8n`; its `config.json` defaults to the one beside it |
| `--vitpose FILE`, `--vitpose-config FILE` | ViTPose safetensors, e.g. `usyd-community/vitpose-base-simple`; `config.json` beside it by default |
| `--pose-out PNG` | also save the OpenPose control image |
| `--prompt`, `--negative` | prompt and negative prompt |
| `--steps N`, `--guidance G`, `--seed N` | 25, 7.5, 0 |
| `--scale S` | `controlnet_conditioning_scale` (1.0) |
| `--control-start F`, `--control-end F` | `control_guidance_start` / `control_guidance_end` (0, 1): step i of N gets the residuals when F_start <= i / N and (i + 1) / N <= F_end |
| `--guess-mode` | diffusers `guess_mode`: the ControlNets run in the prompt pass only, their down residuals scaled from 0.1 up to 1 (`logspace(-1, 0)`); the CFG pass without the prompt gets none. A first ControlNet whose `config.json` sets `global_pool_conditions` (e.g. `lllyasviel/control_v11e_sd15_shuffle`) turns it on for all, as diffusers does |
| `--ip-adapter FILE`, `--ip-image PNG`, `--ip-scale S` | an h94/IP-Adapter `ip-adapter_sd15` image prompt on top of the ControlNet (its `image_encoder/` folder beside FILE), as `pipe.load_ip_adapter` on a `StableDiffusionControlNetPipeline`; scale 1.0 |
| `--width N`, `--height N` | image size, each a multiple of 8 for SD 1.5 (default: the checkpoint's, 512) |
| `--size N` | sets both `--width` and `--height` |
| `--out PNG` | output file (default `controlnet_canny.png`) |
| `--int8` | UNet, ControlNet and text encoder weights in int8 |
| `--gpu`, `--gpu-platform N`, `--gpu-device N` | OpenCL |

With no `--model` / `--controlnet` the run uses random-weight pico fixtures:
it checks wiring and I/O, not image quality.

The IP-Adapter joins through a `TStableDiffusionMultiConditioner`, which runs
several conditioners in one `Denoise`: ControlNets and T2I-Adapters (their
residuals summed, as diffusers `MultiControlNetModel`), at most one
IP-Adapter, and at most one img2img / inpainting / InstructPix2Pix
conditioner. The example exposes several ControlNets plus the IP-Adapter; in
code:

```pascal
Multi := TStableDiffusionMultiConditioner.Create();
Multi.Add(ControlNetCanny);   // TStableDiffusionControlNet
Multi.Add(ControlNetDepth);   // a second one: MultiControlNet
Multi.Add(IPAdapter);         // TStableDiffusionIPAdapter
Pipeline.Conditioner := Multi;
```

## Parity

`TestStableDiffusionControlNetParity` (`tests/TestNeuralPretrained.pas`) runs
the pipeline with the pico ControlNet (a diffusers `ControlNetModel.from_unet`
of `tiny_sd15`, conditioning scale 0.5) for 3 guided DDIM steps and compares
the latents after each step and the decoded image with a diffusers float64
`StableDiffusionControlNetPipeline` run: max |diff| < 1e-4.
`TestStableDiffusionControlNetNonSquare` does the same at 14x10, a 7x5 latent
grid that is neither square nor a multiple of the UNet down factor. The
composed runs (`tests/fixtures/tiny_sd15_composed_io.json`, 3 guided DDIM
steps each) match diffusers float64 to 1e-4 as well:
`TestStableDiffusionIPAdapterControlNetParity` (ControlNet + IP-Adapter),
`TestStableDiffusionControlNetGuidanceWindowParity` (start 0.3, end 0.7),
`TestStableDiffusionControlNetGuessModeParity`,
`TestStableDiffusionMultiControlNetParity` (two control images, scales 0.5 /
0.3, the second from 0.4 of the steps) and
`TestStableDiffusionControlNetAdapterIPParity` (ControlNet + T2I-Adapter +
IP-Adapter; diffusers has no pipeline for a ControlNet and a T2I-Adapter
together, so that oracle adds the adapter features to the ControlNet
pipeline's UNet calls). `global_pool_conditions` has its own oracle
(`tests/fixtures/tiny_sd15_controlnet_global_pool_io.json`, the pico
ControlNet with the flag set): `TestStableDiffusionControlNetGlobalPoolParity`
(alone: guess mode forced, every down / mid residual mean-pooled over H, W),
`TestStableDiffusionMultiControlNetGlobalPoolParity` (two pooled nets) and
`TestStableDiffusionControlNetGlobalPoolSecondParity` (a pooled net after a
plain one: no guess mode, since diffusers reads the flag of the first net
only). The fixtures and the oracles come from
`tools/make_pico_sd_conditioning_fixtures.py`.

## Canny edges

`CannyEdges` (`neural/neuralimageresize.pas`) ports `cv2.Canny(image, low,
high)` with the defaults the diffusers recipe uses (aperture 3, L1 gradient):
3x3 Sobel with replicated borders, the strongest channel per pixel for RGB
input, non-maximum suppression in four directions with cv2's fixed-point
tan(22.5)/tan(67.5) tests, the thresholds compared with `>`, and hysteresis
over 8-connected neighbours. Like `cv2.Canny`, it does not blur; `--canny-blur`
adds a `GaussianBlurImage` pass first. The tests (`TestCannyEdges*` in
`tests/TestNeuralImagePreprocess.pas`) check hand-computed step, square and
hysteresis cases; no `cv2` comparison has been run.

## Pose control image

`--pose-photo` makes the control image that `lllyasviel/sd-controlnet-openpose`
and `lllyasviel/control_v11p_sd15_openpose` expect, from a plain photo:

1. The example resizes the photo to the image size (as `--photo` does).
2. `EstimateMultiPersonPose` (`neural/neuralpretrained.pas`) runs the YOLOv8
   detector (`--yolo`, the checkpoint `examples/YoloDetect` loads), keeps the
   person boxes (class 0, score >= 0.25) and runs top-down ViTPose
   (`--vitpose`) once per box: 17 COCO joints per person, in photo pixels.
3. `DrawOpenPoseBodies` draws them as controlnet_aux's DWpose detector does:
   COCO-17 to OpenPose-18 (the neck is the shoulder midpoint, drawn only when
   both shoulders score > 0.3; every joint needs a score > 0.3), the 17 limbs
   as filled `cv2.ellipse2Poly` polygons 4 pixels wide in the limb color
   x 0.6, then the 18 joint dots (radius 4) in full color, on black.

`TestOpenPoseDrawParity` matches the DWpose drawing code, run with
OpenCV 4.10, value for value on a 96 x 64 canvas with two persons (clipped
limbs and dots, hidden joints). `--pose-out` saves the drawn image;
`examples/PoseEstimation --openpose-out` writes the same image at the photo
size.

Not done yet: a real run with `yolov8n`, `usyd-community/vitpose-base-simple`
and `lllyasviel/sd-controlnet-openpose`. Only the pico fixtures (random
weights) have been through this path.

## Limits

- `--photo` extracts the edges after resizing the photo to the image size;
  the diffusers recipe runs `cv2.Canny` on the original photo and lets the
  pipeline resize the edge map. Resize the photo yourself and pass it at the
  image size for the same result.
- `--pose-photo` detects the poses on the photo resized to the image size, as
  controlnet_aux detects at its `detect_resolution`; a photo whose aspect
  differs from the image is stretched first. ViTPose decides the joints,
  not the OpenPose body model controlnet_aux's `OpenposeDetector` uses, so
  joint positions can differ from that detector's.
- Real checkpoints (`lllyasviel/sd-controlnet-canny` on SD 1.5, an
  `ip-adapter_sd15`) are untested.
- A pooled (`global_pool_conditions`) ControlNet first and a plain one after
  it runs here; diffusers fails on that order (its MultiControlNet sums the
  1x1 pooled residuals in place), so it has no oracle.
- `--guess-mode` is one switch for all ControlNets, as diffusers' pipeline
  argument; `TStableDiffusionControlNet.GuessMode` can differ per net in code.
