# VideoMatting: Robust Video Matting (RVM)

Cuts the people out of a video, frame by frame, with the repo's **Robust Video
Matting** importer (`BuildRVMFromSafeTensors`, `TRVMMatting` in
`neuralpretrained.pas`). RVM (Lin et al., WACV 2022, [*Robust High-Resolution
Video Matting with Temporal Guidance*](https://arxiv.org/abs/2108.11515),
`PeterL1n/RobustVideoMatting`) predicts a foreground colour and an alpha matte
per pixel, with no green screen and no trimap. It is recurrent: four ConvGRU
states carry what the net saw in earlier frames, so the matte of each frame
uses the previous ones.

**Status: coded; tested on CPU with the real `rvm_mobilenetv3.pth`, not yet on
a GPU.** The importer is parity-tested against the official `model.py` on a
random pico net (`TestRVMParity`, 1.2e-7 over 3 frames), and the OpenCL forward
matches the CPU forward (`TestRVMOpenCLParity`). The real MobileNetV3
checkpoint matched the official model to 2.9e-6 on a 128x96 clip (a one-off
check). On a 640x480 4-frame clip on one CPU core, the example took about
1.4 s per frame after the first (ratio 0.75) and about 1.7 GB of RAM.

## How it works

1. The program lists the PNG / JPEG files of `--input`, sorted by name with
   numbers compared by value (`f2.png` before `f10.png`).
2. It builds the net for the first frame's size and `--ratio`. With a ratio
   below 1, the net downsamples the frame, runs MobileNetV3 + LR-ASPP + the
   recurrent decoder at low resolution, and the Deep Guided Filter refiner
   brings the matte back to full resolution.
3. `TRVMMatting.Step` runs one frame and copies the four ConvGRU outputs back
   into the state inputs for the next frame.
4. For each frame it writes `<frame>_rgba.png` (foreground RGB + alpha) and
   `<frame>_green.png` (the foreground over RVM's green,
   `fgr * alpha + green * (1 - alpha)`), and prints the alpha range.

A frame of another size needs another net: the program prints a notice,
rebuilds the net and starts the recurrent state over. An animated `--gif` keeps
the first frame's size and leaves frames of another size out.

## Run

```
./VideoMatting                                                  # pico fixture smoke run
./VideoMatting --model rvm_mobilenetv3.pth --input frames --output matted
./VideoMatting --model rvm_mobilenetv3.pth --input frames --output matted --ratio 0.25 --gif matted.gif --gpu
```

To turn a video into frames and back, use for example ffmpeg:

```
ffmpeg -i clip.mp4 frames/%05d.png
ffmpeg -framerate 30 -i matted/%05d_green.png matted.mp4
```

| option | meaning | default |
| --- | --- | --- |
| `--model FILE` | `rvm_mobilenetv3.pth` (or a `.safetensors` copy of it) | smoke run |
| `--input DIR` | folder of consecutive frames of one clip, PNG / JPEG | required with `--model` |
| `--output DIR` | output folder for `<frame>_rgba.png` and `<frame>_green.png` (created if missing) | required with `--model` |
| `--ratio R` | downsample ratio in (0, 1] for the backbone | see below |
| `--gif FILE` | also write the green-screen frames as an animated GIF | |
| `--fps N` | GIF frame rate | 25 |
| `--gpu`, `--gpu-platform N`, `--gpu-device N` | run the net on OpenCL (a build with `-dOpenCL`, the `.lpi` default) | CPU |

### Choosing `--ratio`

The RVM README recommends:

| resolution | portrait | full body |
| --- | --- | --- |
| up to 512x512 | 1 | 1 |
| 1280x720 | 0.375 | 0.6 |
| 1920x1080 | 0.25 | 0.4 |
| 3840x2160 | 0.125 | 0.2 |

Without `--ratio`, the program uses the portrait column: 1 when the longest side
is at most 512 pixels, else `480 / longest side` (0.375 at 1280x720, 0.25 at
1920x1080). For full-body shots pass the larger value from the table. The
official torch code sizes the low-resolution frame as `floor(size * ratio)`, as
this importer does, but it resizes with the scale `1/ratio` while the net uses
the size ratio; the two agree exactly when `width * ratio` and
`height * ratio` are whole numbers.

## Checkpoint

Download `rvm_mobilenetv3.pth` from the
[RobustVideoMatting releases](https://github.com/PeterL1n/RobustVideoMatting/releases)
and pass it to `--model`; no conversion is needed. The weights are released
under **GPL-3.0**; read the license before you redistribute anything built on
them. This repository does not include them. The ResNet-50 checkpoint
(`rvm_resnet50.pth`) is not supported yet.

With no arguments, the program writes three 44x36 frames and one 40x32 frame of
a disc moving over a gradient into `rvm_smoke_frames/`, mattes them with the
committed random pico net (`tests/fixtures/tiny_rvm_mobilenetv3.pth`) at ratio
0.5, and writes `rvm_smoke_out/` and `rvm_smoke.gif`. The weights are random,
so the matte is meaningless; the smoke run only exercises loading, the
recurrent forward, the rebuild on a size change and the PNG / GIF output. It
must be run from this folder (the fixture path is relative).

## Build

Open `VideoMatting.lpi` in Lazarus, or:

```
lazbuild VideoMatting.lpi
```

The binary lands in `../../bin/<cpu>-<os>/bin/`. For `--gpu`, the library finds
`neural/neural.cl` relative to this folder or to the binary.
