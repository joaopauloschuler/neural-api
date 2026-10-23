# FaceRestoration: GFPGAN and CodeFormer Blind Face Restoration

Restores the faces of a degraded photo (blur, noise, compression, low
resolution) with the repo's **GFPGAN** importer (`BuildGFPGANFromSafeTensors`,
`GFPGANRestoreFace` in `neuralpretrained.pas`) or its **CodeFormer** importer
(`BuildCodeFormerFromSafeTensors`, `CodeFormerRestoreFace`), the **RetinaFace** face
detector importer and a port of facexlib's face alignment and paste-back
(`RestoreFacesInImage`). GFPGAN (Wang et al., CVPR 2021,
[*Towards Real-World Blind Face Restoration with Generative Facial
Prior*](https://arxiv.org/abs/2101.04061), `TencentARC/GFPGAN`) uses a
pretrained StyleGAN2 face generator as a prior: a U-Net cleans the input face
and steers the generator through its style latents and spatial feature
transforms. CodeFormer (Zhou et al., NeurIPS 2022, [*Towards Robust Blind Face
Restoration with Codebook Lookup Transformer*](https://arxiv.org/abs/2206.11253),
`sczhou/CodeFormer`) replaces each latent cell of a degraded face with the
closest entry of a learned codebook of face parts, chosen by a transformer, and
decodes the codes with a VQGAN decoder; `--fidelity` blends encoder features
back in (0 = quality, 1 = identity). `--model` tells the two checkpoints apart
by their tensors (`IsCodeFormerCheckpoint`).

**CodeFormer license: the S-Lab License 1.0, non-commercial use only.** That
license covers the official code and the `codeformer.pth` weights; using the
weights through this program does not change it.

**Status, GFPGAN: coded, not yet tested on a real checkpoint.** The importer is
parity-tested (< 1e-4) against the official `gfpganv1_clean_arch.py` on two
random pico nets (`TestGFPGANParity`, `TestGFPGANSharedLatentParityInfersConfig`),
and the OpenCL forward matches the CPU forward on the same net
(`TestGFPGANOpenCLParity`). No official checkpoint has been run yet, neither
GFPGAN nor the RetinaFace detector.

**Status, CodeFormer: coded, pico parity only.** The importer is
parity-tested (< 1e-4) against the official `codeformer_arch.py` +
`vqgan_arch.py` on a random pico net at two fidelity weights, with and without
AdaIN (`TestCodeFormerParity`, `TestCodeFormerNoAdaINParity`); the OpenCL
forward matches the CPU forward on the same net (`TestCodeFormerOpenCLParity`),
and `TestRestoreFacesInImageCodeFormer` covers the whole-photo path. The real
`codeformer.pth` has not been run yet.

## Two modes

**Whole photo** (the default, what the official `inference_gfpgan.py` does):
the program finds every face with the **RetinaFace** detector (facexlib's
`detection_Resnet50_Final.pth`, or the smaller
`detection_mobilenet0.25_Final.pth`), aligns each face to the FFHQ 5-point
template at 512x512, restores it with GFPGAN or CodeFormer and pastes it back into the photo,
which is upscaled by `--upscale` (2 by default). This is a port of facexlib's
`FaceRestoreHelper` (`RestoreFacesInImage` in `neuralpretrained.pas`).

**Aligned crop** (`--aligned`): the input is already a square crop of one face,
aligned the way GFPGAN was trained (eyes level, face centered, filling the
frame), as the official tool's `--aligned`. A face of another size is resized
bilinearly to the net resolution (512x512 for the official checkpoints).

## How it works

Whole photo:

1. `LoadImageFromFileIntoVolume` reads the photo.
2. `DetectRetinaFaceFaces` runs RetinaFace at the photo's own size (as
   facexlib) and keeps faces scoring above 0.97 (`--det-threshold`), after
   non-maximum suppression at IoU 0.4. Faces whose eyes are closer than 5
   pixels are skipped (`--min-eye-distance`, GFPGAN's `eye_dist_threshold`).
3. `FaceAlignmentMatrix`: a 4-parameter similarity (rotation, uniform scale,
   shift) takes the 5 landmarks (eyes, nose, mouth corners) onto the FFHQ
   template. facexlib uses `cv2.estimateAffinePartial2D(method=LMEDS)`;
   `EstimateSimilarityTransformLMedS` is a deterministic version of it: the
   landmark pair whose transform has the least median error picks the inliers,
   and the transform is the least-squares fit of the inliers.
4. `AlignWarpFace`: bilinear `warpAffine` into a 512x512 crop; pixels outside
   the photo are gray (132, 133, 135), as facexlib.
5. `FaceRestorerRestoreFace` restores the crop with the loaded net (GFPGAN or
   CodeFormer: both take and return a face in [-1, 1] at their resolution).
6. `PasteFaceIntoImage`: the restored face is warped back by the inverse
   transform (scaled by `--upscale`) together with a square mask; the mask is
   eroded by `2*upscale` pixels, then eroded again and Gaussian-blurred by a
   width proportional to the face size (facexlib's `paste_faces_to_input_image`
   without face parsing), and the face is blended with it. Pixels away from
   every face keep the background exactly.
7. The background is the photo resized with cv2's Lanczos (8 taps,
   `ResizeImageLanczos4CV2`), or, with `--bg-upsampler`, the photo upscaled by
   an RRDBNet / Real-ESRGAN checkpoint (`TRRDBNetUpscaler`, tiles of 400
   pixels, as GFPGAN).

The net: a U-Net encoder down to 4x4, a linear layer to the StyleGAN2 style
latents, a U-Net decoder that produces a scale and a shift per resolution, and
the StyleGAN2 decoder, whose features are multiplied by the scale and shifted
after the first convolution of every resolution (CS-SFT, on half of the
channels). `GFPGANPrepareInput` maps 0..255 to [-1, 1];
`GFPGANComposeOutput` clamps the output to [-1, 1] and maps it back.

CodeFormer: a VQGAN encoder down to 16x16 cells, a 9-layer transformer that
picks one of 1024 codebook entries per cell, the codebook lookup, AdaIN of the
looked-up codes onto the encoder features (as `inference_codeformer.py`), and
a VQGAN decoder; at the 32 to 256 resolutions a fusion block adds
`w * (decoder * scale + shift)`, the scale and shift computed from the encoder
features, where `w` is `--fidelity`. The face size is 512; the same
`[-1, 1]` input and output mapping as GFPGAN.

### Differences from inference_gfpgan.py

- GFPGAN's `GFPGANer` builds the helper with `use_parse=True`: it pastes each
  face through a **ParseNet** face-parsing mask (`parsing_parsenet.pth`). That
  net is not ported; this program uses facexlib's square-mask path
  (`use_parse=False`), so the blend edge differs, mostly around hair and neck.
- facexlib works on 8-bit images and `cv2.warpAffine` rounds sample positions
  to 1/32 pixel; here every step is in float. On a smooth synthetic photo the
  aligned crop differs from cv2's warp by 0.03 gray levels on average (1.5 at
  most) for a float image, and by 0.24 on average for the 8-bit one.
- The face mask, its erosion and blur follow cv2 (including cv2's fixed
  Gaussian tables up to 9 taps and its 3x3 default for an empty erosion kernel).
- `inference_gfpgan.py` turns Real-ESRGAN off on a CPU, so a CPU run of it uses
  the Lanczos background, as this program does without `--bg-upsampler`.

## Run

```
./FaceRestoration                                           # pico fixture smoke runs
./FaceRestoration --model GFPGANv1.4.pth --detector detection_Resnet50_Final.pth \
  --input photo.jpg --output photo_restored.png
./FaceRestoration --model GFPGANv1.4.pth --detector detection_Resnet50_Final.pth \
  --input photo.jpg --output photo_restored.png --bg-upsampler RealESRGAN_x2plus.pth --gpu
./FaceRestoration --aligned --model GFPGANv1.4.pth --input face.png --output face_restored.png
./FaceRestoration --model codeformer.pth --fidelity 0.7 --detector detection_Resnet50_Final.pth \
  --input photo.jpg --output photo_restored.png
```

| option | meaning | default |
| --- | --- | --- |
| `--model FILE` | GFPGAN checkpoint (the official `GFPGANv1.4.pth` or `GFPGANv1.3.pth`) or CodeFormer checkpoint (the official `codeformer.pth`), in the `{'params_ema': ...}` release format, read directly, or a `.safetensors` with the same tensor names; the program picks the restorer from the tensor names | smoke run |
| `--config FILE` | optional `config.json`; GFPGAN: `out_size`, `num_style_feat`, `channel_multiplier`, `narrow`, `different_w`, `sft_half`, ...; CodeFormer: `n_head`, `norm_num_groups`, `fidelity_weight`, `adain` | `config.json` next to `--model`; without one, the release settings, with every size read from the tensor shapes |
| `--fidelity W` | CodeFormer only: fusion weight, 0 = best quality .. 1 = closest to the input face | the config's `fidelity_weight`, 0.5 (as `inference_codeformer.py`) |
| `--input IMG` | the photo (with `--aligned`: an aligned, cropped face), PNG / JPEG / BMP | required with `--model` |
| `--output IMG` | the restored photo or face; the format comes from the extension (PNG recommended) | required with `--model` |
| `--detector FILE` | facexlib RetinaFace checkpoint: `detection_Resnet50_Final.pth` (GFPGAN's default) or `detection_mobilenet0.25_Final.pth`, read directly | required without `--aligned` |
| `--upscale N` | output size / photo size | 2 (as `inference_gfpgan.py`) |
| `--bg-upsampler FILE` | RRDBNet / Real-ESRGAN checkpoint for the background, e.g. `RealESRGAN_x2plus.pth` (its output is resized to the `--upscale` size) | Lanczos resize |
| `--bg-tile N` | tile size of the background upsampler, 0 = the whole photo at once | 400 |
| `--det-threshold T` | RetinaFace score threshold | 0.97 (facexlib) |
| `--min-eye-distance P` | skip faces whose eyes are closer than P pixels | 5 (GFPGAN) |
| `--max-faces N` | restore only the N best-scoring faces | all |
| `--save-faces PREFIX` | also write `PREFIX_cropped_NN.png` (the aligned crops) and `PREFIX_restored_NN.png` | off |
| `--aligned` | the input is an aligned, cropped face: no detection, no paste | whole photo |
| `--keep-size` | (`--aligned`) resize the restored face back to the input size | output at the net resolution |
| `--noise MODE` | GFPGAN only: per-pixel StyleGAN2 noise: `random` (fresh noise each run, what the official tool does), `const` (the noise maps stored in the checkpoint: the same output every run) or `none` | `random` |
| `--seed N` | seed of the random noise, for a repeatable `random` run | time based |
| `--int8` | store the restorer's convolution and linear weights in int8 (`TNNet.QuantizeWeightsInt8`): less RAM, small accuracy loss; not yet checked on CodeFormer, where the error can change the chosen codes | FP32 |
| `--gpu`, `--gpu-platform N`, `--gpu-device N` | run the nets (detector, restorer, background upsampler) on OpenCL (a build with `-dOpenCL`, the `.lpi` default) | CPU |

Checkpoints: `GFPGANv1.3.pth` and `GFPGANv1.4.pth` are published on the
TencentARC/GFPGAN GitHub releases page. Both use the "clean" architecture this
importer implements. The older `GFPGANv1.pth` (the original architecture with
compiled CUDA operators) is a different network and is not supported. The
detectors `detection_Resnet50_Final.pth` and `detection_mobilenet0.25_Final.pth`
are published on the xinntao/facexlib GitHub releases page (v0.1.0); GFPGAN
downloads the ResNet-50 one on its first run. `RealESRGAN_x2plus.pth` is on the
xinntao/Real-ESRGAN releases page. `codeformer.pth` (v0.1.0) is on the
sczhou/CodeFormer GitHub releases page.

The detector runs at the photo's own size, as facexlib does, so a large photo
needs a lot of RAM for the ResNet-50 detector; the mobilenet detector is much
lighter.

With no arguments, the program runs two smokes per restorer on the committed
random pico nets (`tests/fixtures/tiny_gfpgan.*`, a 64x64 net,
`tests/fixtures/tiny_codeformer.*`, a 32x32 net, and
`tests/fixtures/tiny_retinaface_mobilenet.pth`): it writes a 40x30 color
gradient to `face_in.png` and restores it as an aligned face into
`gfpgan_out.png` / `codeformer_out.png`, then writes a 256x192 pattern to
`face_photo_in.png` and runs the whole-photo path into `gfpgan_photo_out.png` /
`codeformer_photo_out.png` (detection threshold 0.5, at most 3 faces). The weights are random, so the outputs are meaningless; the
smokes only exercise loading, detection, alignment, the forward pass, the paste
and PNG I/O. It must be run from this folder (the fixture paths are relative).

## Build

Open `FaceRestoration.lpi` in Lazarus, or:

```
lazbuild FaceRestoration.lpi
```

The binary lands in `../../bin/<cpu>-<os>/bin/`. For `--gpu`, the library finds
`neural/neural.cl` relative to this folder or to the binary.
