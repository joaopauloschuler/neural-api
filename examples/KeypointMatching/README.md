# KeypointMatching: SuperPoint + LightGlue, or EfficientLoFTR

Matches two images with the importers in `neural/neuralpretrained.pas`, in one
of two ways:

- **SuperPoint + LightGlue** (the default): SuperPoint finds keypoints and
  LightGlue matches them.
- **EfficientLoFTR** (`--matcher loftr`): detector-free semi-dense matching on
  a 1/8 grid, refined to sub-pixel positions. It needs no keypoint detector, so
  it keeps finding matches on low-texture walls, floors and repeated patterns,
  where SuperPoint fires on few points.

The output is one PNG with the two images side by side and a line for every
match, colored from red (score 0) to green (score 1). Use it for panorama
stitching, visual odometry, or re-ranking image retrieval results.

## Running

```
cd examples/KeypointMatching
lazbuild -B KeypointMatching.lpi
../../bin/x86_64-linux/bin/KeypointMatching                    # pico fixture smoke
../../bin/x86_64-linux/bin/KeypointMatching \
  --model lightglue_superpoint/model.safetensors \
  --image0 left.jpg --image1 right.jpg --output matches.png
../../bin/x86_64-linux/bin/KeypointMatching --matcher loftr      # pico smoke
../../bin/x86_64-linux/bin/KeypointMatching --matcher loftr \
  --model efficientloftr/model.safetensors --width 256 --height 192 \
  --image0 left.jpg --image1 right.jpg --output matches.png
```

| Option | Meaning |
| --- | --- |
| `--matcher NAME` | `lightglue` (SuperPoint + LightGlue, the default) or `loftr` (EfficientLoFTR) |
| `--model FILE` | LightGlue `model.safetensors` (`ETH-CVG/lightglue_superpoint`), or with `--matcher loftr` the EfficientLoFTR `model.safetensors` (`zju-community/efficientloftr`) |
| `--config FILE` | its `config.json`. Default: `config.json` next to `--model` |
| `--superpoint FILE` | a separate SuperPoint `model.safetensors` (`magic-leap-community/superpoint`). Default: the `keypoint_detector.*` tensors inside `--model` |
| `--superpoint-config FILE` | the SuperPoint `config.json`. Default: next to `--superpoint` |
| `--image0 IMG`, `--image1 IMG` | the two images (PNG / JPEG / BMP) |
| `--output PNG` | the side-by-side image with the match lines |
| `--width N`, `--height N` | net input size, multiples of 8, or of 32 with `--matcher loftr` (default 640 x 480, as `LightGlueImageProcessor` and `EfficientLoFTRImageProcessor`) |
| `--max-keypoints N` | keypoints kept per image, highest score first; `-1` keeps all (default: the config) |
| `--keypoint-threshold X` | SuperPoint score threshold (default: the config, 0.005) |
| `--filter-threshold X` | LightGlue match score threshold (default: the config, 0.1) |
| `--depth-confidence X` | early exit: stop after a layer when more than this share of the keypoints is confident. `<= 0` runs every layer (default: the config, 0.95) |
| `--width-confidence X` | point pruning: drop a confident keypoint whose matchability is at most `1 - X`. `<= 0` keeps every keypoint (default: the config, 0.99) |
| `--coarse-threshold X` | `--matcher loftr`: the coarse confidence a match must exceed (default: the config, 0.2) |

## What runs

1. `SuperPointPreprocessImage` resizes each image to `--width x --height` (PIL
   bilinear), scales it to `[0,1]` and converts it to gray, as
   `LightGlueImageProcessor` does.
2. The SuperPoint net (`BuildLightGlueKeypointDetectorFromSafeTensors`, or
   `BuildSuperPointFromSafeTensors` with `--superpoint`) and
   `TSuperPointKeypoints.Detect` find the keypoints: NMS, the score threshold,
   border removal, top-k, and descriptors sampled at each keypoint.
3. `TLightGlueMatcher.Match` (`BuildLightGlueFromSafeTensors`) runs the
   matcher on both keypoint sets. Each layer has two blocks. The first is
   self-attention with a rotary encoding whose angles are a learned linear map
   of the normalized keypoint coordinates. The second is cross-attention to
   the other image. Each block ends in an MLP over `[x, attention]` and a
   residual. The match head scores every pair with a double log-softmax over
   the similarity plus a matchability sigmoid. It keeps a pair when each point
   is the other's best match and the score is above `filter_threshold`.
   After each layer but the last, a per-layer token confidence head scores
   every keypoint, as in transformers. When the share of confident keypoints
   is above `depth_confidence`, the matcher stops and matches with that
   layer's match head. Otherwise it drops each keypoint that is confident but
   whose matchability is at most `1 - width_confidence`. The confidence
   threshold is `0.8 + 0.1 exp(-4 layer / layers)`. Pruned keypoints get no
   match. The output line reports the stop layer and the kept keypoints.
4. `TSuperPointKeypoints.ImageX` / `ImageY` map the matched keypoints back to
   each original image size, as transformers'
   `post_process_keypoint_matching` does, and the example draws them.

`TLightGlueMatcher` does its math on the host with `TNNetPlainOps` (threaded
per-token linears) and `TNNetVolume.DotProductsPacked` (attention and
similarity). It is not a `TNNet`, because the keypoint count changes with every
image pair.

## What runs with `--matcher loftr`

1. `SuperPointPreprocessImage` resizes each image to `--width x --height` (PIL
   bilinear), scales it to `[0,1]` and converts it to gray, as
   `EfficientLoFTRImageProcessor` does.
2. `BuildEfficientLoFTRFromSafeTensors` builds one `TNNet` with two gray
   inputs:
   - **Backbone.** A RepVGG backbone runs on each image, with the same weights
     loaded twice. Each block's 3x3, 1x1 and identity branches are folded into
     one 3x3 convolution by `ReparamRepVGGBlock`. It produces 1/2, 1/4 and 1/8
     feature maps.
   - **Attention.** Four layers of aggregated attention run on the 1/8 maps:
     self-attention on each image, then cross-attention 0 <- 1 and
     1 <- (updated 0). Each block computes queries with a 4x4 stride-4
     depthwise convolution and keys/values with a 4x4 max pool, applies a
     LayerNorm and then 8-head attention (`TNNetFusedSDPA`) on the reduced
     grid. Self-attention adds a 2-D rotary encoding:
     `TNNetAxialRotaryEmbedding` plus a channel reorder. The block upsamples
     the result bilinearly back to the 1/8 grid, then adds
     LayerNorm(MLP([x, attention])) to x.
   - **Fine pyramid.** Per image, a fine-fusion pyramid takes the 1/8 map back
     to full resolution through the 1/4 and 1/2 backbone maps.
3. `TEfficientLoFTRMatcher.Match` runs the net and then three steps on the
   host:
   - It scores every pair of 1/8 cells with a dual softmax: softmax over image
     0 times softmax over image 1, of the scaled similarity divided by
     temperature 0.1.
   - It keeps the mutual nearest neighbours above the threshold (0.2), at
     least 2 cells from the border.
   - It refines each match in two stages, like the original EfficientLoFTR
     code. The first stage takes the best pair inside the 8x8 window of image
     0 and the 10x10 window of image 1. The second stage adds a 3x3
     soft-argmax offset to the image-1 point.
4. `TEfficientLoFTRMatcher.ImageX` / `ImageY` map the matched points back to
   each original image size, and the example draws them with the same
   renderer as LightGlue.

**Deviations from transformers.** The fine refinement follows the original
EfficientLoFTR code (zju3dv/EfficientLoFTR `fine_matching.py`), not
transformers 5.17. transformers' extra batch axis moves the fine dual softmax
and the window-grid gather onto the keypoint axis. It also pairs image-0 rows
ordered by image 1 with image-1 rows ordered by image 0, and it scales the
first-stage features by the sliced width instead of the full fine width. The
coarse confidence and the coarse matches are the same as transformers'.

## Limitations

- transformers only runs the early exit and the pruning together; with
  only one of them on, it fails. Here, early exit alone matches without
  pruning, and pruning alone uses only the matchability (as the original
  LightGlue code does). `--depth-confidence 0 --width-confidence 0` gives
  the all-layers matches, the most accurate setting.
- No OpenCL path.
- The real LightGlue / SuperPoint checkpoints have not been run yet.
- `--matcher loftr` needs about 22 KB of RAM per input pixel on the CPU:
  1.35 GB at 256 x 192, about 6.9 GB at the 640 x 480 default. Most of it is
  the per-layer convolution scratch (the im2col matrix, the padded input copy
  and the raw output), which every 3x3 convolution keeps for the life of the
  net. Use a smaller `--width` / `--height` on a small machine.
- `zju-community/efficientloftr` was checked at 256 x 192 only, against
  transformers on `docs/park.jpg` and a cropped copy of it. All 463 coarse
  matches agree. The confidence differs by up to 9.2e-6 and the keypoints by
  up to 1.5e-5 pixels, with the keypoints checked against the original
  refinement. The forward takes 4.8 s on one core. It has not been user-tested
  on real image pairs.

## Smoke mode (no arguments)

With no arguments, the example writes a 64 x 48 blocky texture
(`keypoint_in0.png`) and the same texture shifted 8 pixels to the right
(`keypoint_in1.png`). It matches them with the committed random-weight pico
fixture `tests/fixtures/tiny_lightglue` and writes `keypoint_matches.png`.
This checks the import, the wiring and the image I/O. It is not a trained
matcher. With `--matcher loftr`, the smoke pair is 128 x 96 and the fixture is
`tests/fixtures/tiny_efficientloftr`.

Tests (`tests/TestNeuralPretrained.pas`): `TestLightGlueParity` (an input
projection from 8 to 16 channels) and `TestLightGlueGQAParity` (an identity
projection with 2 query heads over 1 key/value head). Both compare against the
transformers 5.17 `LightGlueForKeypointMatching` float64 forward, with early
exit and pruning disabled: keypoints and match indices exact, final descriptors
and match scores `< 1e-4`. `TestLightGlueEarlyExitParity` (pruning at layer
0, early exit at layer 1 of 3) and `TestLightGlueGQAPruningParity` (pruning
at layer 0, no early exit) compare the same pairs with both on: stop layer,
kept keypoints, their descriptors, matches and scores. Also
`TestLightGluePruningWithoutEarlyExit`, `TestLightGlueConfigFromJSONFile` and
`TestLightGlueNoKeypoints`. The fixtures are written by
`tools/make_pico_lightglue_fixture.py`.

`TestEfficientLoFTRParity` checks the EfficientLoFTR pico net and matcher.
It compares against the transformers 5.17 `EfficientLoFTRForKeypointMatching`
float64 forward:
- the coarse confidence over every cell pair: `< 1e-4` (measured 9.0e-6);
- the match cells: exact; the match scores: `< 1e-4`;
- the fine windows: `< 1e-4`.

The refined keypoints are compared with the original refinement run on those
windows: `< 1e-4` (equal to float32 rounding). Also
`TestEfficientLoFTRConfigFromJSONFile` and
`TestEfficientLoFTRRejectsInputSize`. The fixture is written by
`tools/make_pico_efficientloftr_fixture.py`.

Coded by Claude (AI).
