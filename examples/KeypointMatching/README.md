# KeypointMatching: SuperPoint + LightGlue

Finds keypoints in two images with **SuperPoint** and matches them with
**LightGlue**, using the importers in `neural/neuralpretrained.pas`. The output
is one PNG with the two images side by side and a line for every match, colored
from red (score 0) to green (score 1). Use it for panorama stitching, visual
odometry, or re-ranking image retrieval results.

## Running

```
cd examples/KeypointMatching
lazbuild -B KeypointMatching.lpi
../../bin/x86_64-linux/bin/KeypointMatching                    # pico fixture smoke
../../bin/x86_64-linux/bin/KeypointMatching \
  --model lightglue_superpoint/model.safetensors \
  --image0 left.jpg --image1 right.jpg --output matches.png
```

| Option | Meaning |
| --- | --- |
| `--model FILE` | LightGlue `model.safetensors` (`ETH-CVG/lightglue_superpoint`) |
| `--config FILE` | its `config.json`. Default: `config.json` next to `--model` |
| `--superpoint FILE` | a separate SuperPoint `model.safetensors` (`magic-leap-community/superpoint`). Default: the `keypoint_detector.*` tensors inside `--model` |
| `--superpoint-config FILE` | the SuperPoint `config.json`. Default: next to `--superpoint` |
| `--image0 IMG`, `--image1 IMG` | the two images (PNG / JPEG / BMP) |
| `--output PNG` | the side-by-side image with the match lines |
| `--width N`, `--height N` | net input size, multiples of 8 (default 640 x 480, as `LightGlueImageProcessor`) |
| `--max-keypoints N` | keypoints kept per image, highest score first; `-1` keeps all (default: the config) |
| `--keypoint-threshold X` | SuperPoint score threshold (default: the config, 0.005) |
| `--filter-threshold X` | LightGlue match score threshold (default: the config, 0.1) |
| `--depth-confidence X` | early exit: stop after a layer when more than this share of the keypoints is confident. `<= 0` runs every layer (default: the config, 0.95) |
| `--width-confidence X` | point pruning: drop a confident keypoint whose matchability is at most `1 - X`. `<= 0` keeps every keypoint (default: the config, 0.99) |

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

## Limitations

- transformers only runs the early exit and the pruning together; with
  only one of them on, it fails. Here, early exit alone matches without
  pruning, and pruning alone uses only the matchability (as the original
  LightGlue code does). `--depth-confidence 0 --width-confidence 0` gives
  the all-layers matches, the most accurate setting.
- No OpenCL path.
- The real checkpoints have not been run yet.

## Smoke mode (no arguments)

With no arguments, the example writes a 64 x 48 blocky texture
(`keypoint_in0.png`) and the same texture shifted 8 pixels to the right
(`keypoint_in1.png`). It matches them with the committed random-weight pico
fixture `tests/fixtures/tiny_lightglue` and writes `keypoint_matches.png`.
This checks the import, the wiring and the image I/O. It is not a trained
matcher.

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

Coded by Claude (AI).
