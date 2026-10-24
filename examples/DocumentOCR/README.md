# DocumentOCR: Text of a Whole Page or Photo

Reads all the text of a page or photo. Two importers from
`neuralpretrained.pas` work together:

1. a **docTR DBNet** text detector (Liao et al., AAAI 2020,
   [*Real-time Scene Text Detection with Differentiable
   Binarization*](https://arxiv.org/abs/1911.08947)) finds the word boxes;
2. a **TrOCR** recognizer (Li et al. 2021,
   [arXiv:2109.10282](https://arxiv.org/abs/2109.10282), for example
   `microsoft/trocr-small-printed`) reads each box.

**Status: coded, not yet tested on real checkpoints.** Both importers are
parity-tested on random pico nets (`TestDBNetMobileNetV3LargeParity`,
`TestDBNetResNet50Parity`, `TestTrOCRParity`), and the pipeline runs end to end
on those pico nets (`--smoke`). No real docTR or TrOCR checkpoint has been run
through this example yet, and the DBNet tensor names are not yet confirmed
against a released docTR checkpoint (see `tasklist.md`).

## How it works

```
page or photo (RGB)
  -> DetectDBNetTextBoxes          docTR resize + pad + normalize, DBNet, DB post-processing
  -> SortDBNetTextBoxesInReadingOrder   lines top to bottom, words left to right
  -> MergeDBNetTextBoxesIntoLines      (--merge-lines) one box per run of words in a line
     (--rotated: DetectDBNetRotatedTextBoxes + SortDBNetRotatedTextBoxesInReadingOrder,
      then CropRotatedRect turns each box upright for TranscribeTrOCRImage)
  -> for each box: TranscribeTrOCRImageRegion
       crop, TrOCR resize + normalize, DeiT encoder once,
       greedy decode with the self-attention KV cache
  -> text, one output line per text line
```

- `DetectDBNetTextBoxes` returns axis-aligned word boxes in image pixels
  (docTR's straight-page post-processing).
- `SortDBNetTextBoxesInReadingOrder` sorts the boxes by their vertical centre
  and starts a new line when a box's centre passes the mean bottom edge of the
  boxes already in the current line. Inside a line, boxes go left to right.
- With `--merge-lines`, `MergeDBNetTextBoxesIntoLines` unions the word boxes
  of each line into one line box, so TrOCR reads a whole line crop, which is
  what it was trained on. A horizontal gap wider than `--line-gap` times the
  mean box height of the run so far (default 2) starts a new box in the same
  line, so two columns stay apart. The line box score is the mean word score.
- With `--rotated` (docTR `assume_straight_pages=False`),
  `DecodeDBNetRotatedTextBoxes` takes the convex hull of each text region
  (`ConvexHullOfPoints`), its minimum-area rectangle by rotating calipers
  (`MinAreaRectOfConvexHull`, as `cv2.minAreaRect`) and grows that rectangle
  on every side by D = hull area x unclip ratio / hull perimeter. docTR
  instead offsets the contour polygon by D with pyclipper (round joins) and
  then takes `cv2.minAreaRect`; for a convex region the two agree up to the
  orientation choice and pyclipper's arc approximation. As in docTR, the score
  is the mean non-zero probability inside the filled outer contour (the region
  plus its holes), and each corner coordinate is clipped to the image, so a box
  at the border can become a non-rectangular quadrilateral. `CropRotatedRect`
  then warps the parallelogram of corners 0, 1, 2 to an upright crop of
  |c1 - c0| x |c2 - c1| pixels, as docTR's `extract_rcrops` does, with bilinear
  sampling (`WarpImageAffine`, white outside the image). `--rotated` and
  `--merge-lines` do not combine.
- The TrOCR decoder is built with `DecSeqLen=1`, so `DecodeTrOCRGreedy` feeds
  one token per step through `TNNetStreamingDecoder` (KV cache) instead of
  re-running the whole prefix every step.

## Run

```
./DocumentOCR --smoke --boxes smoke_boxes.png       # pico fixture smoke run
./DocumentOCR --smoke --merge-lines                  # same, one crop per text line
./DocumentOCR --smoke --rotated --boxes rot.png     # same, rotated boxes
./DocumentOCR --detector db_mobilenet_v3_large.safetensors \
  --recognizer trocr-small-printed/model.safetensors \
  --input page.png --boxes page_boxes.png
```

| option | meaning | default |
| --- | --- | --- |
| `--smoke` | run the committed pico fixtures on a synthetic page | on when neither `--detector` nor `--recognizer` is given |
| `--detector FILE` | docTR DBNet checkpoint (`db_mobilenet_v3_large` or `db_resnet50`) | required outside `--smoke` |
| `--detector-config FILE` | docTR `config.json` (`mean`, `std`, `input_shape`, thresholds) | `config.json` next to `--detector`, else docTR's defaults (1024x1024) |
| `--recognizer FILE` | TrOCR `model.safetensors` (or `pytorch_model.bin`) | required outside `--smoke` |
| `--recognizer-config FILE` | TrOCR `config.json` | `config.json` next to `--recognizer` |
| `--processor FILE` | TrOCR `preprocessor_config.json` | next to `--recognizer`, else a bilinear square resize with mean = std = 0.5 |
| `--tokenizer FILE` | `tokenizer.json` that turns the ids into text | next to `--recognizer`; without one the ids are printed |
| `--input IMG` | page or photo, PNG / JPEG / BMP | required outside `--smoke` |
| `--boxes IMG` | also write the image with the boxes drawn, one colour per text line, numbered in reading order | off |
| `--max-tokens N` | token cap per box | 32 |
| `--bin-threshold X`, `--box-threshold X` | DBNet pixel and box thresholds | the detector config |
| `--merge-lines` | read one crop per text line (word boxes merged) instead of one per word | off |
| `--line-gap FACTOR` | with `--merge-lines`: a gap wider than FACTOR x the box height splits the line | 2 |
| `--rotated` | rotated boxes, each crop turned upright before TrOCR; `--boxes` draws the rotated outlines | off |
| `--int8` | store the weights of all three nets in int8 (`TNNet.QuantizeWeightsInt8`) | FP32 |
| `--gpu`, `--gpu-platform N`, `--gpu-device N` | run the nets on OpenCL (a build with `-dOpenCL`, the `.lpi` default) | CPU |

The smoke run must start from the repository root or from this folder (the
fixtures are found in `tests/fixtures`). The weights are random, so the boxes
and token ids do not mean anything; the smoke run only exercises loading,
detection, reading order, cropping, the cached decode and the box drawing.

Not checked yet:

- docTR publishes PyTorch `.pt` files; converting one to `.safetensors` and
  confirming its tensor names is the open DBNet task in `tasklist.md`.
- The `trocr-small-*` checkpoints ship a SentencePiece `sentencepiece.bpe.model`
  with the fairseq id layout. Whether that file, loaded through `--tokenizer`,
  gives the right text has not been checked; a `tokenizer.json` is the
  intended input.
- docTR detects words, and TrOCR was trained on text lines. Reading single
  words works in principle; `--merge-lines` gives TrOCR line crops instead.
  The character accuracy of the two modes on a real page has not been
  measured.
- `--rotated` has only run on the pico smoke and the hand-made unit tests,
  not on a real rotated page.

Coded by Claude (AI).
