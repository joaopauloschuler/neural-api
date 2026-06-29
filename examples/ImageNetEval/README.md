# ImageNetEval example

End-to-end demo for the ImageNet top-1 / top-5 accuracy harness
`EvaluateImageNet` / `ImageNetReport` (`neural/neuralimagemetrics.pas`). This is
the import-VERIFICATION backstop for the landed classifier importers (ResNet /
ViT / Swin / DINOv2 / MobileNetV3 / VGG / Inception-v3 / EfficientNet) — the CV
analogue of [`../MMLUEval`](../MMLUEval) / [`../PerplexityEval`](../PerplexityEval)
on the LLM side.

Each classifier importer's parity test only compares raw **logits on one or two
tensors**, which catches a transposed weight but **not**:

* a wrong **preprocessing pipeline** (resize / center-crop / normalize), or
* a **label permutation** (the net is right but the class index → label map is off).

Running a folder of labelled ImageNet-val images through the real preprocessing
transform and the real net, then checking top-1 / top-5 against the published
numbers, is exactly that missing backstop.

## What the harness does

`EvaluateImageNet(NN, Samples, NumClasses, K, MaxConfusion)`:

* takes already-preprocessed `TNNetImageNetSample` records (a network-ready
  normalized `(ImageSize, ImageSize, 3)` volume + the gold class index);
* runs `NN.Compute` on each, reads the final layer's `NumClasses`-length output
  (logits or post-softmax probabilities — argmax / top-K is identical either way);
* forms top-1 and top-K via `TopKIndices` (most-confident first, **first-max**
  / lowest-index tie-break, matching `GetClass`);
* tallies **top-1** (argmax == gold) and **top-K** (gold anywhere in the top-K)
  accuracy, skipping any sample whose gold label is outside `0..NumClasses-1`;
* retains up to `MaxConfusion` **top-1 misses** for a confusion sample, each
  flagged top-K-**hit** (the common benign case) vs top-K-**miss**.

`ImageNetReport(Stats, ClassNames, Title)` formats the top-1 / top-5 lines plus
the confusion sample (rows labelled by `ClassNames` when supplied).

The transform itself is `neuraldatasets.PreprocessImageForVisionModel`:
shorter-side resize → center-crop → `(x/255 − csImageNetMean)/csImageNetStd`
(the standard torchvision val transform). `LoadImageForVisionModel` is the
load-from-file convenience wrapper.

## Default SMOKE run (no arguments)

To stay self-contained (no network fetch, no multi-GB download, no real
ImageNet) the default run:

1. builds a small CNN over a `24x24x3` input;
2. renders a tiny **deterministic** synthetic 6-class coloured-pattern set at
   `64x64` and pushes every image through the **same** `PreprocessImageForVisionModel`
   transform the real path uses (shorter-side resize 32 → center-crop 24 →
   ImageNet mean/std), so the smoke exercises the real transform end to end;
3. trains the CNN for 30 epochs (fixed `RandSeed`); and
4. evaluates a held-out synthetic split with the harness.

The synthetic classes are cleanly separable, so the run reports `1.0000 /
1.0000` — the point is the **harness + transform mechanics**, not a real number.
Runs in seconds under `ulimit -v 3000000`.

```bash
ulimit -v 3000000
timeout 300 ./ImageNetEval
```

The scoring path is **checkpoint-agnostic**: swap `BuildSmokeClassifier` for a
`BuildResNetFromSafeTensors` (or any classifier importer), feed
`LoadImageForVisionModel`-produced volumes into the same `TNNetImageNetSample`
records, and the harness is unchanged.

## Running against real ImageNet-val (`--full <dir>`)

`--full <dir>` prints the documented recipe (the binary ships the SMOKE wired so
CI stays self-contained — it deliberately does not bundle a multi-GB importer
call). To run real ImageNet-val:

1. **Folder layout.** Lay out `<dir>` as:

   ```
   <dir>/labels.txt          one "<filename> <class_index>" per line, e.g.
                             ILSVRC2012_val_00000001.JPEG 65
                             ILSVRC2012_val_00000002.JPEG 970
                             ...
   <dir>/ILSVRC2012_val_00000001.JPEG
   <dir>/ILSVRC2012_val_00000002.JPEG
   ...
   ```

   `<class_index>` is the standard 0..999 ImageNet (torchvision) class index for
   that image. (The official devkit ships `ILSVRC2012_validation_ground_truth.txt`
   in 1-based WNID order; map it to the torchvision class ordering your importer
   expects — a label permutation here is precisely the bug this harness exists to
   catch.)

2. **Import a backbone** and read its declared `ImageSize` and the ImageNet
   normalization constants:

   ```pascal
   NN := BuildResNetFromSafeTensors('resnet50.safetensors', Cfg);
   // ResizeSide is the resize-shorter-side target (e.g. 256 for a 224 crop);
   // ImageSize is the importer's center-crop / input size.
   ```

3. **Build samples** with the real transform:

   ```pascal
   LoadImageForVisionModel(FileName, V, ResizeSide, ImageSize,
     csImageNetMean, csImageNetStd);
   Sample.Image := V;
   Sample.GoldLabel := ClassIndex;
   Sample.SourceName := FileName;
   ```

4. **Evaluate and report:**

   ```pascal
   Stats := EvaluateImageNet(NN, Samples, 1000, 5);
   WriteLn(ImageNetReport(Stats, ClassNames, 'ResNet-50 ImageNet-val'));
   ```

Wiring a real checkpoint + the full 50 k-image ImageNet-val run (and comparing
top-1 / top-5 against the published torchvision numbers) is a documented
follow-up — see `tasklist.md`.
