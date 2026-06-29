# VideoAction — imported VideoMAE video classifier

Classifies a short synthetic video **clip** (T frames → an action label) with an
**imported** VideoMAE spatiotemporal transformer
(`BuildVideoMAEFromSafeTensorsEx`, `neuralpretrained.pas`) — a
video-classification importer. Pure CPU, no external dataset, finishes in well
under a second.

## What it shows

The committed pico checkpoint `tests/fixtures/tiny_videomae.safetensors` (a
randomly-initialised HF `VideoMAEForVideoClassification`, the same fixture the
parity test asserts float64-exact against) is loaded end-to-end:

```
tubelet 3-D conv (TNNetConvolution3D)
  -> fixed sin-cos 3-D position table
  -> stock pre-LN transformer encoder (joint space-time attention)
  -> mean-pool over tokens -> fc_norm -> linear classifier -> action logits
```

A clip is a `(num_frames, H, W, C)` volume packed for the importer as
`(W, H, num_frames*C)` — frame `t`'s `C` channels contiguous at depth
`[t*C .. t*C+C)`. The program synthesizes two clips (a bright blob sliding
**right** and one sliding **down**), feeds each through the imported net via
`RunVideoMAELogits`, and prints the per-class action logits + the argmax label.

The pico weights are **random**, so the predicted label is not semantically
meaningful; the point is that the importer + forward path runs on CPU and produces
calibrated logits.

> Related: `examples/VideoActionTiny` is a *from-scratch, trained* counterpart —
> it trains a small `TNNetConvolution3D` stack on a synthetic motion-direction
> task (no importer, no transformer). This example, by contrast, exercises the
> full imported VideoMAE transformer in inference only.

## Build & run

Run from the **repo root** (the importer reads `tests/fixtures/`):

```
lazbuild examples/VideoAction/VideoAction.lpi
./bin/x86_64-linux/bin/VideoAction
```

Or compile directly with fpc:

```
fpc -O3 -Mobjfpc -Sh -Funeural examples/VideoAction/VideoAction.lpr
```

## Output

The VideoMAE config (`VideoMAEConfigToString`), the derived clip / tubelet-token /
class-count summary, and for each of the two clips a line of per-class logits with
the argmax action class.
