# CLIPScore (reference-free text↔image alignment metric)

**CLIPScore** (Hessel et al. 2021,
[arXiv:2104.08718](https://arxiv.org/abs/2104.08718)) is the standard
**reference-free** metric for text-to-image / image-captioning quality: how
well a (generated) image matches its prompt. It complements the **image-only**
FID / IS / KID (`neural/neuralimagemetrics.pas`) with a **semantic**
image↔text score — and needs no reference images at all, just a pretrained
CLIP.

It runs the image through the CLIP **vision** tower and the prompt through the
CLIP **text** tower, L2-normalizes both pooled embeddings, and takes the
clipped, scaled cosine:

```
CLIPScore = w · max(0, cos(image_embed, text_embed)),   w = 2.5
```

(the weight `w = 2.5` is the paper's choice so the score's typical range lands
in `[0, 1]`; the `max(0, ·)` clips the rare negative cosines to zero). For
captioning there is also **RefCLIPScore**, the harmonic mean of CLIPScore with
the best candidate↔reference-caption cosine — rewarding captions that match
**both** the image and the human references.

This is a **metric/helper, not a new layer**. It reuses the landed CLIP
dual-encoder importer `BuildClipFromSafeTensors`
(`neural/neuralpretrained.pas`) and the existing pooling helpers
`ClipTextEosPosition` / `ClipExtractEmbedding` / `ClipSimilarity`. The new
public helpers are:

* `ClipScore(TextNet, VisionNet, ImageInput, TokenIds, EosTokenId, Weight=2.5)`
  — end-to-end: runs both towers, pools (vision row 0, text at the eot row),
  L2-normalizes and returns `Weight · max(0, cosine)`.
* `ClipScoreFromEmbeddings(ImageEmb, TextEmb, Weight=2.5)` — same from two
  already-extracted unit-L2 embeddings.
* `RefClipScoreFromEmbeddings(ImageEmb, TextEmb, RefTextEmb, Weight=2.5)` —
  the RefCLIPScore harmonic-mean variant for captioning.

## Run it (offline, on the committed pico fixture)

```
examples/ClipScore/ClipScore
examples/ClipScore/ClipScore /path/to/clip-vit-base-patch32/model.safetensors
```

The default checkpoint is the committed `tests/fixtures/tiny_clip.*` pico
fixture, so it runs **offline** in under a second on pure CPU. The demo scores
one image against three prompts and shows the **mismatched** prompt scores
**lower** — its negative cosine clips CLIPScore to exactly `0` — while the
matching prompt scores higher; it then prints a RefCLIPScore. (On the random
pico checkpoint only the *known-mismatched* prompt is guaranteed to rank
lowest; on a real trained CLIP the matching caption ranks first. With a real
checkpoint the prompts are still synthetic token-id sequences — wiring the CLIP
BPE tokenizer and the resize/normalize image preprocessing is out of this
demo's scope.)

## Parity

`TestClipScore` (`tests/TestNeuralPretrained.pas`) pins the helper against the
torch float64 oracle that already ships in the CLIP fixture
(`tests/fixtures/tiny_clip_embeds.json`): since HF's
`logits_per_image = exp(logit_scale) · cosine`, the pure cosine is
`logit / exp(logit_scale)` and `CLIPScore = max(0, 2.5 · cosine)`. The
end-to-end `ClipScore` reproduces it to `< 1e-4`, the mismatched prompt clips
to exactly `0`, and `RefClipScoreFromEmbeddings` matches the harmonic-mean
formula to `< 1e-5`.

This example is coded by Claude (AI).
