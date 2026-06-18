# CLIP zero-shot classification

CLIP (Radford et al. 2021, [arXiv:2103.00020](https://arxiv.org/abs/2103.00020))
classifies images **zero-shot**: embed the image with one tower, embed a text
prompt per candidate class with the other, and softmax the scaled cosine
similarities — no task-specific training at all. This example shows that exact
structure on top of `BuildClipFromSafeTensors`
(`neural/neuralpretrained.pas`), which returns the contrastive dual encoder as **two
independent nets** (the T5/Marian two-net convention, but as peers — no
cross-attention):

* **TEXT net** — token embedding + learned positions, **causal** pre-LN
  blocks with quick_gelu (`x*sigmoid(1.702x)`, built as
  `TNNetSwishLearnable(1.702)`), `final_layer_norm`, then the bias-free
  `text_projection` per token. HF's `text_embeds` is the row at the **eot
  position**: `ClipTextEosPosition` implements both modeling_clip pooling
  branches (the legacy `eos_token_id = 2` ARGMAX-of-ids rule every published
  OpenAI CLIP uses, and the fixed first-eos rule).
* **VISION net** — bias-free patch conv (kernel = stride = patch size), the
  learned class token folded into row 0 of the position table,
  `pre_layrnorm`, **bidirectional** pre-LN blocks, `post_layernorm` +
  bias-free `visual_projection` per token. Row 0 (the class token) is HF's
  `image_embeds`. The tower is factored as the reusable
  `BuildClipVisionTower` for future ViT/DINO/SigLIP imports.

Scoring is HF's: `logits_per_image = exp(logit_scale) * cosine`, with
`ClipExtractEmbedding` (slice one row + L2 normalize) and `ClipSimilarity`
(dot product of unit vectors).

## Run (offline, on the committed pico fixture)

From the repo root:

```
lazbuild examples/ClipZeroShot/ClipZeroShot.lpi
examples/ClipZeroShot/ClipZeroShot
```

The default checkpoint is `tests/fixtures/tiny_clip.safetensors` — the pico
random CLIP used by `TestClipParity` (both towers match HF transformers'
float64 oracle to ~7e-7). Output:

```
Zero-shot class probabilities for the test image:
  class prompt #0 (eot mid-sequence)   cosine*scale =  -2.8182   p =   0.79%
  class prompt #1 (eot last)           cosine*scale =   1.2435   p =  46.03%
  class prompt #2 (synthetic)          cosine*scale =   1.3879   p =  53.18%
```

The first two logits are byte-comparable to the fixture's HF
`logits_per_image` reference (`tests/fixtures/tiny_clip_embeds.json`).
Prompt #0 deliberately carries its eot token **mid-sequence**: the causal
text tower provably ignores everything after it, and pooling anywhere else
would change the score.

## Run on a real checkpoint

```
examples/ClipZeroShot/ClipZeroShot /path/to/clip-vit-base-patch32/model.safetensors
```

(`config.json` is read from the same directory; pass it as argument 2 to
override. `pytorch_model.bin` works too.) The prompts stay synthetic token
ids — the CLIP byte-level BPE tokenizer (readable with
`neuralhftokenizer.pas`) and the resize/center-crop/normalize image
preprocessing are out of this demo's scope; real text/images just swap in
real token ids and pixel values.
