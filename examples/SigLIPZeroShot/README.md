# SigLIP zero-shot classification (the sigmoid-loss dual encoder)

SigLIP (Zhai et al. 2023, [arXiv:2303.15343](https://arxiv.org/abs/2303.15343))
is a contrastive image-text **dual encoder** trained with a **sigmoid**
pairwise loss instead of CLIP's softmax contrastive loss. It is the de-facto
**vision tower of modern open VLMs** (LLaVA-style), so this importer unblocks
later VLM work. This example shows the zero-shot scoring structure on top of
`BuildSigLIPFromSafeTensors` (`neural/neuralpretrained.pas`), which returns the
dual encoder as **two independent nets**.

SigLIP reuses CLIP's pre-LN encoder block but is architecturally **DISTINCT**
from CLIP (handled on its own path, NOT force-fit onto the CLIP code):

* **score head** — a learnable `logit_scale` **AND** `logit_bias` (CLIP has
  scale only). `SigLIPLogit` computes `exp(logit_scale) * cosine + logit_bias`
  — HF's `logits_per_image` entry — and the native SigLIP output is the
  per-pair **sigmoid** `1/(1+exp(-logit))`: each class is an INDEPENDENT
  yes/no match, not a softmax over classes.
* **TEXT net** — token embedding + learned positions, **BIDIRECTIONAL** pre-LN
  blocks (no causal mask, unlike CLIP), `final_layer_norm`, then a **biased**
  `head` nn.Linear (`text_model.head`) applied per token. HF's `text_embeds`
  is the **LAST** token's row (NOT CLIP's eos-argmax).
* **VISION net** — a **biased** patch conv (kernel = stride = patch size),
  **NO class token**, learned positions over exactly `num_patches` rows,
  bidirectional pre-LN blocks, `post_layernorm`, then a **Multihead Attention
  Pooling head** (MAP): one learnable probe query (`TNNetSoftPrompt`)
  cross-attends over the patch tokens (`TNNetCrossAttention`), then
  `out = attn + mlp(LayerNorm(attn))`; row 0 is HF's `image_embeds`. The tower
  is factored as the reusable `BuildSigLIPVisionTower`, which offers a
  `pVisionFeatures` skip-pooling / select-hidden-layer mode for LLaVA/VLM
  consumers that want the raw `(num_patches, 1, hidden)` patch states.
* **activation** — `gelu_pytorch_tanh` (the tanh-approx GELU) in both towers
  and the MAP head MLP — not CLIP's quick_gelu.

## Running (offline)

The default checkpoint is the committed pico fixture `tests/fixtures/
tiny_siglip.*`, so it runs with no network access:

```
examples/SigLIPZeroShot/SigLIPZeroShot
examples/SigLIPZeroShot/SigLIPZeroShot /path/to/siglip-base-patch16-224/model.safetensors
```

It embeds one deterministic test image and N class-prompt token sequences and
prints, per class, the raw logit, the native **sigmoid** match probability and
a softmax ranking. With a real checkpoint the prompts are still synthetic
token-id sequences (the SigLIP SentencePiece tokenizer and the image
resize/normalize pipeline are out of this demo's scope; SigLIP pads text to
`max_length`, so the LAST position is meaningful). Parity of the importer is
verified `< 1e-4` against a float64 HF `SiglipModel` oracle in
`TestSigLIPParity` (generator `tools/siglip_tiny_fixture.py`).

## Follow-up

v1 scopes to the **fixed-resolution** siglip / siglip2 base configs.
**NaFlex / variable-resolution** siglip2 (per-image patch counts + attention
pooling over a variable token grid) is an explicit open follow-up.
