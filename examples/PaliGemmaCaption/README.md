# PaliGemmaCaption: prefix-LM vision-language captioning

A generative vision-language captioning demo for the **PaliGemma** importer
(`BuildPaliGemmaFromSafeTensors`, `neural/neuralpretrained.pas`). What makes this
different from `../LlavaDescribe` is the attention regime: PaliGemma is a
**prefix-LM**.

## What it does

`BuildPaliGemmaFromSafeTensors` returns **three nets** — the SigLIP vision tower
(last_hidden_state, with post_layernorm), the single-linear multimodal projector,
and the Gemma language decoder. The example assembles a multimodal prompt
(`[<image>*NumPatches | prompt-text | generated-suffix]`) and greedily decodes a
short caption under the prefix-LM mask.

In a prefix-LM the image tokens **and** the prompt tokens (the "prefix") see each
other with **full bidirectional attention**; only the generated suffix is causal.
`PaliGemmaRunLogits` sets the bidirectional block to `PrefixLen` on every SDPA
layer (`TNNet.SetAttentionPrefixLen`) for the forward, then restores pure causal.
`PrefixLen` stays fixed at the original image+prompt length while the suffix grows.

## Running

Run from the repo root so the default fixture path resolves:

```
cd examples/PaliGemmaCaption
lazbuild PaliGemmaCaption.lpi --build-mode=Release   # or: fpc -Fu../../neural PaliGemmaCaption.lpr
# from the repo root:
examples/PaliGemmaCaption/PaliGemmaCaption
examples/PaliGemmaCaption/PaliGemmaCaption /path/to/paligemma/model.safetensors [config.json]
```

## Notes

- Default checkpoint is the committed **random** pico fixture
  `tests/fixtures/tiny_paligemma.safetensors`, so the "caption" is gibberish: the
  point is the image→text prefix-LM **plumbing**. The image is the fixture's
  deterministic dyadic test pattern (a real pipeline loads + SigLIP-normalizes a
  photo).
- The decoder is **rebuilt at the current prompt length** each step so the mask and
  positions line up (a KV-cache fast path is a follow-up); generation is short
  (`MaxNewTokens = 5`).
- Cap memory on a real checkpoint, e.g. `ulimit -v 12000000`. Inference only,
  pure CPU.

This example is coded by Claude (AI).
