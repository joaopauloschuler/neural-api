# Blip2Caption: the BLIP-2 Q-Former vision-language bridge

A CPU smoke of the **BLIP-2 Q-Former** importer
(`BuildBlip2FromSafeTensors` / `BuildBlip2QFormerFromSafeTensors`,
`neural/neuralpretrained.pas`). BLIP-2 (Li et al. 2023,
[arXiv:2301.12597](https://arxiv.org/abs/2301.12597)) bridges a frozen vision
encoder to a frozen LLM with a small **querying transformer** (the Q-Former).

## What it does

A fixed set of learned **query tokens** is fed through a few BERT-style blocks
that, in each block, **self**-attend among themselves and **cross**-attend into
the (frozen) ViT patch features, distilling the image into N query embeddings. A
`language_projection` linear then maps those query embeddings into the LLM token
space, where a real pipeline splices them ahead of the prompt and a FLAN-T5
decoder produces a caption.

This v1 demo exercises the **new piece**, the Q-Former (interleaved
self/cross-attention + the two-source cross-attention into the ViT features + the
BERT post-LN FFN) and the linear projector. The vision tower
(`BuildClipVisionTower`) and the FLAN-T5 decode tail (`BuildT5FromSafeTensors`,
the `T5EncoderStatesInput` two-net convention) are documented as reuse, not built
here. It feeds the learned `query_tokens` and a **deterministic synthetic** ViT
patch grid, runs the bridge and prints the projected query embeddings that would
splice into the LLM.

## Running

```
cd examples/Blip2Caption
lazbuild Blip2Caption.lpi --build-mode=Release   # or: fpc -Fu../../neural Blip2Caption.lpr
./Blip2Caption                              # committed pico fixture
./Blip2Caption model.safetensors [config.json]   # a real full-blip2 checkpoint
```

## Notes

- Default input is the committed config-faithful **random** pico fixture
  `tests/fixtures/tiny_blip2_full.safetensors` (built from the real HF
  `Blip2QFormerModel` float64 oracle, parity-checked < 1e-4 in
  `TestBlip2QFormerParity` / `TestBlip2FullBridgeParity`). Run from
  `examples/Blip2Caption/` so the relative fixture path resolves.
- Pico weights are random → the printed embeddings are a **wiring/throughput
  smoke, not a trained caption**. The header lists the real-checkpoint recipe.
- Inference only, pure CPU.

Coded by Claude (AI).
