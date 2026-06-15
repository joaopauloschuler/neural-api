# BLIP image captioning

A **generative** vision-language importer of the
**encoder-decoder** kind. A ViT image encoder feeds a BERT-style **causal text
decoder** through **cross-attention**, which autoregressively generates a
caption with greedy decoding.

`BuildBlipForCaptioningFromSafeTensors` (`neural/neuralpretrained.pas`) imports a
HuggingFace `BlipForConditionalGeneration` checkpoint (model_type `blip`, e.g.
`Salesforce/blip-image-captioning-base`) and returns **two nets**:

* **VisionNet** — a ViT image encoder (biased patch conv, a class token folded
  into position row 0, pre-LN encoder blocks, `post_layernorm`). It emits every
  `num_patches+1` post-LN hidden-state row (the `vision_model.last_hidden_state`
  the decoder cross-attends to — no pooling).
* **TextNet** — a BERT-style POST-norm decoder whose **second `TNNetInput`**
  holds the image hidden states (the `T5EncoderStatesInput` convention shared
  with the T5/Marian/Pegasus importers). Each block is: causal self-attention →
  add+LN → **cross-attention to the image** (`TNNetCrossAttention`) → add+LN →
  exact-erf GELU FFN → add+LN; then the BERT LM head
  (`transform = LN(GELU(dense(x)))`, then the vocab decoder).

## Run (offline, on the committed pico fixture)

```
examples/ImageCaptioning/ImageCaptioning
```

or pass a real checkpoint:

```
examples/ImageCaptioning/ImageCaptioning /path/to/blip-image-captioning-base/model.safetensors
```

The demo encodes the fixture's deterministic test image **once**, then
`DecodeBlipCaptionGreedy` rolls out the caption from `bos_token_id`, stopping at
`sep`/`eos`. It prints the generated token ids; decode them with the BLIP
WordPiece tokenizer to get text (decode-to-text plumbing is the follow-up).

## Parity

The importer is float64-parity-verified against real HuggingFace transformers on
a committed pico fixture (`tests/fixtures/tiny_blip.*`, generator
`tools/make_pico_blip_fixture.py`):

* `TestBlipCaptioningParity` — per-position next-token logits over the vocab,
  max |diff| < 1e-4 vs the float64 oracle (given a tiny image + a short input
  token prefix);
* `TestBlipCaptionGreedy` — the autoregressive greedy caption ids match HF
  `generate()` exactly.

Because the fixture is randomly initialized, its caption is meaningless as
quality — this verifies the **import + cross-attention + generation structure**.
Pure CPU, runs in under a second on the fixture.
