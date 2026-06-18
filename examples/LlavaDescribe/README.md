# LLaVA image captioning

LLaVA (Liu et al. 2023, [arXiv:2304.08485](https://arxiv.org/abs/2304.08485))
is the classic **generative vision-language** recipe: a ViT vision tower turns
an image into patch features, a small MLP **projector** maps them into the
language model's embedding space, and those projected **visual tokens are
spliced into the decoder's token-embedding sequence** at the `<image>`
placeholder — then the model decodes text **causally**, exactly like a normal
LLM. No cross-attention; the image is "just more tokens".

This example drives `BuildLlavaFromSafeTensors`
(`neural/neuralpretrained.pas`), an image-in / text-out
importer of the decoder-only-with-projector kind (model_type `llava`:
llava-hf/llava-interleave-qwen-0.5b-hf and siblings). It is distinct from the
contrastive **dual encoders** (CLIP/SigLIP — score image/text, cannot
generate) and from the **encoder-decoder** captioners (BLIP/TrOCR —
cross-attention to the image). LLaVA is returned as **three nets**:

* **Vision net** — `BuildSigLIPVisionTower` / `BuildClipVisionTower` in
  `pVisionFeatures` mode. `vision_feature_layer = -1` runs every encoder block
  but **skips `post_layernorm`** (HF captures `hidden_states[-1]` *before* the
  post-norm); a more negative layer (CLIP's `-2`) drops trailing blocks. The
  output is the `(NumPatches, 1, vision_hidden)` patch features — no CLS row
  (SigLIP), no MAP pooling.
* **Projector net** — `multi_modal_projector.linear_1 → gelu → linear_2`
  (both biased), mapping `vision_hidden → text_hidden`, per token.
* **Text net** — the stock Llama/Qwen2 decoder (`BuildLlamaFromSafeTensors`);
  its `TNNetEmbedding` is fed **externally**.

The new plumbing (the LLaVA analogue of `RunT5`'s external-states feed — the
**embedding-injection** convention):

* `LlavaProjectImage` — vision tower + projector, once, → the visual tokens.
* `LlavaAssembleEmbeddings` — looks up each text token's embedding row and
  splices the projected visual tokens at the `image_token_index` slots into a
  `(SeqLen, 1, text_hidden)` sequence.
* `LlavaRunLogits` — **injects** that sequence into the decoder's
  embedding-layer output and runs the decoder from the next layer onward
  (skipping the token lookup so the splice survives) → next-token logits.

The multimodal prompt is rendered by the **`cfLlava`** chat template
(`neural/neuralchat.pas`): the llava_v1 vicuna preamble +
`"USER: <image>\n… ASSISTANT:"` turns.

## Run (offline, on the committed pico fixture)

From the repo root:

```
lazbuild examples/LlavaDescribe/LlavaDescribe.lpi
examples/LlavaDescribe/LlavaDescribe
```

The default checkpoint is `tests/fixtures/tiny_llava.safetensors` — the pico
random LLaVA (SigLIP vision + Qwen2 text) used by `TestLlavaVisualTokenParity`
and `TestLlavaNextTokenLogitsParity` (the projected visual tokens **and** the
mixed image+text next-token logits both match HF transformers' float64 oracle
to < 1e-4). Because the pico weights are random, the decoded caption ids are
**gibberish** — the demo exercises the image→text **plumbing**, not quality.

## Run on a real checkpoint

```
ulimit -v 12000000
examples/LlavaDescribe/LlavaDescribe /path/to/llava/model.safetensors
```

(`config.json` is read from the same directory; pass it as argument 2 to
override. A sharded `model.safetensors.index.json` or `pytorch_model.bin`
works too.) For a real caption you still need the checkpoint's tokenizer (to
turn the question into ids and decode the answer) and the SigLIP/CLIP image
preprocessing (`ReadClipImageProcessorConfig` + `ClipPreprocessImage`) — both
are out of this demo's plumbing scope. The KV-cache fast decode path (this
demo re-runs the full prompt per generated token) is a follow-up.
