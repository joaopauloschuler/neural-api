# Qwen2-Audio — audio-understanding chat

An **audio-in / text-out** chat demo for **Qwen2-Audio**. A Whisper audio tower
encodes a log-mel spectrogram into frames; the frames are average-pooled,
LayerNorm'd, and **linearly projected** into the language model's embedding
space; the projected audio frames are then **spliced into the decoder's token
sequence** at the special `<|AUDIO|>` placeholder slots, and a Qwen2 language
decoder answers the question by greedy causal decoding.

```
log-mel spectrogram
  -> Whisper audio TOWER            (TowerNet)
  -> avg_pooler -> LayerNorm        (PoolNormNet)
  -> linear PROJECTOR               (ProjectorNet)  -> audio frames in d_model
  -> splice frames at <|AUDIO|> slots in the prompt
  -> Qwen2 language DECODER         (TextNet)       -> greedy next-token answer
```

## The importer

`BuildQwen2AudioFromSafeTensors` (config `TQwen2AudioConfig`, printed by
`Qwen2AudioConfigToString`) returns **four** `TNNet`s — `TowerNet`,
`PoolNormNet`, `ProjectorNet`, `TextNet` — plus a nested config (`Config.Audio`
for the Whisper tower, `Config.Text` for the language model). The example feeds a
log-mel volume through `Qwen2AudioMelToInput`, then drives the whole stack with
`Qwen2AudioRunLogits` (tower → pooler → projector → splice → decoder forward) and
takes the argmax of the last logit row each step. Generation rebuilds the four
nets at the exact current prompt length per token (no KV-cache yet).

## Running

With **no arguments** it runs a **pico smoke test** on the committed fixture
(`tests/fixtures/tiny_qwen2audio.safetensors` + `tiny_qwen2audio_config.json`),
generating 6 greedy tokens from a deterministic test-pattern mel:

```
cd examples/Qwen2AudioChat
# build with lazbuild Qwen2AudioChat.lpi (or fpc), then:
./Qwen2AudioChat
```

The fixture weights are **untrained random**, so the generated token ids are
**gibberish** — the smoke test only proves the plumbing (audio tower → avg_pooler
→ LayerNorm → projector → spliced audio frames → causal decode).

With a **real checkpoint** it takes the safetensors path (and optionally an
explicit config path) as arguments:

```
ulimit -v 14000000
./Qwen2AudioChat /path/to/model.safetensors [/path/to/config.json]
```

The `ulimit -v` (~14 GB shown) bounds RAM for a real checkpoint.

This example is coded by Claude (AI).
