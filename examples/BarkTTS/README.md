# BarkTTS — text-to-speech with the Bark GPT cascade

A **text-to-speech generative** smoke for the Bark importer (`model_type`
`bark`, `suno/bark[-small]`). Bark is the autoregressive GPT-style TTS family:
**three stacked GPT-2-style decoders** chained, then the landed **EnCodec
decoder** (reused from the MusicGen path) turns the codes into a waveform.

```
text+semantic tokens
  -> SEMANTIC model (BarkCausalModel)            -> semantic tokens
  -> COARSE model   (BarkCausalModel)            -> coarse EnCodec codebooks
  -> FINE model     (BarkFineModel, NON-causal over the codebook axis)
       given codebooks 0..idx, predicts codebook idx for every frame at once
  -> EnCodec audio DECODER (BuildEnCodecFromSafeTensors) -> waveform -> WAV
```

## The three sub-models

Each sub-model is a **pre-norm GPT-2 block stack** with a learned positional
embedding and `nn.GELU` (**exact erf**). Unlike GPT-2's HF `Conv1D` (`[in,out]`),
Bark uses `nn.Linear` (`[out,in]`): a fused `att_proj` (`3*hidden` q|k|v),
`out_proj`, and the MLP `in_proj`/`out_proj`, with biases gated by
`config.bias`; the `lm_head`(s) are bias-free.

* **SEMANTIC / COARSE** (`BarkCausalModel`): one embedding table, one `lm_head`,
  **causal** self-attention.
* **FINE** (`BarkFineModel`): the subtle one. It is **NON-causal over the
  codebook axis**: there is one embedding table and one `lm_head` **per
  EnCodec codebook** (`n_codes_total` tables, `n_codes_total - n_codes_given`
  heads). For a target codebook `idx`, the merged input is the **sum of the
  codebook embeddings `0..idx`** (the already-known codebooks), the trunk runs
  with **bidirectional** attention over time, and head `idx - n_codes_given`
  predicts codebook `idx`. The holder (`TBarkSubModel.ComputeFineLogits`) does
  the codebook-sum and per-head selection in Pascal so this is explicit.

## Running

With **no arguments** this runs a self-contained pico smoke on the committed
random fixtures (`tests/fixtures/tiny_bark_*.safetensors` + the matched
`tiny_musicgen_encodec.*` codec), runs all three stages end-to-end, and writes
a short `bark_tts_demo.wav` via `SaveVolumeToWav16`. The weights are **untrained
random** so the clip is noise, not speech — the point is to exercise the full
wiring under the time/memory budget (pure CPU, a fraction of a second).

## Parity

`tools/make_pico_bark_fixture.py` builds the re-randomized pico fixture and a
self-contained float64 HF oracle (`BarkSemanticModel` / `BarkCoarseModel` /
`BarkFineModel` in `.double()`). `TestBarkParity` pins all **three** sub-models'
forward logits to the oracle at `< 1e-4`: the semantic and coarse next-token
logits over fixed id sequences, and the fine model's codebook-conditioned
non-causal logits per `(sequence, codebook_idx)` case. The generator also runs
self-checks proving each quirk (positional embedding, fine codebook
conditioning, fine bidirectional-over-time attention) provably moves the oracle.

## Follow-ups

Real `suno/bark` key-mapping (one nested checkpoint with `semantic` /
`coarse_acoustics` / `fine_acoustics` prefixes), a real tokenizer + voice/history
prompt, and full autoregressive sampling are documented follow-ups; this example
imports the three sub-models from separate files and exercises the deterministic
forward cascade.
