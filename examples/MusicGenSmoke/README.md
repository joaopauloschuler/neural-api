# MusicGenSmoke — text-to-music delay-pattern generation

A **text-to-audio generative** demo. The MusicGen LM
decoder predicts a stack of EnCodec codes autoregressively using the
**delay-pattern** codebook interleaving, decoded back to audio through the
landed EnCodec decoder.

```
text tokens
  -> T5 text ENCODER (BuildT5FromSafeTensors)        [external; not in smoke]
  -> enc_to_dec_proj -> cross-attention conditioning
  -> MusicGen DECODER (pre-norm cross-attention blocks, K summed code
     embeddings, sinusoidal positions, K LM heads)
  -> greedy delay-pattern decode -> a [K][frames] EnCodec code stack
  -> EnCodec audio DECODER (BuildEnCodecFromSafeTensors) -> waveform
```

## The delay pattern

A code at (codebook `k`, frame `f`) is emitted at sequence position `f + k`,
padded with a special token before it appears, so a single set of `K` LM heads
can predict all `K` codebooks causally from one shared transformer.
`MusicGenDelayInterleave` / `MusicGenDelayDeinterleave` (`neuralpretrained.pas`)
match HF `build_delay_pattern_mask` / `apply_delay_pattern_mask` exactly.

## The decoder

`BuildMusicGenFromSafeTensors` (model_type `musicgen`) builds a
`TMusicGenModel` holder whose transformer decoder is the **PRE-norm**
cross-attention block skeleton (the Pegasus path — *not* post-norm BART),
with:

- `K = num_codebooks` separate code-embedding tables, summed per frame;
- HF `cat([cos, sin])` half-split **sinusoidal** position embeddings
  (cos in the first half, sin in the second; base `log(10000)/(half-1)`);
- **bias-free** q/k/v/out and fc1/fc2 linears;
- a final decoder LayerNorm;
- `K` untied LM heads (`lm_heads.{0..K-1}`);
- a biased `enc_to_dec_proj` mapping the T5 hidden size to the decoder hidden
  size before cross-attention.

## Running

With no arguments this runs a self-contained pico smoke test on the committed
random fixture (`tests/fixtures/tiny_musicgen.*`): it builds the decoder, feeds
a fixed pseudo-encoder-state tensor (standing in for the T5 text encoder), and
greedily generates a short code stack through the delay pattern. The weights
are untrained random, so the codes are not meant to sound like music; this only
exercises the importer + delay-pattern generation end to end.

```
MusicGenSmoke
```

Pure CPU, a fraction of a second, no download.

## Parity

The pico fixture (`tools/musicgen_tiny_fixture.py`) pins the contract:

- `TestMusicGenDecoderParity` — the decoder next-token logits
  (`K × T × vocab`) match the HF `transformers` float64 oracle to
  `< 1e-4` (achieved: max `|diff|` = 0.0).
- `TestMusicGenDelayPattern` — the delay-pattern interleave matches HF
  `build_delay_pattern_mask` exactly, and the (de)interleave round-trips.

## Deferred follow-ups

- Stereo (`audio_channels=2`, the interleaved `2*K`-codebook delay layout).
- Wiring the full text-prompt → waveform pipeline (real T5 encoder + EnCodec
  decoder); v1's `Generate` stops at the code stack.
- KV-cache incremental decode (v1 recomputes the whole prefix each step).
- top-k / temperature sampling instead of greedy.

Coded by Claude (AI).
