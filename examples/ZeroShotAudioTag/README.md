# ZeroShotAudioTag — CLAP zero-shot audio tagging

The demo for `BuildClapFromSafeTensors` (`neural/neuralpretrained.pas`), the
audio-domain analogue of the CLIP dual encoder. **CLAP** (clap-htsat-unfused)
embeds an audio clip and a set of free-text labels into a **shared space**; the
clip is tagged with whichever label has the highest cosine similarity — no per-class
training, i.e. **zero-shot**.

```
log-mel spectrogram (time, 1, mel)
  -> HTS-AT (Swin) audio tower      -> audio embedding
N text prompts (token ids)
  -> RoBERTa text tower             -> text embeddings
  -> exp(logit_scale_a) * cosine in the shared space
  -> cosine-similarity matrix + top-1 zero-shot label
```

## The importer

`BuildClapFromSafeTensors(Checkpoint, AudioNet, TextNet, Config, TextSeqLen,
pTrainable, ConfigPath)` returns **two** `TNNet`s — the audio tower and the text
tower — and fills `TClapConfig` (printed by `ClapConfigToString`). Helpers used:
`CreatePretrainedTensorReader` (to read the checkpoint's batch-norm stats),
`ClapBatchNormMelImage` (HTS-AT batch_norm + mel2img affine on the raw log-mel),
`ClipExtractEmbedding` (pull the pooled embedding), `ClapSimilarityMatrix` /
`ClipSimilarity` (scoring). The text input is `(SeqLen, 1, 2)`: token id in channel
0, token-type id (0) in channel 1.

## Build / run

Run **from the repo root** so the fixture path resolves. With **no arguments** it
uses the committed pico fixture (`tests/fixtures/tiny_clap.safetensors` +
`tiny_clap_config.json`, auto-located), fully **offline**:

```
# from the repo root:
examples/ZeroShotAudioTag/ZeroShotAudioTag
examples/ZeroShotAudioTag/ZeroShotAudioTag /path/to/clap-htsat-unfused/model.safetensors [config.json]
```

(Build with `lazbuild examples/ZeroShotAudioTag/ZeroShotAudioTag.lpi` or fpc.)

## Synthetic inputs / scope

This demo exercises the zero-shot **scoring structure**, not a real audio
front-end:

- The clip's log-mel is **synthetic** (a deterministic pattern); a real pipeline
  would compute it with the log-mel frontend in `neural/neuralaudio.pas`.
- The 3 "prompts" are **synthetic token-id sequences** mapped to labels
  (`a dog barking`, `a vacuum cleaner`, `rain on a window`); the RoBERTa BPE
  tokenizer is out of this demo's scope.
- v1 supports `freq_ratio = 1` (spec_size = num_mel_bins) only.

With the random pico fixture the scores are illustrative — real audio + text just
swap in the real frontends through the same path.

## Output

Prints the loaded checkpoint and config, the audio<->text cosine similarity for
each label (raw cosine plus the `exp(logit_scale_a)`-scaled logit from
`ClapSimilarityMatrix`), and the top-1 zero-shot label. Pure CPU, runs in seconds
on the fixture.

This example is coded by Claude (AI).
