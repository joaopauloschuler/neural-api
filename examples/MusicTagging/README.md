# MusicTagging — MERT music-representation embeddings

A **music-representation embedding** demo using the **MERT** encoder
(`m-a-p/MERT-v1-95M`), a self-supervised music understanding model analogous to a
frozen vision backbone. A raw waveform runs through a HuBERT-style conv feature
extractor + transformer trunk; the per-layer hidden states are pooled into one
fixed-size **music embedding** by a learned **weighted-layer-sum** (per-layer
softmax weights) combined with mean pooling over time. The example embeds two
synthesized clips and scores their **cosine similarity**.

```
raw waveform
  -> MERT conv feature extractor + transformer (all N+1 hidden states)
  -> weighted-layer-sum (learned per-layer softmax weights) + mean-pool over time
  -> fixed music embedding
  -> cosine similarity between two clips
```

## The importer

`BuildMERTFromSafeTensorsEx` (config `TMERTConfig`, printed by
`MERTConfigToString`) returns a `TNNet` and the full stack of hidden states as a
`TMERTHiddenStateArray`. `MERTEncoderLength` derives the frame count from the raw
length; `MERTWeightedLayerSum(Config, HiddenLayers, out Emb, MeanPool=true)`
pools the hidden states into the embedding; `CosineSimilarity` scores two
embeddings.

## Running

With **no arguments** it uses the committed pico fixture
(`tests/fixtures/tiny_mert.safetensors` + its `_config.json`) and runs offline in
seconds. With a path argument it loads a real MERT checkpoint
(`<checkpoint-dir>/config.json` is auto-located).

```
cd examples/MusicTagging
# build with lazbuild (or fpc), then:
./MusicTagging                 # pico fixture
./MusicTagging /path/to/MERT   # real checkpoint
```

It synthesizes two 200-sample clips (deterministic sine + noise patterns),
prints each embedding's first dimensions and the cosine similarity, then a
verdict: `> 0.95` very similar, `> 0.5` some shared structure, else musically
distinct. With the random pico fixture the score is illustrative only.

This example is coded by Claude (AI).
