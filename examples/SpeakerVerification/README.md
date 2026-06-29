# SpeakerVerification — ECAPA-TDNN speaker embeddings

A **speaker verification** demo using the **ECAPA-TDNN** speaker-embedding
importer. A companion to [SpeakerDiarization](../SpeakerDiarization): where
Pyannote answers *"who speaks when"*, ECAPA-TDNN turns a whole utterance into a
fixed-length **speaker embedding**, and two embeddings are compared by **cosine
similarity** to answer *"are these the same person?"*

```
log-mel clip
  -> conv_pre (TNNetTDNNConv1D dilated TDNN)
  -> 3x SE-Res2Block (Res2Net dilated cascade + squeeze-excitation, AddSEBlock)
  -> multi-layer aggregation + attentive statistics pooling
     (TNNetAttentiveStatsPooling: context-weighted MEAN and STD over time)
  -> linear -> speaker embedding
  -> cosine similarity between two utterances
```

(AAM-softmax / ArcFace is the training head; verification uses cosine similarity.)

## The importer

`BuildEcapaTdnnFromSafeTensorsEx` (config `TEcapaTdnnConfig`) builds a `TNNet`
from `TNNetTDNNConv1D` dilated convolutions, `AddSEBlock` squeeze-excitation
gating and `TNNetAttentiveStatsPooling`. `EcapaCosineScore(Emb1, Emb2)` scores
two embeddings.

## Running

This example takes **no arguments** — it runs a self-contained smoke test on the
committed pico fixture (`tests/fixtures/tiny_ecapa.safetensors` +
`tiny_ecapa_config.json`). It synthesizes three short log-mel clips — two
phrasings of speaker A (same tonal makeup) and one clip of speaker B (different
tonal makeup) — embeds all three, and checks that the same-speaker pair scores
higher. Pure-CPU, low-RAM.

```
cd examples/SpeakerVerification
# build with lazbuild SpeakerVerification.lpi (or fpc), then:
./SpeakerVerification
```

It prints the config, the same-speaker and different-speaker cosine scores, and a
`VERIFIED` verdict when same > diff. The pico fixture is random, so the ordering
is illustrative only. If the fixture is missing, run
`tools/make_pico_ecapa_fixture.py` first.

Coded by Claude (AI).
