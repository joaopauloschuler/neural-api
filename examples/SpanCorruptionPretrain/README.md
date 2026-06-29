# SpanCorruptionPretrain — a tiny T5-style encoder-decoder pretrained from scratch

A tiny **T5-style ENCODER-DECODER** trained **from scratch** on the T5
**span-corruption** pretraining objective — the natural end-to-end demo of
`neuraldatasets.TNNetSpanCorruptionCollator`.

## The span-corruption task

Take a sequence, mask **contiguous spans** of tokens, collapse each masked span
in the ENCODER input to one unique sentinel id (`<extra_id_0>`, `<extra_id_1>`,
...), and train the DECODER to emit the dropped spans as a sentinel/span stream:

```
encoder in : the <0> the rabbit at dawn
decoder out: <0> fox chases <final_sentinel>
```

`TNNetSpanCorruptionCollator.Collate` produces exactly this
`(corrupted-source, sentinel-target)` pair. The model therefore learns to
**reconstruct the masked spans from their surrounding context** — the pretraining
signal behind T5/BART.

**Tokenization is word-level** (like real T5/SentencePiece, only simpler): each
whole word is one token; the distinct words across the corpus form a small
(~30-token) vocabulary. The corpus is a small, **strongly deterministic** set of
templated sentences (`the fox chases the rabbit at dawn`, `the owl hunts the
mouse at night`, ...) where the subject pins the rest of the sentence, so a
masked word-span is reconstructible from the surrounding words — exactly the
regularity span corruption is meant to capture. Sentinels occupy the **top**
`csNumSentinels` ids (T5 convention: `<extra_id_0> = VocabSize-1`, descending);
ids 0/1 are the decoder-start/pad and EOS specials.

## Architecture

Both branches live in **one `TNNet`** so the loss back-propagates end-to-end
**through the cross-attention into the encoder**. The net has TWO `TNNetInput`
layers (`FLayers[0]`=encoder tokens, `FLayers[1]`=decoder tokens), fed together
with the array form `FNN.Compute([FEncToks, FDecToks])`.

```
ENCODER (bidirectional):
  Input(EncSeqLen,1,1)
   -> TokenAndPositionalEmbedding(VocabSize, d_model=64)
   -> 2x AddTransformerEncoderBlock(Heads=4, d_ff=128, PreNorm, CausalMask=false)
   -> LayerNorm                                  (final encoder hidden states)

DECODER (causal self-attn + cross-attn over the encoder tip):
  Input(DecSeqLen,1,1)
   -> TokenAndPositionalEmbedding(VocabSize, d_model=64)
   -> 2x AddTransformerDecoderBlock(Heads=4, d_ff=128, EncoderOutput=EncTip, PreNorm)
   -> LayerNorm
   -> PointwiseConvLinear(VocabSize)             per-token LM head
   -> PointwiseSoftMax(1)                        softmax across depth
```

Config (sized for a <5 min CPU budget and `ulimit -v 3000000`): `d_model=64`,
4 heads, `d_ff=128`, 2 encoder + 2 decoder blocks, `EncSeqLen=14`,
`DecSeqLen=12`.

## Training

Per-step SGD with momentum (`SetBatchUpdate(false)`, `lr=0.002`, inertia 0.9) for
12000 steps. Each step samples a corpus line, draws a fresh corruption with the
collator, teacher-forces the decoder (start token + shifted target), computes the
per-token cross-entropy over the supervised decoder positions, and
back-propagates. A **loss mask** zeros the gradient on the unsupervised (padded)
decoder rows by copying the model's own output into those `FDesired` rows so the
`(output - desired)` seed is exactly zero there.

## Decoding

`GreedyDecode` is an inlined `DecodeSeq2SeqGreedy` (argmax of the per-token
logits row, autoregressive, padded-causal): the encoder source is fed alongside
the growing decoder prefix, and `Logits.GetClassOnPixel(CurLen-1, 0)` reads the
next token until EOS.

## How to run

```
cd examples/SpanCorruptionPretrain
fpc -O3 -Mobjfpc -Sh -Fu../../neural SpanCorruptionPretrain.lpr
./SpanCorruptionPretrain
```

(or open `SpanCorruptionPretrain.lpi` in Lazarus). Pure CPU; the whole run (build
+ 12000 train steps + decode demo) takes ~2 min.

## Expected output

The per-token cross-entropy falls from ~2.0 toward ~0.04. The demo then prints,
for each line, the corrupted source (spans rendered `<i>`), the gold target span
stream, and the model's reconstruction — clearly **input-dependent** (each
corrupted line yields a different, context-appropriate span stream). It reports
exact-match reconstructions (roughly 10/12 lines) and per-token target accuracy
(~95%); the few misses are honest near-ties (e.g. `at` vs `in`).
