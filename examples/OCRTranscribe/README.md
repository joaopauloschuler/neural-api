# OCRTranscribe — TrOCR optical character recognition

An **OCR** demo using a **TrOCR** VisionEncoderDecoder checkpoint
(`microsoft/trocr-*`). A text-line image is encoded **once** by a DeiT/ViT vision
encoder, then **greedily transcribed** by a BART-style causal text decoder that
cross-attends over all patch features (cls + distillation + patches). Because the
encoder is reused for every decoder token, the architecture is efficient for
document/line transcription.

```
text-line image (ImageSize x ImageSize x NumChannels)
  -> DeiT/ViT vision ENCODER (run once)   -> patch features
  -> BART-style causal text DECODER (cross-attention over patches)
  -> greedy roll-out (start token -> ... -> EOS)
  -> token ids  (decode with the TrOCR byte-level BPE tokenizer for text)
```

## The importer

`BuildTrOCRFromSafeTensors` (config `TTrOCRConfig`, printed by
`TrOCRConfigToString`) returns **two** `TNNet`s — `EncoderNet` and `DecoderNet`.
`DecodeTrOCRGreedy(EncoderNet, DecoderNet, PixelValues, Config, MaxNewTokens)`
encodes the image once and rolls the decoder out token-by-token, stopping at
`Config.EosTokenId` or the token cap.

## Running

With **no arguments** it runs offline on the committed pico fixture
(`tests/fixtures/tiny_trocr.safetensors` + `tiny_trocr_config.json`,
auto-located), feeding a deterministic test-pattern "image":

```
cd examples/OCRTranscribe
# build with lazbuild (or fpc), then:
./OCRTranscribe
```

You can also pass a real checkpoint (and optional config):

```
./OCRTranscribe /path/to/trocr.safetensors [/path/to/config.json]
```

It prints the config and the transcribed **token ids** (the start token is
excluded; EOS terminates). The fixture weights are random, so the ids are
illustrative; decode them with the TrOCR GPT-2 byte-level BPE tokenizer for text.

Coded by Claude (AI).
