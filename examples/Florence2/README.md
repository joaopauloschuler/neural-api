# Florence-2 unified vision (spatial-output-as-text)

The demo for `BuildFlorence2FromSafeTensors` (`neural/neuralpretrained.pas`), the
repo's first **"spatial-output-as-text"** vision-language model (Xiao et al.
2023, *Florence-2*; `microsoft/Florence-2-base` / `-large`). A SINGLE
task-prompted seq2seq head does captioning AND box detection: the input is an
image plus a short **task token** stream (`<CAPTION>`, `<OD>`, ...), a multimodal
projector turns the image feature map into **visual tokens**, those are prepended
to the embedded task-prompt text, a BART encoder runs over the whole
`[visual; text]` sequence, and a BART decoder cross-attends to it and emits the
answer token stream.

The genuinely new idea this demo highlights: **boxes and polygons are emitted as
quantized location tokens** `<loc_0..loc_999>` in the vocabulary (spatial outputs
as text). The example shows the location-token (de)quantization round-trip on a
sample box via `Florence2QuantizeCoord` / `Florence2DequantizeCoord`.

**Scope v1**: the DaViT vision tower is the deferred gap — the importer takes the
tower's `last_hidden_state` feature map as a precomputed input. The committed pico
fixture supplies that feature map (and a sample task prompt) in its `io.json`, so
the demo exercises the image+task → token-stream **plumbing** plus the
coordinate↔loc-token mapping, not a trained caption.

## Build / run

```
cd examples/Florence2
lazbuild Florence2.lpi --build-mode=Release
# run from the repo root so the fixture path resolves:
../../bin/x86_64-linux/bin/Florence2
../../bin/x86_64-linux/bin/Florence2 /path/to/Florence-2-base/model.safetensors
```

## Input

By default the committed pico fixture
`tests/fixtures/tiny_florence2.safetensors` (+ `tiny_florence2_config.json` +
`tiny_florence2_io.json`). The fixture is randomly initialized, so the generated
caption ids are gibberish. Pass a real Florence-2 `.safetensors` path as the
first argument to use your own checkpoint (a real run additionally needs the
DaViT tower feeding real feature maps and the Florence-2 BART tokenizer).

## Output

Prints the config, the task-prompt token ids, the first generated token id
(argmax of one greedy decoder step), and the box→`<loc_>`→box round-trip. Pure
CPU, well under a second on the fixture.
