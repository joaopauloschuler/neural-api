# Semantic segmentation with an imported SegFormer

A tiny, CPU/offline demo of dense **semantic segmentation** using an imported
**SegFormer** (Xie et al. 2021, *"SegFormer: Simple and Efficient Design for
Semantic Segmentation with Transformers"*). SegFormer is a hierarchical
Mix-Transformer (MiT) encoder — overlap-patch embeddings, spatial-reduction
efficient attention, Mix-FFN, no positional embedding — feeding a lightweight
all-MLP decode head that emits a per-pixel class map.

## What it does

The example loads the committed **pico SegFormer parity fixture** (a tiny
MiT-b0-shaped model), runs it on a small synthetic image, takes the per-pixel
**argmax** over the class logits, and renders the resulting label map as a
colored ASCII palette (one glyph per class) at the head resolution
(`input / 4`). It also prints an argmax label histogram (per-class pixel counts).

A real run would load `nvidia/segformer-b0-finetuned-ade-512-512` the same way
and colorize a photograph — the math is identical; only the checkpoint and image
differ.

### Key neural-api pieces

- **`BuildSegformerFromSafeTensors`** (`neuralpretrained`) — imports the MiT
  encoder + all-MLP decode head into a `TNNet`; config via `TSegformerConfig` /
  `SegformerConfigToString`.

## Running

Build with the usual conventions, then run from the example dir or the repo root
(both fixture paths are probed):

```
cd examples/SemanticSegmentation
lazbuild SemanticSegmentation.lpi --build-mode=Release
./SemanticSegmentation
```

(Or compile the `.lpr` with `fpc -Fu../../neural`.)

## Inputs / outputs

- **Checkpoint:** `tests/fixtures/tiny_segformer.safetensors` (+
  `tiny_segformer_config.json`), committed — fully offline, finishes in well
  under a second.
- **Input image:** synthetic radial + diagonal RGB gradient generated in code (no
  image I/O); any real image works the same.
- **Output:** an ASCII label map printed to stdout plus a per-class pixel
  histogram. The pico fixture has 5 classes. There is no file output.

Coded by Joao Paulo Schwarz Schuler with Claude (AI).
