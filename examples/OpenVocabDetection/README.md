# OWL-ViT open-vocabulary object detection

This example demonstrates **zero-shot, open-vocabulary** object detection with
[OWL-ViT](https://huggingface.co/google/owlvit-base-patch32) imported by
`BuildOwlViTFromSafeTensors` (`neural/neuralpretrained.pas`). Unlike DETR (which
detects a **fixed, closed** label set), OWL-ViT scores every **image patch**
against arbitrary **free-text query** embeddings by cosine similarity, so the
"classes" are whatever text you encode — decided at inference time.

## What it does

1. Loads the two nets returned by `BuildOwlViTFromSafeTensors`:
   - the **vision + detection net**: a CLIP ViT image tower (reusing
     `BuildClipVisionTower`) whose post-layernorm patch states are merged with
     the CLS token, layer-normalized, and fed through the per-patch
     **class-embedding head** (`dense0` + learnable `logit_shift`/`logit_scale`)
     and the **box-regression MLP** (3-layer GELU MLP → cxcywh);
   - the **text net**: a CLIP text tower (causal, argmax-of-ids EOS pooling,
     `text_projection`) producing one embedding per free-text query.
2. Runs one tiny image through the vision net.
3. Embeds a couple of free-text query token-id sequences with
   `OwlViTQueryEmbedding`.
4. Scores every `(patch, query)` pair with `DecodeOwlViTDetections`: the match
   logit is `(cos(image_class_embeds[p], query[q]) + shift[p]) *
   (elu(scale_raw[p]) + 1)`, `Score = sigmoid(logit)`, and the box is
   `sigmoid(box_raw + grid_box_bias)` in cxcywh on the 0..1 image.
5. Prints the best-matching patch (score + box) for each query.

## Running

Works **offline** — the default checkpoint is the committed pico fixture
`tests/fixtures/tiny_owlvit.*`, and the image + query token ids are read from
the fixture's `io.json` so the numbers reproduce the parity test
(`TestOwlViTOpenVocabDetectionParity`).

From the repo root:

```
examples/OpenVocabDetection/OpenVocabDetection
examples/OpenVocabDetection/OpenVocabDetection /path/to/owlvit/model.safetensors
```

With a real checkpoint the queries are still synthetic token-id sequences (the
OWL-ViT BPE tokenizer and the image resize/normalize preprocessing are out of
this demo's scope): the point is the open-vocabulary **scoring structure** —
real prompts just swap in real token ids.

## Notes

- OWL-ViT is **CLIP-based and text-conditioned** with **no learned object
  queries** (the distinguishing contrast with DETR): the detections come from
  image patches matched against text, not from a fixed query set.
- The image-conditioned query path (one-shot detection from an exemplar image)
  is not covered here — this demo is the text-query path only.
