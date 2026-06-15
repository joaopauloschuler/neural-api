#!/usr/bin/env python3
"""Generate a tiny RANDOM OWL-ViT open-vocabulary object-detection parity
fixture for tests/TestNeuralPretrained.pas.

No network access: a pico OwlViTForObjectDetection (a CLIP text tower + CLIP ViT
image tower + the per-patch class-embedding head and box-regression MLP head) is
built from a small config and randomly initialized (never downloaded). The
reference forward is the REAL transformers OwlViTForObjectDetection run in
float64 (the package is installed in this environment); the test asserts the
Pascal forward matches the HF per-patch / per-query class `logits` AND the
per-patch `pred_boxes` tensors.

This is the repo's first ZERO-SHOT (open-vocabulary, text-conditioned) detector
(DETR is closed-vocabulary). The output modality is a SET of (box, query-match
score) per image: for every image patch (NOT a learned object query) and every
free-text query the model emits a cosine-similarity match logit and a cxcywh
box. Genuinely new code exercised here:
  - the CLIP ViT image tower with the CLS-token MERGE: post_layernorm hidden
    states, image_embeds = patch_rows * broadcast(class_token_row), then a final
    LayerNorm (the OWL-ViT `layer_norm`);
  - the class-prediction head: dense0 (hidden -> text_dim), L2-normalized and
    matched by cosine against the L2-normalized text-query embeddings, then a
    learnable per-patch logit_shift + ELU-gated logit_scale;
  - the box-regression head: 3-layer GELU MLP -> + grid box_bias -> sigmoid
    cxcywh;
  - the CLIP text tower (causal, EOS = argmax-of-ids pooling, text_projection)
    producing the query embeddings.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/make_pico_owlvit_fixture.py
writes tests/fixtures/tiny_owlvit{.safetensors,_config.json,_io.json}.
"""
import json
import os

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import OwlViTConfig, OwlViTForObjectDetection
from transformers.models.owlvit.configuration_owlvit import (
    OwlViTTextConfig,
    OwlViTVisionConfig,
)

# ---------------- pico config ----------------
# IMAGE 16x16, patch 8 -> a 2x2 = 4-patch feature grid (tiny but exercises the
# grid box_bias and the CLS merge over >1 patch). Text + vision hidden sizes are
# equal so dense0 (vision_hidden -> text_hidden) and the projection both land in
# the same shared dim for the cosine match.
IMAGE = 16
PATCH = 8
HIDDEN = 24            # both text and vision hidden_size
PROJ = 24             # projection_dim == text hidden_size (OWL-ViT invariant)
INTER = 48            # intermediate_size (fc1 width)
NUM_LAYERS = 2
NUM_HEADS = 3
VOCAB = 50
TEXT_MAXPOS = 8
EOS_ID = VOCAB - 1     # argmax pooling picks the EOS row (highest id present)

torch.manual_seed(20260614)
np.random.seed(20260614)

text_cfg = OwlViTTextConfig(
    vocab_size=VOCAB,
    hidden_size=HIDDEN,
    intermediate_size=INTER,
    num_hidden_layers=NUM_LAYERS,
    num_attention_heads=NUM_HEADS,
    max_position_embeddings=TEXT_MAXPOS,
    hidden_act="quick_gelu",
    eos_token_id=EOS_ID,
)
vision_cfg = OwlViTVisionConfig(
    hidden_size=HIDDEN,
    intermediate_size=INTER,
    num_hidden_layers=NUM_LAYERS,
    num_attention_heads=NUM_HEADS,
    image_size=IMAGE,
    patch_size=PATCH,
    num_channels=3,
    hidden_act="quick_gelu",
)
cfg = OwlViTConfig(
    text_config=text_cfg.to_dict(),
    vision_config=vision_cfg.to_dict(),
    projection_dim=PROJ,
)
model = OwlViTForObjectDetection(cfg).eval()

# Re-randomize on an O(1) scale: the HF default init (std ~0.02) makes the
# per-patch logits near-constant (every patch collapses to the same match), so
# the parity test would pass trivially. Larger spreads make the per-patch and
# per-query outputs visibly DISTINCT so the test truly exercises the CLS merge,
# the cosine match and the box grid bias.
with torch.no_grad():
    for name, p in model.named_parameters():
        if "weight" in name and p.dim() >= 2:
            torch.nn.init.normal_(p, mean=0.0, std=0.30)
        elif "bias" in name:
            torch.nn.init.normal_(p, mean=0.0, std=0.05)
        else:
            torch.nn.init.normal_(p, mean=0.0, std=0.30)
    # class_embedding (CLS token) of the vision tower: a normal spread.
    model.owlvit.vision_model.embeddings.class_embedding.normal_(0.0, 0.5)

NUM_QUERIES = 3       # free-text queries scored against the image
GRID = IMAGE // PATCH
NUM_PATCHES = GRID * GRID

# ---------------- float64 reference forward ----------------
model_d = model.double()
cases = []
for case in range(2):
    pixel = np.random.randn(1, 3, IMAGE, IMAGE).astype(np.float64) * 0.8
    # input_ids: NUM_QUERIES queries, each a TEXT_MAXPOS-long row ending in EOS.
    ids = np.random.randint(1, VOCAB - 1, size=(NUM_QUERIES, TEXT_MAXPOS))
    # make every query have a clean EOS at a per-query length, rest padded 0.
    lengths = [TEXT_MAXPOS - 1, TEXT_MAXPOS - 2, TEXT_MAXPOS - 3]
    for q in range(NUM_QUERIES):
        L = lengths[q % len(lengths)]
        ids[q, L] = EOS_ID
        ids[q, L + 1:] = 0
    attn = (ids != 0).astype(np.int64)
    with torch.no_grad():
        out = model_d(
            input_ids=torch.from_numpy(ids),
            pixel_values=torch.from_numpy(pixel),
            attention_mask=torch.from_numpy(attn),
        )
    logits = out.logits.numpy()[0]      # (num_patches, num_queries)
    boxes = out.pred_boxes.numpy()[0]   # (num_patches, 4) cxcywh sigmoid
    img_chw = pixel[0]
    img_xyc = np.transpose(img_chw, (1, 2, 0)).reshape(-1)  # flat (y, x, c)
    cases.append({
        "input": img_xyc.tolist(),
        "input_ids": ids.reshape(-1).tolist(),
        "attention_mask": attn.reshape(-1).tolist(),
        "num_queries": NUM_QUERIES,
        "text_seqlen": TEXT_MAXPOS,
        "num_patches": NUM_PATCHES,
        "logits": logits.reshape(-1).tolist(),   # flat (patch, query)
        "boxes": boxes.reshape(-1).tolist(),      # flat (patch, 4) cxcywh
    })

# ---------------- write ----------------
here = os.path.dirname(os.path.abspath(__file__))
fixtures = os.path.join(here, "..", "tests", "fixtures")
os.makedirs(fixtures, exist_ok=True)

state = model.float().state_dict()
tensors = {}
for k, v in state.items():
    if k.endswith("num_batches_tracked") or k.endswith("position_ids"):
        continue
    tensors[k] = v.detach().cpu().numpy().astype(np.float32)
save_file(tensors, os.path.join(fixtures, "tiny_owlvit.safetensors"))

config = {
    "model_type": "owlvit",
    "projection_dim": PROJ,
    "text_config": {
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "intermediate_size": INTER,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": NUM_HEADS,
        "max_position_embeddings": TEXT_MAXPOS,
        "hidden_act": "quick_gelu",
        "layer_norm_eps": 1e-5,
        "eos_token_id": EOS_ID,
    },
    "vision_config": {
        "hidden_size": HIDDEN,
        "intermediate_size": INTER,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": NUM_HEADS,
        "image_size": IMAGE,
        "patch_size": PATCH,
        "num_channels": 3,
        "hidden_act": "quick_gelu",
        "layer_norm_eps": 1e-5,
    },
}
with open(os.path.join(fixtures, "tiny_owlvit_config.json"), "w") as f:
    json.dump(config, f, indent=2)

with open(os.path.join(fixtures, "tiny_owlvit_io.json"), "w") as f:
    json.dump({"cases": cases}, f)

print("wrote tiny_owlvit.{safetensors,_config.json,_io.json} to", fixtures)
print(f"  image={IMAGE} patch={PATCH} grid={GRID}x{GRID} patches={NUM_PATCHES} "
      f"hidden={HIDDEN} layers={NUM_LAYERS} heads={NUM_HEADS}")
print(f"  queries={NUM_QUERIES} text_seqlen={TEXT_MAXPOS} vocab={VOCAB}")
print(f"  logits shape = ({NUM_PATCHES}, {NUM_QUERIES})")
print(f"  #tensors = {len(tensors)}")
sz = os.path.getsize(os.path.join(fixtures, "tiny_owlvit.safetensors"))
print(f"  safetensors size = {sz} bytes")
