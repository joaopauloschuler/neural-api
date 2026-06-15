#!/usr/bin/env python3
"""Generate a tiny RANDOM VideoMAE video-classification parity fixture for
tests/TestNeuralPretrained.pas.

No network access: a pico VideoMAEForVideoClassification (tubelet 3-D conv patch
embedding + fixed sin-cos 3-D position table + stock pre-LN transformer encoder
+ mean-pool -> fc_norm -> linear classifier) is built from a small config and
randomly initialized (never downloaded). The reference forward is the REAL
transformers VideoMAEForVideoClassification run in float64 (the package is
installed in this environment); the test asserts the Pascal forward matches the
HF classification `logits`.

This is the repo's FIRST video-classification importer (a clip of T frames ->
an action label) - the pay-off of the landed TNNetConvolution3D layer. The only
genuinely new Pascal code is the tubelet-patchifier wiring + the fixed sin-cos
3-D position table; the encoder reuses the CLIP/BERT pre-LN encoder block path.

Architecture exercised (MCG-NJU/videomae-base-finetuned-kinetics shape, shrunk):
  - Tubelet embedding: a 3-D conv (kernel = stride = (tubelet_size, patch,
    patch), no pad) over the (T, H, W, C) clip -> a (T', H', W') grid of
    hidden-dim tokens, T' = T/tubelet_size, H' = W' = IMAGE/PATCH. HF flattens
    the conv output (B, hidden, T', H', W') as flatten(2).transpose(1,2), i.e.
    token order (t, h, w) with t outermost, w innermost.
  - Position: a FIXED (non-learned) sin-cos table get_sinusoid_encoding_table
    (num_tokens, hidden) ADDED to the patch tokens. NOT in the state_dict (a
    registered buffer) - replicated exactly in Pascal.
  - Encoder: NUM_LAYERS stock pre-LN ViT/BERT blocks (separate biased q/k/v,
    1/sqrt(head_dim) joint space-time attention over ALL tokens, biased
    out/fc1/fc2, exact-erf GELU, layernorm_before/after).
  - Head: use_mean_pooling=True -> NO final encoder layernorm; mean over all
    tokens -> fc_norm (LayerNorm) -> classifier (biased Linear) -> logits.

The Pascal input volume packs the clip as (W, H, T*C) with the C channels of
frame t contiguous at depth [t*C .. t*C+C) (the TNNetConvolution3D frame-packing
convention). The fixture emits the input in that flat order:
  idx = (y*W + x)*(T*C) + t*C + c.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/make_pico_videomae_fixture.py
writes tests/fixtures/tiny_videomae{.safetensors,_config.json,_io.json}.
"""
import json
import os

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import VideoMAEConfig, VideoMAEForVideoClassification

# ---------------- pico config ----------------
IMAGE = 8
PATCH = 4
NUM_CHANNELS = 3
NUM_FRAMES = 4
TUBELET = 2
HIDDEN = 16
NUM_LAYERS = 2
NUM_HEADS = 2
INTERMEDIATE = 32
NUM_LABELS = 5
LAYER_NORM_EPS = 1e-6

GRID = IMAGE // PATCH          # H' = W'
TGRID = NUM_FRAMES // TUBELET  # T'
NUM_TOKENS = TGRID * GRID * GRID

torch.manual_seed(7)
np.random.seed(7)

cfg = VideoMAEConfig(
    image_size=IMAGE,
    patch_size=PATCH,
    num_channels=NUM_CHANNELS,
    num_frames=NUM_FRAMES,
    tubelet_size=TUBELET,
    hidden_size=HIDDEN,
    num_hidden_layers=NUM_LAYERS,
    num_attention_heads=NUM_HEADS,
    intermediate_size=INTERMEDIATE,
    num_labels=NUM_LABELS,
    hidden_act="gelu",
    qkv_bias=True,
    use_mean_pooling=True,
    layer_norm_eps=LAYER_NORM_EPS,
)
model = VideoMAEForVideoClassification(cfg).eval()

# Re-randomize every parameter on an O(1) scale: the HF default init (std 0.02)
# makes the encoder output near-constant, so the parity test would pass
# trivially (the ModernBERT gotcha). A modest std keeps the forward in a small
# dynamic range so the float32 Pascal path stays within ~1e-4 of float64.
with torch.no_grad():
    for name, p in model.named_parameters():
        if "weight" in name and p.dim() >= 2:
            torch.nn.init.normal_(p, mean=0.0, std=0.12)
        elif "bias" in name:
            torch.nn.init.normal_(p, mean=0.0, std=0.05)
        else:
            torch.nn.init.normal_(p, mean=0.0, std=0.1)

# ---------------- float64 reference forward ----------------
model_d = model.double()
cases = []
for case in range(2):
    # HF input: (B, num_frames, num_channels, H, W)
    x = (np.random.randn(1, NUM_FRAMES, NUM_CHANNELS, IMAGE, IMAGE)
         .astype(np.float64) * 0.8)
    with torch.no_grad():
        logits = model_d(torch.from_numpy(x)).logits  # (1, num_labels)
    logits = logits.numpy()[0]  # (num_labels,)

    # Pascal input volume order: (W, H, T*C), idx = (y*W+x)*(T*C) + t*C + c
    inp = np.zeros((IMAGE * IMAGE * NUM_FRAMES * NUM_CHANNELS,), np.float64)
    for y in range(IMAGE):
        for xx in range(IMAGE):
            for t in range(NUM_FRAMES):
                for c in range(NUM_CHANNELS):
                    inp[(y * IMAGE + xx) * (NUM_FRAMES * NUM_CHANNELS)
                        + t * NUM_CHANNELS + c] = x[0, t, c, y, xx]
    cases.append({
        "input": inp.tolist(),
        "output": logits.tolist(),
    })

# ---------------- write ----------------
here = os.path.dirname(os.path.abspath(__file__))
fixtures = os.path.join(here, "..", "tests", "fixtures")
os.makedirs(fixtures, exist_ok=True)

state = model.float().state_dict()
tensors = {}
for k, v in state.items():
    if k.endswith("num_batches_tracked"):
        continue
    # position_embeddings is a fixed sin-cos buffer (rebuilt in Pascal), not a
    # parameter - it is not in named_parameters but may appear in state_dict.
    if k.endswith("embeddings.position_embeddings"):
        continue
    tensors[k] = v.detach().cpu().numpy().astype(np.float32)
save_file(tensors, os.path.join(fixtures, "tiny_videomae.safetensors"))

config = {
    "model_type": "videomae",
    "image_size": IMAGE,
    "patch_size": PATCH,
    "num_channels": NUM_CHANNELS,
    "num_frames": NUM_FRAMES,
    "tubelet_size": TUBELET,
    "hidden_size": HIDDEN,
    "num_hidden_layers": NUM_LAYERS,
    "num_attention_heads": NUM_HEADS,
    "intermediate_size": INTERMEDIATE,
    "num_labels": NUM_LABELS,
    "hidden_act": "gelu",
    "qkv_bias": True,
    "use_mean_pooling": True,
    "layer_norm_eps": LAYER_NORM_EPS,
}
with open(os.path.join(fixtures, "tiny_videomae_config.json"), "w") as f:
    json.dump(config, f, indent=2)

with open(os.path.join(fixtures, "tiny_videomae_io.json"), "w") as f:
    json.dump({"cases": cases}, f)

print("wrote tiny_videomae.{safetensors,_config.json,_io.json} to", fixtures)
print(f"  image={IMAGE} patch={PATCH} frames={NUM_FRAMES} tubelet={TUBELET} "
      f"hidden={HIDDEN} layers={NUM_LAYERS} labels={NUM_LABELS}")
print(f"  tokens = {NUM_TOKENS} (T'={TGRID} x H'={GRID} x W'={GRID})")
print(f"  #tensors = {len(tensors)}")
print(f"  case0 logits = {cases[0]['output']}")
