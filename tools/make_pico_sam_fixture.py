#!/usr/bin/env python3
"""Generate a tiny RANDOM Segment-Anything (SAM) ViT image-encoder parity
fixture for tests/TestNeuralPretrained.pas (no network: the model is randomly
built from a pico config, never downloaded).

SAM's image encoder is a ViT-det backbone:
  - a BIASED patch conv (kernel=stride=patch) to a (Grid, Grid, hidden) grid;
  - a learned 2-D ABSOLUTE pos_embed of shape (1, Grid, Grid, hidden) added
    directly (NO cls token, NO flatten);
  - NumLayers pre-LN transformer blocks whose self-attention is WINDOWED
    (window_size, with zero-padding partition) EXCEPT the blocks in
    global_attn_indexes which use GLOBAL attention; every block adds the MViTv2
    DECOMPOSED relative-position bias Q.rel_pos_h + Q.rel_pos_w (query-dependent,
    NOT a Swin bias table);
  - a neck: conv1 1x1 (no bias) -> LayerNorm2d (over channels) -> conv2 3x3 pad1
    (no bias) -> LayerNorm2d -> the (Grid, Grid, output_channels) image embedding.

The reference is the HF float64 image_embeddings for a pinned image. Weights are
re-randomized to O(1) scale (the ModernBERT vacuous-init lesson) so every quirk
(windowing, global blocks, decomposed rel-pos, neck) is visible above 1e-4.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/make_pico_sam_fixture.py
writes tests/fixtures/tiny_sam{.safetensors,_config.json,_embed.json}.
"""
import copy
import json

import torch
from safetensors.torch import save_file
from transformers import SamConfig, SamModel
from transformers.models.sam.configuration_sam import SamVisionConfig

HIDDEN = 24
N_LAYER = 4
N_HEAD = 3                 # head_dim = 8
IMAGE = 32
PATCH = 8                  # 4x4 grid
WINDOW = 2                 # local-window size (grid 4 -> 2x2 windows, padding-free here)
GLOBAL = [2]               # block 2 uses global attention
MLP_DIM = 48
OUT_CH = 16

torch.manual_seed(20260615)

vc = SamVisionConfig(
    hidden_size=HIDDEN, num_hidden_layers=N_LAYER, num_attention_heads=N_HEAD,
    image_size=IMAGE, patch_size=PATCH, num_channels=3, mlp_dim=MLP_DIM,
    window_size=WINDOW, global_attn_indexes=GLOBAL, output_channels=OUT_CH,
    hidden_act='gelu', layer_norm_eps=1e-6, use_abs_pos=True, use_rel_pos=True,
)
cfg = SamConfig(vision_config=vc.to_dict())
model = SamModel(cfg)
enc = model.vision_encoder

# Boost the pico inits so every quirk is visible above the 1e-4 parity gate.
with torch.no_grad():
    enc.pos_embed.normal_(0.0, 0.4)
    enc.patch_embed.projection.weight.normal_(0.0, 0.25)
    enc.patch_embed.projection.bias.normal_(0.0, 0.2)
    for layer in enc.layers:
        layer.attn.qkv.weight.normal_(0.0, 0.35)
        layer.attn.qkv.bias.normal_(0.0, 0.2)
        layer.attn.proj.weight.normal_(0.0, 0.35)
        layer.attn.proj.bias.normal_(0.0, 0.2)
        layer.attn.rel_pos_h.normal_(0.0, 0.5)
        layer.attn.rel_pos_w.normal_(0.0, 0.5)
        layer.mlp.lin1.weight.normal_(0.0, 0.5); layer.mlp.lin1.bias.normal_(0.0, 0.3)
        layer.mlp.lin2.weight.normal_(0.0, 0.3); layer.mlp.lin2.bias.normal_(0.0, 0.2)
        for norm in (layer.layer_norm1, layer.layer_norm2):
            norm.weight.normal_(1.0, 0.25); norm.bias.normal_(0.0, 0.2)
    enc.neck.conv1.weight.normal_(0.0, 0.3)
    enc.neck.conv2.weight.normal_(0.0, 0.3)
    for norm in (enc.neck.layer_norm1, enc.neck.layer_norm2):
        norm.weight.normal_(1.0, 0.25); norm.bias.normal_(0.0, 0.2)

enc = enc.double().eval()

# Pinned input: deterministic dyadic pixel values (exact in f32 and JSON).
pixels = torch.zeros(1, 3, IMAGE, IMAGE, dtype=torch.float64)
for c in range(3):
    for y in range(IMAGE):
        for x in range(IMAGE):
            pixels[0, c, y, x] = (((c * 256 + y * 16 + x) * 5) % 17 - 8) / 8.0

# ---- save weights (vision_encoder.* keys, exactly the importer prefix) ----
sd = {('vision_encoder.' + k): v.to(torch.float32).clone().contiguous()
      for k, v in enc.state_dict().items()}
save_file(sd, 'tests/fixtures/tiny_sam.safetensors')

# Config the importer reads.
with open('tests/fixtures/tiny_sam_config.json', 'w') as f:
    json.dump(cfg.to_dict(), f, indent=1, default=str)

with torch.no_grad():
    out = enc(pixel_values=pixels).last_hidden_state  # [1, OUT_CH, Grid, Grid]
# Reorder to (Grid, Grid, OUT_CH) to match the Pascal (X=row, Y=col, Depth=chan)
# output layout.
emb = out[0].permute(1, 2, 0).contiguous()  # [Grid, Grid, OUT_CH]
grid = emb.shape[0]
with open('tests/fixtures/tiny_sam_embed.json', 'w') as f:
    json.dump({
        'pixels': pixels[0].tolist(),
        'embed': emb.tolist(),
        'grid': grid,
        'out_channels': OUT_CH,
    }, f)
print(f'wrote tiny_sam.safetensors ({len(sd)} tensors) + config + oracle')
print(f'  image_embeddings shape = {list(out.shape)} -> oracle (grid,grid,out)=({grid},{grid},{OUT_CH})')

# ---- fixture self-checks: every SAM quirk must be visible ----
with torch.no_grad():
    base = out

    # pos_embed must matter
    alt = copy.deepcopy(enc)
    alt.pos_embed.zero_()
    d = (alt(pixel_values=pixels).last_hidden_state - base).abs().max()
    assert d > 1e-3, f'pos_embed had no effect ({d})'
    print(f'pos_embed effect: max |diff| = {d:.4f}')

    # decomposed rel-pos must matter
    alt = copy.deepcopy(enc)
    for layer in alt.layers:
        layer.attn.rel_pos_h.zero_(); layer.attn.rel_pos_w.zero_()
    d = (alt(pixel_values=pixels).last_hidden_state - base).abs().max()
    assert d > 1e-3, f'decomposed rel-pos had no effect ({d})'
    print(f'rel-pos effect: max |diff| = {d:.4f}')

    # windowing must matter (force every block global and compare)
    alt = copy.deepcopy(enc)
    for layer in alt.layers:
        layer.window_size = 0
    d = (alt(pixel_values=pixels).last_hidden_state - base).abs().max()
    assert d > 1e-3, f'windowing had no effect ({d})'
    print(f'windowing effect: max |diff| = {d:.4f}')

    # neck conv2 must matter
    alt = copy.deepcopy(enc)
    alt.neck.conv2.weight.zero_()
    d = (alt(pixel_values=pixels).last_hidden_state - base).abs().max()
    assert d > 1e-3, f'neck conv2 had no effect ({d})'
    print(f'neck conv2 effect: max |diff| = {d:.4f}')
print('all fixture self-checks passed')
