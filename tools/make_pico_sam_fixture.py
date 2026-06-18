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

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import SamConfig, SamModel
from transformers.models.sam.configuration_sam import (
    SamVisionConfig, SamPromptEncoderConfig, SamMaskDecoderConfig)

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
    num_pos_feats=OUT_CH // 2,   # random Fourier features for the prompt/grid
)                                # pos-enc; 2*num_pos_feats == OUT_CH == dec dim
# --- prompt encoder + mask decoder sub-configs sized to the PICO tower ---
# The mask decoder operates on the NECK output (OUT_CH channels, Grid x Grid),
# so prompt/decoder hidden_size == OUT_CH and image_embedding_size == Grid.
GRID = IMAGE // PATCH
DEC_HEADS = 2              # OUT_CH=16 -> head_dim = 8
pec = SamPromptEncoderConfig(
    hidden_size=OUT_CH, image_size=IMAGE, patch_size=PATCH,
    mask_input_channels=OUT_CH // 4, num_point_embeddings=4,
    image_embedding_size=GRID, layer_norm_eps=1e-6, hidden_act='gelu',
)
mdc = SamMaskDecoderConfig(
    hidden_size=OUT_CH, hidden_act='relu', mlp_dim=4 * OUT_CH,
    num_hidden_layers=2, num_attention_heads=DEC_HEADS,
    attention_downsample_rate=2, num_multimask_outputs=3,
    iou_head_depth=3, iou_head_hidden_dim=OUT_CH, layer_norm_eps=1e-6,
)
cfg = SamConfig(vision_config=vc.to_dict(),
                prompt_encoder_config=pec.to_dict(),
                mask_decoder_config=mdc.to_dict())
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

# Boost the prompt-encoder + mask-decoder inits to O(1) so every quirk
# (point/not-a-point embeds, two-way cross-attention, transposed-conv upscale,
# hypernetwork dot) is visible above the 1e-4 parity gate.
with torch.no_grad():
    pe = model.prompt_encoder
    pe.shared_embedding.positional_embedding.normal_(0.0, 1.0)
    pe.no_mask_embed.weight.normal_(0.0, 0.5)
    pe.not_a_point_embed.weight.normal_(0.0, 0.5)
    for emb in pe.point_embed:
        emb.weight.normal_(0.0, 0.5)
    # also re-randomize the IMAGE positional embedding source (shared with the
    # model-wide grid pos-enc); it is the same shared_image_embedding module.
    model.shared_image_embedding.positional_embedding.normal_(0.0, 1.0)

    md = model.mask_decoder
    md.iou_token.weight.normal_(0.0, 0.5)
    md.mask_tokens.weight.normal_(0.0, 0.5)
    for layer in md.transformer.layers:
        for attn in (layer.self_attn, layer.cross_attn_token_to_image,
                     layer.cross_attn_image_to_token):
            attn.q_proj.weight.normal_(0.0, 0.35); attn.q_proj.bias.normal_(0.0, 0.2)
            attn.k_proj.weight.normal_(0.0, 0.35); attn.k_proj.bias.normal_(0.0, 0.2)
            attn.v_proj.weight.normal_(0.0, 0.35); attn.v_proj.bias.normal_(0.0, 0.2)
            attn.out_proj.weight.normal_(0.0, 0.35); attn.out_proj.bias.normal_(0.0, 0.2)
        layer.mlp.lin1.weight.normal_(0.0, 0.4); layer.mlp.lin1.bias.normal_(0.0, 0.2)
        layer.mlp.lin2.weight.normal_(0.0, 0.4); layer.mlp.lin2.bias.normal_(0.0, 0.2)
        for ln in (layer.layer_norm1, layer.layer_norm2,
                   layer.layer_norm3, layer.layer_norm4):
            ln.weight.normal_(1.0, 0.25); ln.bias.normal_(0.0, 0.2)
    fa = md.transformer.final_attn_token_to_image
    fa.q_proj.weight.normal_(0.0, 0.35); fa.q_proj.bias.normal_(0.0, 0.2)
    fa.k_proj.weight.normal_(0.0, 0.35); fa.k_proj.bias.normal_(0.0, 0.2)
    fa.v_proj.weight.normal_(0.0, 0.35); fa.v_proj.bias.normal_(0.0, 0.2)
    fa.out_proj.weight.normal_(0.0, 0.35); fa.out_proj.bias.normal_(0.0, 0.2)
    md.transformer.layer_norm_final_attn.weight.normal_(1.0, 0.25)
    md.transformer.layer_norm_final_attn.bias.normal_(0.0, 0.2)
    md.upscale_conv1.weight.normal_(0.0, 0.3); md.upscale_conv1.bias.normal_(0.0, 0.2)
    md.upscale_conv2.weight.normal_(0.0, 0.3); md.upscale_conv2.bias.normal_(0.0, 0.2)
    md.upscale_layer_norm.weight.normal_(1.0, 0.25)
    md.upscale_layer_norm.bias.normal_(0.0, 0.2)
    for mlp in md.output_hypernetworks_mlps:
        mlp.proj_in.weight.normal_(0.0, 0.4); mlp.proj_in.bias.normal_(0.0, 0.2)
        mlp.proj_out.weight.normal_(0.0, 0.4); mlp.proj_out.bias.normal_(0.0, 0.2)
        for l in mlp.layers:
            l.weight.normal_(0.0, 0.4); l.bias.normal_(0.0, 0.2)
    for mlp in (md.iou_prediction_head,):
        mlp.proj_in.weight.normal_(0.0, 0.4); mlp.proj_in.bias.normal_(0.0, 0.2)
        mlp.proj_out.weight.normal_(0.0, 0.4); mlp.proj_out.bias.normal_(0.0, 0.2)
        for l in mlp.layers:
            l.weight.normal_(0.0, 0.4); l.bias.normal_(0.0, 0.2)

# Promote the WHOLE model to float64 so the oracle is exact.
model = model.double().eval()
enc = model.vision_encoder

# Pinned input: deterministic dyadic pixel values (exact in f32 and JSON).
pixels = torch.zeros(1, 3, IMAGE, IMAGE, dtype=torch.float64)
for c in range(3):
    for y in range(IMAGE):
        for x in range(IMAGE):
            pixels[0, c, y, x] = (((c * 256 + y * 16 + x) * 5) % 17 - 8) / 8.0

# ---- save weights: vision_encoder.* + prompt_encoder.* + mask_decoder.* +
# shared_image_embedding.* (exactly the HF SamModel top-level key prefixes). ----
sd = {k: v.to(torch.float32).clone().contiguous()
      for k, v in model.state_dict().items()}
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

# ---------------------------------------------------------------------------
# MASK-DECODER ORACLE: a fixed single POSITIVE point click -> single low-res
# mask logits. The oracle is HF's own SamModel forward (float64), fed the
# precomputed image embedding so it is decoupled from the encoder path.
# input_points are in ORIGINAL image pixel coords (HF processor convention);
# here image_size == IMAGE so no resize is needed.
# ---------------------------------------------------------------------------
PT_X, PT_Y = 19.0, 11.0      # a deterministic interior click (x, y) in pixels
with torch.no_grad():
    img_emb = model.get_image_embeddings(pixel_values=pixels)  # [1, OUT_CH, Grid, Grid]
    input_points = torch.tensor([[[[PT_X, PT_Y]]]], dtype=torch.float64)  # (B,pbs,n,2)
    input_labels = torch.tensor([[[1]]], dtype=torch.long)                # positive
    seg = model(
        image_embeddings=img_emb,
        input_points=input_points,
        input_labels=input_labels,
        multimask_output=False,
    )
    masks = seg.pred_masks  # [B, pbs, num_masks(=1), H, W]
mlogits = masks[0, 0, 0]    # [H, W]  low-res mask logits (H=W=4*Grid)
mh, mw = mlogits.shape
with open('tests/fixtures/tiny_sam_mask.json', 'w') as f:
    json.dump({
        'point': [PT_X, PT_Y],
        'label': 1,
        'mask_logits': mlogits.tolist(),  # [H][W]
        'mask_h': mh, 'mask_w': mw,
        'grid': grid, 'hidden': OUT_CH,
    }, f)
print(f'wrote tiny_sam_mask.json: mask logits shape = ({mh},{mw}); '
      f'range [{float(mlogits.min()):.4f}, {float(mlogits.max()):.4f}]')

# ---------------------------------------------------------------------------
# MASK-DECODER v2 ORACLE: multi-point prompt, box prompt, and the full
# multi-mask (3 masks) + IoU-head scores. Same precomputed image embedding.
# A box is [x0,y0,x1,y1] in image pixels; HF reshapes it to two corner points
# (point_embed[2] top-left, point_embed[3] bottom-right) and appends NO pad.
# ---------------------------------------------------------------------------
v2 = {'grid': grid, 'hidden': OUT_CH, 'mask_h': mh, 'mask_w': mw}

# (a) MULTI-point prompt: one positive + one negative click, single mask.
MP_PTS = [[19.0, 11.0], [5.0, 27.0]]
MP_LAB = [1, 0]
with torch.no_grad():
    seg_mp = model(
        image_embeddings=img_emb,
        input_points=torch.tensor([[MP_PTS]], dtype=torch.float64),  # (B,pbs,n,2)
        input_labels=torch.tensor([[MP_LAB]], dtype=torch.long),
        multimask_output=False)
mp_log = seg_mp.pred_masks[0, 0, 0]  # [H, W]
v2['multipoint'] = {'points': MP_PTS, 'labels': MP_LAB,
                    'mask_logits': mp_log.tolist()}

# (b) BOX prompt (no points), single mask. input_boxes is (B, nb, 4).
BOX = [6.0, 4.0, 26.0, 22.0]   # x0,y0,x1,y1 in image pixels
with torch.no_grad():
    seg_bx = model(
        image_embeddings=img_emb,
        input_boxes=torch.tensor([[BOX]], dtype=torch.float64),
        multimask_output=False)
bx_log = seg_bx.pred_masks[0, 0, 0]  # [H, W]
v2['box'] = {'box': BOX, 'mask_logits': bx_log.tolist()}

# (c) MULTI-MASK output (3 masks) + IoU scores for the single positive click.
with torch.no_grad():
    seg_mm = model(
        image_embeddings=img_emb,
        input_points=torch.tensor([[[[PT_X, PT_Y]]]], dtype=torch.float64),
        input_labels=torch.tensor([[[1]]], dtype=torch.long),
        multimask_output=True)
mm_masks = seg_mm.pred_masks[0, 0]      # [3, H, W]
mm_iou = seg_mm.iou_scores[0, 0]        # [3]
v2['multimask'] = {'point': [PT_X, PT_Y], 'label': 1,
                   'num_masks': int(mm_masks.shape[0]),
                   'mask_logits': mm_masks.tolist(),  # [3][H][W]
                   'iou_scores': mm_iou.tolist()}      # [3]

with open('tests/fixtures/tiny_sam_mask_v2.json', 'w') as f:
    json.dump(v2, f)
print(f'wrote tiny_sam_mask_v2.json: multipoint range '
      f'[{float(mp_log.min()):.4f},{float(mp_log.max()):.4f}]; '
      f'box range [{float(bx_log.min()):.4f},{float(bx_log.max()):.4f}]; '
      f'multimask {list(mm_masks.shape)}; iou {[round(float(v),4) for v in mm_iou]}')

# v2 self-checks: each prompt mode must give a DISTINCT result.
assert (mp_log - mlogits).abs().max() > 1e-3, 'multi-point == single-point'
assert (bx_log - mlogits).abs().max() > 1e-3, 'box == single-point'
assert (mm_masks[0] - mlogits).abs().max() > 1e-3, 'multimask[0] == single'
assert mm_masks.shape[0] == 3, 'multimask must emit 3 masks'

# mask-decoder fixture self-checks: the click must matter.
with torch.no_grad():
    seg2 = model(image_embeddings=img_emb,
                 input_points=torch.tensor([[[[2.0, 2.0]]]], dtype=torch.float64),
                 input_labels=input_labels, multimask_output=False)
    d = (seg2.pred_masks[0, 0, 0] - mlogits).abs().max()
    assert d > 1e-3, f'point location had no effect on the mask ({d})'
    print(f'click-location effect: max |diff| = {d:.4f}')

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
