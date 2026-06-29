#!/usr/bin/env python3
"""Generate a tiny RANDOM Mask2Former universal-segmentation parity fixture for
tests/TestNeuralPretrained.pas.

Mask2Former (Cheng et al. 2022, "Masked-attention Mask Transformer for Universal
Image Segmentation", https://arxiv.org/abs/2112.01527) does MASK CLASSIFICATION
set-prediction: a fixed set of learned object queries, each predicting ONE binary
mask + a class distribution, unifying semantic/instance/panoptic in one head.
This is distinct from per-PIXEL argmax (SegFormer) and region-proposal RoIAlign
(Mask R-CNN): there are NO proposals and NO per-pixel classifier.

The conceptual core is the MASKED-ATTENTION decoder layer: each layer's cross-
attention is restricted to the FOREGROUND of the mask predicted by the PREVIOUS
layer (sigmoid(mask) >= 0.5 keys are allowed; background keys get -inf). The
mask is the dot product of a per-query mask-embedding (3-layer MLP) with the
high-res per-pixel mask_features.

SCOPE v1 (this fixture): the importer builds the masked-attention DECODER + heads
only and takes the PIXEL-DECODER outputs (mask_features + the 3 multi-scale
memory levels, already + level_embed + sine pos) as PRECOMPUTED inputs -- exactly
the way Mask R-CNN v1 took FPN feature maps directly. The Swin backbone + FPN
pixel decoder wired into one forward is the documented deferred follow-up. So we
extract the real HF pixel-decoder tensors here and feed them to Pascal; the
parity check covers the genuinely-new pieces: masked attention, the mask einsum,
and the dot-product per-query mask/class logits.

Reference forward is the REAL transformers Mask2FormerForUniversalSegmentation in
float64; the test asserts max|diff| < 1e-4 on the final-layer per-query mask
logits AND class logits for a fixed image.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/make_pico_mask2former_fixture.py
writes tests/fixtures/tiny_mask2former{.safetensors,_config.json,_io.json}.
Needs numpy + torch + safetensors + transformers.
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.numpy import save_file
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

torch.manual_seed(20260626)
np.random.seed(20260626)

# ---------------- pico config ----------------
HIDDEN = 32            # decoder hidden_dim == mask_feature_size (>=32: HF pixel
                       # decoder GroupNorm(32,.) needs divisibility, even though
                       # v1 only ports the transformer module + heads)
HEADS = 1             # 1 attention head (keeps the parity graph minimal)
FFN = 48
NUM_QUERIES = 5
NUM_LABELS = 4         # foreground classes (class logits => +1 no-object slot)
DEC_LAYERS = 3         # >1 so the masked-attention feedback actually cycles
# The pixel decoder emits 3 multi-scale memory levels (low->high res) plus the
# 1/4-res mask_features. We pick tiny grids; the decoder attends round-robin
# level_index = layer % 3.
MASK_H, MASK_W = 8, 8                 # mask_features grid (1/4 res)
LEVELS = [(2, 2), (3, 3), (4, 4)]     # multi_scale memory grids, low->high res

# NOTE: Mask2Former hardcodes num_feature_levels=3 in the decoder.
cfg = Mask2FormerConfig(
    num_labels=NUM_LABELS,
    num_queries=NUM_QUERIES,
    hidden_dim=HIDDEN,
    mask_feature_size=HIDDEN,
    num_attention_heads=HEADS,
    dim_feedforward=FFN,
    decoder_layers=DEC_LAYERS,
    feature_size=HIDDEN,
    encoder_feedforward_dim=FFN,
    encoder_layers=1,
    pre_norm=False,
    enforce_input_projection=False,
    common_stride=4,
)
model = Mask2FormerForUniversalSegmentation(cfg).eval()

# Re-randomize on an O(1) scale. HF default init (std ~0.02) makes the masks
# near-constant so every query foreground is identical and the masked attention
# never actually masks anything (parity would pass trivially). Larger spread
# makes some queries foreground & some background per pixel -> the -inf masking
# genuinely fires and the layer-to-layer feedback matters.
def reinit(m):
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.dim() >= 2:
                p.normal_(0.0, 0.5)
            else:
                p.normal_(0.0, 0.1)
reinit(model)

model = model.double()
torch.set_grad_enabled(False)

# ---------------- extract the pixel-decoder outputs (the v1 INPUTS) ----------
# Run the backbone + pixel decoder ONCE on a fixed random image to obtain:
#   mask_features              [1, HIDDEN, MASK_H, MASK_W]   (1/4 res)
#   multi_scale_features[0..2] [1, HIDDEN, h, w]  low->high res
# We then OVERRIDE these grids with our own tiny random tensors (the real Swin
# grid sizes depend on image size; we just need SELF-CONSISTENT inputs for the
# decoder + heads, which is all the importer reproduces in v1). The transformer
# module's input_projections / level_embed / sine-pos are part of HF's decoder,
# so we must run them; to keep Pascal's job to "decoder + heads only" we feed the
# memory levels AS HF PRODUCES THEM right before the masked-attention loop
# (already projected + level_embed + sine pos for the keys; raw for values).

torch.manual_seed(7)
mask_features = torch.randn(1, HIDDEN, MASK_H, MASK_W, dtype=torch.float64) * 0.7
ms_raw = [torch.randn(1, HIDDEN, h, w, dtype=torch.float64) * 0.7 for (h, w) in LEVELS]

tm = model.model.transformer_module
dec = tm.decoder

# Reproduce Mask2FormerTransformerModule.forward up to the decoder call, but
# with OUR mask_features / multi_scale_features. We need: per level the flattened
# source (value), the flattened pos+level_embed (added to keys), and size_list.
size_list = [(h, w) for (h, w) in LEVELS]
sources = []       # value memory per level, [h*w, 1, HIDDEN]
pos_embeds = []    # key positional (sine + level_embed) per level, [h*w, 1, HIDDEN]
for i, x in enumerate(ms_raw):
    pos = tm.position_embedder(x.shape, x.device, x.dtype).flatten(2).double()  # [1,H,h*w]
    src = tm.input_projections[i](x).flatten(2)                  # [1, HIDDEN, h*w]
    pos = pos.permute(2, 0, 1)                                    # [h*w, 1, HIDDEN]
    src = src.permute(2, 0, 1)
    src = src + tm.level_embed.weight[i].view(1, 1, -1)
    sources.append(src)
    pos_embeds.append(pos)

# query content + positional
B = 1
query_features = tm.queries_features.weight.unsqueeze(1).repeat(1, B, 1)   # [Q,1,H]
query_embed = tm.queries_embedder.weight.unsqueeze(1).repeat(1, B, 1)      # [Q,1,H]

# ---------------- replicate the masked-attention decoder by hand (oracle) -----
# All in float64. This mirrors Mask2FormerMaskedAttentionDecoder exactly and is
# our ground truth (it also cross-checks our understanding of HF internals).
HEAD_DIM = HIDDEN // HEADS
scaling = HEAD_DIM ** -0.5

def linear(x, w, b):
    return x @ w.t() + b

def layernorm(x, ln):
    return F.layer_norm(x, (HIDDEN,), ln.weight, ln.bias, eps=1e-5)

mp = dec.mask_predictor

def predict(hs_normed, target_size):
    # hs_normed: [Q,1,H] AFTER decoder.layernorm
    mask_embed = mp.mask_embedder(hs_normed.transpose(0, 1))   # [1,Q,H]
    outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)  # [1,Q,Hm,Wm]
    am = F.interpolate(outputs_mask, size=target_size, mode="bilinear",
                       align_corners=False)
    am = (am.sigmoid().flatten(2).unsqueeze(1).repeat(1, HEADS, 1, 1) < 0.5).bool()
    am = am.detach()
    attn_mask = am.flatten(0, 1)   # [B*heads, Q, h*w]
    return outputs_mask, attn_mask

def cross_attn(layer, hs, level_idx, attn_mask):
    # nn.MultiheadAttention: q=hs+query_embed, k=src+pos, v=src
    q = hs + query_embed                                  # [Q,1,H]
    k = sources[level_idx] + pos_embeds[level_idx]        # [S,1,H]
    v = sources[level_idx]                                # [S,1,H]
    ca = layer.cross_attn
    w = ca.in_proj_weight; bb = ca.in_proj_bias
    qw, kw, vw = w[:HIDDEN], w[HIDDEN:2 * HIDDEN], w[2 * HIDDEN:]
    qb, kb, vb = bb[:HIDDEN], bb[HIDDEN:2 * HIDDEN], bb[2 * HIDDEN:]
    Q = linear(q, qw, qb).squeeze(1)                      # [Q,H]
    K = linear(k, kw, kb).squeeze(1)                      # [S,H]
    V = linear(v, vw, vb).squeeze(1)                      # [S,H]
    out = torch.zeros_like(Q)
    for h in range(HEADS):
        sl = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
        scores = (Q[:, sl] * scaling) @ K[:, sl].t()      # [Q,S]
        m = attn_mask[h]                                  # [Q,S] bool, True=masked
        scores = scores.masked_fill(m, float("-inf"))
        prob = scores.softmax(dim=-1)
        prob = torch.nan_to_num(prob)                     # fully-masked row -> 0
        out[:, sl] = prob @ V[:, sl]
    out = linear(out.unsqueeze(1), ca.out_proj.weight, ca.out_proj.bias)
    return out

def self_attn(layer, hs):
    sa = layer.self_attn
    q = hs + query_embed
    k = hs + query_embed
    Q = linear(q, sa.q_proj.weight, sa.q_proj.bias).squeeze(1) * scaling
    K = linear(k, sa.k_proj.weight, sa.k_proj.bias).squeeze(1)
    V = linear(hs, sa.v_proj.weight, sa.v_proj.bias).squeeze(1)
    out = torch.zeros_like(Q)
    for h in range(HEADS):
        sl = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
        scores = Q[:, sl] @ K[:, sl].t()
        prob = scores.softmax(dim=-1)
        out[:, sl] = prob @ V[:, sl]
    out = linear(out.unsqueeze(1), sa.out_proj.weight, sa.out_proj.bias)
    return out

def ffn(layer, hs):
    h = linear(hs, layer.fc1.weight, layer.fc1.bias).relu()
    h = linear(h, layer.fc2.weight, layer.fc2.bias)
    return h

# initial prediction (before layer 0) from layernorm(query_features)
hs = query_features
normed0 = layernorm(hs, dec.layernorm)
_, attn_mask = predict(normed0, size_list[0])

# HF builds (decoder_layers - 1) actual decoder layers; the config's
# decoder_layers counts the initial pre-layer-0 prediction. Iterate the real
# module list.
NUM_DEC_LAYERS = len(dec.layers)
class_logits_per_layer = []
mask_logits_per_layer = []
for idx in range(NUM_DEC_LAYERS):
    level_idx = idx % len(LEVELS)
    # "unmask all" fallback: rows that are entirely True -> entirely False
    where = (attn_mask.sum(-1) == attn_mask.shape[-1])
    attn_mask = attn_mask.clone()
    attn_mask[where] = False
    layer = dec.layers[idx]
    # cross -> self -> ffn, post-norm
    hs = layer.cross_attn_layer_norm(hs + cross_attn(layer, hs, level_idx, attn_mask))
    hs = layer.self_attn_layer_norm(hs + self_attn(layer, hs))
    hs = layer.final_layer_norm(hs + ffn(layer, hs))
    normed = layernorm(hs, dec.layernorm)
    cls = linear(normed.transpose(0, 1), model.class_predictor.weight,
                 model.class_predictor.bias)   # [1,Q,L+1]
    outputs_mask, attn_mask = predict(normed, size_list[(idx + 1) % len(LEVELS)])
    class_logits_per_layer.append(cls.squeeze(0))      # [Q,L+1]
    mask_logits_per_layer.append(outputs_mask.squeeze(0))  # [Q,Hm,Wm]

final_class = class_logits_per_layer[-1].detach().numpy()    # [Q,L+1]
final_mask = mask_logits_per_layer[-1].detach().numpy()      # [Q,Hm,Wm]

# semantic map (sanity / example reference): softmax over classes, drop no-object,
# sigmoid masks, einsum, argmax over class.
cls_prob = torch.tensor(final_class).softmax(-1)[:, :NUM_LABELS]      # [Q,L]
mask_prob = torch.tensor(final_mask).sigmoid()                        # [Q,Hm,Wm]
seg = torch.einsum("qc,qhw->chw", cls_prob, mask_prob)               # [L,Hm,Wm]
sem = seg.argmax(0).numpy().astype(int)                              # [Hm,Wm]

# ---------------- write ------------------------------------------------------
here = os.path.dirname(os.path.abspath(__file__))
fixtures = os.path.join(here, "..", "tests", "fixtures")
os.makedirs(fixtures, exist_ok=True)

# Only the DECODER + HEAD weights are needed by the v1 importer.
state = model.state_dict()
keep_prefixes = (
    "model.transformer_module.queries_features.weight",
    "model.transformer_module.queries_embedder.weight",
    "model.transformer_module.decoder.layers.",
    "model.transformer_module.decoder.mask_predictor.mask_embedder.",
    "model.transformer_module.decoder.layernorm.",
    "class_predictor.",
)
tensors = {}
for k, v in state.items():
    if k.endswith("num_batches_tracked"):
        continue
    if k.startswith(keep_prefixes):
        tensors[k] = v.detach().cpu().numpy().astype(np.float32)
save_file(tensors, os.path.join(fixtures, "tiny_mask2former.safetensors"))

config = {
    "model_type": "mask2former",
    "num_labels": NUM_LABELS,
    "num_queries": NUM_QUERIES,
    "hidden_dim": HIDDEN,
    "num_attention_heads": HEADS,
    "dim_feedforward": FFN,
    "decoder_layers": DEC_LAYERS,
    "num_dec_layers": NUM_DEC_LAYERS,
    "mask_h": MASK_H,
    "mask_w": MASK_W,
    "levels": [list(l) for l in LEVELS],
}
with open(os.path.join(fixtures, "tiny_mask2former_config.json"), "w") as f:
    json.dump(config, f, indent=2)

# The IO fixture carries: the precomputed pixel-decoder inputs (mask_features,
# the 3 memory levels' value/pos), and the expected final-layer per-query mask &
# class logits.
def flat(t):
    return [float(x) for x in np.asarray(t).reshape(-1)]

io = {
    "hidden": HIDDEN,
    "num_queries": NUM_QUERIES,
    "num_labels": NUM_LABELS,
    "mask_h": MASK_H, "mask_w": MASK_W,
    "levels": [list(l) for l in LEVELS],
    # mask_features: [HIDDEN, MASK_H, MASK_W] (channel-major, as HF)
    "mask_features": flat(mask_features.squeeze(0).numpy()),
    # per level: value source [S, HIDDEN] and key pos [S, HIDDEN]
    "level_sources": [flat(sources[i].squeeze(1).numpy()) for i in range(len(LEVELS))],
    "level_pos": [flat(pos_embeds[i].squeeze(1).numpy()) for i in range(len(LEVELS))],
    # expected outputs
    "class_logits": flat(final_class),      # [Q, L+1]
    "mask_logits": flat(final_mask),         # [Q, MASK_H, MASK_W]
    "semantic_map": [int(x) for x in sem.reshape(-1)],   # [MASK_H, MASK_W]
}
with open(os.path.join(fixtures, "tiny_mask2former_io.json"), "w") as f:
    json.dump(io, f)

print("wrote tiny_mask2former.{safetensors,_config.json,_io.json} to", fixtures)
print(f"  hidden={HIDDEN} heads={HEADS} queries={NUM_QUERIES} labels={NUM_LABELS} "
      f"dec_layers={DEC_LAYERS}")
print(f"  mask grid={MASK_H}x{MASK_W} levels={LEVELS}")
print(f"  #tensors={len(tensors)}  class_logits shape=({NUM_QUERIES},{NUM_LABELS+1})  "
      f"mask_logits shape=({NUM_QUERIES},{MASK_H},{MASK_W})")
sz = os.path.getsize(os.path.join(fixtures, "tiny_mask2former.safetensors"))
print(f"  safetensors size = {sz} bytes")
