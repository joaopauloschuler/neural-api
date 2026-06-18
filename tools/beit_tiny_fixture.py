#!/usr/bin/env python3
"""Generate a tiny RANDOM BEiT / data2vec-vision parity fixture for
tests/TestNeuralPretrained.pas (no network: the model is randomly built from a
pico config, never downloaded).

BEiT is a DISTINCT ViT backbone family from the plain ViT/DINOv2 and Swin:
  - FULL global attention with a per-LAYER learned relative_position_bias_table
    gathered by a fixed cls-aware relative-position index and ADDED to EVERY
    block's attention scores (NO windowing/shift);
  - LayerScale (lambda_1 / lambda_2) gating BOTH residual branches;
  - NO absolute position embedding; a learnable cls token prepended;
  - query / value projections biased, KEY projection bias-free (a BEiT trait);
  - layer_norm_eps 1e-12; exact-erf "gelu".

This generator builds a tiny HF BeitModel (use_mean_pooling=False so a final
encoder LayerNorm is applied and last_hidden_state is the directly-comparable
output), then RE-KEYS its state_dict to the CLASSIC published key scheme
(beit.encoder.layer.N.attention.attention.{query,key,value}, attention.output.
dense, intermediate.dense, output.dense, lambda_1/2,
attention.relative_position_bias.relative_position_bias_table) that the importer
targets and that the real microsoft/beit-* and facebook/data2vec-vision-*
safetensors use.

The reference is the HF float64 last_hidden_state (post-final-LayerNorm token
hidden states, cls row 0 + patch rows) for a pinned image.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/beit_tiny_fixture.py
writes tests/fixtures/tiny_beit{.safetensors,_config.json,_hidden.json}.
"""
import copy
import json

import torch
from safetensors.torch import save_file
from transformers import BeitConfig, BeitModel

HIDDEN = 16
N_LAYER = 2
N_HEAD = 4                    # head_dim = 4
IMAGE = 12
PATCH = 4                     # 3x3 = 9 patches + 1 cls = 10 tokens
INTERMEDIATE = 64

torch.manual_seed(20260615)

cfg = BeitConfig(
    hidden_size=HIDDEN, intermediate_size=INTERMEDIATE,
    num_hidden_layers=N_LAYER, num_attention_heads=N_HEAD,
    image_size=IMAGE, patch_size=PATCH, num_channels=3,
    hidden_act='gelu', layer_norm_eps=1e-12,
    attention_probs_dropout_prob=0.0, hidden_dropout_prob=0.0,
    drop_path_rate=0.0,
    use_absolute_position_embeddings=False,
    use_relative_position_bias=True,
    use_shared_relative_position_bias=False,
    use_mean_pooling=False,           # -> final encoder LayerNorm is applied
    layer_scale_init_value=0.1,
)
model = BeitModel(cfg, add_pooling_layer=False)

# Boost the pico inits so every quirk (LayerScale, rel-pos bias, cls token,
# patch bias) is visible above the 1e-4 parity gate (the ModernBERT
# vacuous-init lesson).
with torch.no_grad():
    emb = model.embeddings
    emb.cls_token.normal_(0.0, 0.6)
    emb.patch_embeddings.projection.weight.normal_(0.0, 0.25)
    emb.patch_embeddings.projection.bias.normal_(0.0, 0.2)
    for layer in model.layers:
        a = layer.attention
        a.q_proj.weight.normal_(0.0, 0.35); a.q_proj.bias.normal_(0.0, 0.2)
        a.k_proj.weight.normal_(0.0, 0.35)          # bias-free
        a.v_proj.weight.normal_(0.0, 0.35); a.v_proj.bias.normal_(0.0, 0.2)
        a.o_proj.weight.normal_(0.0, 0.35); a.o_proj.bias.normal_(0.0, 0.2)
        layer.mlp.fc1.weight.normal_(0.0, 0.5); layer.mlp.fc1.bias.normal_(0.0, 0.3)
        layer.mlp.fc2.weight.normal_(0.0, 0.3); layer.mlp.fc2.bias.normal_(0.0, 0.2)
        for norm in (layer.layernorm_before, layer.layernorm_after):
            norm.weight.normal_(1.0, 0.25); norm.bias.normal_(0.0, 0.2)
        layer.lambda_1.normal_(0.5, 0.3)
        layer.lambda_2.normal_(0.5, 0.3)
        # rel-pos bias table: make it clearly non-zero so it matters.
        layer.relative_position_bias.relative_position_bias_table.normal_(0.0, 0.8)
    model.layernorm.weight.normal_(1.0, 0.25)
    model.layernorm.bias.normal_(0.0, 0.2)
model = model.double().eval()

# Pinned input: deterministic dyadic pixel values (exact in f32 and JSON).
pixels = torch.zeros(1, 3, IMAGE, IMAGE, dtype=torch.float64)
for c in range(3):
    for y in range(IMAGE):
        for x in range(IMAGE):
            pixels[0, c, y, x] = (((c * 256 + y * 16 + x) * 5) % 17 - 8) / 8.0

# ---- RE-KEY the v5 state_dict to the classic published BEiT scheme ----
v5 = {k: v.to(torch.float32).clone().contiguous()
      for k, v in model.state_dict().items()}
classic = {}
classic['beit.embeddings.cls_token'] = v5['embeddings.cls_token']
classic['beit.embeddings.patch_embeddings.projection.weight'] = \
    v5['embeddings.patch_embeddings.projection.weight']
classic['beit.embeddings.patch_embeddings.projection.bias'] = \
    v5['embeddings.patch_embeddings.projection.bias']
classic['beit.layernorm.weight'] = v5['layernorm.weight']
classic['beit.layernorm.bias'] = v5['layernorm.bias']
for i in range(N_LAYER):
    s = f'layers.{i}.'
    d = f'beit.encoder.layer.{i}.'
    classic[d + 'lambda_1'] = v5[s + 'lambda_1']
    classic[d + 'lambda_2'] = v5[s + 'lambda_2']
    classic[d + 'attention.attention.query.weight'] = v5[s + 'attention.q_proj.weight']
    classic[d + 'attention.attention.query.bias'] = v5[s + 'attention.q_proj.bias']
    classic[d + 'attention.attention.key.weight'] = v5[s + 'attention.k_proj.weight']
    classic[d + 'attention.attention.value.weight'] = v5[s + 'attention.v_proj.weight']
    classic[d + 'attention.attention.value.bias'] = v5[s + 'attention.v_proj.bias']
    classic[d + 'attention.output.dense.weight'] = v5[s + 'attention.o_proj.weight']
    classic[d + 'attention.output.dense.bias'] = v5[s + 'attention.o_proj.bias']
    classic[d + 'attention.attention.relative_position_bias.relative_position_bias_table'] = \
        v5[s + 'relative_position_bias.relative_position_bias_table']
    classic[d + 'layernorm_before.weight'] = v5[s + 'layernorm_before.weight']
    classic[d + 'layernorm_before.bias'] = v5[s + 'layernorm_before.bias']
    classic[d + 'layernorm_after.weight'] = v5[s + 'layernorm_after.weight']
    classic[d + 'layernorm_after.bias'] = v5[s + 'layernorm_after.bias']
    classic[d + 'intermediate.dense.weight'] = v5[s + 'mlp.fc1.weight']
    classic[d + 'intermediate.dense.bias'] = v5[s + 'mlp.fc1.bias']
    classic[d + 'output.dense.weight'] = v5[s + 'mlp.fc2.weight']
    classic[d + 'output.dense.bias'] = v5[s + 'mlp.fc2.bias']

save_file(classic, 'tests/fixtures/tiny_beit.safetensors')

# Config: emit the classic-scheme config (the importer reads it).
cfg_out = cfg.to_dict()
with open('tests/fixtures/tiny_beit_config.json', 'w') as f:
    json.dump(cfg_out, f, indent=1, default=str)

with torch.no_grad():
    hidden = model(pixel_values=pixels).last_hidden_state    # [1][N_TOK][HIDDEN]
n_tok = hidden.shape[1]
with open('tests/fixtures/tiny_beit_hidden.json', 'w') as f:
    json.dump({
        'pixels': pixels[0].tolist(),
        'hidden': hidden[0].tolist(),
        'num_tokens': n_tok,
        'hidden_size': HIDDEN,
    }, f)
print(f'wrote tiny_beit.safetensors ({len(classic)} tensors) + config + oracle')
print(f'  last_hidden_state shape = {list(hidden.shape)}')
for k in sorted(classic):
    print(f'  {k} {list(classic[k].shape)}')

# ---- fixture self-checks: every BEiT quirk must be visible ----
with torch.no_grad():
    base = hidden

    class TanhGELU(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.gelu(x, approximate='tanh')
    alt = copy.deepcopy(model)
    for layer in alt.layers:
        layer.mlp.activation_fn = TanhGELU().double()
    d = (alt(pixel_values=pixels).last_hidden_state - base).abs().max()
    assert d > 1.5e-4, f'exact-gelu vs tanh-gelu indistinguishable ({d})'
    print(f'gelu-vs-tanh: max |diff| = {d:.4f}')

    # rel-pos bias must matter
    alt = copy.deepcopy(model)
    for layer in alt.layers:
        layer.relative_position_bias.relative_position_bias_table.zero_()
    d = (alt(pixel_values=pixels).last_hidden_state - base).abs().max()
    assert d > 1e-3, f'relative_position_bias had no effect ({d})'
    print(f'rel-pos-bias effect: max |diff| = {d:.4f}')

    # LayerScale must matter
    alt = copy.deepcopy(model)
    for layer in alt.layers:
        layer.lambda_1.fill_(1.0); layer.lambda_2.fill_(1.0)
    d = (alt(pixel_values=pixels).last_hidden_state - base).abs().max()
    assert d > 1e-3, f'LayerScale had no effect ({d})'
    print(f'LayerScale effect: max |diff| = {d:.4f}')

    # cls token must matter
    alt = copy.deepcopy(model)
    alt.embeddings.cls_token.zero_()
    d = (alt(pixel_values=pixels).last_hidden_state - base).abs().max()
    assert d > 1e-3, f'cls_token had no effect ({d})'
    print(f'cls_token effect: max |diff| = {d:.4f}')

    # patch bias must matter
    alt = copy.deepcopy(model)
    alt.embeddings.patch_embeddings.projection.bias.zero_()
    d = (alt(pixel_values=pixels).last_hidden_state - base).abs().max()
    assert d > 1e-3, f'patch projection bias had no effect ({d})'
    print(f'patch bias effect: max |diff| = {d:.4f}')

    # key bias-free trait: q/v biased, k not. Confirm there is no k_proj.bias.
    assert model.layers[0].attention.k_proj.bias is None, 'k_proj should be bias-free'
    print('k_proj is bias-free (BEiT trait) confirmed')
print('all fixture self-checks passed')
