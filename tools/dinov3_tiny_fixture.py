#!/usr/bin/env python3
"""Generate a tiny RANDOM DINOv3 parity fixture for
tests/TestNeuralPretrained.pas (no network access: the model is randomly
initialized from a pico config, never downloaded).

DINOv3 (facebook/dinov3-*, HF model_type "dinov3_vit", class DINOv3ViTModel)
is the successor to DINOv2. Same pre-LN ViT trunk + LayerScale, but THREE
deltas the importer must reproduce:

  1. 2-D AXIAL RoPE on the patch positions INSTEAD of a learned absolute
     position table. inv_freq[j] = base**(-(4j)/head_dim), j in
     [0, head_dim/4); patch n at grid (i,k) gets coord
     (2*(i+.5)/Hp - 1, 2*(k+.5)/Wp - 1); angles =
     2*pi * coord[:,None] * inv_freq -> flatten -> tile(2) (length head_dim);
     q,k rotated with rotate_half(x)=cat(-x[h:],x[:h]). RoPE is applied ONLY
     to the patch tokens (the CLS + register tokens pass through unrotated)
     and the SAME (cos,sin) is reused in every layer. base = rope_theta = 100.
  2. REGISTER TOKENS: num_register_tokens learnable tokens prepended right
     AFTER the CLS token: order is [CLS][reg_0..reg_{R-1}][patches].
  3. Gram-anchoring-trained weights (no architectural change).

Attention is BERT-unfused: separate q_proj/k_proj/v_proj/o_proj with
query/value/proj biased but KEY UNBIASED (key_bias=False). MLP is the plain
up_proj/down_proj + gelu (use_gated_mlp=False); LayerScale on each branch;
final layernorm "norm"; output = last_hidden_state over ALL tokens.

The reference is computed by HF transformers in float64 (the oracle
convention of the committed fixtures): the post-final-layernorm
last_hidden_state for the pinned image.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/dinov3_tiny_fixture.py
writes tests/fixtures/tiny_dinov3{.safetensors,_config.json,_hidden.json}.
Needs torch + transformers (>=5.11, ships dinov3_vit) + safetensors.
"""
import copy
import json

import torch
from safetensors.torch import save_file
from transformers import DINOv3ViTModel, DINOv3ViTConfig

HIDDEN = 16
N_LAYER = 2
N_HEAD = 4                    # head_dim = 4 -> head_dim/4 = 1 freq per axis
IMAGE = 12
PATCH = 4                     # 3x3 = 9 patches; +1 CLS +2 register = 12 tokens
INTERMEDIATE = 64             # mlp_ratio 4
N_REGISTER = 2
ROPE_THETA = 100.0

torch.manual_seed(20260627)

cfg = DINOv3ViTConfig(
    hidden_size=HIDDEN, intermediate_size=INTERMEDIATE,
    num_hidden_layers=N_LAYER, num_attention_heads=N_HEAD,
    image_size=IMAGE, patch_size=PATCH, num_channels=3,
    hidden_act='gelu', layer_norm_eps=1e-6,
    num_register_tokens=N_REGISTER, rope_theta=ROPE_THETA,
    use_gated_mlp=False, layerscale_value=1.0,
    query_bias=True, key_bias=False, value_bias=True, proj_bias=True,
    mlp_bias=True, attention_dropout=0.0, drop_path_rate=0.0,
)
model = DINOv3ViTModel(cfg)

# HF inits with tiny stds at pico width; boost so every quirk (esp. the
# LayerScale gates, register tokens and RoPE) is visible in the oracle
# (the ModernBERT vacuous-init lesson).
with torch.no_grad():
    emb = model.embeddings
    emb.cls_token.normal_(0.0, 0.6)
    emb.register_tokens.normal_(0.0, 0.6)
    emb.patch_embeddings.weight.normal_(0.0, 0.25)
    emb.patch_embeddings.bias.normal_(0.0, 0.2)
    for layer in model.model.layer:
        a = layer.attention
        for proj in (a.q_proj, a.k_proj, a.v_proj, a.o_proj):
            proj.weight.normal_(0.0, 0.35)
            if proj.bias is not None:
                proj.bias.normal_(0.0, 0.2)
        layer.mlp.up_proj.weight.normal_(0.0, 0.5)
        layer.mlp.up_proj.bias.normal_(0.0, 0.3)
        layer.mlp.down_proj.weight.normal_(0.0, 0.3)
        layer.mlp.down_proj.bias.normal_(0.0, 0.2)
        for norm in (layer.norm1, layer.norm2):
            norm.weight.normal_(1.0, 0.25)
            norm.bias.normal_(0.0, 0.2)
        layer.layer_scale1.lambda1.normal_(0.5, 0.3)
        layer.layer_scale2.lambda1.normal_(0.5, 0.3)
    model.norm.weight.normal_(1.0, 0.25)
    model.norm.bias.normal_(0.0, 0.2)
model = model.double().eval()

assert model.embeddings.patch_embeddings.bias is not None, \
    'DINOv3 patch_embeddings must be biased'
assert model.model.layer[0].attention.k_proj.bias is None, \
    'DINOv3 k_proj must be UNbiased (key_bias=False)'

# Pinned input: deterministic dyadic pixel values (exact in f32 and JSON).
pixels = torch.zeros(1, 3, IMAGE, IMAGE, dtype=torch.float64)
for c in range(3):
    for y in range(IMAGE):
        for x in range(IMAGE):
            pixels[0, c, y, x] = (((c * 256 + y * 16 + x) * 5) % 17 - 8) / 8.0

sd = {k: v.to(torch.float32).clone().contiguous()
      for k, v in model.state_dict().items()
      if not k.endswith('position_ids') and not k.endswith('inv_freq')
      and not k.endswith('mask_token')}
save_file(sd, 'tests/fixtures/tiny_dinov3.safetensors')
with open('tests/fixtures/tiny_dinov3_config.json', 'w') as f:
    json.dump(cfg.to_dict(), f, indent=1, default=str)

with torch.no_grad():
    hidden = model(pixel_values=pixels).last_hidden_state   # [1][N_TOK][HIDDEN]
n_tok = hidden.shape[1]
with open('tests/fixtures/tiny_dinov3_hidden.json', 'w') as f:
    json.dump({
        'pixels': pixels[0].tolist(),                    # [3][IMAGE][IMAGE]
        'hidden': hidden[0].tolist(),                    # [N_TOK][HIDDEN]
        'num_tokens': n_tok,
        'hidden_size': HIDDEN,
        'num_register_tokens': N_REGISTER,
    }, f)
print(f'wrote tiny_dinov3.safetensors ({len(sd)} tensors) + config + oracle')
print(f'  last_hidden_state shape = {list(hidden.shape)} (1 CLS + '
      f'{N_REGISTER} reg + 9 patch = {n_tok} tokens)')
for k in sorted(sd):
    print(f'  {k} {list(sd[k].shape)}')

# ---- fixture self-checks: every DINOv3 quirk must be visible ----
with torch.no_grad():
    base = hidden

    # 1. exact-erf "gelu" vs tanh-GELU / quick_gelu distinguishable > 1e-4.
    class TanhGELU(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.gelu(x, approximate='tanh')

    class QuickGELU(torch.nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(1.702 * x)
    for wrong, act in (('gelu_tanh', TanhGELU().double()),
                       ('quick_gelu', QuickGELU().double())):
        alt = copy.deepcopy(model)
        for layer in alt.model.layer:
            layer.mlp.act_fn = act
        d = (alt(pixel_values=pixels).last_hidden_state - base).abs().max()
        assert d > 1.5e-4, f'exact-gelu vs {wrong} indistinguishable ({d})'
        print(f'gelu-vs-{wrong}: max |diff| = {d:.4f}')

    # 2. LayerScale must matter.
    alt = copy.deepcopy(model)
    for layer in alt.model.layer:
        layer.layer_scale1.lambda1.fill_(1.0)
        layer.layer_scale2.lambda1.fill_(1.0)
    d = (alt(pixel_values=pixels).last_hidden_state - base).abs().max()
    assert d > 1e-3, f'LayerScale had no effect ({d})'
    print(f'LayerScale effect on hidden: max |diff| = {d:.4f}')

    # 3. RoPE must matter: zeroing inv_freq (=no rotation, cos=1,sin=0) must
    # move the patch-token hidden states.
    alt = copy.deepcopy(model)
    alt.rope_embeddings.inv_freq.zero_()
    d = (alt(pixel_values=pixels).last_hidden_state - base).abs().max()
    assert d > 1e-3, f'RoPE had no effect ({d})'
    print(f'RoPE effect on hidden: max |diff| = {d:.4f}')

    # 4. register tokens must matter.
    alt = copy.deepcopy(model)
    alt.embeddings.register_tokens.zero_()
    d = (alt(pixel_values=pixels).last_hidden_state - base).abs().max()
    assert d > 1e-3, f'register tokens had no effect ({d})'
    print(f'register-token effect on hidden: max |diff| = {d:.4f}')

    # 5. CLS token must matter.
    alt = copy.deepcopy(model)
    alt.embeddings.cls_token.zero_()
    d = (alt(pixel_values=pixels).last_hidden_state - base).abs().max()
    assert d > 1e-3, f'cls_token had no effect ({d})'
    print(f'cls_token effect on hidden: max |diff| = {d:.4f}')

    # 6. patch projection BIAS must matter.
    alt = copy.deepcopy(model)
    alt.embeddings.patch_embeddings.bias.zero_()
    d = (alt(pixel_values=pixels).last_hidden_state - base).abs().max()
    assert d > 1e-3, f'patch projection bias had no effect ({d})'
    print(f'patch bias effect on hidden: max |diff| = {d:.4f}')
print('all fixture self-checks passed')
