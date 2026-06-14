#!/usr/bin/env python3
"""Generate a tiny RANDOM DINOv2 parity fixture for
tests/TestNeuralPretrained.pas (no network access needed: the model is
randomly initialized from a pico config, never downloaded).

One fixture, KB-scale, pinned in tests/fixtures/:

  tiny_dinov2.*: Dinov2Model (the facebook/dinov2-small architecture,
      model_type "dinov2") with every DINOv2 trait the importer must
      reproduce:
        - a learnable CLS token (embeddings.cls_token) PREPENDED to the
          patch sequence;
        - a BIASED patch_embeddings.projection Conv2d(stride = kernel =
          patch_size);
        - learned absolute positions over num_patches+1 rows
          (embeddings.position_embeddings), added directly at the native
          square resolution (HF interpolate_pos_encoding is a no-op);
        - pre-LN encoder blocks (norm1 / norm2, BERT-style
          attention.attention.{query,key,value} + attention.output.dense +
          mlp.fc1 + mlp.fc2);
        - LayerScale on EVERY block: per-channel learnable scales
          (layer_scale1.lambda1 / layer_scale2.lambda1) multiplying each
          residual branch output before the skip add;
        - exact erf "gelu" activation (the Dinov2Config hidden_act default);
        - layer_norm_eps = 1e-6 (NOT ViT's 1e-12);
        - a final layernorm; the OUTPUT is the CLS + patch token hidden
          states (NO classifier head).

The reference is computed by HF transformers in float64 (the oracle
convention of the committed fixtures): the post-final-layernorm
last_hidden_state for the pinned image (CLS row 0 + patch rows).

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/dinov2_tiny_fixture.py
writes tests/fixtures/tiny_dinov2{.safetensors,_config.json,_hidden.json}.
Needs torch + transformers + safetensors.
"""
import copy
import json

import torch
from safetensors.torch import save_file
from transformers import Dinov2Config, Dinov2Model

HIDDEN = 16
N_LAYER = 2
N_HEAD = 4                    # head_dim = 4
IMAGE = 12
PATCH = 4                     # 3x3 = 9 patches + 1 CLS = 10 tokens
MLP_RATIO = 4                 # intermediate = 64

torch.manual_seed(20260614)

cfg = Dinov2Config(
    hidden_size=HIDDEN, mlp_ratio=MLP_RATIO,
    num_hidden_layers=N_LAYER, num_attention_heads=N_HEAD,
    image_size=IMAGE, patch_size=PATCH, num_channels=3,
    hidden_act='gelu', layer_norm_eps=1e-6,
    attention_probs_dropout_prob=0.0, hidden_dropout_prob=0.0,
    layerscale_value=1.0, use_swiglu_ffn=False,
)
model = Dinov2Model(cfg)

# HF inits with tiny stds at this pico width; boost so every quirk (esp. the
# LayerScale gates) is visible in the oracle (the ModernBERT vacuous-init
# lesson).
with torch.no_grad():
    emb = model.embeddings
    emb.cls_token.normal_(0.0, 0.6)
    emb.patch_embeddings.projection.weight.normal_(0.0, 0.25)
    emb.patch_embeddings.projection.bias.normal_(0.0, 0.2)
    emb.position_embeddings.normal_(0.0, 0.4)
    for layer in model.encoder.layer:
        a = layer.attention.attention
        for proj in (a.query, a.key, a.value):
            proj.weight.normal_(0.0, 0.35)
            proj.bias.normal_(0.0, 0.2)
        layer.attention.output.dense.weight.normal_(0.0, 0.35)
        layer.attention.output.dense.bias.normal_(0.0, 0.2)
        layer.mlp.fc1.weight.normal_(0.0, 0.5)
        layer.mlp.fc1.bias.normal_(0.0, 0.3)
        layer.mlp.fc2.weight.normal_(0.0, 0.3)
        layer.mlp.fc2.bias.normal_(0.0, 0.2)
        for norm in (layer.norm1, layer.norm2):
            norm.weight.normal_(1.0, 0.25)
            norm.bias.normal_(0.0, 0.2)
        # LayerScale lambda1: HF inits to layerscale_value (1.0); randomize so
        # the per-channel scale is non-trivial and the importer must load it.
        layer.layer_scale1.lambda1.normal_(0.5, 0.3)
        layer.layer_scale2.lambda1.normal_(0.5, 0.3)
    model.layernorm.weight.normal_(1.0, 0.25)
    model.layernorm.bias.normal_(0.0, 0.2)
model = model.double().eval()

assert model.embeddings.patch_embeddings.projection.bias is not None, \
    'DINOv2 patch_embeddings.projection must be biased'

# Pinned input: deterministic dyadic pixel values (exact in f32 and JSON).
pixels = torch.zeros(1, 3, IMAGE, IMAGE, dtype=torch.float64)
for c in range(3):
    for y in range(IMAGE):
        for x in range(IMAGE):
            pixels[0, c, y, x] = (((c * 256 + y * 16 + x) * 5) % 17 - 8) / 8.0

sd = {k: v.to(torch.float32).clone().contiguous()
      for k, v in model.state_dict().items()
      if not k.endswith('position_ids')}
save_file(sd, 'tests/fixtures/tiny_dinov2.safetensors')
with open('tests/fixtures/tiny_dinov2_config.json', 'w') as f:
    json.dump(cfg.to_dict(), f, indent=1, default=str)

with torch.no_grad():
    hidden = model(pixel_values=pixels).last_hidden_state   # [1][N_TOK][HIDDEN]
n_tok = hidden.shape[1]
with open('tests/fixtures/tiny_dinov2_hidden.json', 'w') as f:
    json.dump({
        'pixels': pixels[0].tolist(),                    # [3][IMAGE][IMAGE]
        'hidden': hidden[0].tolist(),                    # [N_TOK][HIDDEN]
        'num_tokens': n_tok,
        'hidden_size': HIDDEN,
    }, f)
print(f'wrote tiny_dinov2.safetensors ({len(sd)} tensors) + config + oracle')
print(f'  last_hidden_state shape = {list(hidden.shape)}')
for k in sorted(sd):
    print(f'  {k} {list(sd[k].shape)}')

# ---- fixture self-checks: every DINOv2 quirk must be visible ----
with torch.no_grad():
    base = hidden

    # 1. exact-erf "gelu" vs tanh-GELU and quick_gelu must be distinguishable
    # above the 1e-4 parity gate.
    class TanhGELU(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.gelu(x, approximate='tanh')

    class QuickGELU(torch.nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(1.702 * x)
    for wrong, act in (('gelu_tanh', TanhGELU().double()),
                       ('quick_gelu', QuickGELU().double())):
        alt = copy.deepcopy(model)
        for layer in alt.encoder.layer:
            layer.mlp.activation = act
        d = (alt(pixel_values=pixels).last_hidden_state - base).abs().max()
        assert d > 1.5e-4, f'exact-gelu vs {wrong} indistinguishable ({d})'
        print(f'gelu-vs-{wrong}: max |diff| = {d:.4f}')

    # 2. LayerScale must matter: setting lambda1 to ones (the no-LayerScale
    # baseline) must move the hidden states.
    alt = copy.deepcopy(model)
    for layer in alt.encoder.layer:
        layer.layer_scale1.lambda1.fill_(1.0)
        layer.layer_scale2.lambda1.fill_(1.0)
    d = (alt(pixel_values=pixels).last_hidden_state - base).abs().max()
    assert d > 1e-3, f'LayerScale had no effect ({d})'
    print(f'LayerScale effect on hidden: max |diff| = {d:.4f}')

    # 3. CLS token must matter (row 0 is the global feature).
    alt = copy.deepcopy(model)
    alt.embeddings.cls_token.zero_()
    d = (alt(pixel_values=pixels).last_hidden_state - base).abs().max()
    assert d > 1e-3, f'cls_token had no effect ({d})'
    print(f'cls_token effect on hidden: max |diff| = {d:.4f}')

    # 4. patch_embeddings projection BIAS must matter.
    alt = copy.deepcopy(model)
    alt.embeddings.patch_embeddings.projection.bias.zero_()
    d = (alt(pixel_values=pixels).last_hidden_state - base).abs().max()
    assert d > 1e-3, f'patch projection bias had no effect ({d})'
    print(f'patch bias effect on hidden: max |diff| = {d:.4f}')
print('all fixture self-checks passed')
