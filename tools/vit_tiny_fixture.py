#!/usr/bin/env python3
"""Generate a tiny RANDOM ViT image-classification parity fixture for
tests/TestNeuralPretrained.pas (no network access needed: the model is
randomly initialized from a pico config, never downloaded).

One fixture, KB-scale, pinned in tests/fixtures/:

  tiny_vit.*: ViTForImageClassification (the google/vit-base-patch16-224
      architecture, model_type "vit") with every ViT trait the importer
      must reproduce:
        - a learnable CLS token (embeddings.cls_token) PREPENDED to the
          patch sequence;
        - a BIASED patch_embeddings.projection Conv2d(stride = kernel =
          patch_size) (CLIP's is bias-free);
        - learned absolute positions over num_patches+1 rows
          (embeddings.position_embeddings);
        - pre-LN encoder blocks (layernorm_before / layernorm_after,
          BERT-style attention.attention.{query,key,value} +
          attention.output.dense + intermediate.dense + output.dense);
        - exact erf "gelu" activation (the ViTConfig hidden_act default);
        - a final layernorm then a classifier nn.Linear reading the CLS
          row -> num_labels class logits.

The reference is computed by HF transformers in float64 (the oracle
convention of the committed fixtures): the class logits for the pinned
image.

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/vit_tiny_fixture.py
writes tests/fixtures/tiny_vit{.safetensors,_config.json,_logits.json}.
Needs torch + transformers + safetensors.
"""
import copy
import json

import torch
from safetensors.torch import save_file
from transformers import ViTConfig, ViTForImageClassification

HIDDEN = 16
INTER = 32
N_LAYER = 2
N_HEAD = 4                    # head_dim = 4
IMAGE = 12
PATCH = 4                     # 3x3 = 9 patches + 1 CLS = 10 tokens
NUM_LABELS = 5

torch.manual_seed(20260614)

cfg = ViTConfig(
    hidden_size=HIDDEN, intermediate_size=INTER,
    num_hidden_layers=N_LAYER, num_attention_heads=N_HEAD,
    image_size=IMAGE, patch_size=PATCH, num_channels=3,
    hidden_act='gelu', layer_norm_eps=1e-12,
    attention_probs_dropout_prob=0.0, hidden_dropout_prob=0.0,
    num_labels=NUM_LABELS,
)
model = ViTForImageClassification(cfg)

# HF inits with tiny stds at this pico width; boost so every quirk is
# visible in the oracle (the ModernBERT vacuous-init lesson).
with torch.no_grad():
    emb = model.vit.embeddings
    emb.cls_token.normal_(0.0, 0.6)
    emb.patch_embeddings.projection.weight.normal_(0.0, 0.25)
    emb.patch_embeddings.projection.bias.normal_(0.0, 0.2)
    emb.position_embeddings.normal_(0.0, 0.4)
    # transformers >=5 refactored ViT module layout (state_dict keys
    # vit.layers.N.attention.{q,k,v,o}_proj / mlp.fc1 / mlp.fc2). The
    # importer also accepts the legacy encoder.layer.N.attention.attention.*
    # names of the published google/vit-base-patch16-224 checkpoint.
    for layer in model.vit.layers:
        a = layer.attention
        for proj in (a.q_proj, a.k_proj, a.v_proj, a.o_proj):
            proj.weight.normal_(0.0, 0.35)
            proj.bias.normal_(0.0, 0.2)
        layer.mlp.fc1.weight.normal_(0.0, 0.5)
        layer.mlp.fc1.bias.normal_(0.0, 0.3)
        layer.mlp.fc2.weight.normal_(0.0, 0.3)
        layer.mlp.fc2.bias.normal_(0.0, 0.2)
        for norm in (layer.layernorm_before, layer.layernorm_after):
            norm.weight.normal_(1.0, 0.25)
            norm.bias.normal_(0.0, 0.2)
    model.vit.layernorm.weight.normal_(1.0, 0.25)
    model.vit.layernorm.bias.normal_(0.0, 0.2)
    model.classifier.weight.normal_(0.0, 0.4)
    model.classifier.bias.normal_(0.0, 0.2)
model = model.double().eval()

assert model.vit.embeddings.patch_embeddings.projection.bias is not None, \
    'ViT patch_embeddings.projection must be biased'

# Pinned input: deterministic dyadic pixel values (exact in f32 and JSON).
pixels = torch.zeros(1, 3, IMAGE, IMAGE, dtype=torch.float64)
for c in range(3):
    for y in range(IMAGE):
        for x in range(IMAGE):
            pixels[0, c, y, x] = (((c * 256 + y * 16 + x) * 5) % 17 - 8) / 8.0

sd = {k: v.to(torch.float32).clone().contiguous()
      for k, v in model.state_dict().items()
      if not k.endswith('position_ids')}
save_file(sd, 'tests/fixtures/tiny_vit.safetensors')
with open('tests/fixtures/tiny_vit_config.json', 'w') as f:
    json.dump(cfg.to_dict(), f, indent=1, default=str)

with torch.no_grad():
    logits = model(pixel_values=pixels).logits          # [1][NUM_LABELS]
with open('tests/fixtures/tiny_vit_logits.json', 'w') as f:
    json.dump({
        'pixels': pixels[0].tolist(),                    # [3][IMAGE][IMAGE]
        'logits': logits.tolist(),                       # [1][NUM_LABELS]
        'num_labels': NUM_LABELS,
    }, f)
print(f'wrote tiny_vit.safetensors ({len(sd)} tensors) + config + oracle')
for k in sorted(sd):
    print(f'  {k} {list(sd[k].shape)}')

# ---- fixture self-checks: every ViT quirk must be visible ----
with torch.no_grad():
    base = logits

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
        for layer in alt.vit.layers:
            layer.mlp.activation_fn = act
        d = (alt(pixel_values=pixels).logits - base).abs().max()
        assert d > 1.5e-4, f'exact-gelu vs {wrong} indistinguishable ({d})'
        print(f'gelu-vs-{wrong}: max |diff| = {d:.4f}')

    # 2. CLS token must matter: zeroing it must move the logits (the head
    # reads the CLS row).
    alt = copy.deepcopy(model)
    alt.vit.embeddings.cls_token.zero_()
    d = (alt(pixel_values=pixels).logits - base).abs().max()
    assert d > 1e-3, f'cls_token had no effect ({d})'
    print(f'cls_token effect on logits: max |diff| = {d:.4f}')

    # 3. patch_embeddings projection BIAS must matter (CLIP has none).
    alt = copy.deepcopy(model)
    alt.vit.embeddings.patch_embeddings.projection.bias.zero_()
    d = (alt(pixel_values=pixels).logits - base).abs().max()
    assert d > 1e-3, f'patch projection bias had no effect ({d})'
    print(f'patch bias effect on logits: max |diff| = {d:.4f}')

    # 4. classifier BIAS must matter.
    alt = copy.deepcopy(model)
    alt.classifier.bias.zero_()
    d = (alt(pixel_values=pixels).logits - base).abs().max()
    assert d > 1e-3, f'classifier bias had no effect ({d})'
    print(f'classifier bias effect on logits: max |diff| = {d:.4f}')

    # 5. the head reads the CLS row (row 0), NOT a patch row: the logits must
    # equal classifier(layernorm(last_hidden_state)[:, 0]).
    hs = model.vit(pixel_values=pixels).last_hidden_state[:, 0]
    ref = model.classifier(hs)
    assert (ref - base).abs().max() < 1e-10, \
        'classifier does not read the CLS (row 0) after final layernorm'
    print('CLS-row classifier head check passed')
print('all fixture self-checks passed')
