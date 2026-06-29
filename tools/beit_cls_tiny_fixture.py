#!/usr/bin/env python3
"""Generate tiny RANDOM BEiT / data2vec-vision IMAGE-CLASSIFICATION parity
fixtures for tests/TestNeuralPretrained.pas (no network: the model is randomly
built from a pico config, never downloaded).

This is the classification counterpart of tools/beit_tiny_fixture.py (which
covers the bare-encoder last_hidden_state). Here we build a tiny HF
BeitForImageClassification and compare the (1,num_labels) class LOGITS, which
exercise the full pooler + classifier head the Pascal
BuildBeitFromSafeTensorsForImageClassification appends:

  BeitPooler (HF, EXACT):
    use_mean_pooling=True  -> layernorm(hidden_states[:, 1:, :].mean(1))
                              (mean over PATCH tokens, cls excluded; the encoder
                               final 'layernorm' is Identity in this mode - the
                               LayerNorm lives in pooler.layernorm)
    use_mean_pooling=False -> hidden_states[:, 0]  (cls row, after the encoder's
                               final 'layernorm'; NO pooler LayerNorm)
  classifier nn.Linear(hidden, num_labels) -> class logits.

We emit BOTH modes:
  use_mean_pooling=True  (the PUBLISHED microsoft/beit-* / data2vec-vision-ft
                          default) -> tiny_beit_cls_mean.*
  use_mean_pooling=False                                  -> tiny_beit_cls_cls.*

Each fixture re-keys the v5 state_dict to the CLASSIC published BEiT scheme
(beit.encoder.layer.N.*, beit.pooler.layernorm.*, beit.layernorm.*,
classifier.*) the importer targets and the real checkpoints use.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/beit_cls_tiny_fixture.py
writes, for each mode, tests/fixtures/tiny_beit_cls_{mode}{.safetensors,
_config.json,_logits.json}.
"""
import json

import torch
from safetensors.torch import save_file
from transformers import BeitConfig, BeitForImageClassification

HIDDEN = 16
N_LAYER = 2
N_HEAD = 4                    # head_dim = 4
IMAGE = 12
PATCH = 4                     # 3x3 = 9 patches + 1 cls = 10 tokens
INTERMEDIATE = 64
NUM_LABELS = 5


def build(use_mean_pooling: bool):
    torch.manual_seed(20260627)
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
        use_mean_pooling=use_mean_pooling,
        layer_scale_init_value=0.1,
        num_labels=NUM_LABELS,
    )
    model = BeitForImageClassification(cfg)

    # Boost the pico inits so every quirk (LayerScale, rel-pos bias, cls token,
    # patch bias, pooler LN, classifier) is visible above the 1e-4 parity gate.
    with torch.no_grad():
        beit = model.beit
        emb = beit.embeddings
        emb.cls_token.normal_(0.0, 0.6)
        emb.patch_embeddings.projection.weight.normal_(0.0, 0.25)
        emb.patch_embeddings.projection.bias.normal_(0.0, 0.2)
        for layer in beit.layers:
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
            layer.relative_position_bias.relative_position_bias_table.normal_(0.0, 0.8)
        if use_mean_pooling:
            beit.pooler.layernorm.weight.normal_(1.0, 0.25)
            beit.pooler.layernorm.bias.normal_(0.0, 0.2)
        else:
            beit.layernorm.weight.normal_(1.0, 0.25)
            beit.layernorm.bias.normal_(0.0, 0.2)
        model.classifier.weight.normal_(0.0, 0.4)
        model.classifier.bias.normal_(0.0, 0.3)
    model = model.double().eval()
    return cfg, model


def rekey(model):
    v5 = {k: v.to(torch.float32).clone().contiguous()
          for k, v in model.state_dict().items()}
    classic = {}
    classic['beit.embeddings.cls_token'] = v5['beit.embeddings.cls_token']
    classic['beit.embeddings.patch_embeddings.projection.weight'] = \
        v5['beit.embeddings.patch_embeddings.projection.weight']
    classic['beit.embeddings.patch_embeddings.projection.bias'] = \
        v5['beit.embeddings.patch_embeddings.projection.bias']
    if 'beit.layernorm.weight' in v5:          # use_mean_pooling=False
        classic['beit.layernorm.weight'] = v5['beit.layernorm.weight']
        classic['beit.layernorm.bias'] = v5['beit.layernorm.bias']
    if 'beit.pooler.layernorm.weight' in v5:   # use_mean_pooling=True
        classic['beit.pooler.layernorm.weight'] = v5['beit.pooler.layernorm.weight']
        classic['beit.pooler.layernorm.bias'] = v5['beit.pooler.layernorm.bias']
    classic['classifier.weight'] = v5['classifier.weight']
    classic['classifier.bias'] = v5['classifier.bias']
    for i in range(N_LAYER):
        s = f'beit.layers.{i}.'
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
    return classic


# Pinned input: deterministic dyadic pixel values (exact in f32 and JSON).
pixels = torch.zeros(1, 3, IMAGE, IMAGE, dtype=torch.float64)
for c in range(3):
    for y in range(IMAGE):
        for x in range(IMAGE):
            pixels[0, c, y, x] = (((c * 256 + y * 16 + x) * 5) % 17 - 8) / 8.0

for mode, ump in (('mean', True), ('cls', False)):
    cfg, model = build(ump)
    classic = rekey(model)
    stem = f'tests/fixtures/tiny_beit_cls_{mode}'
    save_file(classic, stem + '.safetensors')
    with open(stem + '_config.json', 'w') as f:
        json.dump(cfg.to_dict(), f, indent=1, default=str)
    with torch.no_grad():
        logits = model(pixel_values=pixels).logits        # [1][NUM_LABELS]
    with open(stem + '_logits.json', 'w') as f:
        json.dump({
            'pixels': pixels[0].tolist(),
            'logits': logits[0].tolist(),
            'num_labels': NUM_LABELS,
            'use_mean_pooling': ump,
        }, f)
    print(f'wrote {stem}.* ({len(classic)} tensors, use_mean_pooling={ump})')
    print(f'  logits = {[round(v, 5) for v in logits[0].tolist()]}')

    # ---- self-check: the head must matter (zeroing classifier -> bias only) ----
    with torch.no_grad():
        import copy
        alt = copy.deepcopy(model)
        alt.classifier.weight.zero_()
        d = (alt(pixel_values=pixels).logits - logits).abs().max()
        assert d > 1e-2, f'classifier weight had no effect ({d})'
        # pooler/cls path must matter: perturbing the relevant norm shifts logits
        alt = copy.deepcopy(model)
        if ump:
            alt.beit.pooler.layernorm.bias.add_(1.0)
        else:
            alt.beit.layernorm.bias.add_(1.0)
        d = (alt(pixel_values=pixels).logits - logits).abs().max()
        assert d > 1e-3, f'pooler/final LayerNorm had no effect ({d})'
    print('  self-checks passed (classifier + pooler/final-LN both matter)')
print('all classification fixtures written')
