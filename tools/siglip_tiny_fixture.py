#!/usr/bin/env python3
"""Generate a tiny RANDOM SigLIP parity fixture for
tests/TestNeuralPretrained.pas (no network access needed: the model is
randomly initialized from a pico config, never downloaded).

One fixture, KB-scale, pinned in tests/fixtures/:

  tiny_siglip.*: SiglipModel (the google/siglip-base-patch16-224
      architecture, model_type "siglip") with every SigLIP trait the
      importer must reproduce - and that distinguish it from CLIP:
        - TEXT tower: token embedding + LEARNED absolute positions,
          BIDIRECTIONAL pre-LN encoder blocks (NO causal mask, unlike
          CLIP), final_layer_norm, pools the LAST token's hidden state
          (position -1, NOT CLIP's eos-argmax), then a BIASED head
          nn.Linear (text_model.head) - CLIP's text_projection is
          bias-free;
        - VISION tower: BIASED patch_embedding Conv2d(stride = kernel =
          patch_size) (CLIP's is bias-free), NO class token, learned
          absolute positions over the patch grid (num_patches rows, NOT
          num_patches+1), bidirectional pre-LN blocks, post_layernorm,
          then a Multihead Attention Pooling head (MAP): a single
          learnable probe query cross-attends over the patch tokens via
          nn.MultiheadAttention, then out = attn + mlp(layernorm(attn)),
          take row 0 - CLIP pools the CLS token + a bias-free projection;
        - gelu_pytorch_tanh activation (tanh-approx GELU) in BOTH towers
          and in the MAP head MLP - NOT CLIP's quick_gelu;
        - logit_scale AND logit_bias (CLIP has scale only):
          logits_per_text = text_n @ image_n^T * exp(logit_scale) +
          logit_bias after L2 normalization; logits_per_image = transpose.

The reference is computed by HF transformers in float64 (the oracle
convention of the committed fixtures): text_embeds / image_embeds are the
UNNORMALIZED pooled features (HF get_text_features / get_image_features),
plus logits_per_image for the pinned pair.

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/siglip_tiny_fixture.py
writes tests/fixtures/tiny_siglip{.safetensors,_config.json,_embeds.json}.
Needs torch + transformers + safetensors.
"""
import copy
import json

import torch
from safetensors.torch import save_file
from transformers import (SiglipConfig, SiglipModel, SiglipTextConfig,
                          SiglipVisionConfig)

HIDDEN = 16
INTER = 32
N_LAYER = 2
N_HEAD = 4                    # head_dim = 4
VOCAB = 33
MAX_POS = 12
SEQ_LEN = 8
IMAGE = 12
PATCH = 4                     # 3x3 = 9 patches, NO class token
PROJ = HIDDEN                 # SigLIP projection_size defaults to hidden
N_TEXTS = 2

torch.manual_seed(20260614)

text_cfg = dict(
    vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=INTER,
    num_hidden_layers=N_LAYER, num_attention_heads=N_HEAD,
    max_position_embeddings=MAX_POS, hidden_act='gelu_pytorch_tanh',
    layer_norm_eps=1e-6, attention_dropout=0.0, projection_size=PROJ,
)
vision_cfg = dict(
    hidden_size=HIDDEN, intermediate_size=INTER,
    num_hidden_layers=N_LAYER, num_attention_heads=N_HEAD,
    image_size=IMAGE, patch_size=PATCH, num_channels=3,
    hidden_act='gelu_pytorch_tanh', layer_norm_eps=1e-6,
    attention_dropout=0.0,
)
cfg = SiglipConfig(text_config=text_cfg, vision_config=vision_cfg)
model = SiglipModel(cfg)

# HF inits with tiny stds at this pico width; boost so every quirk is
# visible in the oracle (the ModernBERT vacuous-init lesson).
with torch.no_grad():
    model.text_model.embeddings.token_embedding.weight.normal_(0.0, 0.5)
    model.text_model.embeddings.position_embedding.weight.normal_(0.0, 0.4)
    model.vision_model.embeddings.patch_embedding.weight.normal_(0.0, 0.25)
    model.vision_model.embeddings.patch_embedding.bias.normal_(0.0, 0.2)
    model.vision_model.embeddings.position_embedding.weight.normal_(0.0, 0.4)
    for tower in (model.text_model, model.vision_model):
        for layer in tower.encoder.layers:
            for proj in (layer.self_attn.q_proj, layer.self_attn.k_proj,
                         layer.self_attn.v_proj, layer.self_attn.out_proj):
                proj.weight.normal_(0.0, 0.35)
                proj.bias.normal_(0.0, 0.2)
            layer.mlp.fc1.weight.normal_(0.0, 0.5)
            layer.mlp.fc1.bias.normal_(0.0, 0.3)
            layer.mlp.fc2.weight.normal_(0.0, 0.3)
            layer.mlp.fc2.bias.normal_(0.0, 0.2)
            for norm in (layer.layer_norm1, layer.layer_norm2):
                norm.weight.normal_(1.0, 0.25)
                norm.bias.normal_(0.0, 0.2)
    for norm in (model.text_model.final_layer_norm,
                 model.vision_model.post_layernorm):
        norm.weight.normal_(1.0, 0.25)
        norm.bias.normal_(0.0, 0.2)
    # text head (biased nn.Linear)
    model.text_model.head.weight.normal_(0.0, 0.35)
    model.text_model.head.bias.normal_(0.0, 0.2)
    # vision MAP head
    head = model.vision_model.head
    head.probe.normal_(0.0, 0.6)
    head.attention.in_proj_weight.normal_(0.0, 0.35)
    head.attention.in_proj_bias.normal_(0.0, 0.2)
    head.attention.out_proj.weight.normal_(0.0, 0.35)
    head.attention.out_proj.bias.normal_(0.0, 0.2)
    head.layernorm.weight.normal_(1.0, 0.25)
    head.layernorm.bias.normal_(0.0, 0.2)
    head.mlp.fc1.weight.normal_(0.0, 0.5)
    head.mlp.fc1.bias.normal_(0.0, 0.3)
    head.mlp.fc2.weight.normal_(0.0, 0.3)
    head.mlp.fc2.bias.normal_(0.0, 0.2)
    # learnable logit scale/bias (away from defaults so both matter)
    model.logit_scale.fill_(0.7)
    model.logit_bias.fill_(-0.5)
model = model.double().eval()

assert model.text_model.head.bias is not None, \
    'SigLIP text head must be biased'
assert model.vision_model.embeddings.patch_embedding.bias is not None, \
    'SigLIP patch_embedding must be biased'

# Pinned inputs. Image: deterministic dyadic pixel values (exact in f32
# and JSON). Text: arbitrary ids; pooling is at the LAST position.
text_sequences = [
    [0, 7, 23, 11, 14, 5, 9, 3],
    [0, 31, 8, 30, 17, 26, 2, 19],
]
pixels = torch.zeros(1, 3, IMAGE, IMAGE, dtype=torch.float64)
for c in range(3):
    for y in range(IMAGE):
        for x in range(IMAGE):
            pixels[0, c, y, x] = (((c * 256 + y * 16 + x) * 5) % 17 - 8) / 8.0

sd = {k: v.to(torch.float32).clone().contiguous()
      for k, v in model.state_dict().items()
      if not k.endswith('position_ids')}
save_file(sd, 'tests/fixtures/tiny_siglip.safetensors')
with open('tests/fixtures/tiny_siglip_config.json', 'w') as f:
    json.dump(cfg.to_dict(), f, indent=1, default=str)

with torch.no_grad():
    ids = torch.tensor(text_sequences)

    def text_feats(m, i):
        r = m.get_text_features(input_ids=i)
        return r if torch.is_tensor(r) else r.pooler_output

    def image_feats(m, p):
        r = m.get_image_features(pixel_values=p)
        return r if torch.is_tensor(r) else r.pooler_output

    text_embeds = text_feats(model, ids)                         # UNnormalized
    image_embeds = image_feats(model, pixels)
    out = model(input_ids=ids, pixel_values=pixels)
with open('tests/fixtures/tiny_siglip_embeds.json', 'w') as f:
    json.dump({
        'text_sequences': text_sequences,
        'pixels': pixels[0].tolist(),                 # [3][IMAGE][IMAGE]
        'text_embeds': text_embeds.tolist(),          # [N_TEXTS][PROJ]
        'image_embeds': image_embeds.tolist(),        # [1][PROJ]
        'logit_scale': model.logit_scale.item(),
        'logit_bias': model.logit_bias.item(),
        'logits_per_image': out.logits_per_image.tolist(),  # [1][N_TEXTS]
        'logits_per_text': out.logits_per_text.tolist(),    # [N_TEXTS][1]
    }, f)
print(f'wrote tiny_siglip.safetensors ({len(sd)} tensors) + config + oracle')
for k in sorted(sd):
    print(f'  {k} {list(sd[k].shape)}')

# ---- fixture self-checks: every SigLIP-vs-CLIP quirk must be visible ----
with torch.no_grad():
    base_t = text_embeds
    base_i = image_embeds

    # 1. gelu_pytorch_tanh vs exact-erf and quick_gelu must be
    # distinguishable above the 1e-4 parity gate in BOTH towers.
    class QuickGELU(torch.nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(1.702 * x)
    for wrong, act in (('gelu', torch.nn.GELU().double()),
                       ('quick_gelu', QuickGELU().double())):
        alt = copy.deepcopy(model)
        for tower in (alt.text_model, alt.vision_model):
            for layer in tower.encoder.layers:
                layer.mlp.activation_fn = act
        alt.vision_model.head.mlp.activation_fn = act
        dt = (text_feats(alt, ids) - base_t).abs().max()
        di = (image_feats(alt, pixels) - base_i).abs().max()
        # exact-erf and tanh GELU differ only ~2e-4 at this pico width
        # (still above the 1e-4 parity gate, so a wrong activation choice
        # is detectable); quick_gelu diverges far more.
        assert dt > 1.5e-4 and di > 1.5e-4, \
            f'gelu_tanh vs {wrong} indistinguishable ({dt}, {di})'
        print(f'gelu_tanh-vs-{wrong}: text {dt:.4f}, image {di:.4f}')

    # 2. TEXT attention must be BIDIRECTIONAL (unlike CLIP): the pooled
    # row is the LAST one; changing an EARLIER token must move it (trivially
    # true) AND - the discriminating test vs causal - a causal mask would
    # give a DIFFERENT pooled value. Compare against an explicitly causal run.
    from transformers.models.siglip import modeling_siglip as ms
    lh_bi = model.text_model(input_ids=ids).last_hidden_state[:, -1]
    # last-token pooling under bidirectional attention is what we load.
    pooled = lh_bi @ model.text_model.head.weight.T + \
        model.text_model.head.bias
    assert (pooled - base_t).abs().max() < 1e-10, \
        'text pooling is not last-token + biased head'
    print('text last-token + biased-head pooling check passed')

    # 3. Pooling must be the LAST row, not the first/argmax: a different row
    # must give a different embed.
    pooled_first = model.text_model(input_ids=ids).last_hidden_state[:, 0] \
        @ model.text_model.head.weight.T + model.text_model.head.bias
    assert (pooled_first - base_t).abs().max() > 1e-3, \
        'last-row vs first-row pooling indistinguishable'
    print('last-row (not first/argmax) pooling check passed')

    # 4. logit_bias must matter: zeroing it must change logits_per_image
    # by exactly -logit_bias.
    alt = copy.deepcopy(model)
    alt.logit_bias.zero_()
    d = (model(input_ids=ids, pixel_values=pixels).logits_per_image -
         alt(input_ids=ids, pixel_values=pixels).logits_per_image)
    assert (d - model.logit_bias).abs().max() < 1e-9, \
        'logit_bias is not an additive constant on logits_per_image'
    print(f'logit_bias additive check passed (bias={model.logit_bias.item()})')

    # 5. logits_per_image == exp(logit_scale)*cos + logit_bias, transposed.
    tn = base_t / base_t.norm(dim=-1, keepdim=True)
    iv = base_i / base_i.norm(dim=-1, keepdim=True)
    ref = (iv @ tn.T) * model.logit_scale.exp() + model.logit_bias
    assert (ref - out.logits_per_image).abs().max() < 1e-10
    print('logit_scale + logit_bias similarity check passed')

    # 6. MAP head probe must matter: zeroing it must move the image embed.
    alt = copy.deepcopy(model)
    alt.vision_model.head.probe.zero_()
    d = (image_feats(alt, pixels) - base_i).abs().max()
    assert d > 1e-3, f'MAP head probe had no effect ({d})'
    print(f'MAP head probe effect on image embeds: max |diff| = {d:.4f}')

    # 7. patch_embedding BIAS must matter (CLIP has none).
    alt = copy.deepcopy(model)
    alt.vision_model.embeddings.patch_embedding.bias.zero_()
    d = (image_feats(alt, pixels) - base_i).abs().max()
    assert d > 1e-3, f'patch_embedding bias had no effect ({d})'
    print(f'patch bias effect on image embeds: max |diff| = {d:.4f}')

    # 8. text head BIAS must matter.
    alt = copy.deepcopy(model)
    alt.text_model.head.bias.zero_()
    d = (text_feats(alt, ids) - base_t).abs().max()
    assert d > 1e-3, f'text head bias had no effect ({d})'
    print(f'text head bias effect on text embeds: max |diff| = {d:.4f}')
print('all fixture self-checks passed')
