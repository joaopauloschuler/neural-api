#!/usr/bin/env python3
"""Generate a tiny RANDOM CLIP parity fixture for
tests/TestNeuralPretrained.pas (no network access needed: the model is
randomly initialized from a pico config, never downloaded).

One fixture, KB-scale, pinned in tests/fixtures/:

  tiny_clip.*: CLIPModel (the openai/clip-vit-base-patch32 architecture,
      model_type "clip") with every CLIP trait the importer must reproduce:
        - TEXT tower: token embedding + LEARNED absolute positions, CAUSAL
          pre-LN encoder blocks (layer_norm1 -> biased q/k/v/out self-attn
          with 1/sqrt(head_dim) scaling -> residual; layer_norm2 ->
          fc1/quick_gelu/fc2 -> residual), final_layer_norm, EOS pooling
          (eos_token_id = 2, the legacy pre-#24773 config every published
          OpenAI CLIP carries: pooled = hidden state at ARGMAX(input_ids),
          i.e. at the highest token id = the eot token), then the bias-free
          text_projection;
        - VISION tower: bias-FREE patch_embedding Conv2d(stride =
          kernel = patch_size), learned class_embedding prepended as token
          0, learned absolute positions over [cls]+patches, pre_layrnorm
          (the historical HF spelling), BIDIRECTIONAL pre-LN blocks (same
          shape as text), post_layernorm on the CLASS token, then the
          bias-free visual_projection;
        - quick_gelu activation x*sigmoid(1.702x) in BOTH towers (HF
          QuickGELUActivation - NOT the exact erf gelu, NOT the tanh
          approximation);
        - logit_scale: logits_per_image = exp(logit_scale) *
          image_embeds_n @ text_embeds_n^T after L2 normalization.

The reference is computed by HF transformers in float64 (the oracle
convention of the committed fixtures): text_embeds / image_embeds are the
UNNORMALIZED pooled+projected features (HF get_text_features /
get_image_features), plus logits_per_image for the pinned pair.

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/clip_tiny_fixture.py
writes tests/fixtures/tiny_clip{.safetensors,_config.json,_embeds.json}.
Needs torch + transformers + safetensors.
"""
import copy
import json

import torch
from safetensors.torch import save_file
from transformers import CLIPConfig, CLIPModel, CLIPTextConfig, CLIPVisionConfig

HIDDEN = 16
INTER = 32
N_LAYER = 2
N_HEAD = 4                    # head_dim = 4
VOCAB = 33                    # eot = 32, the HIGHEST id (argmax pooling)
EOT = VOCAB - 1
MAX_POS = 12
SEQ_LEN = 8
IMAGE = 16
PATCH = 8                     # 2x2 = 4 patches + [CLS] = 5 positions
PROJ = 8
N_TEXTS = 2

torch.manual_seed(20260612)

text_cfg = dict(
    vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=INTER,
    num_hidden_layers=N_LAYER, num_attention_heads=N_HEAD,
    max_position_embeddings=MAX_POS, hidden_act='quick_gelu',
    layer_norm_eps=1e-5, attention_dropout=0.0,
    bos_token_id=0, eos_token_id=2,  # legacy id: forces the ARGMAX pooling
    pad_token_id=1,
)
vision_cfg = dict(
    hidden_size=HIDDEN, intermediate_size=INTER,
    num_hidden_layers=N_LAYER, num_attention_heads=N_HEAD,
    image_size=IMAGE, patch_size=PATCH, num_channels=3,
    hidden_act='quick_gelu', layer_norm_eps=1e-5, attention_dropout=0.0,
)
cfg = CLIPConfig(text_config=text_cfg, vision_config=vision_cfg,
                 projection_dim=PROJ, logit_scale_init_value=2.6592)
model = CLIPModel(cfg)

# HF inits with tiny stds at this pico width; boost so every quirk is
# visible in the oracle: O(1) attention scores, FFN pre-activations in the
# region where quick_gelu and the erf/tanh gelus genuinely differ,
# layer-norm gains and biases away from (1, 0), a class embedding and
# position tables that move the embeds (the ModernBERT vacuous-init lesson).
with torch.no_grad():
    model.text_model.embeddings.token_embedding.weight.normal_(0.0, 0.5)
    model.text_model.embeddings.position_embedding.weight.normal_(0.0, 0.4)
    model.vision_model.embeddings.class_embedding.normal_(0.0, 0.8)
    model.vision_model.embeddings.patch_embedding.weight.normal_(0.0, 0.25)
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
                 model.vision_model.pre_layrnorm,
                 model.vision_model.post_layernorm):
        norm.weight.normal_(1.0, 0.25)
        norm.bias.normal_(0.0, 0.2)
    model.text_projection.weight.normal_(0.0, 0.35)
    model.visual_projection.weight.normal_(0.0, 0.35)
model = model.double().eval()

assert model.vision_model.embeddings.patch_embedding.bias is None, \
    'CLIP patch_embedding must be bias-free'
assert model.text_projection.bias is None and \
    model.visual_projection.bias is None, 'CLIP projections must be bias-free'

# Pinned inputs. Text: eot (the HIGHEST id) at DIFFERENT positions, with
# real lower-id tokens after it in sequence 0 - pooling at the wrong
# position (e.g. the last token) FAILS parity. Image: deterministic dyadic
# pixel values (exact in f32 and JSON).
text_sequences = [
    [0, 7, 23, 11, EOT, 5, 9, 3],
    [0, 31, 8, 30, 17, 26, 2, EOT],
]
pixels = torch.zeros(1, 3, IMAGE, IMAGE, dtype=torch.float64)
for c in range(3):
    for y in range(IMAGE):
        for x in range(IMAGE):
            pixels[0, c, y, x] = (((c * 256 + y * 16 + x) * 5) % 17 - 8) / 8.0

sd = {k: v.to(torch.float32).clone().contiguous()
      for k, v in model.state_dict().items()
      if not k.endswith('position_ids')}
save_file(sd, 'tests/fixtures/tiny_clip.safetensors')
with open('tests/fixtures/tiny_clip_config.json', 'w') as f:
    json.dump(cfg.to_dict(), f, indent=1, default=str)

with torch.no_grad():
    ids = torch.tensor(text_sequences)
    # newer transformers returns the full output object; the projected
    # UNnormalized features sit in pooler_output either way
    def text_feats(m, i):
        r = m.get_text_features(input_ids=i)
        return r if torch.is_tensor(r) else r.pooler_output

    def image_feats(m, p):
        r = m.get_image_features(pixel_values=p)
        return r if torch.is_tensor(r) else r.pooler_output

    text_embeds = text_feats(model, ids)                         # UNnormalized
    image_embeds = image_feats(model, pixels)
    out = model(input_ids=ids, pixel_values=pixels)
with open('tests/fixtures/tiny_clip_embeds.json', 'w') as f:
    json.dump({
        'text_sequences': text_sequences,
        'pixels': pixels[0].tolist(),                 # [3][IMAGE][IMAGE]
        'text_embeds': text_embeds.tolist(),          # [N_TEXTS][PROJ]
        'image_embeds': image_embeds.tolist(),        # [1][PROJ]
        'logit_scale': model.logit_scale.item(),
        'logits_per_image': out.logits_per_image.tolist(),  # [1][N_TEXTS]
    }, f)
print(f'wrote tiny_clip.safetensors ({len(sd)} tensors) + config + oracle')
for k in sorted(sd):
    print(f'  {k} {list(sd[k].shape)}')

# ---- fixture self-checks: every quirk must be visible in the oracle ----
with torch.no_grad():
    base_t = text_embeds
    base_i = image_embeds

    # 1. quick_gelu vs the exact-erf and tanh gelus must be distinguishable
    # above the 1e-4 parity gate in BOTH towers.
    for wrong in ('gelu', 'gelu_tanh'):
        alt = copy.deepcopy(model)
        act = (torch.nn.GELU().double() if wrong == 'gelu'
               else torch.nn.GELU(approximate='tanh').double())
        for tower in (alt.text_model, alt.vision_model):
            for layer in tower.encoder.layers:
                layer.mlp.activation_fn = act
        dt = (text_feats(alt, ids) - base_t).abs().max()
        di = (image_feats(alt, pixels) -
              base_i).abs().max()
        assert dt > 1e-3 and di > 1e-3, \
            f'quick_gelu vs {wrong} indistinguishable ({dt}, {di})'
        print(f'quick_gelu-vs-{wrong}: text {dt:.4f}, image {di:.4f}')

    # 2. TEXT attention must be CAUSAL: changing a token AFTER the eot of
    # sequence 0 must NOT move its pooled embedding.
    ids2 = ids.clone()
    ids2[0, -1] = (ids2[0, -1] + 5) % (VOCAB - 2)
    d = (text_feats(model, ids2)[0] -
         base_t[0]).abs().max()
    assert d < 1e-12, f'text attention is not causal at the eot row ({d})'
    print('text causality (post-eot token has no effect) check passed')
    # ... while a token BEFORE the eot must move it.
    ids3 = ids.clone()
    ids3[0, 1] = (ids3[0, 1] + 5) % (VOCAB - 2)
    d = (text_feats(model, ids3)[0] -
         base_t[0]).abs().max()
    assert d > 1e-3, f'pre-eot token had no effect ({d})'

    # 3. The pooling position must be the ARGMAX row (eos_token_id == 2
    # legacy path), not the last row: sequence 0 carries its eot mid-way.
    lh = model.text_model(input_ids=ids).last_hidden_state
    pooled_argmax = lh[0, 4] @ model.text_projection.weight.T
    assert (pooled_argmax - base_t[0]).abs().max() < 1e-12, \
        'pooling is not at the argmax position'
    pooled_last = lh[0, -1] @ model.text_projection.weight.T
    assert (pooled_last - base_t[0]).abs().max() > 1e-3, \
        'argmax vs last-row pooling indistinguishable'
    print('argmax (eot-position) pooling check passed')

    # 4. VISION attention must be BIDIRECTIONAL: the class token (position
    # 0, the pooled one) must see the patches - zeroing the LAST patch's
    # pixels must move the image embedding.
    px2 = pixels.clone()
    px2[0, :, 8:, 8:] = 0.0
    d = (image_feats(model, px2) - base_i).abs().max()
    assert d > 1e-3, f'patch content had no effect on the CLS pooling ({d})'
    print(f'bidirectional CLS<-patch effect: max |diff| = {d:.4f}')

    # 5. class_embedding, the position tables and pre_layrnorm must matter.
    for name, mutate in (
        ('class_embedding',
         lambda m: m.vision_model.embeddings.class_embedding.zero_()),
        ('vision positions',
         lambda m: m.vision_model.embeddings.position_embedding
         .weight.zero_()),
        # NOT a constant shift (a pre-LN stack is invariant to one: every
        # LayerNorm re-centers and the residual carries it to the final
        # norm) - a PER-CHANNEL shift survives.
        ('pre_layrnorm',
         lambda m: m.vision_model.pre_layrnorm.bias.add_(
             torch.linspace(-1.0, 1.0, HIDDEN, dtype=torch.float64))),
    ):
        alt = copy.deepcopy(model)
        mutate(alt)
        d = (image_feats(alt, pixels) -
             base_i).abs().max()
        assert d > 1e-3, f'{name} had no effect ({d})'
        print(f'{name} effect on image embeds: max |diff| = {d:.4f}')
    alt = copy.deepcopy(model)
    alt.text_model.embeddings.position_embedding.weight.zero_()
    d = (text_feats(alt, ids) - base_t).abs().max()
    assert d > 1e-3, f'text positions had no effect ({d})'
    print(f'text position effect on text embeds: max |diff| = {d:.4f}')

    # 6. logits_per_image must equal exp(logit_scale) * normalized cosine.
    tn = base_t / base_t.norm(dim=-1, keepdim=True)
    iv = base_i / base_i.norm(dim=-1, keepdim=True)
    ref = model.logit_scale.exp() * iv @ tn.T
    assert (ref - out.logits_per_image).abs().max() < 1e-10
    print('logit_scale similarity check passed')
print('all fixture self-checks passed')
