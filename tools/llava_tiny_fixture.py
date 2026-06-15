#!/usr/bin/env python3
"""Generate a tiny RANDOM LLaVA parity fixture for
tests/TestNeuralPretrained.pas (no network access: the model is randomly
initialized from a pico config, never downloaded).

LLaVA = the classic generative vision-language recipe (the
llava-interleave-qwen-0.5b-hf architecture, scaled down): a SigLIP ViT
vision tower -> a 2-layer MLP projector (gelu) -> the projected visual
tokens SPLICED into the language decoder's token-embedding sequence at the
<image> placeholder positions, then ordinary CAUSAL decoding through a
Qwen2 LM. This is the FIRST image-in / text-out importer in the repo.

The pico LLaVA here pairs:
  - VISION: a SigLIP vision tower (model_type "siglip_vision_model",
    vision_use_head=False so there is NO MAP pooling head), image 12,
    patch 4 -> 9 patch tokens, NO class token, learned positions,
    post_layernorm. The LLaVA "vision feature" is hidden_states[-1] =
    the post_layernorm last_hidden_state (vision_feature_layer=-1) with
    select_strategy="full" (keep ALL 9 patch rows - SigLIP has no CLS to
    crop). This is exactly BuildSigLIPVisionTower(pVisionFeatures=True,
    SelectHiddenLayer=0).
  - PROJECTOR: multi_modal_projector.linear_1 (vision_hidden -> text_hidden)
    -> gelu -> linear_2 (text_hidden -> text_hidden), both biased.
  - TEXT: a Qwen2 decoder (biased q/k/v, GQA, RoPE, SwiGLU, RMSNorm),
    the stock BuildLlamaFromSafeTensors path.

The committed fixture (KB-scale) in tests/fixtures/:
  tiny_llava.safetensors  - the full LLaVA state_dict (HF transformers
      5.x "model."-prefixed keys: model.vision_tower.*,
      model.multi_modal_projector.*, model.language_model.*, lm_head.*).
  tiny_llava_config.json  - the LlavaConfig.to_dict().
  tiny_llava_logits.json  - the float64 oracle: the pinned pixel tensor,
      the pinned token id sequence (with image_token_index at the image
      slots), the projected visual tokens [9][text_hidden], and the
      next-token logits [seq][vocab] for the mixed image+text prompt.

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/llava_tiny_fixture.py
Needs torch + transformers + safetensors.
"""
import copy
import json

import torch
from safetensors.torch import save_file
from transformers import LlavaConfig, LlavaForConditionalGeneration

HID_T = 32          # text hidden_size
INTER_T = 64        # text intermediate_size (SwiGLU width)
N_LAYER_T = 2
N_HEAD_T = 4        # text head_dim = 8
N_KV_T = 2          # GQA
VOCAB = 50
MAX_POS = 64

HID_V = 24          # vision hidden_size
INTER_V = 48
N_LAYER_V = 2
N_HEAD_V = 4        # vision head_dim = 6
IMAGE = 12
PATCH = 4           # 3x3 = 9 patch tokens, NO class token
NPATCH = (IMAGE // PATCH) ** 2

IMAGE_TOKEN = 49    # image_token_index (last real id, < VOCAB)

torch.manual_seed(20260615)

cfg = LlavaConfig(
    text_config=dict(
        model_type="qwen2", hidden_size=HID_T, intermediate_size=INTER_T,
        num_hidden_layers=N_LAYER_T, num_attention_heads=N_HEAD_T,
        num_key_value_heads=N_KV_T, vocab_size=VOCAB,
        max_position_embeddings=MAX_POS, rms_norm_eps=1e-6,
        rope_theta=10000.0, tie_word_embeddings=False),
    vision_config=dict(
        model_type="siglip_vision_model", hidden_size=HID_V,
        intermediate_size=INTER_V, num_hidden_layers=N_LAYER_V,
        num_attention_heads=N_HEAD_V, image_size=IMAGE, patch_size=PATCH,
        num_channels=3, hidden_act="gelu_pytorch_tanh", layer_norm_eps=1e-6,
        vision_use_head=False),
    vision_feature_layer=-1, vision_feature_select_strategy="full",
    image_token_index=IMAGE_TOKEN, projector_hidden_act="gelu")

model = LlavaForConditionalGeneration(cfg)

# HF inits with tiny stds at this pico width; boost so every component is
# visible above the 1e-4 parity gate (the ModernBERT vacuous-init lesson).
vt = model.model.vision_tower
lm = model.model.language_model
with torch.no_grad():
    vt.embeddings.patch_embedding.weight.normal_(0.0, 0.25)
    vt.embeddings.patch_embedding.bias.normal_(0.0, 0.2)
    vt.embeddings.position_embedding.weight.normal_(0.0, 0.4)
    for layer in vt.encoder.layers:
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
    vt.post_layernorm.weight.normal_(1.0, 0.25)
    vt.post_layernorm.bias.normal_(0.0, 0.2)

    # projector (both biased)
    proj = model.model.multi_modal_projector
    proj.linear_1.weight.normal_(0.0, 0.3)
    proj.linear_1.bias.normal_(0.0, 0.2)
    proj.linear_2.weight.normal_(0.0, 0.3)
    proj.linear_2.bias.normal_(0.0, 0.2)

    # Qwen2 LM
    lm.embed_tokens.weight.normal_(0.0, 0.4)
    for layer in lm.layers:
        sa = layer.self_attn
        for p in (sa.q_proj, sa.k_proj, sa.v_proj):
            p.weight.normal_(0.0, 0.3)
            p.bias.normal_(0.0, 0.2)          # Qwen2 biased q/k/v
        sa.o_proj.weight.normal_(0.0, 0.3)
        layer.mlp.gate_proj.weight.normal_(0.0, 0.3)
        layer.mlp.up_proj.weight.normal_(0.0, 0.3)
        layer.mlp.down_proj.weight.normal_(0.0, 0.3)
        layer.input_layernorm.weight.normal_(1.0, 0.2)
        layer.post_attention_layernorm.weight.normal_(1.0, 0.2)
    lm.norm.weight.normal_(1.0, 0.2)
    model.lm_head.weight.normal_(0.0, 0.3)

model = model.double().eval()

assert model.model.multi_modal_projector.linear_1.bias is not None
assert vt.embeddings.patch_embedding.bias is not None

# ---- pinned inputs ----
# Mixed prompt: [t t <image>*9 t t]  (text, image block, text).
text_pre = [1, 7]
text_post = [12, 3]
ids_list = text_pre + [IMAGE_TOKEN] * NPATCH + text_post
ids = torch.tensor([ids_list])
SEQ = len(ids_list)

# Deterministic dyadic pixel values (exact in f32 + JSON), channel-first.
pix = torch.zeros(1, 3, IMAGE, IMAGE, dtype=torch.float64)
for c in range(3):
    for y in range(IMAGE):
        for x in range(IMAGE):
            pix[0, c, y, x] = (((c * 256 + y * 16 + x) * 5) % 17 - 8) / 8.0

with torch.no_grad():
    # projected visual tokens (Step 3 oracle)
    img_out = model.model.get_image_features(
        pixel_values=pix, vision_feature_layer=-1,
        vision_feature_select_strategy="full")
    visual_tokens = img_out.pooler_output[0]      # [1][9][HID_T] -> take [0]
    if visual_tokens.dim() == 3:
        visual_tokens = visual_tokens[0]
    assert visual_tokens.shape == (NPATCH, HID_T), visual_tokens.shape

    # full forward -> next-token logits (Step 4 oracle)
    out = model(input_ids=ids, pixel_values=pix)
    logits = out.logits[0]                         # [SEQ][VOCAB]
    assert logits.shape == (SEQ, VOCAB), logits.shape

# ---- save weights + config ----
sd = {k: v.to(torch.float32).clone().contiguous()
      for k, v in model.state_dict().items()
      if not k.endswith('position_ids')}
save_file(sd, 'tests/fixtures/tiny_llava.safetensors')
cfg_dict = cfg.to_dict()
# Newer transformers nest rope params under text_config.rope_parameters; the
# Pascal reader reads a flat text_config.rope_theta. Pin the effective value
# explicitly so the importer is unambiguous regardless of transformers version.
rp = cfg_dict['text_config'].get('rope_parameters') or \
    cfg_dict['text_config'].get('rope_scaling') or {}
cfg_dict['text_config']['rope_theta'] = rp.get('rope_theta', 10000.0)
with open('tests/fixtures/tiny_llava_config.json', 'w') as f:
    json.dump(cfg_dict, f, indent=1, default=str)

with open('tests/fixtures/tiny_llava_logits.json', 'w') as f:
    json.dump({
        'image_token_index': IMAGE_TOKEN,
        'num_patches': NPATCH,
        'token_ids': ids_list,
        'pixels': pix[0].tolist(),                 # [3][IMAGE][IMAGE]
        'visual_tokens': visual_tokens.tolist(),   # [9][HID_T]
        'logits': logits.tolist(),                 # [SEQ][VOCAB]
    }, f)
print(f'wrote tiny_llava.safetensors ({len(sd)} tensors) + config + oracle')
print(f'  seq_len={SEQ} num_patches={NPATCH} vocab={VOCAB}')

# ---- fixture self-checks: every LLaVA-distinguishing piece must matter ----
with torch.no_grad():
    base_vt = visual_tokens.clone()
    base_logits = logits.clone()

    # 1. projector linear_2 bias must matter (it is the visible projector tail).
    alt = copy.deepcopy(model)
    alt.model.multi_modal_projector.linear_2.bias.zero_()
    f = alt.model.get_image_features(pixel_values=pix, vision_feature_layer=-1,
                                     vision_feature_select_strategy="full")
    fv = f.pooler_output[0]
    if fv.dim() == 3:
        fv = fv[0]
    d = (fv - base_vt).abs().max()
    assert d > 1e-3, f'projector linear_2 bias had no effect ({d})'
    print(f'projector bias effect on visual tokens: max |diff| = {d:.4f}')

    # 2. projector gelu (not identity) must matter.
    alt = copy.deepcopy(model)
    alt.model.multi_modal_projector.act = torch.nn.Identity()
    f = alt.model.get_image_features(pixel_values=pix, vision_feature_layer=-1,
                                     vision_feature_select_strategy="full")
    fv = f.pooler_output[0]
    if fv.dim() == 3:
        fv = fv[0]
    d = (fv - base_vt).abs().max()
    assert d > 1e-3, f'projector gelu vs identity indistinguishable ({d})'
    print(f'projector gelu effect on visual tokens: max |diff| = {d:.4f}')

    # 3. the visual tokens must actually flow into the logits: zeroing the
    # projector output must move the next-token logits at/after the image
    # block (the splice is real, not a no-op).
    alt = copy.deepcopy(model)
    alt.model.multi_modal_projector.linear_2.weight.zero_()
    alt.model.multi_modal_projector.linear_2.bias.zero_()
    lo = alt(input_ids=ids, pixel_values=pix).logits[0]
    d = (lo - base_logits).abs().max()
    assert d > 1e-3, f'visual tokens do not affect the logits ({d})'
    print(f'visual-token effect on logits: max |diff| = {d:.4f}')

    # 4. CRITICAL: vision_feature_layer=-1 selects encoder hidden_states[-1],
    # which is captured BEFORE post_layernorm. Zeroing post_layernorm gain/bias
    # must therefore have NO effect on the visual tokens - the importer must
    # likewise SKIP post_layernorm (BuildSigLIPVisionTower SelectHiddenLayer =
    # num_layers, which returns the raw encoder output). Assert the no-op.
    alt = copy.deepcopy(model)
    alt.model.vision_tower.post_layernorm.weight.normal_(3.0, 1.0)
    alt.model.vision_tower.post_layernorm.bias.normal_(2.0, 1.0)
    f = alt.model.get_image_features(pixel_values=pix, vision_feature_layer=-1,
                                     vision_feature_select_strategy="full")
    fv = f.pooler_output[0]
    if fv.dim() == 3:
        fv = fv[0]
    d = (fv - base_vt).abs().max()
    assert d < 1e-12, (f'post_layernorm UNEXPECTEDLY affected the -1 feature '
                       f'({d}); the importer must skip post_layernorm')
    print('vision feature is PRE-post_layernorm (encoder hidden_states[-1]): '
          'confirmed no-op')
print('all fixture self-checks passed')
