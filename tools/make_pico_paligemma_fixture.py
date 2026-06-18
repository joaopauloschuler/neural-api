#!/usr/bin/env python3
"""Generate a tiny RANDOM PaliGemma parity fixture for
tests/TestNeuralPretrained.pas (no network access: the model is randomly
initialized from a pico config, never downloaded).

PaliGemma = the SigLIP-ViT vision tower -> a SINGLE linear multimodal
projector -> the projected visual tokens SPLICED into a GEMMA decoder's
token-embedding sequence at the <image> placeholder positions. The genuinely
NEW behaviour versus LLaVA is the PREFIX-LM attention mask: the image tokens
AND the prompt tokens (token_type_id 0, the "prefix") see each other with FULL
BIDIRECTIONAL attention, while ONLY the generated suffix (token_type_id 1) is
causal. This is google/paligemma-3b-mix-224 scaled down.

The pico PaliGemma here pairs:
  - VISION: a SigLIP vision tower (model_type "siglip_vision_model",
    image 12, patch 4 -> 9 patch tokens, NO class token, learned positions,
    post_layernorm APPLIED). PaliGemma's vision feature is the SigLIP
    last_hidden_state WITH post_layernorm (BuildSigLIPVisionTower
    pVisionFeatures=True, SelectHiddenLayer=0).
  - PROJECTOR: a SINGLE biased linear multi_modal_projector.linear
    (vision_hidden -> text_hidden = vision_config.projection_dim). NO gelu,
    NO second layer (unlike LLaVA's 2-layer MLP).
  - TEXT: a Gemma decoder (head_dim decoupled from hidden/heads, RoPE,
    RMSNorm with +1 gain, sqrt(hidden) embedding scale, GeGLU-tanh MLP),
    the stock BuildGemmaFromSafeTensors path.

The PREFIX-LM mask is exercised by a prompt whose last token is a generated
suffix token (token_type_id 1): if the importer used a causal-everywhere mask
(as LLaVA does) the prefix positions would NOT see the future prefix tokens
and the next-token logits would differ -> the parity test would FAIL. The
fixture self-checks below prove the bidirectional block is load-bearing.

The committed fixture (KB-scale) in tests/fixtures/:
  tiny_paligemma.safetensors  - the full PaliGemma state_dict (HF
      transformers 5.x "model."-prefixed keys: model.vision_tower.*,
      model.multi_modal_projector.*, model.language_model.*, lm_head.*).
  tiny_paligemma_config.json  - the PaliGemmaConfig.to_dict().
  tiny_paligemma_logits.json  - the float64 oracle: the pinned pixel tensor,
      the pinned token id sequence (image_token_index at the image slots),
      the token_type_ids (0 prefix / 1 suffix), the prefix length, the
      projected visual tokens [9][text_hidden], and the next-token logits
      [seq][vocab] for the mixed image+text PREFIX-LM prompt.

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/make_pico_paligemma_fixture.py
Needs torch + transformers + safetensors.
"""
import copy
import json

import torch
from safetensors.torch import save_file
from transformers import PaliGemmaConfig, PaliGemmaForConditionalGeneration

HID_T = 32          # text hidden_size
INTER_T = 64        # text intermediate_size (GeGLU width)
N_LAYER_T = 2
N_HEAD_T = 4
N_KV_T = 1          # GQA / MQA
HEAD_DIM_T = 8      # Gemma decouples head_dim from hidden/heads
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

cfg = PaliGemmaConfig(
    text_config=dict(
        model_type="gemma", hidden_size=HID_T, intermediate_size=INTER_T,
        num_hidden_layers=N_LAYER_T, num_attention_heads=N_HEAD_T,
        num_key_value_heads=N_KV_T, head_dim=HEAD_DIM_T, vocab_size=VOCAB,
        max_position_embeddings=MAX_POS, rms_norm_eps=1e-6,
        rope_theta=10000.0, tie_word_embeddings=False),
    vision_config=dict(
        model_type="siglip_vision_model", hidden_size=HID_V,
        intermediate_size=INTER_V, num_hidden_layers=N_LAYER_V,
        num_attention_heads=N_HEAD_V, image_size=IMAGE, patch_size=PATCH,
        num_channels=3, hidden_act="gelu_pytorch_tanh", layer_norm_eps=1e-6,
        projection_dim=HID_T),
    projection_dim=HID_T,
    image_token_index=IMAGE_TOKEN)

model = PaliGemmaForConditionalGeneration(cfg)

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

    # projector (single biased linear)
    proj = model.model.multi_modal_projector
    proj.linear.weight.normal_(0.0, 0.3)
    proj.linear.bias.normal_(0.0, 0.2)

    # Gemma LM. Gemma RMSNorm stores (gain - 1); HF adds 1 at runtime, so a
    # normal(0,.2) weight here corresponds to a runtime gain ~ 1.
    lm.embed_tokens.weight.normal_(0.0, 0.4)
    for layer in lm.layers:
        sa = layer.self_attn
        for p in (sa.q_proj, sa.k_proj, sa.v_proj, sa.o_proj):
            p.weight.normal_(0.0, 0.3)
        layer.mlp.gate_proj.weight.normal_(0.0, 0.3)
        layer.mlp.up_proj.weight.normal_(0.0, 0.3)
        layer.mlp.down_proj.weight.normal_(0.0, 0.3)
        layer.input_layernorm.weight.normal_(0.0, 0.2)
        layer.post_attention_layernorm.weight.normal_(0.0, 0.2)
    lm.norm.weight.normal_(0.0, 0.2)
    model.lm_head.weight.normal_(0.0, 0.3)

model = model.double().eval()

assert model.model.multi_modal_projector.linear.bias is not None
assert vt.embeddings.patch_embedding.bias is not None

# ---- pinned inputs ----
# Mixed PREFIX-LM prompt: [<image>*9  t t  | s]  where the image block and the
# two prompt text tokens form the PREFIX (token_type_id 0, full bidirectional),
# and the trailing token is a generated SUFFIX (token_type_id 1, causal).
# PaliGemma's processor lays the image tokens FIRST, then the prompt, then the
# generated answer.
img_block = [IMAGE_TOKEN] * NPATCH
prompt_text = [7, 12]
suffix = [3]
ids_list = img_block + prompt_text + suffix
ids = torch.tensor([ids_list])
SEQ = len(ids_list)
PREFIX_LEN = len(img_block) + len(prompt_text)   # token_type_id 0 count
# token_type_ids: 0 for the prefix (image+prompt), 1 for the suffix.
token_type_ids = torch.tensor([[0] * PREFIX_LEN + [1] * len(suffix)])

# Deterministic dyadic pixel values (exact in f32 + JSON), channel-first.
pix = torch.zeros(1, 3, IMAGE, IMAGE, dtype=torch.float64)
for c in range(3):
    for y in range(IMAGE):
        for x in range(IMAGE):
            pix[0, c, y, x] = (((c * 256 + y * 16 + x) * 5) % 17 - 8) / 8.0

with torch.no_grad():
    # projected visual tokens (Step 3 oracle)
    img_out = model.model.get_image_features(pixel_values=pix)
    visual_tokens = img_out.pooler_output
    if visual_tokens.dim() == 3:
        visual_tokens = visual_tokens[0]
    assert visual_tokens.shape == (NPATCH, HID_T), visual_tokens.shape

    # full forward -> next-token logits (Step 4 oracle). token_type_ids drives
    # the PREFIX-LM block-bidirectional mask.
    out = model(input_ids=ids, pixel_values=pix, token_type_ids=token_type_ids)
    logits = out.logits[0]                         # [SEQ][VOCAB]
    assert logits.shape == (SEQ, VOCAB), logits.shape

# ---- save weights + config ----
sd = {k: v.to(torch.float32).clone().contiguous()
      for k, v in model.state_dict().items()
      if not k.endswith('position_ids')}
save_file(sd, 'tests/fixtures/tiny_paligemma.safetensors')
cfg_dict = cfg.to_dict()
rp = cfg_dict['text_config'].get('rope_parameters') or \
    cfg_dict['text_config'].get('rope_scaling') or {}
cfg_dict['text_config']['rope_theta'] = rp.get('rope_theta', 10000.0)
with open('tests/fixtures/tiny_paligemma_config.json', 'w') as f:
    json.dump(cfg_dict, f, indent=1, default=str)

with open('tests/fixtures/tiny_paligemma_logits.json', 'w') as f:
    json.dump({
        'image_token_index': IMAGE_TOKEN,
        'num_patches': NPATCH,
        'prefix_len': PREFIX_LEN,
        'token_ids': ids_list,
        'token_type_ids': token_type_ids[0].tolist(),
        'pixels': pix[0].tolist(),                 # [3][IMAGE][IMAGE]
        'visual_tokens': visual_tokens.tolist(),   # [9][HID_T]
        'logits': logits.tolist(),                 # [SEQ][VOCAB]
    }, f)
print(f'wrote tiny_paligemma.safetensors ({len(sd)} tensors) + config + oracle')
print(f'  seq_len={SEQ} num_patches={NPATCH} prefix_len={PREFIX_LEN} '
      f'vocab={VOCAB}')

# ---- fixture self-checks: every PaliGemma-distinguishing piece must matter ----
with torch.no_grad():
    base_vt = visual_tokens.clone()
    base_logits = logits.clone()

    # 1. projector bias must matter.
    alt = copy.deepcopy(model)
    alt.model.multi_modal_projector.linear.bias.zero_()
    fv = alt.model.get_image_features(pixel_values=pix).pooler_output
    if fv.dim() == 3:
        fv = fv[0]
    d = (fv - base_vt).abs().max()
    assert d > 1e-3, f'projector bias had no effect ({d})'
    print(f'projector bias effect on visual tokens: max |diff| = {d:.4f}')

    # 2. the visual tokens must actually flow into the logits.
    alt = copy.deepcopy(model)
    alt.model.multi_modal_projector.linear.weight.zero_()
    alt.model.multi_modal_projector.linear.bias.zero_()
    lo = alt(input_ids=ids, pixel_values=pix,
             token_type_ids=token_type_ids).logits[0]
    d = (lo - base_logits).abs().max()
    assert d > 1e-3, f'visual tokens do not affect the logits ({d})'
    print(f'visual-token effect on logits: max |diff| = {d:.4f}')

    # 3. PaliGemma's vision feature INCLUDES post_layernorm (the SigLIP
    # last_hidden_state). Perturbing post_layernorm MUST move the visual
    # tokens - the importer must APPLY post_layernorm (the opposite of LLaVA's
    # -1 feature). Assert the effect is real.
    alt = copy.deepcopy(model)
    alt.model.vision_tower.post_layernorm.weight.normal_(3.0, 1.0)
    alt.model.vision_tower.post_layernorm.bias.normal_(2.0, 1.0)
    fv = alt.model.get_image_features(pixel_values=pix).pooler_output
    if fv.dim() == 3:
        fv = fv[0]
    d = (fv - base_vt).abs().max()
    assert d > 1e-3, (f'post_layernorm had no effect ({d}); PaliGemma uses the '
                      f'SigLIP last_hidden_state WITH post_layernorm')
    print('vision feature is POST-post_layernorm (SigLIP last_hidden_state): '
          f'confirmed effect (max |diff| = {d:.4f})')

    # 4. CRITICAL: the PREFIX-LM bidirectional mask must be load-bearing. Re-run
    # the SAME forward with a PURELY CAUSAL mask (token_type_ids = all 1, so the
    # whole sequence is causal). If the logits are unchanged the prompt does not
    # exercise the bidirectional block and the parity test could pass with a
    # wrong (causal-everywhere) importer mask. Demand a real difference.
    causal_tt = torch.ones_like(token_type_ids)
    lo = model(input_ids=ids, pixel_values=pix,
               token_type_ids=causal_tt).logits[0]
    d = (lo - base_logits).abs().max()
    assert d > 1e-3, (f'PREFIX-LM mask indistinguishable from causal ({d}); the '
                      f'prompt must exercise the bidirectional prefix block')
    print(f'PREFIX-LM vs causal-everywhere: max |diff| in logits = {d:.4f} '
          f'(the bidirectional prefix block is load-bearing)')
print('all fixture self-checks passed')
