#!/usr/bin/env python3
"""Generate a tiny RANDOM Florence-2 unified-vision parity fixture for
tests/TestNeuralPretrained.pas.

No network access: a pico Florence2ForConditionalGeneration (a tiny DaViT
vision backbone + the multimodal projector + a BART-style seq2seq language
model with a tied LM head) is built from a small config and randomly
initialized (never downloaded). The reference forward is the REAL
transformers Florence2 classes run in float64 (transformers 5.11 ships
Florence-2 natively, so trust_remote_code is NOT needed); the test asserts the
Pascal forward matches the per-position next-token decoder `logits` over the
vocabulary for a fixed image + task-prompt + decoder prefix.

Florence-2 is the repo's FIRST "spatial-output-as-text" VLM: ONE task-prompted
seq2seq head does captioning AND detection. The genuinely new ideas vs the
tracked single-task PaliGemma/LLaVA:
  - the encoder is a TEXT BART encoder fed a VISUAL-TOKEN PREFIX: the image
    features are projected to visual tokens, prepended to the (embedded) task-
    prompt text tokens via image-token scatter, and the whole [visual; text]
    sequence is run through the encoder. The decoder cross-attends to that.
  - boxes/polygons are emitted as quantized location tokens <loc_0..loc_999>
    in the vocabulary (spatial outputs as text). The (de)quantization is a pure
    coordinate<->token-id mapping the Pascal side reproduces exactly.

SCOPE v1 (this fixture): the multimodal projector + the BART encoder
(visual-prefix) + the BART decoder, pinned to the REAL HF Florence2 float64.
The DaViT vision tower itself is the DEFERRED gap: the Pascal importer takes
the DaViT feature map (last_hidden_state, B x C x H x W) as a PRECOMPUTED input
- exactly the way the tracked Qwen2-VL task takes the merged visual tokens as
input v1. This fixture therefore emits BOTH the DaViT feature map (the Pascal
input) AND the reference decoder logits (the parity target), so the projector +
encoder + decoder math is matched against real HF down to < 1e-4 while the
tower is reproduced by the oracle only.

Projector traits the importer reproduces:
  - a learned 2D ABSOLUTE position embedding (row table + column table; for
    grid cell (h,w) the embedding is cat(column[w][:C//2], row[h][C//2:]));
  - added to the feature map, flatten H*W -> tokens of dim C;
  - a fixed cosine 1D "temporal" embedding added (single frame -> row 0);
  - the visual tokens = cat([ spatial-mean(1 token), the H*W per-cell tokens ]);
  - a bias-free image_projection Linear then a biased image_proj_norm LayerNorm.

LM (BART) traits (same as the landed BART/TrOCR importer):
  - learned ABSOLUTE positions with BART's +2 offset over the WHOLE
    [visual; text] encoder sequence and the decoder sequence;
  - layernorm_embedding after token+position; POST-norm blocks (residual add
    THEN biased LayerNorm eps 1e-5); exact-erf GELU FFN; all q/k/v/out/fc
    Linears biased; scale_embedding (sqrt(d_model)) on the TEXT token
    embeddings only (the scattered visual tokens are NOT re-scaled);
  - the LM head tied to shared embeddings, NO final_logits_bias.

The weights are saved under the landed BART importer key convention
(model.shared.weight, model.encoder.*, model.decoder.*, +
multi_modal_projector.*), so the Florence-2 importer reuses the BART machinery
verbatim and only adds the visual-prefix encoder + projector + location tokens.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/make_pico_florence2_fixture.py
writes tests/fixtures/tiny_florence2{.safetensors,_config.json,_io.json}.
"""
import json
import math

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers.models.florence2.configuration_florence2 import (
    Florence2Config, Florence2VisionConfig)
from transformers.models.florence2.modeling_florence2 import (
    Florence2ForConditionalGeneration)

torch.manual_seed(1234)
np.random.seed(1234)

# ---------------- pico config ----------------
IMG = 32
DMODEL = 16
VOCAB = 60
IMAGE_TOKEN_ID = 59       # the <image> placeholder scattered with visual tokens
PROJ_DIM = 16             # vision projection_dim == d_model so it slots in

vcfg = Florence2VisionConfig(
    in_channels=3,
    depths=(1, 1), patch_size=(7, 3), patch_stride=(4, 2),
    patch_padding=(3, 1), patch_prenorm=(False, True),
    embed_dim=(8, 16), num_heads=(2, 2), num_groups=(2, 2),
    window_size=4, projection_dim=PROJ_DIM,
    max_temporal_embeddings=8, max_position_embeddings=16,
    mlp_ratio=2.0, activation_function="gelu", drop_path_rate=0.0,
)
text_config = dict(
    vocab_size=VOCAB, d_model=DMODEL,
    encoder_layers=2, decoder_layers=2,
    encoder_attention_heads=2, decoder_attention_heads=2,
    encoder_ffn_dim=32, decoder_ffn_dim=32,
    max_position_embeddings=40, activation_function="gelu",
    model_type="bart", bos_token_id=0, eos_token_id=2, pad_token_id=1,
    decoder_start_token_id=2, scale_embedding=True, dropout=0.0,
    attention_dropout=0.0, activation_dropout=0.0, encoder_layerdrop=0.0,
    decoder_layerdrop=0.0,
)
cfg = Florence2Config(vision_config=vcfg.to_dict(), text_config=text_config,
                      image_token_id=IMAGE_TOKEN_ID, vocab_size=VOCAB)

model = Florence2ForConditionalGeneration(cfg).double().eval()


# ---------------- re-randomize for visibility ----------------
# HF's default init (std 0.02, LN gain 1, bias 0) makes every quirk vanish
# below 1e-4; boost so attention scores spread, GELU vs ReLU differs, LN gains
# and biases matter, and the projector means/positions are all exercised.
def boost(t, scale):
    with torch.no_grad():
        t.copy_(torch.randn_like(t) * scale)


with torch.no_grad():
    for name, p in model.named_parameters():
        if p.dim() >= 2:
            boost(p, 0.45)
        elif 'norm' in name and 'weight' in name:
            # LayerNorm gains: spread around 1 so they are not the identity.
            p.copy_(1.0 + torch.randn_like(p) * 0.30)
        else:
            boost(p, 0.30)
    # The fixed cosine temporal buffer is recomputed by the module; leave it.

# ---------------- inputs ----------------
img = torch.randn(1, 3, IMG, IMG, dtype=torch.float64)
# A short task prompt after the visual tokens: pretend an <OD> detection task.
task_ids = [5, 6, 7, 8]
dec_ids = [2, 10, 11, 12, 13]   # decoder prefix (starts with eos = start token)

with torch.no_grad():
    # DaViT feature map -> the PASCAL INPUT (precomputed-tower shortcut).
    fmap = model.model.vision_tower(img).last_hidden_state  # (1, C, H, W)
    # Projected visual tokens (the projector output the encoder consumes).
    visual_tokens = model.model.get_image_features(img).pooler_output  # (1,N,D)
    n_img = visual_tokens.shape[1]
    input_ids = torch.tensor([[IMAGE_TOKEN_ID] * n_img + task_ids])
    dec_tensor = torch.tensor([dec_ids])
    out = model(input_ids=input_ids, pixel_values=img,
                decoder_input_ids=dec_tensor)
    logits = out.logits[0]                # (dec_len, vocab)
    enc_hidden = out.encoder_last_hidden_state[0]   # (enc_len, d_model)

C, H, W = int(fmap.shape[1]), int(fmap.shape[2]), int(fmap.shape[3])
print(f"DaViT feature map (C,H,W) = ({C},{H},{W}); visual tokens = {n_img}; "
      f"enc_len = {enc_hidden.shape[0]}; dec_len = {logits.shape[0]}")

# ---------------- self-checks (the oracle must EXERCISE every trait) ----------
sd = model.state_dict()


def assert_affects(perturb_keys, label):
    saved = {k: sd[k].clone() for k in perturb_keys}
    with torch.no_grad():
        for k in perturb_keys:
            sd[k].add_(torch.randn_like(sd[k]) * 0.5)
        l2 = model(input_ids=input_ids, pixel_values=img,
                   decoder_input_ids=dec_tensor).logits[0]
    diff = (l2 - logits).abs().max().item()
    with torch.no_grad():
        for k in perturb_keys:
            sd[k].copy_(saved[k])
    assert diff > 1e-3, f"{label} does not affect logits (diff {diff})"
    print(f"  [ok] {label} affects logits (max diff {diff:.4f})")


assert_affects(['model.multi_modal_projector.image_projection.weight'],
               'projector image_projection')
assert_affects(['model.multi_modal_projector.image_proj_norm.weight'],
               'projector LayerNorm gain')
assert_affects(
    ['model.multi_modal_projector.image_position_embed.row_embeddings.weight'],
    'projector 2D row position embed')
assert_affects(
    ['model.multi_modal_projector.image_position_embed.column_embeddings.weight'],
    'projector 2D column position embed')
assert_affects(['model.language_model.encoder.layers.0.self_attn.q_proj.weight'],
               'encoder self-attn (visual prefix mixes)')
assert_affects(['model.language_model.decoder.layers.0.encoder_attn.q_proj.weight'],
               'decoder cross-attn to encoder states')
assert_affects(['model.language_model.encoder.embed_positions.weight'],
               'encoder learned +2 positions')

# Decoder causality: changing a LATER decoder token must not move an
# earlier-position logit.
with torch.no_grad():
    alt = torch.tensor([dec_ids[:-1] + [40]])
    l_alt = model(input_ids=input_ids, pixel_values=img,
                  decoder_input_ids=alt).logits[0]
causal_early = (l_alt[:-1] - logits[:-1]).abs().max().item()
assert causal_early < 1e-9, f"decoder not causal (early diff {causal_early})"
print(f"  [ok] decoder self-attention is causal (early diff {causal_early:.2e})")

# Image must affect the logits (the visual prefix is actually consumed).
with torch.no_grad():
    l_noimg = model(input_ids=input_ids,
                    pixel_values=torch.zeros_like(img),
                    decoder_input_ids=dec_tensor).logits[0]
img_diff = (l_noimg - logits).abs().max().item()
assert img_diff > 1e-3, f"image does not affect logits (diff {img_diff})"
print(f"  [ok] image affects logits (max diff {img_diff:.4f})")

# ---------------- location-token (de)quantization reference ----------------
# Florence-2 emits boxes as <loc_i> tokens: a coordinate in [0,1] maps to
# bin = round(coord * (BINS - 1)); the token id is LOC_BASE + bin. The Pascal
# helpers must round-trip these. We emit a few reference pairs.
BINS = 1000
LOC_BASE = 100   # pico vocab is tiny, but the MAPPING is what is tested
loc_examples = []
for coord in [0.0, 0.1234, 0.5, 0.75, 0.999, 1.0]:
    b = int(round(coord * (BINS - 1)))
    loc_examples.append({"coord": coord, "bin": b})
# A reference box (x0,y0,x1,y1) in normalized coords -> 4 bins.
ref_box = [0.10, 0.20, 0.80, 0.90]
ref_box_bins = [int(round(c * (BINS - 1))) for c in ref_box]

# ---------------- save weights under the BART importer key convention --------
# The Florence-2 importer reuses BuildBartFromSafeTensors machinery, which
# expects model.shared.weight + model.encoder.* + model.decoder.* and the
# projector under multi_modal_projector.*.
out_sd = {}
for k, v in sd.items():
    arr = v.detach().cpu().to(torch.float32).numpy()
    if k == 'model.language_model.shared.weight':
        out_sd['model.shared.weight'] = arr
    elif k.startswith('model.language_model.encoder.'):
        out_sd['model.encoder.' + k[len('model.language_model.encoder.'):]] = arr
    elif k.startswith('model.language_model.decoder.'):
        out_sd['model.decoder.' + k[len('model.language_model.decoder.'):]] = arr
    elif k.startswith('model.multi_modal_projector.'):
        out_sd['multi_modal_projector.' + k[len('model.multi_modal_projector.'):]] = arr
    # vision_tower.* weights are intentionally DROPPED (deferred tower).
    # embed_tokens aliases are tied; skip to avoid the importer's unexpected
    # tensor guard.

# Drop the tied per-stack embed_tokens aliases (importer rebuilds from shared).
for drop in list(out_sd.keys()):
    if drop.endswith('embed_tokens.weight'):
        del out_sd[drop]

save_file(out_sd, 'tests/fixtures/tiny_florence2.safetensors')

# ---------------- config json (BART text config + Florence extras) -----------
config_json = {
    "model_type": "florence2",
    "image_token_id": IMAGE_TOKEN_ID,
    "vision_feature_channels": C,
    "vision_feature_height": H,
    "vision_feature_width": W,
    "projection_dim": PROJ_DIM,
    "max_temporal_embeddings": vcfg.max_temporal_embeddings,
    "vision_max_position_embeddings": vcfg.max_position_embeddings,
    "text_config": {
        "model_type": "bart",
        "d_model": DMODEL,
        "encoder_layers": 2, "decoder_layers": 2,
        "encoder_attention_heads": 2, "decoder_attention_heads": 2,
        "encoder_ffn_dim": 32, "decoder_ffn_dim": 32,
        "vocab_size": VOCAB, "max_position_embeddings": 40,
        "activation_function": "gelu",
        "bos_token_id": 0, "eos_token_id": 2, "pad_token_id": 1,
        "decoder_start_token_id": 2, "scale_embedding": True,
    },
    "location_token": {"num_bins": BINS, "loc_base": LOC_BASE},
}
with open('tests/fixtures/tiny_florence2_config.json', 'w') as f:
    json.dump(config_json, f, indent=2)

# ---------------- io json (oracle) -------------------------------------------
io = {
    "feature_map_chw": fmap[0].detach().cpu().numpy().tolist(),  # (C,H,W)
    "feature_c": C, "feature_h": H, "feature_w": W,
    "visual_tokens": visual_tokens[0].detach().cpu().numpy().tolist(),  # (N,D)
    "num_visual_tokens": n_img,
    "task_ids": task_ids,
    "dec_ids": dec_ids,
    "enc_hidden": enc_hidden.detach().cpu().numpy().tolist(),  # (enc_len, D)
    "logits": logits.detach().cpu().numpy().tolist(),          # (dec_len, vocab)
    "loc_examples": loc_examples,
    "ref_box": ref_box,
    "ref_box_bins": ref_box_bins,
    "num_bins": BINS, "loc_base": LOC_BASE,
}
with open('tests/fixtures/tiny_florence2_io.json', 'w') as f:
    json.dump(io, f)

print("wrote tests/fixtures/tiny_florence2{.safetensors,_config.json,_io.json}")
