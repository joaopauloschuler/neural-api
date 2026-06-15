#!/usr/bin/env python3
"""Generate a tiny RANDOM Qwen2-Audio parity fixture for
tests/TestNeuralPretrained.pas (no network access: the model is randomly
initialized from a pico config, never downloaded).

Qwen2-Audio = the AUDIO analogue of the landed vision-language importers
(LLaVA / PaliGemma): a frozen Whisper-style log-mel conv+transformer audio
ENCODER -> a single biased Linear multimodal projector
(audio d_model -> text hidden_size) -> the projected audio frames SPLICED
into the language decoder's token-embedding sequence at the <|AUDIO|>
placeholder positions, then ordinary CAUSAL decoding through a Qwen2 LM.

The pico Qwen2-Audio here pairs:
  - AUDIO tower (audio_tower.*): a Whisper encoder - conv1 (k3,p1) -> gelu ->
    conv2 (k3,p1,stride2) -> gelu, transpose, + FIXED-table embed_positions,
    then encoder_layers pre-norm transformer blocks. The Qwen2-Audio TAIL
    (NEW vs plain Whisper): permute, AvgPool1d(2, stride2) over the frame
    axis (halves the frames again), permute, then a final LayerNorm. So the
    mel input length 2*max_source_positions -> max_source_positions frames
    after conv2 -> max_source_positions//2 audio tokens after avg_pooler.
  - PROJECTOR (multi_modal_projector.linear): one biased Linear
    (d_model -> text hidden_size).
  - TEXT (language_model.*): a Qwen2 decoder (biased q/k/v, GQA, RoPE,
    SwiGLU, RMSNorm) - the stock BuildLlamaFromSafeTensors path.

The committed fixture (KB-scale) in tests/fixtures/:
  tiny_qwen2audio.safetensors  - the full state_dict (HF transformers 5.x
      "model."-prefixed keys: model.audio_tower.*,
      model.multi_modal_projector.*, model.language_model.*, lm_head.*).
  tiny_qwen2audio_config.json  - the Qwen2AudioConfig.to_dict().
  tiny_qwen2audio_logits.json  - the float64 oracle: the pinned mel input
      [num_mel_bins][mel_len], the pinned token id sequence (with
      audio_token_index at the audio slots), the projected audio tokens
      [n_audio][text_hidden], and the next-token logits [seq][vocab].

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/qwen2audio_tiny_fixture.py
Needs torch + transformers + safetensors.
"""
import copy
import json

import torch
from safetensors.torch import save_file
from transformers import Qwen2AudioConfig, Qwen2AudioForConditionalGeneration

# ---- text (Qwen2) ----
HID_T = 32          # text hidden_size
INTER_T = 64        # text intermediate_size (SwiGLU width)
N_LAYER_T = 2
N_HEAD_T = 4        # text head_dim = 8
N_KV_T = 2          # GQA
VOCAB = 50
MAX_POS = 64

# ---- audio (Whisper encoder) ----
D_MODEL = 24        # audio d_model
ENC_LAYERS = 2
ENC_HEADS = 4       # audio head_dim = 6
ENC_FFN = 48
N_MEL = 16          # num_mel_bins (encoder input depth)
MAX_SRC = 6         # max_source_positions -> mel_len = 12, audio tokens = 3
MEL_LEN = 2 * MAX_SRC
N_AUDIO = MAX_SRC // 2   # frames after conv2(/2) then avg_pooler(/2)

AUDIO_TOKEN = 49    # audio_token_index (last real id, < VOCAB)

torch.manual_seed(20260615)

cfg = Qwen2AudioConfig(
    audio_config=dict(
        model_type="qwen2_audio_encoder", d_model=D_MODEL,
        encoder_layers=ENC_LAYERS, encoder_attention_heads=ENC_HEADS,
        encoder_ffn_dim=ENC_FFN, num_mel_bins=N_MEL,
        max_source_positions=MAX_SRC, vocab_size=51865,
        activation_function="gelu", scale_embedding=False),
    text_config=dict(
        model_type="qwen2", hidden_size=HID_T, intermediate_size=INTER_T,
        num_hidden_layers=N_LAYER_T, num_attention_heads=N_HEAD_T,
        num_key_value_heads=N_KV_T, vocab_size=VOCAB,
        max_position_embeddings=MAX_POS, rms_norm_eps=1e-6,
        rope_theta=10000.0, tie_word_embeddings=False),
    audio_token_index=AUDIO_TOKEN)

model = Qwen2AudioForConditionalGeneration(cfg)

# HF inits with tiny stds at this pico width; boost so every component is
# visible above the 1e-4 parity gate (the ModernBERT vacuous-init lesson).
at = model.model.audio_tower
lm = model.model.language_model
with torch.no_grad():
    at.conv1.weight.normal_(0.0, 0.3)
    at.conv1.bias.normal_(0.0, 0.2)
    at.conv2.weight.normal_(0.0, 0.3)
    at.conv2.bias.normal_(0.0, 0.2)
    # embed_positions is a FIXED (non-trainable) sinusoid table in Whisper but
    # a plain Embedding here; randomize it so the position add is visible.
    at.embed_positions.weight.normal_(0.0, 0.4)
    for layer in at.layers:
        sa = layer.self_attn
        for proj in (sa.q_proj, sa.v_proj, sa.out_proj):
            proj.weight.normal_(0.0, 0.35)
            proj.bias.normal_(0.0, 0.2)
        sa.k_proj.weight.normal_(0.0, 0.35)   # Whisper k_proj is bias-free
        layer.fc1.weight.normal_(0.0, 0.4)
        layer.fc1.bias.normal_(0.0, 0.2)
        layer.fc2.weight.normal_(0.0, 0.3)
        layer.fc2.bias.normal_(0.0, 0.2)
        for norm in (layer.self_attn_layer_norm, layer.final_layer_norm):
            norm.weight.normal_(1.0, 0.25)
            norm.bias.normal_(0.0, 0.2)
    at.layer_norm.weight.normal_(1.0, 0.25)
    at.layer_norm.bias.normal_(0.0, 0.2)

    # projector (single biased linear)
    proj = model.model.multi_modal_projector
    proj.linear.weight.normal_(0.0, 0.3)
    proj.linear.bias.normal_(0.0, 0.2)

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

assert model.model.multi_modal_projector.linear.bias is not None
assert model.model.audio_tower.layers[0].self_attn.k_proj.bias is None

# ---- pinned inputs ----
# Mixed prompt: [t t <|AUDIO|>*N_AUDIO t t]  (text, audio block, text).
text_pre = [1, 7]
text_post = [12, 3]
ids_list = text_pre + [AUDIO_TOKEN] * N_AUDIO + text_post
ids = torch.tensor([ids_list])
SEQ = len(ids_list)

# Deterministic dyadic mel values (exact in f32 + JSON): [num_mel_bins][mel_len].
mel = torch.zeros(1, N_MEL, MEL_LEN, dtype=torch.float64)
for b in range(N_MEL):
    for t in range(MEL_LEN):
        mel[0, b, t] = (((b * 13 + t * 7) * 5) % 17 - 8) / 8.0

# feature_attention_mask: full length, no padding (single full-length clip).
feat_mask = torch.ones(1, MEL_LEN, dtype=torch.long)

with torch.no_grad():
    # projected audio tokens (intermediate oracle): run the audio tower then
    # the projector exactly as Qwen2AudioModel.forward does.
    audio_out = model.model.audio_tower(mel)
    selected = audio_out.last_hidden_state           # [1][N_AUDIO][D_MODEL]
    audio_tokens = model.model.multi_modal_projector(selected)[0]  # [N_AUDIO][HID_T]
    assert audio_tokens.shape == (N_AUDIO, HID_T), audio_tokens.shape

    # full forward -> next-token logits (final oracle)
    out = model(input_ids=ids, input_features=mel,
                feature_attention_mask=feat_mask)
    logits = out.logits[0]                           # [SEQ][VOCAB]
    assert logits.shape == (SEQ, VOCAB), logits.shape

# ---- save weights + config ----
sd = {k: v.to(torch.float32).clone().contiguous()
      for k, v in model.state_dict().items()
      if not k.endswith('position_ids')}
save_file(sd, 'tests/fixtures/tiny_qwen2audio.safetensors')
cfg_dict = cfg.to_dict()
# Pin the effective rope_theta flat (newer transformers nest it).
rp = cfg_dict['text_config'].get('rope_parameters') or \
    cfg_dict['text_config'].get('rope_scaling') or {}
cfg_dict['text_config']['rope_theta'] = rp.get('rope_theta', 10000.0)
with open('tests/fixtures/tiny_qwen2audio_config.json', 'w') as f:
    json.dump(cfg_dict, f, indent=1, default=str)

with open('tests/fixtures/tiny_qwen2audio_logits.json', 'w') as f:
    json.dump({
        'audio_token_index': AUDIO_TOKEN,
        'num_audio_tokens': N_AUDIO,
        'num_mel_bins': N_MEL,
        'mel_len': MEL_LEN,
        'token_ids': ids_list,
        'mel': mel[0].tolist(),                  # [num_mel_bins][mel_len]
        'audio_tokens': audio_tokens.tolist(),   # [N_AUDIO][HID_T]
        'logits': logits.tolist(),               # [SEQ][VOCAB]
    }, f)
print(f'wrote tiny_qwen2audio.safetensors ({len(sd)} tensors) + config + oracle')
print(f'  seq_len={SEQ} num_audio={N_AUDIO} mel={N_MEL}x{MEL_LEN} vocab={VOCAB}')

# ---- fixture self-checks: every Qwen2-Audio-distinguishing piece must matter ----
with torch.no_grad():
    base_audio = audio_tokens.clone()
    base_logits = logits.clone()

    # 1. projector bias must matter.
    alt = copy.deepcopy(model)
    alt.model.multi_modal_projector.linear.bias.zero_()
    s = alt.model.audio_tower(mel).last_hidden_state
    fa = alt.model.multi_modal_projector(s)[0]
    d = (fa - base_audio).abs().max()
    assert d > 1e-3, f'projector bias had no effect ({d})'
    print(f'projector bias effect on audio tokens: max |diff| = {d:.4f}')

    # 2. the avg_pooler must matter: replacing it with identity (no pooling)
    # changes the token count, so instead perturb the post-pool layer_norm and
    # confirm it moves the audio tokens (the pool+norm tail is real).
    alt = copy.deepcopy(model)
    alt.model.audio_tower.layer_norm.weight.normal_(3.0, 1.0)
    alt.model.audio_tower.layer_norm.bias.normal_(2.0, 1.0)
    s = alt.model.audio_tower(mel).last_hidden_state
    fa = alt.model.multi_modal_projector(s)[0]
    d = (fa - base_audio).abs().max()
    assert d > 1e-3, f'audio tower final layer_norm had no effect ({d})'
    print(f'audio tower layer_norm effect on audio tokens: max |diff| = {d:.4f}')

    # 3. the audio tokens must actually flow into the logits: zeroing the
    # projector output must move the logits at/after the audio block.
    alt = copy.deepcopy(model)
    alt.model.multi_modal_projector.linear.weight.zero_()
    alt.model.multi_modal_projector.linear.bias.zero_()
    lo = alt(input_ids=ids, input_features=mel,
             feature_attention_mask=feat_mask).logits[0]
    d = (lo - base_logits).abs().max()
    assert d > 1e-3, f'audio tokens do not affect the logits ({d})'
    print(f'audio-token effect on logits: max |diff| = {d:.4f}')
print('all fixture self-checks passed')
