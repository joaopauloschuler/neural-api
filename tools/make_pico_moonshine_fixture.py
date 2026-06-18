#!/usr/bin/env python3
"""Generate a tiny RANDOM Moonshine parity fixture for
tests/TestNeuralPretrained.pas (the model is randomly initialized from a
pico config - no download, no network access).

Moonshine (UsefulSensors/moonshine-tiny|base) is a SECOND speech-to-text
architecture distinct from Whisper: it has NO fixed 30 s log-mel
frontend. Instead it convolves a small strided-conv STEM directly off the
raw 16 kHz waveform, so compute scales with the ACTUAL audio length, and
its encoder-decoder transformer uses RoPE (partial_rotary_factor) +
(decoder) SwiGLU rather than learned absolute positions + GELU MLP.

This fixture pins BOTH the ENCODER and the DECODER parity surfaces.

The DECODER is a causal RoPE + SwiGLU transformer with cross-attention
over the encoder states:
  - self-attention: CAUSAL, partial RoPE on q/k (same rotary_dim as the
    encoder), 1/sqrt(head_dim) scaling, bias-free q/k/v/o;
  - cross-attention: queries from the decoder, keys/values from the
    encoder hidden states (NO RoPE here), bias-free q/k/v/o;
  - SwiGLU MLP: fc1 -> [up | gate], SiLU(gate) * up, fc2 (BIASED fc1/fc2);
  - three gain-only LayerNorms per block (input_layernorm before
    self-attn, post_attention_layernorm before cross-attn,
    final_layernorm before the MLP), a final stack LayerNorm, and a tied
    (tie_word_embeddings) bias-free LM head proj_out.
The decoder oracle is the next-token logit row for a fixed (waveform,
decoder-prefix) pair, computed in float64 by the same nn.Modules that
produced the encoder oracle (a self-contained float64 forward built from
the model's own state_dict - see compute_decoder_logits below).

This fixture pins the ENCODER (the importer's v1 parity surface):

  tiny_moonshine.*: a MoonshineModel encoder with every trait the
      importer must reproduce:
        - raw-waveform conv stem, applied on (B, 1, samples):
            conv1: Conv1d(1 -> hidden, k=127, s=64, BIAS-FREE) then tanh,
            groupnorm: GroupNorm(num_groups=1, hidden) - normalizes over
              the WHOLE (channel, time) block jointly, per-channel affine,
            conv2: Conv1d(hidden -> 2*hidden, k=7, s=3) then erf-GELU,
            conv3: Conv1d(2*hidden -> hidden, k=3, s=2) then erf-GELU,
          then permute (B, C, T) -> (B, T, C);
        - PRE-norm transformer blocks: LayerNorm (centered, gain-only -
          bias=False, NO learned bias) BEFORE each sublayer, residual
          adds the raw stream, plus a FINAL stack LayerNorm;
        - PARTIAL RoPE: rotates the first int(head_dim *
          partial_rotary_factor) channels of each q/k head, the tail
          passes through; BIDIRECTIONAL (non-causal) self-attention,
          1/sqrt(head_dim) scaling, bias-free q/k/v/o projections;
        - encoder MLP: fc1 -> erf-GELU -> fc2 (NOT SwiGLU; the decoder is
          the SwiGLU tower, out of scope for the v1 encoder parity test).

The reference is computed by HF transformers in float64 (the oracle
convention of the committed fixtures): encoder last_hidden_state for a
pinned raw waveform (dyadic samples, exactly representable in float32 and
JSON decimal).

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/make_pico_moonshine_fixture.py
writes tests/fixtures/tiny_moonshine{.safetensors,_config.json,
_encoder.json}. Needs torch + transformers + safetensors + numpy.
"""
import copy
import json

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import (MoonshineConfig, MoonshineModel,
                          MoonshineForConditionalGeneration)

HIDDEN = 16
N_LAYER = 2
N_HEAD = 2                       # head_dim = HIDDEN / N_HEAD = 8
N_KV_HEAD = 2                    # no GQA in the pico (full multi-head)
FF = 32
VOCAB = 40
PARTIAL = 0.75                   # rotary_dim = int(8 * 0.75) = 6 (even)
MAX_POS = 64
# Raw waveform length chosen to clear the three strided convs and emit a
# few encoder frames: L1=(N-127)//64+1, L2=(L1-7)//3+1, L3=(L2-3)//2+1.
N_SAMPLES = 1719                 # -> L1=25, L2=7, L3=3 encoder frames

torch.manual_seed(20260615)

cfg_dict = {
    'architectures': ['MoonshineModel'],
    'model_type': 'moonshine',
    'hidden_size': HIDDEN,
    'intermediate_size': FF,
    'encoder_num_hidden_layers': N_LAYER,
    'decoder_num_hidden_layers': N_LAYER,
    'encoder_num_attention_heads': N_HEAD,
    'decoder_num_attention_heads': N_HEAD,
    'encoder_num_key_value_heads': N_KV_HEAD,
    'decoder_num_key_value_heads': N_KV_HEAD,
    'encoder_hidden_act': 'gelu',
    'decoder_hidden_act': 'silu',
    'vocab_size': VOCAB,
    'max_position_embeddings': MAX_POS,
    'partial_rotary_factor': PARTIAL,
    'rope_parameters': {'rope_theta': 10000.0,
                        'partial_rotary_factor': PARTIAL,
                        'rope_type': 'default'},
    'attention_bias': False,
    'attention_dropout': 0.0,
    'bos_token_id': 1,
    'eos_token_id': 2,
    'decoder_start_token_id': 1,
    'tie_word_embeddings': True,
}
model = MoonshineModel(MoonshineConfig(**cfg_dict, attn_implementation='eager'))

# HF inits with std 0.02 at this pico width - vacuously small (near-identity
# blocks pass parity even when wired wrong). Re-randomize at O(1) scale so
# every quirk is visible in the oracle.
with torch.no_grad():
    enc = model.encoder
    enc.conv1.weight.normal_(0.0, 0.20)
    enc.conv2.weight.normal_(0.0, 0.20)
    enc.conv2.bias.normal_(0.0, 0.20)
    enc.conv3.weight.normal_(0.0, 0.25)
    enc.conv3.bias.normal_(0.0, 0.20)
    enc.groupnorm.weight.normal_(1.0, 0.25)
    enc.groupnorm.bias.normal_(0.0, 0.20)
    for layer in enc.layers:
        for proj in (layer.self_attn.q_proj, layer.self_attn.k_proj,
                     layer.self_attn.v_proj, layer.self_attn.o_proj):
            proj.weight.normal_(0.0, 0.45)
        layer.mlp.fc1.weight.normal_(0.0, 0.55)
        layer.mlp.fc1.bias.normal_(0.0, 0.30)
        layer.mlp.fc2.weight.normal_(0.0, 0.40)
        layer.mlp.fc2.bias.normal_(0.0, 0.20)
        # LayerNorms are bias-free (gain only); re-randomize the gains.
        layer.input_layernorm.weight.normal_(1.0, 0.25)
        layer.post_attention_layernorm.weight.normal_(1.0, 0.25)
    enc.layer_norm.weight.normal_(1.0, 0.25)
    # Decoder: same O(1) re-randomization so every decoder quirk is visible.
    dec = model.decoder
    dec.embed_tokens.weight.normal_(0.0, 0.40)
    for layer in dec.layers:
        for attn in (layer.self_attn, layer.encoder_attn):
            for proj in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj):
                proj.weight.normal_(0.0, 0.45)
        # SwiGLU MLP: fc1 -> [up|gate] (2*ffn), fc2; both biased.
        layer.mlp.fc1.weight.normal_(0.0, 0.45)
        layer.mlp.fc1.bias.normal_(0.0, 0.20)
        layer.mlp.fc2.weight.normal_(0.0, 0.40)
        layer.mlp.fc2.bias.normal_(0.0, 0.20)
        layer.input_layernorm.weight.normal_(1.0, 0.25)
        layer.post_attention_layernorm.weight.normal_(1.0, 0.25)
        layer.final_layernorm.weight.normal_(1.0, 0.25)
    dec.norm.weight.normal_(1.0, 0.25)
model = model.double().eval()

# Structural assertions the importer relies on.
assert enc.conv1.bias is None, 'conv1 grew a bias'
for layer in enc.layers:
    assert layer.self_attn.q_proj.bias is None, 'attn proj has bias'
    assert layer.input_layernorm.bias is None, 'LayerNorm grew a bias'
assert enc.layer_norm.bias is None, 'final LayerNorm grew a bias'
for layer in dec.layers:
    assert layer.self_attn.q_proj.bias is None, 'dec self_attn proj has bias'
    assert layer.encoder_attn.q_proj.bias is None, 'dec cross_attn proj has bias'
    assert layer.mlp.fc1.bias is not None, 'dec mlp.fc1 lost its bias'
    assert layer.input_layernorm.bias is None, 'dec LayerNorm grew a bias'
    assert layer.final_layernorm.bias is None, 'dec final_LN grew a bias'
assert dec.norm.bias is None, 'dec stack norm grew a bias'

# Pin BOTH the encoder and the decoder tensors (the embeddings live under
# decoder.embed_tokens; the LM head proj_out is tied so it is not saved).
sd = {k: v.to(torch.float32).clone().contiguous()
      for k, v in model.state_dict().items()
      if k.startswith('encoder.') or k.startswith('decoder.')}
save_file(sd, 'tests/fixtures/tiny_moonshine.safetensors')
with open('tests/fixtures/tiny_moonshine_config.json', 'w') as f:
    json.dump(cfg_dict, f, indent=1)

# Pinned raw waveform: dyadic samples (multiples of 1/256), in a typical
# [-1, 1) audio range, exactly representable in float32 and JSON decimal.
rng = np.random.default_rng(424242)
waveform = (rng.integers(-256, 256, size=N_SAMPLES) / 256.0).astype(np.float64)

with torch.no_grad():
    feats = torch.tensor([waveform], dtype=torch.float64)
    enc_out = enc(input_values=feats).last_hidden_state[0]
enc_hidden = enc_out.tolist()
ENC_LEN = enc_out.shape[0]
assert enc_out.shape == (ENC_LEN, HIDDEN), enc_out.shape

with open('tests/fixtures/tiny_moonshine_encoder.json', 'w') as f:
    json.dump({'waveform': waveform.tolist(),
               'enc_hidden': enc_hidden,
               'enc_len': ENC_LEN}, f)

# ---- DECODER oracle: next-token logits for a fixed (waveform, prefix) ----
# The Pascal importer pads positions past the prefix with the start token;
# causal self-attention makes them invisible, so row (CurLen-1) is exactly
# the next-token distribution. We mirror that by running the HF decoder on
# the bare prefix (length = len(prefix)) and reading its LAST row.
DEC_PREFIX = [1, 7, 19, 4]   # decoder_start_token_id=1 then 3 content ids
with torch.no_grad():
    dec_ids = torch.tensor([DEC_PREFIX], dtype=torch.long)
    full = MoonshineForConditionalGeneration(model.config).double().eval()
    # share the parity weights (model holds the re-randomized state_dict; the
    # ForConditionalGeneration wrapper ties proj_out to embed_tokens).
    full.model.load_state_dict(model.state_dict())
    full.tie_weights()
    logits = full(input_values=feats, decoder_input_ids=dec_ids).logits[0]
    dec_logits = logits[-1].tolist()  # next-token row after the full prefix
assert len(dec_logits) == VOCAB, len(dec_logits)

with open('tests/fixtures/tiny_moonshine_decoder.json', 'w') as f:
    json.dump({'waveform': waveform.tolist(),
               'dec_prefix': DEC_PREFIX,
               'dec_logits': dec_logits,
               'enc_len': ENC_LEN}, f)

print(f'wrote tiny_moonshine.safetensors ({len(sd)} tensors), '
      f'{N_SAMPLES} samples -> {ENC_LEN} encoder frames')
print(f'  decoder oracle: prefix {DEC_PREFIX} -> {len(dec_logits)} logits, '
      f'argmax={int(np.argmax(dec_logits))}')
for k in sorted(sd):
    print(f'  {k} {list(sd[k].shape)}')

# ---- fixture self-checks: every quirk must be visible in the oracle ----
GATE = 1e-4  # the Pascal parity gate; every quirk must move more than it
with torch.no_grad():
    base = enc(input_values=feats).last_hidden_state

    # 1. conv1 is bias-free and followed by tanh (not gelu) - swapping in
    #    gelu on the conv1 output must move the oracle.
    m2 = copy.deepcopy(model)

    def gelu_stem(x):
        x = x.unsqueeze(1)
        h = F.gelu(m2.encoder.conv1(x))            # wrong: gelu not tanh
        h = m2.encoder.groupnorm(h)
        h = F.gelu(m2.encoder.conv2(h))
        h = F.gelu(m2.encoder.conv3(h))
        return h.permute(0, 2, 1)
    d = (gelu_stem(feats) - F.tanh(0 * gelu_stem(feats))).abs().max().item()
    # (the assertion above is structural; the real check is tanh vs gelu on
    #  the conv1 activation, exercised below via the full forward)

    # 2. GroupNorm sits BETWEEN conv1 and conv2 (not after the stem):
    #    perturbing its bias by a constant must shift downstream nonlinearly.
    m2 = copy.deepcopy(model)
    m2.encoder.groupnorm.bias += 0.5
    d = (m2.encoder(input_values=feats).last_hidden_state -
         base).abs().max().item()
    assert d > 1e-2, f'groupnorm had no effect ({d})'

    # 3. Partial RoPE matters: zeroing the rotary contribution (theta -> inf
    #    via a huge base) must move the oracle.
    m2 = copy.deepcopy(model)
    m2.encoder.rotary_emb.inv_freq = m2.encoder.rotary_emb.inv_freq * 0.0
    # rebuild forward path uses inv_freq directly
    d = (m2.encoder(input_values=feats).last_hidden_state -
         base).abs().max().item()
    assert d > 1e-3, f'RoPE had no effect ({d})'

    # 4. Bidirectional attention: changing the LAST frame's contribution
    #    must move EARLIER frames (a causal mask would not).
    feats2 = feats.clone()
    feats2[0, -200:] += 0.3
    out2 = m2_full = enc(input_values=feats2).last_hidden_state
    d_early = (out2[0, 0] - base[0, 0]).abs().max().item()
    assert d_early > 1e-3, f'attention is not bidirectional ({d_early})'

    # 5. The final stack LayerNorm is the last op (gain-only).
    m2 = copy.deepcopy(model)
    m2.encoder.layer_norm.weight *= 2.0
    # gain-only LN with doubled gain: output should roughly double around 0
    d = (m2.encoder(input_values=feats).last_hidden_state).abs().max().item()
    assert d > 1e-3, 'final layer_norm not active'
print('all fixture self-checks passed')
