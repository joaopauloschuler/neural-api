#!/usr/bin/env python3
"""Generate a tiny RANDOM Moonshine *GQA* parity fixture for
tests/TestNeuralPretrained.pas.

This is the sibling of make_pico_moonshine_fixture.py, but it pins the
GROUPED-QUERY-ATTENTION slice path that the base pico leaves dormant
(the base pico sets num_key_value_heads == num_attention_heads, so the
K/V-head replication degenerates to GroupSize=1 and is never exercised
by an oracle).

Here BOTH towers use 4 query heads over 2 K/V heads (GQA, GroupSize=2):
  - encoder: num_attention_heads = 4, num_key_value_heads = 2
  - decoder: num_attention_heads = 4, num_key_value_heads = 2

so the importer's `KVGroup := HeadCnt div GroupSize` slice path - which
broadcasts each K/V head to a group of query heads in self-attention,
cross-attention AND the encoder - is actually verified against a float64
HF oracle at the 1e-4 importer-parity gate.

(NOTE: transformers 5.x MoonshineAttention.__init__ does an in-place
`config.update({"num_key_value_heads": ...})` on the SHARED config, and
the encoder layers - which build first - leave config.num_key_value_heads
== encoder_num_key_value_heads, which the decoder then reads via the
attribute_map. So an asymmetric decoder kv head count is silently
clobbered by HF; we keep enc==dec==2 to stay parity-honest. GroupSize=2
on both towers is exactly the K/V-replication path the base pico
(GroupSize=1) never exercises.)

Dimensions are chosen so head_dim is even and rotary_dim is even
(the importer rejects odd head_dim / odd rotary_dim):
  hidden_size = 32, heads = 4  -> head_dim = 8
  partial_rotary_factor = 0.75 -> rotary_dim = int(8 * 0.75) = 6 (even)

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/make_pico_moonshine_gqa_fixture.py
writes tests/fixtures/tiny_moonshine_gqa{.safetensors,_config.json,
_encoder.json,_decoder.json}. Needs torch + transformers + safetensors
+ numpy.
"""
import copy
import json

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import (MoonshineConfig, MoonshineModel,
                          MoonshineForConditionalGeneration)

HIDDEN = 32
N_LAYER = 2
N_HEAD = 4                       # head_dim = HIDDEN / N_HEAD = 8
ENC_KV_HEAD = 2                  # encoder GQA: 4 query heads -> 2 kv heads
DEC_KV_HEAD = 2                  # decoder GQA: 4 query heads -> 2 kv heads
                                 # (HF clobbers an asymmetric value; see top)
FF = 32
VOCAB = 40
PARTIAL = 0.75                   # rotary_dim = int(8 * 0.75) = 6 (even)
MAX_POS = 64
# Raw waveform length chosen to clear the three strided convs and emit a
# few encoder frames: L1=(N-127)//64+1, L2=(L1-7)//3+1, L3=(L2-3)//2+1.
N_SAMPLES = 1719                 # -> L1=25, L2=7, L3=3 encoder frames

torch.manual_seed(20260626)

cfg_dict = {
    'architectures': ['MoonshineModel'],
    'model_type': 'moonshine',
    'hidden_size': HIDDEN,
    'intermediate_size': FF,
    'encoder_num_hidden_layers': N_LAYER,
    'decoder_num_hidden_layers': N_LAYER,
    'encoder_num_attention_heads': N_HEAD,
    'decoder_num_attention_heads': N_HEAD,
    'encoder_num_key_value_heads': ENC_KV_HEAD,
    'decoder_num_key_value_heads': DEC_KV_HEAD,
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
# transformers 5.11 MoonshineAttention.forward reshapes the QUERY by
# num_key_value_heads instead of num_attention_heads (modeling_moonshine.py
# line ~303): a real upstream bug that crashes / mis-shapes every GQA
# Moonshine (kv != heads). The math is otherwise correct (eager_attention_
# forward repeat_kv's the K/V groups). We monkeypatch ONLY that one reshape
# to num_attention_heads so HF serves a faithful GQA oracle - this is the
# exact grouped-query semantics our Pascal importer reproduces.
import transformers.models.moonshine.modeling_moonshine as _msm


def _gqa_attention_forward(self, hidden_states, position_embeddings=None,
                           attention_mask=None, past_key_values=None,
                           key_value_states=None, **kwargs):
    kwargs.pop('position_ids', None)
    kwargs.pop('use_cache', None)
    bsz, q_len = hidden_states.shape[:-1]
    # FIX: query uses num_attention_heads (not num_key_value_heads).
    query_states = (self.q_proj(hidden_states)
                    .view(bsz, q_len, self.config.num_attention_heads,
                          self.head_dim).transpose(1, 2))
    is_cross_attention = key_value_states is not None
    current_states = key_value_states if is_cross_attention else hidden_states
    key_states = (self.k_proj(current_states)
                  .view(bsz, -1, self.config.num_key_value_heads,
                        self.head_dim).transpose(1, 2))
    value_states = (self.v_proj(current_states)
                    .view(bsz, -1, self.config.num_key_value_heads,
                          self.head_dim).transpose(1, 2))
    if not is_cross_attention:
        cos, sin = position_embeddings
        query_states, key_states = _msm.apply_rotary_pos_emb(
            query_states, key_states, cos, sin)
    attention_interface = _msm.eager_attention_forward
    is_causal = self.is_causal and attention_mask is None and q_len > 1
    attn_output, attn_weights = attention_interface(
        self, query_states, key_states, value_states, attention_mask,
        dropout=0.0, scaling=self.scaling, is_causal=is_causal, **kwargs)
    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


_msm.MoonshineAttention.forward = _gqa_attention_forward

model = MoonshineModel(MoonshineConfig(**cfg_dict, attn_implementation='eager'))

# HF inits with std 0.02 at this pico width - vacuously small (near-identity
# blocks pass parity even when wired wrong). Re-randomize at O(1) scale so
# every quirk - including the GQA K/V-head broadcast - is visible.
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
        layer.input_layernorm.weight.normal_(1.0, 0.25)
        layer.post_attention_layernorm.weight.normal_(1.0, 0.25)
    enc.layer_norm.weight.normal_(1.0, 0.25)
    dec = model.decoder
    dec.embed_tokens.weight.normal_(0.0, 0.40)
    for layer in dec.layers:
        for attn in (layer.self_attn, layer.encoder_attn):
            for proj in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj):
                proj.weight.normal_(0.0, 0.45)
        layer.mlp.fc1.weight.normal_(0.0, 0.45)
        layer.mlp.fc1.bias.normal_(0.0, 0.20)
        layer.mlp.fc2.weight.normal_(0.0, 0.40)
        layer.mlp.fc2.bias.normal_(0.0, 0.20)
        layer.input_layernorm.weight.normal_(1.0, 0.25)
        layer.post_attention_layernorm.weight.normal_(1.0, 0.25)
        layer.final_layernorm.weight.normal_(1.0, 0.25)
    dec.norm.weight.normal_(1.0, 0.25)
model = model.double().eval()

# Structural assertions: the GQA shapes must actually be smaller on K/V.
HEAD_DIM = HIDDEN // N_HEAD
for layer in enc.layers:
    assert layer.self_attn.k_proj.weight.shape[0] == ENC_KV_HEAD * HEAD_DIM, \
        'encoder k_proj is not GQA-shaped'
    assert layer.self_attn.q_proj.weight.shape[0] == N_HEAD * HEAD_DIM
for layer in dec.layers:
    assert layer.self_attn.k_proj.weight.shape[0] == DEC_KV_HEAD * HEAD_DIM, \
        'decoder self_attn k_proj is not GQA-shaped'
    assert layer.encoder_attn.k_proj.weight.shape[0] == DEC_KV_HEAD * HEAD_DIM, \
        'decoder cross_attn k_proj is not GQA-shaped'
    assert layer.self_attn.q_proj.weight.shape[0] == N_HEAD * HEAD_DIM
assert enc.conv1.bias is None
for layer in enc.layers:
    assert layer.self_attn.q_proj.bias is None
    assert layer.input_layernorm.bias is None
assert enc.layer_norm.bias is None
for layer in dec.layers:
    assert layer.self_attn.q_proj.bias is None
    assert layer.encoder_attn.q_proj.bias is None
    assert layer.mlp.fc1.bias is not None
    assert layer.input_layernorm.bias is None
    assert layer.final_layernorm.bias is None
assert dec.norm.bias is None

sd = {k: v.to(torch.float32).clone().contiguous()
      for k, v in model.state_dict().items()
      if k.startswith('encoder.') or k.startswith('decoder.')}
save_file(sd, 'tests/fixtures/tiny_moonshine_gqa.safetensors')
with open('tests/fixtures/tiny_moonshine_gqa_config.json', 'w') as f:
    json.dump(cfg_dict, f, indent=1)

rng = np.random.default_rng(424242)
waveform = (rng.integers(-256, 256, size=N_SAMPLES) / 256.0).astype(np.float64)

with torch.no_grad():
    feats = torch.tensor([waveform], dtype=torch.float64)
    enc_out = enc(input_values=feats).last_hidden_state[0]
enc_hidden = enc_out.tolist()
ENC_LEN = enc_out.shape[0]
assert enc_out.shape == (ENC_LEN, HIDDEN), enc_out.shape

with open('tests/fixtures/tiny_moonshine_gqa_encoder.json', 'w') as f:
    json.dump({'waveform': waveform.tolist(),
               'enc_hidden': enc_hidden,
               'enc_len': ENC_LEN}, f)

# ---- DECODER oracle: next-token logits for a fixed (waveform, prefix) ----
DEC_PREFIX = [1, 7, 19, 4]   # decoder_start_token_id=1 then 3 content ids
with torch.no_grad():
    dec_ids = torch.tensor([DEC_PREFIX], dtype=torch.long)
    full = MoonshineForConditionalGeneration(model.config).double().eval()
    full.model.load_state_dict(model.state_dict())
    full.tie_weights()
    logits = full(input_values=feats, decoder_input_ids=dec_ids).logits[0]
    dec_logits = logits[-1].tolist()
assert len(dec_logits) == VOCAB, len(dec_logits)

with open('tests/fixtures/tiny_moonshine_gqa_decoder.json', 'w') as f:
    json.dump({'waveform': waveform.tolist(),
               'dec_prefix': DEC_PREFIX,
               'dec_logits': dec_logits,
               'enc_len': ENC_LEN}, f)

print(f'wrote tiny_moonshine_gqa.safetensors ({len(sd)} tensors), '
      f'{N_SAMPLES} samples -> {ENC_LEN} encoder frames')
print(f'  enc GQA {N_HEAD}q/{ENC_KV_HEAD}kv, dec GQA {N_HEAD}q/{DEC_KV_HEAD}kv')
print(f'  decoder oracle: prefix {DEC_PREFIX} -> {len(dec_logits)} logits, '
      f'argmax={int(np.argmax(dec_logits))}')

# ---- fixture self-checks: the GQA broadcast must be visible in the oracle ----
with torch.no_grad():
    base = enc(input_values=feats).last_hidden_state

    # GQA broadcast: perturbing ONE encoder kv head's k_proj must move the
    # oracle. With ENC_KV_HEAD=2 each kv head feeds 2 query heads; if the
    # importer mis-mapped query head -> kv head this would not reproduce.
    m2 = copy.deepcopy(model)
    m2.encoder.layers[0].self_attn.k_proj.weight[:HEAD_DIM, :] += 0.3
    d = (m2.encoder(input_values=feats).last_hidden_state -
         base).abs().max().item()
    assert d > 1e-3, f'encoder kv head 0 had no effect ({d})'

    # Decoder GQA: each kv head feeds 2 query heads. Perturb the decoder
    # self_attn k_proj and confirm the logits move (a mis-mapped query->kv
    # head assignment would not reproduce the oracle).
    m2 = copy.deepcopy(model)
    m2.decoder.layers[0].self_attn.k_proj.weight += 0.3
    fullp = MoonshineForConditionalGeneration(m2.config).double().eval()
    fullp.model.load_state_dict(m2.state_dict())
    fullp.tie_weights()
    lp = fullp(input_values=feats, decoder_input_ids=dec_ids).logits[0][-1]
    d = (lp - torch.tensor(dec_logits, dtype=torch.float64)).abs().max().item()
    assert d > 1e-3, f'decoder MQA kv head had no effect ({d})'
print('all GQA fixture self-checks passed')
