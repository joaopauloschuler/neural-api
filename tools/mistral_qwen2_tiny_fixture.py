#!/usr/bin/env python3
"""Generate tiny RANDOM Mistral and Qwen2 parity fixtures for
tests/TestNeuralPretrained.pas (no network access needed: the models are
randomly initialized from a pico config, never downloaded).

Two fixtures, both ~5 KB, pinned in tests/fixtures/:

  tiny_mistral.*: MistralForCausalLM with sliding_window=4 SMALLER than the
      16-token test sequences, so the sliding-window attention path is
      genuinely exercised (the script ASSERTS that the windowed logits
      differ from a full-attention run on the same weights).
  tiny_qwen2.*:   Qwen2ForCausalLM with NONZERO random q/k/v projection
      biases (HF zero-inits them, so the script re-randomizes and ASSERTS
      that zeroing the biases changes the logits).

Reference logits are computed by HF transformers in float64 (the oracle
convention of the committed fixtures).

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/mistral_qwen2_tiny_fixture.py
writes tests/fixtures/tiny_{mistral,qwen2}{.safetensors,_config.json,
_logits.json}. Needs torch + transformers + safetensors.
"""
import json

import torch
from safetensors.torch import save_file
from transformers import (MistralConfig, MistralForCausalLM,
                          Qwen2Config, Qwen2ForCausalLM)

N_LAYER = 2
N_HEAD = 2
N_KV_HEAD = 1
HEAD_DIM = 4
D_MODEL = N_HEAD * HEAD_DIM  # 8
D_FF = 12
MAX_POS = 16
N_SEQUENCES = 3
SLIDING_WINDOW = 4  # < MAX_POS so the window genuinely masks


def dump(model, cfg_dict, out_prefix, vocab):
    model = model.double().eval()
    sd = {k: v.to(torch.float32).contiguous()
          for k, v in model.state_dict().items()}
    save_file(sd, out_prefix + '.safetensors')
    with open(out_prefix + '_config.json', 'w') as f:
        json.dump(cfg_dict, f, indent=1)
    sequences = [[(7 * i + 3 * s + s * s) % vocab for i in range(MAX_POS)]
                 for s in range(N_SEQUENCES)]
    with torch.no_grad():
        logits = [model(input_ids=torch.tensor([seq])).logits[0].tolist()
                  for seq in sequences]
    with open(out_prefix + '_logits.json', 'w') as f:
        json.dump({'sequences': sequences, 'logits': logits}, f)
    print(f'wrote {out_prefix}.safetensors ({len(sd)} tensors) '
          f'+ config + logits ({N_SEQUENCES} sequences of {MAX_POS})')
    return sequences


# ------------------------------ Mistral ------------------------------------
torch.manual_seed(20260612)
VOCAB_M = 11
mistral_cfg = {
    'architectures': ['MistralForCausalLM'],
    'model_type': 'mistral',
    'hidden_size': D_MODEL,
    'intermediate_size': D_FF,
    'num_hidden_layers': N_LAYER,
    'num_attention_heads': N_HEAD,
    'num_key_value_heads': N_KV_HEAD,
    'head_dim': HEAD_DIM,
    'vocab_size': VOCAB_M,
    'max_position_embeddings': MAX_POS,
    'rms_norm_eps': 1e-5,
    'rope_theta': 10000.0,
    'sliding_window': SLIDING_WINDOW,
    'tie_word_embeddings': False,
    'hidden_act': 'silu',
}
mistral = MistralForCausalLM(
    MistralConfig(**mistral_cfg, attn_implementation='eager'))
seqs = dump(mistral, mistral_cfg, 'tests/fixtures/tiny_mistral', VOCAB_M)

# The sliding window MUST change the logits vs full attention on the same
# weights, otherwise the fixture would not test the window path at all.
full_cfg = dict(mistral_cfg)
full_cfg['sliding_window'] = None
full = MistralForCausalLM(
    MistralConfig(**full_cfg, attn_implementation='eager'))
full.load_state_dict(mistral.state_dict())
full = full.double().eval()
with torch.no_grad():
    lw = mistral(input_ids=torch.tensor([seqs[0]])).logits
    lf = full(input_ids=torch.tensor([seqs[0]])).logits
window_effect = (lw - lf).abs().max().item()
assert window_effect > 1e-3, \
    f'sliding window had no effect on the logits ({window_effect})'
print(f'sliding-window effect on logits: max |diff| = {window_effect:.4f}')

# ------------------------------ Qwen2 --------------------------------------
torch.manual_seed(20260613)
VOCAB_Q = 12
qwen2_cfg = {
    'architectures': ['Qwen2ForCausalLM'],
    'model_type': 'qwen2',
    'hidden_size': D_MODEL,
    'intermediate_size': D_FF,
    'num_hidden_layers': N_LAYER,
    'num_attention_heads': N_HEAD,
    'num_key_value_heads': N_KV_HEAD,
    'vocab_size': VOCAB_Q,
    'max_position_embeddings': MAX_POS,
    'rms_norm_eps': 1e-5,
    'rope_theta': 10000.0,
    'use_sliding_window': False,
    'tie_word_embeddings': False,
    'hidden_act': 'silu',
}
qwen2 = Qwen2ForCausalLM(
    Qwen2Config(**qwen2_cfg, attn_implementation='eager'))
# HF zero-inits the q/k/v biases; the fixture must carry NONZERO biases or
# the bias-loading path would be vacuously "verified".
with torch.no_grad():
    for layer in qwen2.model.layers:
        for proj in (layer.self_attn.q_proj, layer.self_attn.k_proj,
                     layer.self_attn.v_proj):
            proj.bias.normal_(0.0, 0.5)
seqs = dump(qwen2, qwen2_cfg, 'tests/fixtures/tiny_qwen2', VOCAB_Q)

# Zeroing the biases MUST change the logits.
nobias = Qwen2ForCausalLM(
    Qwen2Config(**qwen2_cfg, attn_implementation='eager'))
sd = {k: v.clone() for k, v in qwen2.state_dict().items()}
for k in sd:
    if k.endswith(('q_proj.bias', 'k_proj.bias', 'v_proj.bias')):
        sd[k].zero_()
nobias.load_state_dict(sd)
nobias = nobias.double().eval()
with torch.no_grad():
    lb = qwen2(input_ids=torch.tensor([seqs[0]])).logits
    l0 = nobias(input_ids=torch.tensor([seqs[0]])).logits
bias_effect = (lb - l0).abs().max().item()
assert bias_effect > 1e-3, \
    f'q/k/v biases had no effect on the logits ({bias_effect})'
print(f'qkv-bias effect on logits: max |diff| = {bias_effect:.4f}')
