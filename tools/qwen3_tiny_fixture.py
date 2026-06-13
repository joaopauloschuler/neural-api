#!/usr/bin/env python3
"""Generate a tiny RANDOM Qwen3 parity fixture for
tests/TestNeuralPretrained.pas (no network access needed: the model is
randomly initialized from a pico config, never downloaded).

One fixture, ~5 KB, pinned in tests/fixtures/:

  tiny_qwen3.*: Qwen3ForCausalLM with the two Qwen3 deltas on the Qwen2
      path genuinely exercised:
      - DECOUPLED head_dim: head_dim=6 with hidden=8, heads=2, so
        num_heads*head_dim = 12 != hidden_size = 8 (the Qwen3-0.6B shape
        quirk; q_proj is [12, 8], o_proj is [8, 12]);
      - per-head q/k RMSNorm BEFORE RoPE: HF ones-inits the q_norm/k_norm
        gains, which would make the gain-loading path vacuous, so the
        script re-randomizes them and ASSERTS that resetting them back to
        ones changes the logits.
      GQA (1 kv head < 2 query heads) and tied embeddings are also on.

Reference logits are computed by HF transformers in float64 (the oracle
convention of the committed fixtures).

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/qwen3_tiny_fixture.py
writes tests/fixtures/tiny_qwen3{.safetensors,_config.json,_logits.json}.
Needs torch + transformers (>= the Qwen3 release) + safetensors.
"""
import json

import torch
from safetensors.torch import save_file
from transformers import Qwen3Config, Qwen3ForCausalLM

N_LAYER = 2
N_HEAD = 2
N_KV_HEAD = 1
HEAD_DIM = 6           # != D_MODEL // N_HEAD = 4: decoupled head_dim
D_MODEL = 8
D_FF = 12
MAX_POS = 16
N_SEQUENCES = 3
VOCAB = 13

torch.manual_seed(20260614)
qwen3_cfg = {
    'architectures': ['Qwen3ForCausalLM'],
    'model_type': 'qwen3',
    'hidden_size': D_MODEL,
    'intermediate_size': D_FF,
    'num_hidden_layers': N_LAYER,
    'num_attention_heads': N_HEAD,
    'num_key_value_heads': N_KV_HEAD,
    'head_dim': HEAD_DIM,
    'vocab_size': VOCAB,
    'max_position_embeddings': MAX_POS,
    'rms_norm_eps': 1e-5,
    'rope_theta': 10000.0,
    'attention_bias': False,
    'use_sliding_window': False,
    'tie_word_embeddings': True,
    'hidden_act': 'silu',
}
model = Qwen3ForCausalLM(Qwen3Config(**qwen3_cfg,
                                     attn_implementation='eager'))
assert N_HEAD * HEAD_DIM != D_MODEL, 'head_dim decoupling not exercised'

# HF ones-inits the q_norm/k_norm RMSNorm gains; the fixture must carry
# NON-ONE gains or the gain-loading path would be vacuously "verified".
with torch.no_grad():
    for layer in model.model.layers:
        layer.self_attn.q_norm.weight.normal_(1.0, 0.5)
        layer.self_attn.k_norm.weight.normal_(1.0, 0.5)

model = model.double().eval()
sd = {k: v.to(torch.float32).contiguous()
      for k, v in model.state_dict().items()}
save_file(sd, 'tests/fixtures/tiny_qwen3.safetensors')
with open('tests/fixtures/tiny_qwen3_config.json', 'w') as f:
    json.dump(qwen3_cfg, f, indent=1)
sequences = [[(7 * i + 3 * s + s * s) % VOCAB for i in range(MAX_POS)]
             for s in range(N_SEQUENCES)]
with torch.no_grad():
    logits = [model(input_ids=torch.tensor([seq])).logits[0].tolist()
              for seq in sequences]
with open('tests/fixtures/tiny_qwen3_logits.json', 'w') as f:
    json.dump({'sequences': sequences, 'logits': logits}, f)
print(f'wrote tiny_qwen3.safetensors ({len(sd)} tensors) '
      f'+ config + logits ({N_SEQUENCES} sequences of {MAX_POS})')

# Resetting the q/k norm gains to ones MUST change the logits, otherwise
# the fixture would not test the shared-gain loading path at all.
ones = Qwen3ForCausalLM(Qwen3Config(**qwen3_cfg,
                                    attn_implementation='eager'))
sd_ones = {k: v.clone() for k, v in model.state_dict().items()}
for k in sd_ones:
    if k.endswith(('q_norm.weight', 'k_norm.weight')):
        sd_ones[k] = torch.ones_like(sd_ones[k])
ones.load_state_dict(sd_ones)
ones = ones.double().eval()
with torch.no_grad():
    lg = model(input_ids=torch.tensor([sequences[0]])).logits
    l1 = ones(input_ids=torch.tensor([sequences[0]])).logits
norm_effect = (lg - l1).abs().max().item()
assert norm_effect > 1e-3, \
    f'q/k norm gains had no effect on the logits ({norm_effect})'
print(f'qk-norm gain effect on logits: max |diff| = {norm_effect:.4f}')
