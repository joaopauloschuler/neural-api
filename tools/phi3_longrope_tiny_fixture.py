#!/usr/bin/env python3
"""Generate a tiny RANDOM Phi-3 LongRoPE parity fixture for
tests/TestNeuralPretrained.pas (no network: the model is randomly
initialized from a pico config, never downloaded).

This exercises the rope_scaling type "longrope" (Phi-3 / Phi-4-mini 128k
variants) import path on the Llama backbone:
  - rope_scaling = {"rope_type": "longrope", "short_factor": [...],
    "long_factor": [...], ...} with original_max_position_embeddings SMALLER
    than the test sequence length so HF's Phi3RotaryEmbedding selects the
    LONG factor table (long_inv_freq) at forward time;
  - a NON-TRIVIAL long_factor (> 1 per entry) so the per-frequency division
    visibly moves the logits versus unscaled RoPE;
  - the long attention scaling sqrt(1 + ln(factor)/ln(orig)) with
    factor = max_pos/orig (HF's default when no explicit attention_factor),
    applied to cos/sin.
head_dim = hidden/num_heads = 8, partial_rotary_factor = 0.75 ->
rotary_dim = 6 -> long_factor / short_factor have rotary_dim/2 = 3 entries.

Reference logits are computed by HF transformers in float64.

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/phi3_longrope_tiny_fixture.py
writes tests/fixtures/tiny_phi3_longrope{.safetensors,_config.json,
_logits.json}.

Coded by Claude (AI).
"""
import json
import math

import torch
from safetensors.torch import save_file
from transformers import Phi3Config, Phi3ForCausalLM

N_LAYER = 2
N_HEAD = 4
N_KV_HEAD = 2
D_MODEL = 32            # head_dim = 8; rotary_dim = int(8 * 0.75) = 6
D_FF = 24
ORIG_MAX_POS = 8       # pretraining context; SMALLER than the 16-token seqs
MAX_POS = 32           # extended (long) context
N_SEQUENCES = 3
SEQ_LEN = 16           # > ORIG_MAX_POS -> HF uses the LONG factor table
VOCAB = 13

# rotary_dim/2 = 3 entries; non-trivial (> 1) so the per-frequency division
# changes the angles meaningfully.
SHORT_FACTOR = [1.0, 1.0, 1.0]
LONG_FACTOR = [1.0, 1.3, 1.8]

torch.manual_seed(20260613)
phi3_cfg = {
    'architectures': ['Phi3ForCausalLM'],
    'model_type': 'phi3',
    'hidden_size': D_MODEL,
    'intermediate_size': D_FF,
    'num_hidden_layers': N_LAYER,
    'num_attention_heads': N_HEAD,
    'num_key_value_heads': N_KV_HEAD,
    'vocab_size': VOCAB,
    'max_position_embeddings': MAX_POS,
    'original_max_position_embeddings': ORIG_MAX_POS,
    'rms_norm_eps': 1e-5,
    'rope_theta': 10000.0,
    'partial_rotary_factor': 0.75,
    'rope_scaling': {
        'rope_type': 'longrope',
        'short_factor': SHORT_FACTOR,
        'long_factor': LONG_FACTOR,
    },
    'tie_word_embeddings': True,
    'hidden_act': 'silu',
    'bos_token_id': 1,
    'eos_token_id': 2,
    'pad_token_id': 0,
}


def make_model(cfg_overrides=None):
    cfg = dict(phi3_cfg)
    if cfg_overrides:
        cfg.update(cfg_overrides)
    torch.manual_seed(20260613)
    return Phi3ForCausalLM(Phi3Config(**cfg, attn_implementation='eager'))


model = make_model()
# HF's std-0.02 init keeps a pico net almost linear; rescale every weight to
# O(1) so the long-factor RoPE division moves the logits well above the 1e-4
# parity gate.
with torch.no_grad():
    for p in model.parameters():
        if p.dim() >= 2:
            p.mul_(8.0)
sd_f32 = {k: v.to(torch.float32).contiguous()
          for k, v in model.state_dict().items()}
model = model.double().eval()

# Tied checkpoints do not serialize the redundant lm_head.weight.
sd_saved = {k: v for k, v in sd_f32.items() if k != 'lm_head.weight'}
save_file(sd_saved, 'tests/fixtures/tiny_phi3_longrope.safetensors')
with open('tests/fixtures/tiny_phi3_longrope_config.json', 'w') as f:
    json.dump(phi3_cfg, f, indent=1)

sequences = [[(7 * i + 3 * s + s * s) % VOCAB for i in range(SEQ_LEN)]
             for s in range(N_SEQUENCES)]
with torch.no_grad():
    logits = [model(input_ids=torch.tensor([seq])).logits[0].tolist()
              for seq in sequences]
with open('tests/fixtures/tiny_phi3_longrope_logits.json', 'w') as f:
    json.dump({'sequences': sequences, 'logits': logits}, f)
print(f'wrote tiny_phi3_longrope.safetensors ({len(sd_saved)} tensors) '
      f'+ config + logits ({N_SEQUENCES} sequences of {SEQ_LEN})')

# Sanity: report the long attention scaling and assert HF picked the long
# table (the long factor MUST move the logits vs a short-only run, i.e. the
# 16-token sequence must trigger the long-table switch).
factor = MAX_POS / ORIG_MAX_POS
attn = math.sqrt(1 + math.log(factor) / math.log(ORIG_MAX_POS))
print(f'long attention scaling = {attn:.6f} (factor={factor})')

base = torch.tensor([logits[0]])


def logits_of(cfg_overrides):
    m = make_model(cfg_overrides)
    m.load_state_dict(sd_f32)
    m = m.double().eval()
    with torch.no_grad():
        return m(input_ids=torch.tensor([sequences[0]])).logits


# The long_factor MUST matter: replacing it with an all-ones table (pure
# RoPE on the long path, but attention scaling unchanged) has to move the
# logits, or the per-frequency division would be vacuously verified.
ones_cfg = {'rope_scaling': {'rope_type': 'longrope',
                             'short_factor': SHORT_FACTOR,
                             'long_factor': [1.0, 1.0, 1.0]}}
long_effect = (base - logits_of(ones_cfg)).abs().max().item()
assert long_effect > 1e-3, \
    f'long_factor had no effect on the logits ({long_effect})'
print(f'long_factor effect on logits: max |diff| = {long_effect:.4f}')
