#!/usr/bin/env python3
"""Generate a tiny RANDOM Phi-3 parity fixture for
tests/TestNeuralPretrained.pas (no network access needed: the model is
randomly initialized from a pico config, never downloaded).

One fixture, ~40 KB, pinned in tests/fixtures/:

  tiny_phi3.*: Phi3ForCausalLM with every phi3 delta on the Llama path
      genuinely exercised:
      - FUSED projections: self_attn.qkv_proj.weight (q|k|v packed rows)
        and mlp.gate_up_proj.weight (gate|up packed) - the row-block
        slicing path, including the rotate_half permutation applied
        AFTER slicing;
      - GQA: 4 query heads sharing 2 kv heads (the Phi-4-mini shape at
        24q/8kv, scaled down);
      - PARTIAL rotary: partial_rotary_factor=0.75 with head_dim=8 ->
        rotary_dim=6 < 8 (the Phi-4-mini quirk; the script asserts the
        same weights under full rotary move the logits);
      - TIED embeddings (Phi-4-mini ties lm_head to embed_tokens);
      - sliding_window=4 SMALLER than the 16-token test sequences (the
        script asserts windowed logits differ from full attention).

Reference logits are computed by HF transformers in float64 (the oracle
convention of the committed fixtures).

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/phi3_tiny_fixture.py
writes tests/fixtures/tiny_phi3{.safetensors,_config.json,_logits.json}.
Needs torch + transformers (>= the Phi-3 release) + safetensors.
"""
import json

import torch
from safetensors.torch import save_file
from transformers import Phi3Config, Phi3ForCausalLM

N_LAYER = 2
N_HEAD = 4
N_KV_HEAD = 2
D_MODEL = 32           # head_dim = 8; rotary_dim = int(8 * 0.75) = 6
D_FF = 24
MAX_POS = 16
N_SEQUENCES = 3
VOCAB = 13
WINDOW = 4             # < MAX_POS: the sliding-window mask is exercised

torch.manual_seed(20260612)
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
    'rms_norm_eps': 1e-5,
    'rope_theta': 10000.0,
    'partial_rotary_factor': 0.75,
    'sliding_window': WINDOW,
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
    torch.manual_seed(20260612)  # identical weights for every variant
    return Phi3ForCausalLM(Phi3Config(**cfg, attn_implementation='eager'))


model = make_model()
# HF's std-0.02 init keeps a pico net almost linear; rescale every weight
# to O(1) so each delta (window, partial rotary, gate/up order) moves the
# logits well above the 1e-4 parity gate (the ModernBERT lesson).
with torch.no_grad():
    for p in model.parameters():
        if p.dim() >= 2:
            p.mul_(8.0)
sd_f32 = {k: v.to(torch.float32).contiguous()
          for k, v in model.state_dict().items()}
model = model.double().eval()

# Tied checkpoints do not serialize the redundant lm_head.weight (HF
# safetensors refuses shared tensors): drop it like the real exports do.
sd_saved = {k: v for k, v in sd_f32.items() if k != 'lm_head.weight'}
save_file(sd_saved, 'tests/fixtures/tiny_phi3.safetensors')
with open('tests/fixtures/tiny_phi3_config.json', 'w') as f:
    json.dump(phi3_cfg, f, indent=1)
sequences = [[(7 * i + 3 * s + s * s) % VOCAB for i in range(MAX_POS)]
             for s in range(N_SEQUENCES)]
with torch.no_grad():
    logits = [model(input_ids=torch.tensor([seq])).logits[0].tolist()
              for seq in sequences]
with open('tests/fixtures/tiny_phi3_logits.json', 'w') as f:
    json.dump({'sequences': sequences, 'logits': logits}, f)
print(f'wrote tiny_phi3.safetensors ({len(sd_f32)} tensors) '
      f'+ config + logits ({N_SEQUENCES} sequences of {MAX_POS})')


def logits_of_variant(cfg_overrides):
    m = make_model(cfg_overrides)
    m.load_state_dict(sd_f32)
    m = m.double().eval()
    with torch.no_grad():
        return m(input_ids=torch.tensor([sequences[0]])).logits


base = torch.tensor([logits[0]])

# (a) the sliding window MUST matter at these lengths, otherwise the
# fixture would not test the window convention at all.
window_effect = (base - logits_of_variant({'sliding_window': None})) \
    .abs().max().item()
assert window_effect > 1e-3, \
    f'sliding window had no effect on the logits ({window_effect})'
print(f'sliding-window effect on logits: max |diff| = {window_effect:.4f}')

# (b) partial rotary MUST matter: the same weights under full rotary
# (factor 1.0) have to move the logits, or the pass-through tail would be
# vacuously "verified".
rot_effect = (base - logits_of_variant({'partial_rotary_factor': 1.0})) \
    .abs().max().item()
assert rot_effect > 1e-3, \
    f'partial rotary had no effect on the logits ({rot_effect})'
print(f'partial-rotary effect on logits: max |diff| = {rot_effect:.4f}')
