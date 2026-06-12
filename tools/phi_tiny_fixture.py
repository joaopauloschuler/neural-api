#!/usr/bin/env python3
"""Generate tiny RANDOM Phi parity fixtures for
tests/TestNeuralPretrained.pas (no network access needed: the model is
randomly initialized from a pico config, never downloaded).

Pinned in tests/fixtures/ (~17 KB total):

  tiny_phi.safetensors + tiny_phi_config.json + tiny_phi_logits.json:
      PhiForCausalLM (microsoft/phi-1 / phi-1_5 / phi-2 architecture) with
      the quirks that distinguish Phi:
        - SHARED-LN parallel residual: ONE input_layernorm feeds BOTH the
          attention and the MLP branch, x := x + Attn(LN(x)) + MLP(LN(x))
          (the GPT-J layout, but with biases everywhere);
        - PARTIAL rotary with the NeoX rotate_half pair layout:
          partial_rotary_factor=0.5 of head_dim=8 -> RoPE on the first 4
          dims of each q/k head, pairing (first half, second half) - so
          q/k rows need the rotate_half permutation at weight-load time
          (unlike GPT-J's interleaved layout);
        - SEPARATE q/k/v projections WITH bias + biased attn dense;
        - mlp fc1/fc2 WITH bias, hidden_act 'gelu_new';
        - final_layernorm + UNTIED lm_head WITH bias;
        - intermediate_size=24, deliberately NOT 4*hidden, to catch a
          hardcoded 4x.

Reference logits are computed by HF transformers in float64 (the oracle
convention of the committed fixtures). The script ASSERTS that the partial
rotary slice genuinely changes the logits (vs partial_rotary_factor=1.0
and vs a narrower slice), so the parity test pins the 0.5 wiring.

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/phi_tiny_fixture.py
Needs torch + transformers + safetensors.
"""
import json

import torch
from safetensors.torch import save_file
from transformers import PhiConfig, PhiForCausalLM

N_LAYER = 2
N_HEAD = 2
HEAD_DIM = 8
D_MODEL = N_HEAD * HEAD_DIM  # 16
INTERMEDIATE = 24  # deliberately NOT 4*hidden
MAX_POS = 16
N_SEQUENCES = 3
VOCAB = 11
PARTIAL_ROTARY = 0.5  # rotary_ndims = 4 of head_dim 8

torch.manual_seed(20260612)

cfg_dict = {
    'architectures': ['PhiForCausalLM'],
    'model_type': 'phi',
    'hidden_size': D_MODEL,
    'intermediate_size': INTERMEDIATE,
    'num_hidden_layers': N_LAYER,
    'num_attention_heads': N_HEAD,
    'vocab_size': VOCAB,
    'max_position_embeddings': MAX_POS,
    'partial_rotary_factor': PARTIAL_ROTARY,
    'rope_theta': 10000.0,
    'layer_norm_eps': 1e-5,
    'hidden_act': 'gelu_new',
    'qk_layernorm': False,
    'tie_word_embeddings': False,
}


def make_model(overrides=None):
    d = dict(cfg_dict)
    if overrides:
        d.update(overrides)
    return PhiForCausalLM(PhiConfig(**d, attn_implementation='eager'))


model = make_model()
# HF inits with std 0.02; at pico width the attention scores are then ~0 and
# softmax is near-uniform, making RoPE (and the partial-rotary slice)
# numerically invisible. Boost q/k so the scores are O(1) and the rotary
# path genuinely matters.
with torch.no_grad():
    for layer in model.model.layers:
        layer.self_attn.q_proj.weight.normal_(0.0, 0.5)
        layer.self_attn.k_proj.weight.normal_(0.0, 0.5)
        layer.self_attn.q_proj.bias.normal_(0.0, 0.2)
        layer.self_attn.k_proj.bias.normal_(0.0, 0.2)
model = model.double().eval()

sd = {k: v.to(torch.float32).contiguous()
      for k, v in model.state_dict().items()}
save_file(sd, 'tests/fixtures/tiny_phi.safetensors')
with open('tests/fixtures/tiny_phi_config.json', 'w') as f:
    json.dump(cfg_dict, f, indent=1)

sequences = [[(7 * i + 3 * s + s * s) % VOCAB for i in range(MAX_POS)]
             for s in range(N_SEQUENCES)]


def logits_of(m, seqs):
    with torch.no_grad():
        return [m(input_ids=torch.tensor([seq])).logits[0] for seq in seqs]


ref = logits_of(model, sequences)
with open('tests/fixtures/tiny_phi_logits.json', 'w') as f:
    json.dump({'sequences': sequences,
               'logits': [l.tolist() for l in ref]}, f)
print(f'wrote tiny_phi.safetensors ({len(sd)} tensors) '
      f'+ config + logits ({N_SEQUENCES} sequences of {MAX_POS})')

# ---- the PARTIAL rotary must matter: the same weights under a different
# partial_rotary_factor must change the logits (otherwise the fixture would
# not pin the 0.5 slice).
for other_factor in (0.25, 1.0):
    other = make_model({'partial_rotary_factor': other_factor})
    other.load_state_dict(model.state_dict())
    other = other.double().eval()
    effect = max((a - b).abs().max().item()
                 for a, b in zip(ref, logits_of(other, sequences)))
    assert effect > 1e-3, (f'partial_rotary_factor {PARTIAL_ROTARY} vs '
                           f'{other_factor} had no effect ({effect})')
    print(f'partial_rotary_factor {PARTIAL_ROTARY} vs {other_factor} '
          f'effect on logits: max |diff| = {effect:.4f}')

print('tensor names:')
for k in sorted(sd):
    print(' ', k, list(sd[k].shape))
