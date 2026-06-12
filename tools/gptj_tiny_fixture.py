#!/usr/bin/env python3
"""Generate tiny RANDOM GPT-J parity fixtures for
tests/TestNeuralPretrained.pas (no network access needed: the model is
randomly initialized from a pico config, never downloaded).

Pinned in tests/fixtures/ (~15 KB total):

  tiny_gptj.safetensors + tiny_gptj_config.json + tiny_gptj_logits.json:
      GPTJForCausalLM with the quirks that distinguish GPT-J:
        - SHARED-LN parallel residual: ONE LayerNorm (ln_1) feeds BOTH the
          attention and the MLP branch, x := x + Attn(LN(x)) + MLP(LN(x))
          (unlike GPT-NeoX's two-LayerNorm parallel form);
        - PARTIAL rotary with the INTERLEAVED (GPT-J) pair layout:
          rotary_dim=4 of head_dim=8 -> RoPE on the first 4 dims of each
          q/k head, pairing (0,1),(2,3) - NO rotate_half permutation
          (unlike gpt_neox/llama);
        - SEPARATE bias-free q/k/v projections + bias-free out_proj;
        - lm_head WITH bias, UNTIED (tie_word_embeddings=false);
        - activation_function 'gelu_new' (the tanh approximation);
        - n_inner=24, deliberately NOT 4*hidden, to catch a hardcoded 4x.

Reference logits are computed by HF transformers in float64 (the oracle
convention of the committed fixtures). The script ASSERTS that the partial
rotary slice genuinely changes the logits (vs rotary_dim=head_dim and vs a
narrower slice), so the parity test pins the rotary_dim=4 wiring.

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/gptj_tiny_fixture.py
Needs torch + transformers + safetensors.
"""
import json

import torch
from safetensors.torch import save_file
from transformers import GPTJConfig, GPTJForCausalLM

N_LAYER = 2
N_HEAD = 2
HEAD_DIM = 8
D_MODEL = N_HEAD * HEAD_DIM  # 16
N_INNER = 24  # deliberately NOT 4*hidden
MAX_POS = 16
N_SEQUENCES = 3
VOCAB = 11
ROTARY_DIM = 4  # < head_dim 8 -> the partial-interleaved rotary slice

torch.manual_seed(20260612)

cfg_dict = {
    'architectures': ['GPTJForCausalLM'],
    'model_type': 'gptj',
    'n_embd': D_MODEL,
    'n_inner': N_INNER,
    'n_layer': N_LAYER,
    'n_head': N_HEAD,
    'vocab_size': VOCAB,
    'n_positions': MAX_POS,
    'rotary_dim': ROTARY_DIM,
    'layer_norm_epsilon': 1e-5,
    'activation_function': 'gelu_new',
    'tie_word_embeddings': False,
}


def make_model(overrides=None):
    d = dict(cfg_dict)
    if overrides:
        d.update(overrides)
    return GPTJForCausalLM(GPTJConfig(**d, attn_implementation='eager'))


model = make_model()
# HF inits with std 0.02; at pico width the attention scores are then ~0 and
# softmax is near-uniform, making RoPE (and the partial-rotary slice)
# numerically invisible. Boost q/k so the scores are O(1) and the rotary
# path genuinely matters.
with torch.no_grad():
    for layer in model.transformer.h:
        layer.attn.q_proj.weight.normal_(0.0, 0.5)
        layer.attn.k_proj.weight.normal_(0.0, 0.5)
model = model.double().eval()

sd = {k: v.to(torch.float32).contiguous()
      for k, v in model.state_dict().items()}
save_file(sd, 'tests/fixtures/tiny_gptj.safetensors')
with open('tests/fixtures/tiny_gptj_config.json', 'w') as f:
    json.dump(cfg_dict, f, indent=1)

sequences = [[(7 * i + 3 * s + s * s) % VOCAB for i in range(MAX_POS)]
             for s in range(N_SEQUENCES)]


def logits_of(m, seqs):
    with torch.no_grad():
        return [m(input_ids=torch.tensor([seq])).logits[0] for seq in seqs]


ref = logits_of(model, sequences)
with open('tests/fixtures/tiny_gptj_logits.json', 'w') as f:
    json.dump({'sequences': sequences,
               'logits': [l.tolist() for l in ref]}, f)
print(f'wrote tiny_gptj.safetensors ({len(sd)} tensors) '
      f'+ config + logits ({N_SEQUENCES} sequences of {MAX_POS})')

# ---- the PARTIAL rotary must matter: the same weights under a different
# rotary_dim must change the logits (otherwise the fixture would not pin
# the rotary_dim=4 slice).
for other_dim in (2, HEAD_DIM):
    other = make_model({'rotary_dim': other_dim})
    other.load_state_dict(model.state_dict())
    other = other.double().eval()
    effect = max((a - b).abs().max().item()
                 for a, b in zip(ref, logits_of(other, sequences)))
    assert effect > 1e-3, \
        f'rotary_dim {ROTARY_DIM} vs {other_dim} had no effect ({effect})'
    print(f'rotary_dim {ROTARY_DIM} vs {other_dim} effect on logits: '
          f'max |diff| = {effect:.4f}')

print('tensor names:')
for k in sorted(sd):
    print(' ', k, list(sd[k].shape))
