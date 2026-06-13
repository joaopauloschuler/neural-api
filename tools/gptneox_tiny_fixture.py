#!/usr/bin/env python3
"""Generate tiny RANDOM GPT-NeoX (Pythia-architecture) parity fixtures for
tests/TestNeuralPretrained.pas (no network access needed: the model is
randomly initialized from a pico config, never downloaded).

Pinned in tests/fixtures/ (~15 KB total):

  tiny_gptneox.safetensors + tiny_gptneox_config.json +
  tiny_gptneox_logits.json:
      GPTNeoXForCausalLM with the quirks that distinguish GPT-NeoX:
        - PARALLEL residual with TWO LayerNorms
          (use_parallel_residual=true, the Pythia default):
          x := x + Attn(LN1(x)) + MLP(LN2(x));
        - PARTIAL rotary: rotary_pct=0.25 over head_dim=8 -> only the
          first 2 dims of each head get RoPE (rotate_half layout within
          the rotary slice);
        - ONE fused query_key_value projection with PER-HEAD
          [q|k|v] interleaving (view(.., heads, 3*head_dim));
        - UNTIED embed_in / embed_out (tie_word_embeddings=false);
        - hidden_act 'gelu' = the EXACT erf form;
        - intermediate_size=24, deliberately NOT 4*hidden, to catch a
          hardcoded 4x anywhere.

  tiny_gptneox_seq_config.json + tiny_gptneox_seq_logits.json:
      the SAME weights with use_parallel_residual=false (sequential
      residual) - reuses tiny_gptneox.safetensors, only config + reference
      logits differ.

Reference logits are computed by HF transformers in float64 (the oracle
convention of the committed fixtures). The script ASSERTS that the partial
rotary, the parallel-vs-sequential choice and RoPE itself all genuinely
change the logits, so the parity tests cover each ingredient.

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/gptneox_tiny_fixture.py
Needs torch + transformers + safetensors.
"""
import json

import torch
from safetensors.torch import save_file
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

N_LAYER = 2
N_HEAD = 2
HEAD_DIM = 8
D_MODEL = N_HEAD * HEAD_DIM  # 16
INTERMEDIATE = 24  # deliberately NOT 4*hidden
MAX_POS = 16
N_SEQUENCES = 3
VOCAB = 11
ROTARY_PCT = 0.25  # head_dim 8 -> rotary_ndims 2 (the Pythia fraction)

torch.manual_seed(20260612)

cfg_dict = {
    'architectures': ['GPTNeoXForCausalLM'],
    'model_type': 'gpt_neox',
    'hidden_size': D_MODEL,
    'intermediate_size': INTERMEDIATE,
    'num_hidden_layers': N_LAYER,
    'num_attention_heads': N_HEAD,
    'vocab_size': VOCAB,
    'max_position_embeddings': MAX_POS,
    'rotary_pct': ROTARY_PCT,
    'rotary_emb_base': 10000,
    'layer_norm_eps': 1e-5,
    'hidden_act': 'gelu',  # the EXACT erf form
    'use_parallel_residual': True,
    'tie_word_embeddings': False,
}


def make_model(overrides=None):
    d = dict(cfg_dict)
    if overrides:
        d.update(overrides)
    return GPTNeoXForCausalLM(GPTNeoXConfig(**d, attn_implementation='eager'))


model = make_model()
# HF inits with std 0.02; at pico width the attention scores are then ~0 and
# softmax is near-uniform, making RoPE (and the partial-rotary slice)
# numerically invisible. Boost q/k rows of the fused query_key_value so the
# scores are O(1) and the rotary path genuinely matters.
with torch.no_grad():
    for layer in model.gpt_neox.layers:
        w = layer.attention.query_key_value.weight  # [3*hidden, hidden]
        w3 = w.view(N_HEAD, 3, HEAD_DIM, D_MODEL)
        w3[:, 0].normal_(0.0, 0.5)  # q
        w3[:, 1].normal_(0.0, 0.5)  # k
model = model.double().eval()

sd = {k: v.to(torch.float32).contiguous()
      for k, v in model.state_dict().items()}
save_file(sd, 'tests/fixtures/tiny_gptneox.safetensors')
with open('tests/fixtures/tiny_gptneox_config.json', 'w') as f:
    json.dump(cfg_dict, f, indent=1)

sequences = [[(7 * i + 3 * s + s * s) % VOCAB for i in range(MAX_POS)]
             for s in range(N_SEQUENCES)]


def logits_of(m, seqs):
    with torch.no_grad():
        return [m(input_ids=torch.tensor([seq])).logits[0] for seq in seqs]


ref = logits_of(model, sequences)
with open('tests/fixtures/tiny_gptneox_logits.json', 'w') as f:
    json.dump({'sequences': sequences,
               'logits': [l.tolist() for l in ref]}, f)
print(f'wrote tiny_gptneox.safetensors ({len(sd)} tensors) '
      f'+ config + logits ({N_SEQUENCES} sequences of {MAX_POS})')

# ---- sequential variant: same weights, use_parallel_residual=false ----
seq_cfg = dict(cfg_dict)
seq_cfg['use_parallel_residual'] = False
seq_model = make_model({'use_parallel_residual': False})
seq_model.load_state_dict(model.state_dict())
seq_model = seq_model.double().eval()
seq_ref = logits_of(seq_model, sequences)
with open('tests/fixtures/tiny_gptneox_seq_config.json', 'w') as f:
    json.dump(seq_cfg, f, indent=1)
with open('tests/fixtures/tiny_gptneox_seq_logits.json', 'w') as f:
    json.dump({'sequences': sequences,
               'logits': [l.tolist() for l in seq_ref]}, f)
parallel_effect = max((a - b).abs().max().item()
                      for a, b in zip(ref, seq_ref))
assert parallel_effect > 1e-3, \
    f'parallel vs sequential residual had no effect ({parallel_effect})'
print(f'parallel-vs-sequential effect on logits: '
      f'max |diff| = {parallel_effect:.4f}')

# ---- the PARTIAL rotary must matter: rotary_pct=1.0 with the same weights
# must change the logits (otherwise the fixture would not pin the 0.25 slice).
full_rot = make_model({'rotary_pct': 1.0})
full_rot.load_state_dict(model.state_dict())
full_rot = full_rot.double().eval()
pct_effect = max((a - b).abs().max().item()
                 for a, b in zip(ref, logits_of(full_rot, sequences)))
assert pct_effect > 1e-3, \
    f'rotary_pct had no effect on the logits ({pct_effect})'
print(f'partial-rotary (0.25 vs 1.0) effect on logits: '
      f'max |diff| = {pct_effect:.4f}')

print('tensor names:')
for k in sorted(sd):
    print(' ', k, list(sd[k].shape))
