#!/usr/bin/env python3
"""Generate a tiny RANDOM DistilBERT parity fixture for
tests/TestNeuralPretrained.pas (no network access needed: the model is
randomly initialized from a pico config, never downloaded).

One fixture, ~12 KB, pinned in tests/fixtures/:

  tiny_distilbert.*: DistilBertModel (model_type "distilbert") with the
      traits that distinguish DistilBERT from the landed BERT importer:
        - DIFFERENT tensor names: blocks live at transformer.layer.N. with
          attention.q_lin/k_lin/v_lin/out_lin, sa_layer_norm, ffn.lin1,
          ffn.lin2, output_layer_norm;
        - DIFFERENT config keys: n_layers, n_heads, dim, hidden_dim;
        - NO token-type (segment) embeddings and NO pooler;
        - otherwise the exact BERT post-LN math: word + learned-position
          embeddings, embedding LayerNorm (eps 1e-12, HARDCODED in
          modeling_distilbert), POST-LN blocks, BIDIRECTIONAL attention,
          biased nn.Linear everywhere, activation "gelu" = the EXACT erf
          form (the script asserts exact-vs-tanh GELU is visible).

The reference is computed by HF transformers in float64 (the oracle
convention of the committed fixtures): last_hidden_state for every
sequence. The JSON keeps a "token_types" key of ZEROS so the shared
Pascal parity helper (which feeds a (SeqLen,1,2) volume) can be reused -
DistilBERT ignores that channel.

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/distilbert_tiny_fixture.py
writes tests/fixtures/tiny_distilbert{.safetensors,_config.json,_hidden.json}.
Needs torch + transformers + safetensors.
"""
import json

import torch
from safetensors.torch import save_file
from transformers import DistilBertConfig, DistilBertModel

N_LAYER = 2
N_HEAD = 2
HEAD_DIM = 4
D_MODEL = N_HEAD * HEAD_DIM  # 8
INTERMEDIATE = 16
MAX_POS = 16
N_SEQUENCES = 3
VOCAB = 11

torch.manual_seed(20260612)

cfg_dict = {
    'architectures': ['DistilBertModel'],
    'model_type': 'distilbert',
    'dim': D_MODEL,
    'n_layers': N_LAYER,
    'n_heads': N_HEAD,
    'hidden_dim': INTERMEDIATE,
    'vocab_size': VOCAB,
    'max_position_embeddings': MAX_POS,
    'activation': 'gelu',
    'sinusoidal_pos_embds': False,
}
model = DistilBertModel(DistilBertConfig(**cfg_dict,
                                         attn_implementation='eager'))
# HF inits weights with std 0.02; at this pico width the attention scores
# are ~0 and softmax near-uniform. Boost q/k so the (non-causal) attention
# pattern is O(1)-structured, and the FFN input so exact-vs-tanh GELU is
# visible (same trick as tools/bert_tiny_fixture.py).
with torch.no_grad():
    for layer in model.transformer.layer:
        layer.attention.q_lin.weight.normal_(0.0, 0.7)
        layer.attention.k_lin.weight.normal_(0.0, 0.7)
        layer.attention.v_lin.weight.normal_(0.0, 0.5)
        layer.ffn.lin1.weight.normal_(0.0, 1.3)
model = model.double().eval()

sd = {k: v.to(torch.float32).contiguous()
      for k, v in model.state_dict().items()}
save_file(sd, 'tests/fixtures/tiny_distilbert.safetensors')
with open('tests/fixtures/tiny_distilbert_config.json', 'w') as f:
    json.dump(cfg_dict, f, indent=1)

sequences = [[(7 * i + 3 * s + s * s) % VOCAB for i in range(MAX_POS)]
             for s in range(N_SEQUENCES)]
# DistilBERT has no token-type embeddings: keep an all-zero "token_types"
# key so the shared (SeqLen,1,2)-input Pascal parity helper can be reused.
token_types = [[0] * MAX_POS for _ in range(N_SEQUENCES)]

hidden = []
with torch.no_grad():
    for seq in sequences:
        out = model(input_ids=torch.tensor([seq]))
        hidden.append(out.last_hidden_state[0].tolist())
with open('tests/fixtures/tiny_distilbert_hidden.json', 'w') as f:
    json.dump({'sequences': sequences, 'token_types': token_types,
               'hidden': hidden}, f)
print(f'wrote tiny_distilbert.safetensors ({len(sd)} tensors) '
      f'+ config + hidden states ({N_SEQUENCES} sequences of {MAX_POS})')

# ---- fixture self-checks: every quirk must be visible in the reference ----
expected_block_keys = {'attention.q_lin', 'attention.k_lin',
                       'attention.v_lin', 'attention.out_lin',
                       'sa_layer_norm', 'ffn.lin1', 'ffn.lin2',
                       'output_layer_norm'}
got_block_keys = {k[len('transformer.layer.0.'):].rsplit('.', 1)[0]
                  for k in sd if k.startswith('transformer.layer.0.')}
assert got_block_keys == expected_block_keys, got_block_keys
assert not any('token_type' in k or 'pooler' in k for k in sd), \
    'unexpected token-type/pooler tensors in a DistilBertModel export'
with torch.no_grad():
    base = model(input_ids=torch.tensor([sequences[0]]))
    # 1. bidirectionality must matter: changing the LAST token must move
    # position 0 (a causal importer would fail parity at position 0).
    perturbed = list(sequences[0])
    perturbed[-1] = (perturbed[-1] + 1) % VOCAB
    pert = model(input_ids=torch.tensor([perturbed]))
    causal_effect = (base.last_hidden_state[0, 0] -
                     pert.last_hidden_state[0, 0]).abs().max().item()
    assert causal_effect > 1e-3, \
        f'last token did not affect position 0 ({causal_effect})'
    print(f'bidirectional flow (last token -> position 0): '
          f'max |diff| = {causal_effect:.4f}')
    # 2. exact-vs-tanh GELU must differ in the reference beyond the 2e-5
    # Pascal parity gate (so a tanh-approximated importer fails).
    tanh_cfg = DistilBertConfig(
        **{**cfg_dict, 'activation': 'gelu_pytorch_tanh'},
        attn_implementation='eager')
    tanh_model = DistilBertModel(tanh_cfg)
    tanh_model.load_state_dict(model.state_dict())
    tanh_model = tanh_model.double().eval()
    tanh_out = tanh_model(input_ids=torch.tensor([sequences[0]]))
    gelu_effect = (base.last_hidden_state - tanh_out.last_hidden_state) \
        .abs().max().item()
    assert gelu_effect > 2.5e-5, \
        f'exact vs tanh GELU invisible in the fixture ({gelu_effect})'
    print(f'exact-vs-tanh GELU effect: max |diff| = {gelu_effect:.2e}')
