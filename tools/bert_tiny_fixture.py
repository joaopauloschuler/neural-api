#!/usr/bin/env python3
"""Generate a tiny RANDOM BERT parity fixture for
tests/TestNeuralPretrained.pas (no network access needed: the model is
randomly initialized from a pico config, never downloaded).

One fixture, ~15 KB, pinned in tests/fixtures/:

  tiny_bert.*: BertModel (the plain encoder, model_type "bert") with the
      traits that distinguish the encoder family from the landed decoders:
        - learned absolute position embeddings + token-type (segment)
          embeddings + word embeddings, summed, then an embedding
          LayerNorm (eps 1e-12, the BERT default);
        - POST-LN blocks: x := LN(x + Attn(x)); x := LN(x + FFN(x));
        - BIDIRECTIONAL (non-causal) self-attention - the script ASSERTS
          that a causal mask would change the hidden states, so the
          parity test genuinely covers the non-causal path;
        - nn.Linear ([out, in]) with biases EVERYWHERE (q/k/v, attention
          output, both FFN linears);
        - hidden_act "gelu" - the EXACT erf form, NOT the tanh
          approximation (the script asserts the two differ here);
        - a pooler head: pooler_output = tanh(dense(h[CLS])).

The reference is computed by HF transformers in float64 (the oracle
convention of the committed fixtures): the final hidden states
(last_hidden_state) for every sequence plus the pooler output. The test
inputs USE nonzero token_type ids (a segment boundary mid-sequence, at a
different position per sequence) and the script ASSERTS the token-type
embeddings affect the hidden states.

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/bert_tiny_fixture.py
writes tests/fixtures/tiny_bert{.safetensors,_config.json,_hidden.json}.
Needs torch + transformers + safetensors.
"""
import json

import torch
from safetensors.torch import save_file
from transformers import BertConfig, BertModel

N_LAYER = 2
N_HEAD = 2
HEAD_DIM = 4
D_MODEL = N_HEAD * HEAD_DIM  # 8
INTERMEDIATE = 16
MAX_POS = 16
N_SEQUENCES = 3
VOCAB = 11
TYPE_VOCAB = 2

torch.manual_seed(20260612)

cfg_dict = {
    'architectures': ['BertModel'],
    'model_type': 'bert',
    'hidden_size': D_MODEL,
    'num_hidden_layers': N_LAYER,
    'num_attention_heads': N_HEAD,
    'intermediate_size': INTERMEDIATE,
    'vocab_size': VOCAB,
    'max_position_embeddings': MAX_POS,
    'type_vocab_size': TYPE_VOCAB,
    'hidden_act': 'gelu',
    'layer_norm_eps': 1e-12,
}
model = BertModel(BertConfig(**cfg_dict, attn_implementation='eager'))
# HF inits weights with std 0.02; at this pico width the attention scores
# are ~0 and softmax near-uniform. Boost q/k so the (non-causal) attention
# pattern is O(1)-structured and the bidirectional path genuinely matters.
with torch.no_grad():
    for layer in model.encoder.layer:
        layer.attention.self.query.weight.normal_(0.0, 0.7)
        layer.attention.self.key.weight.normal_(0.0, 0.7)
        layer.attention.self.value.weight.normal_(0.0, 0.5)
        # FFN pre-activations of O(1-2): the region where exact (erf) and
        # tanh-approximated GELU genuinely differ (~3e-4), so the parity
        # test discriminates the two (self-check 3 below).
        layer.intermediate.dense.weight.normal_(0.0, 1.3)
model = model.double().eval()

sd = {k: v.to(torch.float32).contiguous()
      for k, v in model.state_dict().items()}
save_file(sd, 'tests/fixtures/tiny_bert.safetensors')
with open('tests/fixtures/tiny_bert_config.json', 'w') as f:
    json.dump(cfg_dict, f, indent=1)

sequences = [[(7 * i + 3 * s + s * s) % VOCAB for i in range(MAX_POS)]
             for s in range(N_SEQUENCES)]
# Nonzero token-type usage: segment boundary mid-sequence, moved per
# sequence (positions >= 5+2s belong to segment B).
token_types = [[1 if i >= 5 + 2 * s else 0 for i in range(MAX_POS)]
               for s in range(N_SEQUENCES)]

hidden = []
pooler = []
with torch.no_grad():
    for seq, tt in zip(sequences, token_types):
        out = model(input_ids=torch.tensor([seq]),
                    token_type_ids=torch.tensor([tt]))
        hidden.append(out.last_hidden_state[0].tolist())
        pooler.append(out.pooler_output[0].tolist())
with open('tests/fixtures/tiny_bert_hidden.json', 'w') as f:
    json.dump({'sequences': sequences, 'token_types': token_types,
               'hidden': hidden, 'pooler': pooler}, f)
print(f'wrote tiny_bert.safetensors ({len(sd)} tensors) '
      f'+ config + hidden states ({N_SEQUENCES} sequences of {MAX_POS})')

# ---- fixture self-checks: every quirk must be visible in the reference ----
with torch.no_grad():
    base = model(input_ids=torch.tensor([sequences[0]]),
                 token_type_ids=torch.tensor([token_types[0]]))
    # 1. token-type embeddings must matter (all-zero segments => different).
    flat = model(input_ids=torch.tensor([sequences[0]]),
                 token_type_ids=torch.zeros(1, MAX_POS, dtype=torch.long))
    tt_effect = (base.last_hidden_state - flat.last_hidden_state) \
        .abs().max().item()
    assert tt_effect > 1e-3, f'token_type had no effect ({tt_effect})'
    print(f'token-type effect on hidden states: max |diff| = {tt_effect:.4f}')
    # 2. bidirectionality must matter: in a CAUSAL model, changing the LAST
    # token cannot move the hidden state at position 0. Assert it does here,
    # so an importer that wrongly applied a causal mask fails the parity test
    # at position 0 already.
    perturbed = list(sequences[0])
    perturbed[-1] = (perturbed[-1] + 1) % VOCAB
    pert_out = model(input_ids=torch.tensor([perturbed]),
                     token_type_ids=torch.tensor([token_types[0]]))
    causal_effect = (base.last_hidden_state[0, 0] -
                     pert_out.last_hidden_state[0, 0]).abs().max().item()
    assert causal_effect > 1e-3, \
        f'last token did not affect position 0 ({causal_effect})'
    print(f'bidirectional flow (last token -> position 0): '
          f'max |diff| = {causal_effect:.4f}')
    # 3. exact-vs-tanh GELU must differ in the reference (so the parity
    # test catches an importer that used the tanh approximation).
    tanh_cfg = BertConfig(**{**cfg_dict, 'hidden_act': 'gelu_pytorch_tanh'},
                          attn_implementation='eager')
    tanh_model = BertModel(tanh_cfg)
    tanh_model.load_state_dict(model.state_dict())
    tanh_model = tanh_model.double().eval()
    tanh_out = tanh_model(input_ids=torch.tensor([sequences[0]]),
                          token_type_ids=torch.tensor([token_types[0]]))
    gelu_effect = (base.last_hidden_state - tanh_out.last_hidden_state) \
        .abs().max().item()
    # The Pascal parity gate is 2e-5: a tanh-approximated importer would
    # show this full effect and fail it.
    assert gelu_effect > 2.5e-5, \
        f'exact vs tanh GELU invisible in the fixture ({gelu_effect})'
    print(f'exact-vs-tanh GELU effect: max |diff| = {gelu_effect:.2e}')
