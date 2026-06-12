#!/usr/bin/env python3
"""Generate a tiny RANDOM GPT2ForSequenceClassification parity fixture for
tests/TestNeuralPretrained.pas (no network access needed: the model is
randomly initialized from a pico config, never downloaded).

A SEPARATE script from tools/gpt2_tiny_fixture.py on purpose: the landed
tiny_gpt2.* fixtures stay byte-identical. Unlike that stdlib-only script,
this one uses HF transformers as the float64 oracle (the convention of the
newer committed fixtures).

Fixture trio in tests/fixtures/:
  tiny_gpt2_seqcls.safetensors  - a random fine-tuned-classifier-shaped
      checkpoint: the transformer.* trunk plus the top-level score.weight
      ([num_labels=3, n_embd]; NO bias - HF defines none). No lm_head.
  tiny_gpt2_seqcls_config.json  - model_type gpt2 (with n_head, which the
      checkpoint shapes cannot reveal), architectures
      ['GPT2ForSequenceClassification'] (what BuildFromPretrained
      dispatches on) and id2label {0: neg, 1: neu, 2: pos}.
  tiny_gpt2_seqcls_logits.json  - float64 oracle CLASS logits for pinned
      full-length sequences, the pinned pooled_position used for
      last-token pooling, and the argmax label strings.

POOLING: HF GPT2ForSequenceClassification applies score to EVERY hidden
state and returns the logits at the LAST NON-PAD position (the causal
trunk only lets information flow left-to-right, so only that position has
seen the whole text) - NOT the [CLS]/position-0 pooling of
BertForSequenceClassification (see tools/bert_seqcls_tiny_fixture.py).
The pinned sequences are full-length with no pad tokens (pad_token_id=0,
tokens drawn from 1..vocab-1), so pooled_position = seq_len - 1 = 15; the
script ASSERTS HF pooled exactly there and that position-0 pooling would
give different logits.

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/gpt2_seqcls_tiny_fixture.py
Needs torch + transformers + safetensors.
"""
import json

import torch
from safetensors.torch import save_file
from transformers import GPT2Config, GPT2ForSequenceClassification

N_LAYER = 2
N_HEAD = 2
N_EMBD = 8
N_CTX = 16
N_SEQUENCES = 3
VOCAB = 11
PAD_TOKEN_ID = 0  # never used in the pinned sequences
ID2LABEL = {0: 'neg', 1: 'neu', 2: 'pos'}

torch.manual_seed(20260612)

cfg_dict = {
    'architectures': ['GPT2ForSequenceClassification'],
    'model_type': 'gpt2',
    'n_layer': N_LAYER,
    'n_head': N_HEAD,
    'n_embd': N_EMBD,
    'n_positions': N_CTX,
    'vocab_size': VOCAB,
    'pad_token_id': PAD_TOKEN_ID,
    'num_labels': len(ID2LABEL),
    'id2label': ID2LABEL,
    'label2id': {v: k for k, v in ID2LABEL.items()},
}
model = GPT2ForSequenceClassification(
    GPT2Config(**cfg_dict, attn_implementation='eager'))
# Boost q/k (structured attention at pico width, as in the other fixture
# generators) and the score head (clear class-logit margins).
with torch.no_grad():
    for block in model.transformer.h:
        w = block.attn.c_attn.weight  # HF Conv1D [in, out] = [d, 3d]
        w[:, :2 * N_EMBD].normal_(0.0, 0.7)   # q and k columns
        w[:, 2 * N_EMBD:].normal_(0.0, 0.4)   # v columns
    model.score.weight.normal_(0.0, 0.9)
model = model.double().eval()

sd = {k: v.to(torch.float32).contiguous()
      for k, v in model.state_dict().items()}
save_file(sd, 'tests/fixtures/tiny_gpt2_seqcls.safetensors')
with open('tests/fixtures/tiny_gpt2_seqcls_config.json', 'w') as f:
    json.dump(cfg_dict, f, indent=1)

# Full-length pinned sequences, tokens 1..VOCAB-1 (no pad anywhere).
sequences = [[1 + (5 * i + 2 * s + s * s) % (VOCAB - 1)
              for i in range(N_CTX)] for s in range(N_SEQUENCES)]
POOLED_POSITION = N_CTX - 1

logits = []
labels = []
with torch.no_grad():
    for seq in sequences:
        ids = torch.tensor([seq])
        out = model(input_ids=ids)
        row = out.logits[0]
        # Self-check 1: HF pooled at the LAST position - score applied to
        # the final hidden state must reproduce out.logits exactly.
        hidden = model.transformer(input_ids=ids).last_hidden_state
        manual = model.score(hidden[0, POOLED_POSITION])
        assert (manual - row).abs().max().item() < 1e-12, \
            'HF did not pool at the last position'
        # Self-check 2: position-0 pooling (the BERT convention) must give
        # VISIBLY different logits, so the parity test discriminates the
        # two pooling conventions.
        first = model.score(hidden[0, 0])
        pool_diff = (first - row).abs().max().item()
        assert pool_diff > 1e-2, \
            f'first-vs-last pooling indistinguishable ({pool_diff})'
        logits.append(row.tolist())
        labels.append(ID2LABEL[int(row.argmax())])
        top2 = torch.topk(row, 2).values
        margin = float(top2[0] - top2[1])
        assert margin > 1e-2, f'top-2 logit margin too small ({margin})'
with open('tests/fixtures/tiny_gpt2_seqcls_logits.json', 'w') as f:
    json.dump({'sequences': sequences, 'pooled_position': POOLED_POSITION,
               'logits': logits, 'labels': labels,
               'id2label': {str(k): v for k, v in ID2LABEL.items()}}, f)
print(f'wrote tiny_gpt2_seqcls.safetensors ({len(sd)} tensors) '
      f'+ config + logits; pooled_position={POOLED_POSITION}, '
      f'labels = {labels}')
assert len(set(labels)) >= 2, 'all sequences got the same class'
