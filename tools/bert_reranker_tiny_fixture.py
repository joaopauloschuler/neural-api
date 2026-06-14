#!/usr/bin/env python3
"""Generate a tiny RANDOM cross-encoder RERANKER parity fixture for
tests/TestNeuralPretrained.pas (no network: the model is randomly
initialized from a pico config, never downloaded).

A cross-encoder reranker (cross-encoder/ms-marco-MiniLM-L-6-v2,
BAAI/bge-reranker-base, ...) is a BertForSequenceClassification with
num_labels=1: it encodes a (query, document) PAIR JOINTLY as
  [CLS] query [SEP] document [SEP]
with token_type_ids 0 over "[CLS] query [SEP]" and 1 over "document [SEP]",
and emits ONE relevance logit from the [CLS] (row 0) position.

The thing UNDER TEST is the SEGMENT-id=1 path: the Pascal importer already
wires channel 1 of its (SeqLen,1,2) input into the token_type_embeddings
table, but until the reranker was added NOTHING exercised a per-position
segment id of 1 in a parity test. So this fixture pins a PAIR whose second
segment carries token_type id 1, and the oracle logit must match the
Pascal CrossEncoderScore [CLS] logit to < 1e-4.

Fixture trio in tests/fixtures/:
  tiny_bert_reranker.safetensors  - random num_labels=1 classifier-shaped
      checkpoint (bert.* trunk incl. bert.pooler.dense.* + classifier.{weight,
      bias} of shape [1, hidden]).
  tiny_bert_reranker_config.json  - model_type bert, architectures
      ['BertForSequenceClassification'], num_labels 1.
  tiny_bert_reranker_logits.json  - float64 oracle: the pinned PAIR's
      input_ids + token_type_ids (already laid out [CLS] q [SEP] d [SEP]),
      the raw [CLS] relevance logit, and its sigmoid.

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/bert_reranker_tiny_fixture.py
Needs torch + transformers + safetensors.
"""
import json

import torch
from safetensors.torch import save_file
from transformers import BertConfig, BertForSequenceClassification

N_LAYER = 2
N_HEAD = 2
HEAD_DIM = 4
D_MODEL = N_HEAD * HEAD_DIM  # 8
INTERMEDIATE = 16
MAX_POS = 16
VOCAB = 11
TYPE_VOCAB = 2

torch.manual_seed(20260614)

cfg_dict = {
    'architectures': ['BertForSequenceClassification'],
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
    'num_labels': 1,
    'id2label': {0: 'relevance'},
    'label2id': {'relevance': 0},
}
model = BertForSequenceClassification(
    BertConfig(**cfg_dict, attn_implementation='eager'))
# Same O(1) structure boosts as the other tiny BERT fixtures.
with torch.no_grad():
    for layer in model.bert.encoder.layer:
        layer.attention.self.query.weight.normal_(0.0, 0.7)
        layer.attention.self.key.weight.normal_(0.0, 0.7)
        layer.attention.self.value.weight.normal_(0.0, 0.5)
        layer.intermediate.dense.weight.normal_(0.0, 1.3)
    model.classifier.weight.normal_(0.0, 1.5)
    model.classifier.bias.normal_(0.0, 0.5)
    # Boost the token_type (segment) embeddings so segment id 1 visibly
    # shapes the logit - makes the "segment-1 path is live" self-check below
    # a comfortable margin above the 1e-4 parity gate, not a near-tie.
    model.bert.embeddings.token_type_embeddings.weight.normal_(0.0, 1.0)
model = model.double().eval()

# A pinned (query, doc) PAIR already tokenized + laid out as the Pascal
# BertTokenizePair would: [CLS] q... [SEP] d... [SEP], with token_type 0 on
# "[CLS] q [SEP]" and 1 on "d [SEP]". Ids are arbitrary in-vocab tokens;
# [CLS]=VOCAB-2, [SEP]=VOCAB-1 by this fixture's convention (only the
# token_type=1 path matters for the test, not the special-token identities).
CLS, SEP = VOCAB - 2, VOCAB - 1  # 9, 10
query_ids = [3, 7, 1]
doc_ids = [2, 5, 8, 4, 6]
input_ids = [CLS] + query_ids + [SEP] + doc_ids + [SEP]
token_type_ids = ([0] * (len(query_ids) + 2)) + ([1] * (len(doc_ids) + 1))
assert len(input_ids) == len(token_type_ids)
# pad the REST to MAX_POS so the saved fixture is full-length like the net.
pad_len = MAX_POS - len(input_ids)
assert pad_len >= 0
input_ids_full = input_ids + [0] * pad_len           # [PAD] = id 0
token_type_ids_full = token_type_ids + [0] * pad_len

sd = {k: v.to(torch.float32).contiguous()
      for k, v in model.state_dict().items()}
save_file(sd, 'tests/fixtures/tiny_bert_reranker.safetensors')
with open('tests/fixtures/tiny_bert_reranker_config.json', 'w') as f:
    json.dump(cfg_dict, f, indent=1)
# f32-rounded weights -> f64 oracle uses EXACTLY them.
model.load_state_dict({k: v.to(torch.float64) for k, v in sd.items()})

with torch.no_grad():
    out = model(input_ids=torch.tensor([input_ids_full]),
                token_type_ids=torch.tensor([token_type_ids_full]))
    logit = float(out.logits[0, 0])
    prob = float(torch.sigmoid(out.logits[0, 0]))

# Self-check: with token_type ALL ZERO the logit MUST differ (proves the
# segment-1 embedding actually shapes the output - else the test is vacuous).
with torch.no_grad():
    out0 = model(input_ids=torch.tensor([input_ids_full]),
                 token_type_ids=torch.zeros(1, MAX_POS, dtype=torch.long))
    logit0 = float(out0.logits[0, 0])
assert abs(logit - logit0) > 1e-3, (
    f'segment-1 path is vacuous: seg0 logit {logit0} == seg-mixed {logit}')

with open('tests/fixtures/tiny_bert_reranker_logits.json', 'w') as f:
    json.dump({'input_ids': input_ids_full,
               'token_type_ids': token_type_ids_full,
               'real_tokens': len(input_ids),
               'logit': logit, 'prob': prob,
               'logit_all_seg0': logit0}, f)
print(f'wrote tiny_bert_reranker.safetensors ({len(sd)} tensors) '
      f'+ config + logits')
print(f'[CLS] relevance logit (seg-mixed) = {logit:.6f}  prob = {prob:.6f}')
print(f'[CLS] logit if all-seg0           = {logit0:.6f}  '
      f'(delta {abs(logit - logit0):.4f} proves segment-1 path is live)')
