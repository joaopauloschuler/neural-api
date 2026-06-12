#!/usr/bin/env python3
"""Generate a tiny RANDOM DistilBertForSequenceClassification parity
fixture for tests/TestNeuralPretrained.pas (no network access needed: the
model is randomly initialized from a pico config, never downloaded).

A SEPARATE script from tools/distilbert_tiny_fixture.py on purpose: the
landed tiny_distilbert.* fixtures stay byte-identical.

Fixture trio in tests/fixtures/:
  tiny_distilbert_seqcls.safetensors - a random fine-tuned-classifier-
      shaped checkpoint: the distilbert.* trunk plus the top-level
      pre_classifier.{weight,bias} ([dim, dim]) and
      classifier.{weight,bias} ([num_labels=3, dim]). NO pooler.
  tiny_distilbert_seqcls_config.json - model_type distilbert,
      architectures ['DistilBertForSequenceClassification'] (what
      BuildFromPretrained dispatches on) and id2label
      {0: neg, 1: neu, 2: pos}.
  tiny_distilbert_seqcls_logits.json - float64 oracle CLASS logits (HF
      transformers) for pinned sequences, plus the argmax label strings
      for the id2label round-trip assertion.

POOLING: HF DistilBertForSequenceClassification computes
  logits = classifier(dropout(ReLU(pre_classifier(hidden[:, 0]))))
i.e. pre_classifier dense + ReLU over the [CLS] hidden state at POSITION
0 - NO pooler (see transformers modeling_distilbert.py; dropout is
identity in eval). The Pascal net applies the head per token, so its ROW
0 must equal these logits. The JSON keeps an all-zero "token_types" key
so the shared (SeqLen,1,2)-input Pascal parity helper can be reused -
DistilBERT ignores that channel.

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/distilbert_seqcls_tiny_fixture.py
Needs torch + transformers + safetensors.
"""
import json

import torch
from safetensors.torch import save_file
from transformers import (DistilBertConfig,
                          DistilBertForSequenceClassification)

N_LAYER = 2
N_HEAD = 2
HEAD_DIM = 4
D_MODEL = N_HEAD * HEAD_DIM  # 8
INTERMEDIATE = 16
MAX_POS = 16
N_SEQUENCES = 3
VOCAB = 11
ID2LABEL = {0: 'neg', 1: 'neu', 2: 'pos'}

torch.manual_seed(20260612)

cfg_dict = {
    'architectures': ['DistilBertForSequenceClassification'],
    'model_type': 'distilbert',
    'dim': D_MODEL,
    'n_layers': N_LAYER,
    'n_heads': N_HEAD,
    'hidden_dim': INTERMEDIATE,
    'vocab_size': VOCAB,
    'max_position_embeddings': MAX_POS,
    'activation': 'gelu',
    'sinusoidal_pos_embds': False,
    'num_labels': len(ID2LABEL),
    'id2label': ID2LABEL,
    'label2id': {v: k for k, v in ID2LABEL.items()},
}
model = DistilBertForSequenceClassification(
    DistilBertConfig(**cfg_dict, attn_implementation='eager'))
# Same boosts as tools/distilbert_tiny_fixture.py (O(1) attention
# structure, FFN pre-activations where exact-vs-tanh GELU differs).
with torch.no_grad():
    for layer in model.distilbert.transformer.layer:
        layer.attention.q_lin.weight.normal_(0.0, 0.7)
        layer.attention.k_lin.weight.normal_(0.0, 0.7)
        layer.attention.v_lin.weight.normal_(0.0, 0.5)
        layer.ffn.lin1.weight.normal_(0.0, 1.3)
    # Boost pre_classifier too: a std-0.02 dense at this pico width feeds
    # the ReLU near-zero inputs and the kink would barely matter.
    model.pre_classifier.weight.normal_(0.0, 1.0)
model = model.double().eval()

sequences = [[(7 * i + 3 * s + s * s + 5) % VOCAB for i in range(MAX_POS)]
             for s in range(N_SEQUENCES)]
# No token-type embeddings: zeros, only so the shared Pascal parity
# helper (which feeds a (SeqLen,1,2) volume) can be reused.
token_types = [[0] * MAX_POS for _ in range(N_SEQUENCES)]


def classify_all():
    logits, labels, min_margin = [], [], float('inf')
    with torch.no_grad():
        for seq in sequences:
            out = model(input_ids=torch.tensor([seq]))
            row = out.logits[0]
            logits.append(row.tolist())
            labels.append(ID2LABEL[int(row.argmax())])
            top2 = torch.topk(row, 2).values
            min_margin = min(min_margin, float(top2[0] - top2[1]))
    return logits, labels, min_margin


# Deterministically scan classifier seeds for distinct argmax labels with
# top-2 margins far above the 1e-4 f32 parity gate (argmax robustness) -
# same trick as tools/bert_seqcls_tiny_fixture.py.
for cls_seed in range(1000, 1100):
    torch.manual_seed(cls_seed)
    with torch.no_grad():
        model.classifier.weight.normal_(0.0, 1.5)
        model.classifier.bias.normal_(0.0, 0.5)
    logits, labels, min_margin = classify_all()
    if len(set(labels)) >= 2 and min_margin > 1e-2:
        print(f'classifier seed {cls_seed}: labels={labels}, '
              f'min top-2 margin={min_margin:.3f}')
        break
else:
    raise AssertionError('no classifier seed gave distinct, well-separated '
                         'labels')

sd = {k: v.to(torch.float32).contiguous()
      for k, v in model.state_dict().items()}
assert not any('pooler' in k for k in sd), \
    'unexpected pooler tensors in a DistilBertForSequenceClassification ' \
    'export'
assert 'pre_classifier.weight' in sd and 'classifier.weight' in sd
save_file(sd, 'tests/fixtures/tiny_distilbert_seqcls.safetensors')
with open('tests/fixtures/tiny_distilbert_seqcls_config.json', 'w') as f:
    json.dump(cfg_dict, f, indent=1)
# The saved weights are f32-rounded; the f64 oracle must use EXACTLY them.
model.load_state_dict({k: v.to(torch.float64) for k, v in sd.items()})
logits, labels, min_margin = classify_all()
assert min_margin > 1e-2 and len(set(labels)) >= 2
with open('tests/fixtures/tiny_distilbert_seqcls_logits.json', 'w') as f:
    json.dump({'sequences': sequences, 'token_types': token_types,
               'logits': logits, 'labels': labels,
               'id2label': {str(k): v for k, v in ID2LABEL.items()}}, f)
print(f'wrote tiny_distilbert_seqcls.safetensors ({len(sd)} tensors) '
      f'+ config + logits; labels = {labels}')

# Self-checks: the logits must equal classifier(ReLU(pre_classifier(
# h[:, 0]))) - the [CLS]-position path - and the ReLU must really clip
# something (otherwise a tanh-headed importer bug could pass).
with torch.no_grad():
    seq = sequences[0]
    hidden = model.distilbert(
        input_ids=torch.tensor([seq])).last_hidden_state
    pre = model.pre_classifier(hidden[:, 0])
    assert float(pre.min()) < -1e-2, \
        f'ReLU never clips (min pre-activation {float(pre.min())})'
    manual = model.classifier(torch.relu(pre))[0]
    diff = (manual - torch.tensor(logits[0], dtype=torch.float64)) \
        .abs().max().item()
    assert diff < 1e-12, f'[CLS] pre_classifier self-check failed ({diff})'
    print(f'[CLS] pre_classifier+ReLU self-check: max |diff| = {diff:.2e}')
