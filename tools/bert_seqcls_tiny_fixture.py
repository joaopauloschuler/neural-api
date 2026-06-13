#!/usr/bin/env python3
"""Generate a tiny RANDOM BertForSequenceClassification parity fixture for
tests/TestNeuralPretrained.pas (no network access needed: the model is
randomly initialized from a pico config, never downloaded).

A SEPARATE script from tools/bert_tiny_fixture.py on purpose: the landed
tiny_bert.* fixtures stay byte-identical.

Fixture trio in tests/fixtures/:
  tiny_bert_seqcls.safetensors  - a random fine-tuned-classifier-shaped
      checkpoint: the bert.* trunk (INCLUDING bert.pooler.dense.*) plus the
      top-level classifier.{weight,bias} ([num_labels=3, hidden]).
  tiny_bert_seqcls_config.json  - model_type bert, architectures
      ['BertForSequenceClassification'] (what BuildFromPretrained
      dispatches on) and id2label {0: neg, 1: neu, 2: pos}.
  tiny_bert_seqcls_logits.json  - float64 oracle CLASS logits (HF
      transformers) for pinned sequences with nonzero token-type ids, plus
      the argmax label strings for the id2label round-trip assertion.

POOLING: HF BertForSequenceClassification computes
  logits = classifier(dropout(pooler(hidden[:, 0])))
i.e. the dense+tanh pooler over the [CLS] hidden state at POSITION 0 (see
transformers modeling_bert.py; dropout is identity in eval). The Pascal
net applies pooler+classifier per token, so its ROW 0 must equal these
logits. (GPT2ForSequenceClassification pools the LAST non-pad token
instead - see tools/gpt2_seqcls_tiny_fixture.py.)

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/bert_seqcls_tiny_fixture.py
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
N_SEQUENCES = 3
VOCAB = 11
TYPE_VOCAB = 2
ID2LABEL = {0: 'neg', 1: 'neu', 2: 'pos'}

torch.manual_seed(20260612)

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
    'num_labels': len(ID2LABEL),
    'id2label': ID2LABEL,
    'label2id': {v: k for k, v in ID2LABEL.items()},
}
model = BertForSequenceClassification(
    BertConfig(**cfg_dict, attn_implementation='eager'))
# Same boosts as tools/bert_tiny_fixture.py (O(1) attention structure,
# FFN pre-activations where exact-vs-tanh GELU differs) plus a boosted
# classifier so the class-logit margins are far above the parity gate.
with torch.no_grad():
    for layer in model.bert.encoder.layer:
        layer.attention.self.query.weight.normal_(0.0, 0.7)
        layer.attention.self.key.weight.normal_(0.0, 0.7)
        layer.attention.self.value.weight.normal_(0.0, 0.5)
        layer.intermediate.dense.weight.normal_(0.0, 1.3)
model = model.double().eval()

# Different pinned inputs than tiny_bert (offset 5 in the token recipe);
# nonzero token types so the segment branch shapes the logits too.
sequences = [[(7 * i + 3 * s + s * s + 5) % VOCAB for i in range(MAX_POS)]
             for s in range(N_SEQUENCES)]
token_types = [[1 if i >= 4 + 3 * s else 0 for i in range(MAX_POS)]
               for s in range(N_SEQUENCES)]


def classify_all():
    logits, labels, min_margin = [], [], float('inf')
    with torch.no_grad():
        for seq, tt in zip(sequences, token_types):
            out = model(input_ids=torch.tensor([seq]),
                        token_type_ids=torch.tensor([tt]))
            row = out.logits[0]
            logits.append(row.tolist())
            labels.append(ID2LABEL[int(row.argmax())])
            top2 = torch.topk(row, 2).values
            min_margin = min(min_margin, float(top2[0] - top2[1]))
    return logits, labels, min_margin


# [CLS] representations of same-length pico sequences are similar, so a
# single random classifier draw often maps every sequence to one class.
# Deterministically scan classifier seeds for distinct argmax labels with
# top-2 margins far above the 1e-4 f32 parity gate (argmax robustness).
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
save_file(sd, 'tests/fixtures/tiny_bert_seqcls.safetensors')
with open('tests/fixtures/tiny_bert_seqcls_config.json', 'w') as f:
    json.dump(cfg_dict, f, indent=1)
# The saved weights are f32-rounded; the f64 oracle must use EXACTLY them.
model.load_state_dict({k: v.to(torch.float64) for k, v in sd.items()})
logits, labels, min_margin = classify_all()
assert min_margin > 1e-2 and len(set(labels)) >= 2
with open('tests/fixtures/tiny_bert_seqcls_logits.json', 'w') as f:
    json.dump({'sequences': sequences, 'token_types': token_types,
               'logits': logits, 'labels': labels,
               'id2label': {str(k): v for k, v in ID2LABEL.items()}}, f)
print(f'wrote tiny_bert_seqcls.safetensors ({len(sd)} tensors) '
      f'+ config + logits; labels = {labels}')

# Self-check: the logits must equal classifier(pooler(h[:, 0])) - i.e. the
# [CLS]-position pooling path, NOT a mean/last-token pool.
with torch.no_grad():
    seq, tt = sequences[0], token_types[0]
    hidden = model.bert(input_ids=torch.tensor([seq]),
                        token_type_ids=torch.tensor([tt])).last_hidden_state
    pooled = torch.tanh(model.bert.pooler.dense(hidden[:, 0]))
    manual = model.classifier(pooled)[0]
    diff = (manual - torch.tensor(logits[0], dtype=torch.float64)) \
        .abs().max().item()
    assert diff < 1e-12, f'[CLS] pooling self-check failed ({diff})'
    print(f'[CLS]-pooling self-check: max |diff| = {diff:.2e}')
