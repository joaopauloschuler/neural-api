#!/usr/bin/env python3
"""Generate a tiny RANDOM RobertaForSequenceClassification parity fixture
for tests/TestNeuralPretrained.pas (no network access needed: the model is
randomly initialized from a pico config, never downloaded).

A SEPARATE script from tools/roberta_tiny_fixture.py on purpose: the
landed tiny_roberta.* fixtures stay byte-identical.

Fixture trio in tests/fixtures/:
  tiny_roberta_seqcls.safetensors - a random fine-tuned-classifier-shaped
      checkpoint: the roberta.* trunk (NO pooler: HF builds the trunk with
      add_pooling_layer=False) plus the top-level
      classifier.dense.{weight,bias} ([hidden, hidden]) and
      classifier.out_proj.{weight,bias} ([num_labels=3, hidden]).
  tiny_roberta_seqcls_config.json - model_type roberta, architectures
      ['RobertaForSequenceClassification'] (what BuildFromPretrained
      dispatches on) and id2label {0: neg, 1: neu, 2: pos}.
  tiny_roberta_seqcls_logits.json - float64 oracle CLASS logits (HF
      transformers) for pinned sequences, plus the argmax label strings
      for the id2label round-trip assertion.

POOLING: HF RobertaForSequenceClassification computes
  logits = out_proj(dropout(tanh(dense(dropout(hidden[:, 0])))))
i.e. the RobertaClassificationHead (classifier.dense + tanh +
classifier.out_proj) over the <s> hidden state at POSITION 0 - NO pooler
(see transformers modeling_roberta.py; dropout is identity in eval). The
Pascal net applies the head per token, so its ROW 0 must equal these
logits. Sequences are MAX_POS-2 = 14 tokens (the full usable context of
the offset position table, see tools/roberta_tiny_fixture.py) with NO pad
token; "token_types" is zeros (type_vocab_size = 1).

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/roberta_seqcls_tiny_fixture.py
Needs torch + transformers + safetensors.
"""
import json

import torch
from safetensors.torch import save_file
from transformers import RobertaConfig, RobertaForSequenceClassification

N_LAYER = 2
N_HEAD = 2
HEAD_DIM = 4
D_MODEL = N_HEAD * HEAD_DIM  # 8
INTERMEDIATE = 16
MAX_POS = 16
PAD_ID = 1
OFFSET = PAD_ID + 1            # first usable position row = 2
SEQ_LEN = MAX_POS - OFFSET     # 14: the full usable context
N_SEQUENCES = 3
VOCAB = 11
ID2LABEL = {0: 'neg', 1: 'neu', 2: 'pos'}

torch.manual_seed(20260612)

cfg_dict = {
    'architectures': ['RobertaForSequenceClassification'],
    'model_type': 'roberta',
    'hidden_size': D_MODEL,
    'num_hidden_layers': N_LAYER,
    'num_attention_heads': N_HEAD,
    'intermediate_size': INTERMEDIATE,
    'vocab_size': VOCAB,
    'max_position_embeddings': MAX_POS,
    'type_vocab_size': 1,
    'pad_token_id': PAD_ID,
    'hidden_act': 'gelu',
    'layer_norm_eps': 1e-5,
    'num_labels': len(ID2LABEL),
    'id2label': ID2LABEL,
    'label2id': {v: k for k, v in ID2LABEL.items()},
}
model = RobertaForSequenceClassification(
    RobertaConfig(**cfg_dict, attn_implementation='eager'))
# Same boosts as tools/roberta_tiny_fixture.py, including the LOUD
# never-read padding-position rows 0/1.
with torch.no_grad():
    for layer in model.roberta.encoder.layer:
        layer.attention.self.query.weight.normal_(0.0, 0.7)
        layer.attention.self.key.weight.normal_(0.0, 0.7)
        layer.attention.self.value.weight.normal_(0.0, 0.5)
        layer.intermediate.dense.weight.normal_(0.0, 1.3)
    model.roberta.embeddings.position_embeddings.weight[:OFFSET] \
        .normal_(0.0, 10.0)
    # Boost classifier.dense so the tanh actually bends (a std-0.02 dense
    # at this pico width keeps tanh in its linear region and a ReLU-headed
    # importer bug could pass).
    model.classifier.dense.weight.normal_(0.0, 1.0)
model = model.double().eval()

# NO pad token (id 1) inside the sequences: HF would give it a padding
# position id and the consecutive-positions importer premise would break.
sequences = [[tid if tid != PAD_ID else (tid + 3) % VOCAB
              for tid in ((7 * i + 3 * s + s * s + 5) % VOCAB
                          for i in range(SEQ_LEN))]
             for s in range(N_SEQUENCES)]
# type_vocab_size = 1: token-type ids can only be 0 (the constant row).
token_types = [[0] * SEQ_LEN for _ in range(N_SEQUENCES)]


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


# Deterministically scan out_proj seeds for distinct argmax labels with
# top-2 margins far above the 1e-4 f32 parity gate (argmax robustness) -
# same trick as tools/bert_seqcls_tiny_fixture.py.
for cls_seed in range(1000, 1100):
    torch.manual_seed(cls_seed)
    with torch.no_grad():
        model.classifier.out_proj.weight.normal_(0.0, 1.5)
        model.classifier.out_proj.bias.normal_(0.0, 0.5)
    logits, labels, min_margin = classify_all()
    if len(set(labels)) >= 2 and min_margin > 1e-2:
        print(f'out_proj seed {cls_seed}: labels={labels}, '
              f'min top-2 margin={min_margin:.3f}')
        break
else:
    raise AssertionError('no out_proj seed gave distinct, well-separated '
                         'labels')

sd = {k: v.to(torch.float32).contiguous()
      for k, v in model.state_dict().items()}
assert not any('pooler' in k for k in sd), \
    'unexpected pooler tensors in a RobertaForSequenceClassification export'
assert 'classifier.dense.weight' in sd and 'classifier.out_proj.weight' in sd
save_file(sd, 'tests/fixtures/tiny_roberta_seqcls.safetensors')
with open('tests/fixtures/tiny_roberta_seqcls_config.json', 'w') as f:
    json.dump(cfg_dict, f, indent=1)
# The saved weights are f32-rounded; the f64 oracle must use EXACTLY them.
model.load_state_dict({k: v.to(torch.float64) for k, v in sd.items()})
logits, labels, min_margin = classify_all()
assert min_margin > 1e-2 and len(set(labels)) >= 2
with open('tests/fixtures/tiny_roberta_seqcls_logits.json', 'w') as f:
    json.dump({'sequences': sequences, 'token_types': token_types,
               'logits': logits, 'labels': labels,
               'id2label': {str(k): v for k, v in ID2LABEL.items()}}, f)
print(f'wrote tiny_roberta_seqcls.safetensors ({len(sd)} tensors) '
      f'+ config + logits; labels = {labels}')

# Self-checks: the logits must equal
# out_proj(tanh(dense(h[:, 0]))) - the <s>-position head path - the tanh
# must really saturate somewhere, and HF must use positions 2..SEQ_LEN+1.
with torch.no_grad():
    seq = sequences[0]
    ids = torch.tensor([seq])
    pos_ids = model.roberta.embeddings.create_position_ids_from_input_ids(
        ids, model.roberta.embeddings.padding_idx)
    assert pos_ids[0].tolist() == list(range(OFFSET, OFFSET + SEQ_LEN)), \
        f'HF position ids are not {OFFSET}..{OFFSET + SEQ_LEN - 1}'
    hidden = model.roberta(input_ids=ids).last_hidden_state
    pre = model.classifier.dense(hidden[:, 0])
    assert float(pre.abs().max()) > 1.0, \
        f'tanh never bends (max |pre-activation| {float(pre.abs().max())})'
    manual = model.classifier.out_proj(torch.tanh(pre))[0]
    diff = (manual - torch.tensor(logits[0], dtype=torch.float64)) \
        .abs().max().item()
    assert diff < 1e-12, f'<s> classification-head self-check failed ({diff})'
    print(f'<s> dense+tanh+out_proj self-check: max |diff| = {diff:.2e}')
