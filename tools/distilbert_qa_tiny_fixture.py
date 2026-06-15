#!/usr/bin/env python3
"""Generate a tiny RANDOM DistilBertForQuestionAnswering parity fixture for
tests/TestNeuralPretrained.pas (no network access: the model is randomly
initialized from a pico config, never downloaded).

The released extractive-QA checkpoints (distilbert-base-cased-distilled-
squad, deepset/roberta-base-squad2) are ~250MB - far too large to commit.
This builds a config-faithful pico DistilBertForQuestionAnswering with
re-randomized O(1)-scale weights and pins the HF float64 span logits, the
same recipe as tools/distilbert_seqcls_tiny_fixture.py.

Fixture trio in tests/fixtures/:
  tiny_distilbert_qa.safetensors - the distilbert.* trunk plus the
      top-level qa_outputs.{weight,bias} ([2, dim] / [2]) span head. No
      pooler, no pre_classifier.
  tiny_distilbert_qa_config.json - model_type distilbert,
      architectures ['DistilBertForQuestionAnswering'].
  tiny_distilbert_qa_logits.json - float64 oracle START and END logits
      (HF transformers) for pinned sequences, plus an all-zero
      "token_types" key so the shared (SeqLen,1,2) parity helper applies.

HF DistilBertForQuestionAnswering computes
  logits = qa_outputs(hidden)            # [B, T, 2]
  start_logits, end_logits = logits.split(1, dim=-1)
i.e. one [dim -> 2] Linear PER TOKEN, column 0 = start, column 1 = end.
The Pascal AddQuestionAnsweringHead emits (SeqLen,1,2) with the SAME
column convention, so its full per-position output must match these.

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/distilbert_qa_tiny_fixture.py
Needs torch + transformers + safetensors.
"""
import json

import torch
from safetensors.torch import save_file
from transformers import (DistilBertConfig,
                          DistilBertForQuestionAnswering)

N_LAYER = 2
N_HEAD = 2
HEAD_DIM = 4
D_MODEL = N_HEAD * HEAD_DIM  # 8
INTERMEDIATE = 16
MAX_POS = 16
N_SEQUENCES = 3
VOCAB = 11

torch.manual_seed(20260613)

cfg_dict = {
    'architectures': ['DistilBertForQuestionAnswering'],
    'model_type': 'distilbert',
    'dim': D_MODEL,
    'n_layers': N_LAYER,
    'n_heads': N_HEAD,
    'hidden_dim': INTERMEDIATE,
    'vocab_size': VOCAB,
    'max_position_embeddings': MAX_POS,
    'activation': 'gelu',
    'sinusoidal_pos_embds': False,
    'num_labels': 2,
}
model = DistilBertForQuestionAnswering(
    DistilBertConfig(**cfg_dict, attn_implementation='eager'))
# Same O(1) boosts as tools/distilbert_seqcls_tiny_fixture.py: a std-0.02
# init at this pico width is numerically vacuous and a head/trunk bug
# could hide under it.
with torch.no_grad():
    for layer in model.distilbert.transformer.layer:
        layer.attention.q_lin.weight.normal_(0.0, 0.7)
        layer.attention.k_lin.weight.normal_(0.0, 0.7)
        layer.attention.v_lin.weight.normal_(0.0, 0.5)
        layer.ffn.lin1.weight.normal_(0.0, 1.3)
    # Boost the span head so start != end and the two rows are clearly
    # distinct (a row0/row1 swap in the importer would then fail loudly).
    model.qa_outputs.weight.normal_(0.0, 1.2)
    model.qa_outputs.bias.normal_(0.0, 0.5)
model = model.double().eval()

sequences = [[(7 * i + 3 * s + s * s + 5) % VOCAB for i in range(MAX_POS)]
             for s in range(N_SEQUENCES)]
token_types = [[0] * MAX_POS for _ in range(N_SEQUENCES)]


def span_logits_all():
    starts, ends = [], []
    with torch.no_grad():
        for seq in sequences:
            out = model(input_ids=torch.tensor([seq]))
            starts.append(out.start_logits[0].tolist())
            ends.append(out.end_logits[0].tolist())
    return starts, ends


sd = {k: v.to(torch.float32).contiguous()
      for k, v in model.state_dict().items()}
assert not any('pooler' in k for k in sd), \
    'unexpected pooler tensors in a DistilBertForQuestionAnswering export'
assert 'qa_outputs.weight' in sd and 'qa_outputs.bias' in sd
assert tuple(sd['qa_outputs.weight'].shape) == (2, D_MODEL)
save_file(sd, 'tests/fixtures/tiny_distilbert_qa.safetensors')
with open('tests/fixtures/tiny_distilbert_qa_config.json', 'w') as f:
    json.dump(cfg_dict, f, indent=1)
# The saved weights are f32-rounded; the f64 oracle must use EXACTLY them.
model.load_state_dict({k: v.to(torch.float64) for k, v in sd.items()})
starts, ends = span_logits_all()
with open('tests/fixtures/tiny_distilbert_qa_logits.json', 'w') as f:
    json.dump({'sequences': sequences, 'token_types': token_types,
               'start_logits': starts, 'end_logits': ends}, f)
print(f'wrote tiny_distilbert_qa.safetensors ({len(sd)} tensors) '
      f'+ config + logits')

# Self-check: HF really splits qa_outputs columns into start/end, and the
# two columns differ (so a swap is detectable).
with torch.no_grad():
    seq = sequences[0]
    hidden = model.distilbert(
        input_ids=torch.tensor([seq])).last_hidden_state
    logits = model.qa_outputs(hidden)  # [1, T, 2]
    manual_start = logits[0, :, 0]
    manual_end = logits[0, :, 1]
    ds = (manual_start - torch.tensor(starts[0], dtype=torch.float64)) \
        .abs().max().item()
    de = (manual_end - torch.tensor(ends[0], dtype=torch.float64)) \
        .abs().max().item()
    assert ds < 1e-12 and de < 1e-12, \
        f'qa_outputs column-split self-check failed ({ds}, {de})'
    colgap = (manual_start - manual_end).abs().max().item()
    assert colgap > 1e-2, \
        f'start/end columns nearly identical ({colgap}) - swap undetectable'
    print(f'qa_outputs split self-check ok; start/end column gap = '
          f'{colgap:.3f}')
