#!/usr/bin/env python3
"""Generate a tiny RANDOM DeBERTa-v2/v3 parity fixture for
tests/TestNeuralPretrained.pas (no network access needed: the model is
randomly initialized from a pico config, never downloaded - the same
recipe as tools/modernbert_tiny_fixture.py / tools/bert_tiny_fixture.py).

Two fixtures, ~10 KB each, pinned in tests/fixtures/:

  tiny_debertav2.*: a config-faithful pico DebertaV2Model (model_type
      "deberta-v2") exercising the deberta-v3 family deltas:
      - DISENTANGLED ATTENTION: pos_att_type=["p2c","c2p"] (scale_factor 3),
        share_att_key=true, relative_attention=true. The score is
        content-to-content + content-to-position + position-to-content,
        the position terms projecting a SEPARATE rel_embeddings table
        (norm_rel_ebd="layer_norm") through the SAME query_proj/key_proj as
        content. position_buckets and max_relative_positions are set SMALL
        (so both the log-bucket clamp AND the bucketing genuinely bite over
        the pico seq len).
      - position_biased_input=false: NO absolute position table, NO
        token-type add (type_vocab_size=0); embedding path = word emb ->
        LayerNorm.
      - conv-after-embeddings ABSENT (conv_kernel_size unset).
      - exact erf GELU (hidden_act "gelu").
      Reference DebertaV2Model last_hidden_state in float64 under "hidden"
      (with token_types all-zero, like the BERT fixture).

  tiny_debertav2_seqcls.*: a DebertaV2ForSequenceClassification on the SAME
      trunk + a ContextPooler (dense+gelu on row 0) + classifier. Reference
      logits under "logits" so the test reuses the seq-cls scoring path.

HF _init_weights std=0.02 makes pico hidden states vacuously tiny (the
ModernBERT lesson) - re-randomize every carried weight at O(1) scale and
give the LayerNorms non-trivial gains/betas so a skipped gain breaks parity.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/deberta_v2_tiny_fixture.py
Needs torch + transformers + safetensors.
"""
import json

import torch
from safetensors.torch import save_file
from transformers import (DebertaV2Config, DebertaV2Model,
                          DebertaV2ForSequenceClassification)

N_LAYER = 3
N_HEAD = 2
D_MODEL = 8            # head_dim = 4
D_FF = 10
MAX_POS = 16
POS_BUCKETS = 4       # SMALL: log bucketing + clamp both bite over seq len
MAX_REL = 6
N_SEQUENCES = 3
VOCAB = 17
NUM_LABELS = 2

torch.manual_seed(20260613)
deberta_cfg = dict(
    architectures=['DebertaV2Model'],
    model_type='deberta-v2',
    hidden_size=D_MODEL,
    intermediate_size=D_FF,
    num_hidden_layers=N_LAYER,
    num_attention_heads=N_HEAD,
    vocab_size=VOCAB,
    max_position_embeddings=MAX_POS,
    layer_norm_eps=1e-7,
    relative_attention=True,
    position_buckets=POS_BUCKETS,
    max_relative_positions=MAX_REL,
    pos_att_type=['p2c', 'c2p'],
    share_att_key=True,
    norm_rel_ebd='layer_norm',
    position_biased_input=False,
    type_vocab_size=0,
    hidden_act='gelu',
    pad_token_id=0,
)


def rerandomize(model):
    """O(1)-scale weights so the disentangled attention is non-vacuous."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith('LayerNorm.weight') or name.endswith('.weight') \
                    and 'LayerNorm' in name:
                p.normal_(1.0, 0.4)
            elif name.endswith('LayerNorm.bias'):
                p.normal_(0.0, 0.3)
            elif 'rel_embeddings' in name:
                p.normal_(0.0, 1.0)
            elif name.endswith('word_embeddings.weight'):
                p.normal_(0.0, 1.0)
            elif name.endswith('.weight'):
                p.normal_(0.0, 0.5)
            elif name.endswith('.bias'):
                p.normal_(0.0, 0.2)


# ----------------------- base encoder fixture --------------------------
base = DebertaV2Model(DebertaV2Config(
    **{k: v for k, v in deberta_cfg.items() if k != 'architectures'}))
rerandomize(base)
base = base.double().eval()

sd = {k: v.to(torch.float32).contiguous() for k, v in base.state_dict().items()}
save_file(sd, 'tests/fixtures/tiny_debertav2.safetensors')
with open('tests/fixtures/tiny_debertav2_config.json', 'w') as f:
    json.dump(deberta_cfg, f, indent=1)

sequences = [[(5 * i + 2 * s + s * s) % VOCAB for i in range(MAX_POS)]
             for s in range(N_SEQUENCES)]
token_types = [[0] * MAX_POS for _ in range(N_SEQUENCES)]
with torch.no_grad():
    hidden = [base(input_ids=torch.tensor([seq])).last_hidden_state[0].tolist()
              for seq in sequences]
with open('tests/fixtures/tiny_debertav2_hidden.json', 'w') as f:
    json.dump({'sequences': sequences, 'token_types': token_types,
               'hidden': hidden}, f)
print(f'wrote tiny_debertav2.safetensors ({len(sd)} tensors) + hidden states')

# Non-vacuity: max abs hidden must be O(1), not ~1e-3.
ref = torch.tensor(hidden)
print(f'base hidden max |val| = {ref.abs().max().item():.4f}')
assert ref.abs().max().item() > 0.1, 'hidden states are vacuously tiny'

# Disentangled-attention non-vacuity: dropping the position terms must move
# the output (otherwise the c2p/p2c import path would vacuously verify).
plain_cfg = {k: v for k, v in deberta_cfg.items() if k != 'architectures'}
plain_cfg['pos_att_type'] = []
plain = DebertaV2Model(DebertaV2Config(**plain_cfg))
# load shared params (drop the per-layer pos-only tensors that don't exist
# when pos_att_type is empty - share_att_key reuses query/key so the state
# dicts match except the encoder rel_embeddings, which we keep).
common = {k: v for k, v in base.state_dict().items()
          if k in plain.state_dict()}
plain.load_state_dict(common, strict=False)
plain = plain.double().eval()
with torch.no_grad():
    alt = plain(input_ids=torch.tensor([sequences[0]])).last_hidden_state
eff = (torch.tensor(hidden[0]) - alt[0]).abs().max().item()
print(f'disentangled position terms: max |diff| vs no-pos = {eff:.4f}')
assert eff > 1e-2, 'position terms had no effect (vacuous fixture)'

# ----------------------- seq-cls head fixture --------------------------
torch.manual_seed(424242)
seqcls = DebertaV2ForSequenceClassification(DebertaV2Config(
    **{k: v for k, v in deberta_cfg.items() if k != 'architectures'},
    num_labels=NUM_LABELS))
# Reuse the base trunk weights, then re-randomize the head at O(1) scale.
seqcls.deberta.load_state_dict(base.state_dict())
with torch.no_grad():
    seqcls.pooler.dense.weight.normal_(0.0, 0.5)
    seqcls.pooler.dense.bias.normal_(0.0, 0.2)
    seqcls.classifier.weight.normal_(0.0, 0.5)
    seqcls.classifier.bias.normal_(0.0, 0.2)
seqcls = seqcls.double().eval()

sd2 = {k: v.to(torch.float32).contiguous()
       for k, v in seqcls.state_dict().items()}
save_file(sd2, 'tests/fixtures/tiny_debertav2_seqcls.safetensors')
cfg2 = {**deberta_cfg, 'architectures': ['DebertaV2ForSequenceClassification'],
        'num_labels': NUM_LABELS, 'id2label': {'0': 'NEG', '1': 'POS'},
        'label2id': {'NEG': 0, 'POS': 1}}
with open('tests/fixtures/tiny_debertav2_seqcls_config.json', 'w') as f:
    json.dump(cfg2, f, indent=1)
with torch.no_grad():
    logits = [seqcls(input_ids=torch.tensor([seq])).logits[0].tolist()
              for seq in sequences]
    id2label = {0: 'NEG', 1: 'POS'}
    labels = [id2label[int(torch.tensor(l).argmax())] for l in logits]
with open('tests/fixtures/tiny_debertav2_seqcls_logits.json', 'w') as f:
    json.dump({'sequences': sequences, 'token_types': token_types,
               'logits': logits, 'labels': labels}, f)
print(f'wrote tiny_debertav2_seqcls.safetensors ({len(sd2)} tensors) + logits')
print(f'seqcls logits[0] = {logits[0]}')
