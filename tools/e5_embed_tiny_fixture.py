#!/usr/bin/env python3
"""Generate a tiny RANDOM E5-style embedding parity fixture for
tests/TestNeuralPretrained.pas (no network access needed: the model is a
randomly initialized pico BertModel, never downloaded).

WHY SYNTHESIZE INSTEAD OF SLICE: the E5 / BGE retriever forward is just a
plain BertModel encoder followed by a POOLING + L2-NORMALIZE step. The
math is identical at any width, so a pico random BertModel exercises the
exact code path that a real intfloat/e5-small checkpoint would. We
synthesize (rather than download a multi-hundred-MB checkpoint) so the
fixture stays a few KB and the test runs in well under a second.

WHAT THIS PINS: the genuinely-deferred half of the text-embedding task -
that the imported encoder's MEAN-pooled + L2-NORMALIZED sentence vector
(the E5 recipe) matches the HF float64 oracle within 1e-4. sentence-
transformers is NOT required: for E5/BGE its SentenceTransformer is
exactly {AutoModel forward -> mean (E5) or CLS (BGE) pool -> L2 normalize}
in float64, which we reproduce here directly with transformers AutoModel.

The INSTRUCTION PREFIXES ("query: " / "passage: " for E5) change the
TOKEN IDS, not the pooling math. Since the committed fixture has no real
tokenizer.json, we BAKE the prefix into the token-id sequences here (the
prefix is represented by a couple of distinct leading token ids), and the
Pascal test feeds those ids straight through the imported net. The
prefix-string table itself (EmbedInstructionPrefix) is unit-tested
separately by TestEmbedInstructionPrefixTable.

emits, in tests/fixtures/:
  tiny_e5.safetensors      - pico BertModel weights (model_type "bert")
  tiny_e5_config.json      - the pico config
  tiny_e5_embed.json       - {pooling, sequences[], real_tokens[],
                             embeddings[]} float64 oracle: for each
                             sequence the mean-pool over its REAL tokens
                             then L2-normalize.

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/e5_embed_tiny_fixture.py
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
VOCAB = 13
TYPE_VOCAB = 2

torch.manual_seed(20260613)

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
    'hidden_act': 'gelu',  # exact erf form, like real E5
    'layer_norm_eps': 1e-12,
}
model = BertModel(BertConfig(**cfg_dict, attn_implementation='eager'))
# Boost q/k/v/ffn so the pico attention pattern is O(1)-structured (std-0.02
# init leaves attention near-uniform and the embedding nearly degenerate).
with torch.no_grad():
    for layer in model.encoder.layer:
        layer.attention.self.query.weight.normal_(0.0, 0.7)
        layer.attention.self.key.weight.normal_(0.0, 0.7)
        layer.attention.self.value.weight.normal_(0.0, 0.5)
        layer.intermediate.dense.weight.normal_(0.0, 1.3)
model = model.double().eval()

sd = {k: v.to(torch.float32).contiguous()
      for k, v in model.state_dict().items()
      if not k.startswith('pooler.')}  # encoder only, no pooler head
save_file(sd, 'tests/fixtures/tiny_e5.safetensors')
with open('tests/fixtures/tiny_e5_config.json', 'w') as f:
    json.dump(cfg_dict, f, indent=1)

# Two sentence "families": a query and two passages. Token ids 1..2 stand
# in for the baked instruction-prefix tokens ("query: " on the query,
# "passage: " on the passages) - distinct leading ids that shift the
# vector exactly as the real prefixes would. Each sequence is padded with
# id 0 ([PAD]) up to MAX_POS, and real_tokens records the unpadded length.
PAD = 0
raw = [
    # (label, prefix_ids, body_ids)
    ('query',    [1, 2], [5, 8, 3, 9, 4]),
    ('passage0', [1, 3], [5, 8, 3, 9, 4, 7]),     # near-paraphrase of query body
    ('passage1', [1, 3], [11, 6, 12, 10, 2, 8, 5]),  # unrelated
]
sequences = []
real_tokens = []
for _label, prefix, body in raw:
    ids = prefix + body
    real_tokens.append(len(ids))
    ids = ids + [PAD] * (MAX_POS - len(ids))
    sequences.append(ids)

embeddings = []
with torch.no_grad():
    for ids, n in zip(sequences, real_tokens):
        out = model(input_ids=torch.tensor([ids]),
                    token_type_ids=torch.zeros(1, MAX_POS, dtype=torch.long))
        h = out.last_hidden_state[0]            # (MAX_POS, D_MODEL) float64
        pooled = h[:n].mean(dim=0)              # mean over REAL tokens only
        pooled = pooled / pooled.norm()         # L2 normalize (E5 recipe)
        embeddings.append(pooled.tolist())

with open('tests/fixtures/tiny_e5_embed.json', 'w') as f:
    json.dump({'pooling': 'mean',
               'sequences': sequences,
               'real_tokens': real_tokens,
               'embeddings': embeddings}, f)

print(f'wrote tiny_e5.safetensors ({len(sd)} tensors) + config + '
      f'embed oracle ({len(sequences)} sequences of {MAX_POS})')

# ---- fixture self-checks ----
import math
# 1. embeddings are unit vectors.
for emb in embeddings:
    norm = math.sqrt(sum(x * x for x in emb))
    assert abs(norm - 1.0) < 1e-9, f'embedding not unit ({norm})'
# 2. query is closer to passage0 (shared body) than passage1 (unrelated):
#    a genuine retrieval signal the Pascal demo also exercises.
def cos(a, b):
    return sum(x * y for x, y in zip(a, b))
s0 = cos(embeddings[0], embeddings[1])
s1 = cos(embeddings[0], embeddings[2])
print(f'cos(query,passage0)={s0:.4f}  cos(query,passage1)={s1:.4f}')
assert s0 > s1, 'planted retrieval signal absent (passage0 must rank first)'
# 3. mean-pool genuinely differs from CLS-pool (so the pooling MODE matters).
with torch.no_grad():
    h0 = model(input_ids=torch.tensor([sequences[0]]),
               token_type_ids=torch.zeros(1, MAX_POS, dtype=torch.long)
               ).last_hidden_state[0]
cls = h0[0] / h0[0].norm()
mean = torch.tensor(embeddings[0], dtype=torch.float64)
poolgap = (cls - mean).abs().max().item()
assert poolgap > 1e-2, f'mean and CLS pooling indistinguishable ({poolgap})'
print(f'mean vs CLS pooling max |diff| = {poolgap:.4f} (pooling mode matters)')
