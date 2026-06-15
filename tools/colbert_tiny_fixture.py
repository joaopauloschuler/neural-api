#!/usr/bin/env python3
"""Generate a tiny RANDOM ColBERT late-interaction parity fixture for
tests/TestNeuralPretrained.pas (no network access: a pico BertModel is
randomly initialized from a config, never downloaded - the established
fixture convention).

ColBERT (colbert-ir/colbertv2.0) = the stock BERT encoder + a single EXTRA
head: a [hidden -> dim] dense with NO bias (the "linear" tensor, dim 128 in
the released checkpoints; we use a small dim here). Each contextual token
hidden state is projected and L2-normalized; there is NO pooling. A
(query, doc) pair is scored by MaxSim late interaction
    score = sum_{q in query} max_{d in doc} <E_q, E_d>.

This synthesizes a config-faithful PICO checkpoint (BertModel weights +
linear.weight) and computes the HF-side oracle in float64:
  - the per-token projected + L2-normalized query matrix (ALL SeqLen rows,
    because ColBERT pads queries with [MASK] and those rows DO count),
  - the per-token projected + L2-normalized doc matrix (the non-[PAD]
    prefix only),
  - the MaxSim score between them.
The committed safetensors stores the f32-rounded weights; the Pascal
importer + ColBERTEmbedTokens/ColBERTMaxSimScore must match the float64
reference within 1e-4.

We SYNTHESIZE the checkpoint (no real colbertv2.0 download): the real
checkpoint is ~440 MB and the parity logic is identical at any width. The
test exercises the exact ColBERT forward (BERT encoder + bias-free
projection + per-row L2 norm + MaxSim), which is what the fixture must pin.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/colbert_tiny_fixture.py
writes tests/fixtures/tiny_colbert{.safetensors,_config.json,_score.json}.
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
PROJ_DIM = 5  # ColBERT projection dim (128 in real checkpoints)
SEQ_LEN = 12  # net context (queries are [MASK]-padded to this length)

# bert-base-uncased marker convention reproduced at pico scale:
#   [CLS]=10 [SEP]=11 [MASK]=12 [PAD]=0 [Q]=[unused0]=1 [D]=[unused1]=2
CLS, SEP, MASK, PAD, QMARK, DMARK = 10, 11, 12, 0, 1, 2

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
    'hidden_act': 'gelu',
    'layer_norm_eps': 1e-12,
}
model = BertModel(BertConfig(**cfg_dict, attn_implementation='eager'))
# HF std-0.02 init makes pico attention scores ~0 (near-uniform softmax);
# boost q/k/v + FFN so the encoder output is O(1)-structured (same recipe
# as bert_tiny_fixture.py).
with torch.no_grad():
    for layer in model.encoder.layer:
        layer.attention.self.query.weight.normal_(0.0, 0.7)
        layer.attention.self.key.weight.normal_(0.0, 0.7)
        layer.attention.self.value.weight.normal_(0.0, 0.5)
        layer.intermediate.dense.weight.normal_(0.0, 1.3)
model = model.double().eval()

# ColBERT projection head: bias-free Linear(hidden -> PROJ_DIM).
linear = torch.nn.Linear(D_MODEL, PROJ_DIM, bias=False).double()
with torch.no_grad():
    linear.weight.normal_(0.0, 0.8)

# ---- the two pinned inputs (a query and a document) ----
# Query: [CLS][Q] tok... [SEP] then [MASK]-padded to SEQ_LEN (all rows count).
q_content = [3, 7, 4]
query_ids = [CLS, QMARK] + q_content + [SEP]
query_ids = query_ids + [MASK] * (SEQ_LEN - len(query_ids))
assert len(query_ids) == SEQ_LEN
# Document: [CLS][D] tok... [SEP] then [PAD] (pad rows skipped).
d_content = [5, 8, 6, 9, 4]
doc_real = [CLS, DMARK] + d_content + [SEP]
doc_real_len = len(doc_real)
doc_ids = doc_real + [PAD] * (SEQ_LEN - doc_real_len)
assert len(doc_ids) == SEQ_LEN


def embed_tokens(ids, real_len):
    """BERT encode -> project -> L2 normalize per row; keep real_len rows."""
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]),
                    token_type_ids=torch.zeros(1, SEQ_LEN, dtype=torch.long))
        h = out.last_hidden_state[0]          # (SEQ_LEN, D_MODEL)
        proj = linear(h)                      # (SEQ_LEN, PROJ_DIM)
        proj = proj[:real_len]
        proj = proj / proj.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return proj


q_mat = embed_tokens(query_ids, SEQ_LEN)        # all rows
d_mat = embed_tokens(doc_ids, doc_real_len)     # non-pad prefix

# MaxSim: sum over query rows of max over doc rows of <q, d>.
sim = q_mat @ d_mat.T                            # (q_rows, d_rows)
maxsim = sim.max(dim=1).values.sum().item()

# ---- save checkpoint (BERT weights + linear.weight), f32 ----
sd = {k: v.to(torch.float32).contiguous()
      for k, v in model.state_dict().items()}
sd['linear.weight'] = linear.weight.detach().to(torch.float32).contiguous()
save_file(sd, 'tests/fixtures/tiny_colbert.safetensors')
with open('tests/fixtures/tiny_colbert_config.json', 'w') as f:
    json.dump(cfg_dict, f, indent=1)

ref = {
    'seq_len': SEQ_LEN,
    'proj_dim': PROJ_DIM,
    'markers': {'cls': CLS, 'sep': SEP, 'mask': MASK, 'pad': PAD,
                'q': QMARK, 'd': DMARK},
    'query_ids': query_ids,
    'doc_ids': doc_ids,
    'query_real_tokens': SEQ_LEN,
    'doc_real_tokens': doc_real_len,
    'query_mat': q_mat.tolist(),
    'doc_mat': d_mat.tolist(),
    'maxsim': maxsim,
}
with open('tests/fixtures/tiny_colbert_score.json', 'w') as f:
    json.dump(ref, f)
print(f'wrote tiny_colbert.safetensors ({len(sd)} tensors, proj_dim={PROJ_DIM})'
      f' + config + score (MaxSim = {maxsim:.6f})')

# ---- fixture self-checks ----
# 1. every row of the projected matrices is unit norm.
qn = q_mat.norm(dim=-1)
dn = d_mat.norm(dim=-1)
assert torch.allclose(qn, torch.ones_like(qn), atol=1e-9), qn
assert torch.allclose(dn, torch.ones_like(dn), atol=1e-9), dn
print(f'row norms: query in [{qn.min():.6f},{qn.max():.6f}], '
      f'doc in [{dn.min():.6f},{dn.max():.6f}]')
# 2. MaxSim is asymmetric / order-sensitive: a shuffled doc gives a
#    DIFFERENT per-row argmax structure but identical score (max is
#    permutation-invariant) - instead assert the score differs from a plain
#    mean-pool cosine, so the test genuinely covers late interaction.
q_pool = (q_mat.mean(0) / q_mat.mean(0).norm())
d_pool = (d_mat.mean(0) / d_mat.mean(0).norm())
cos = (q_pool @ d_pool).item()
assert abs(maxsim - cos) > 1e-3, (maxsim, cos)
print(f'MaxSim {maxsim:.4f} vs mean-pool cosine {cos:.4f} '
      f'(late interaction is distinct)')
# 3. the projection head genuinely mixes channels (not identity-like): the
#    projected query differs from the truncated raw hidden state.
with torch.no_grad():
    raw = model(input_ids=torch.tensor([query_ids]),
                token_type_ids=torch.zeros(1, SEQ_LEN, dtype=torch.long)
                ).last_hidden_state[0]
proj_effect = (linear(raw)[:, :min(PROJ_DIM, D_MODEL)] -
               raw[:, :min(PROJ_DIM, D_MODEL)]).abs().max().item()
assert proj_effect > 1e-2, proj_effect
print(f'projection head effect: max |diff| = {proj_effect:.4f}')
