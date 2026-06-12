#!/usr/bin/env python3
"""Slice a REAL Llama-architecture checkpoint (SmolLM2-135M, TinyLlama, ...)
down to a pico parity fixture (~10 KB) for tests/TestNeuralPretrained.pas.

Unlike slice_llama.py (which only cuts layers/vocab and keeps the full
hidden width), this also slices the HIDDEN dimensions so the fixture is
small enough to commit, while every kept value is a genuine pretrained
weight (a consistent sub-slab, never random):

  layers   : first N_LAYER decoder blocks
  hidden   : first D_MODEL channels of the residual stream
  heads    : N_HEAD query heads + N_KV_HEAD kv heads, first HEAD_DIM dims
             of each. The kept q heads are the FIRST heads of the kept kv
             groups, so the GQA sharing structure stays genuine.
  MLP      : first D_FF rows of gate/up
  vocab    : first VOCAB rows of embed_tokens

The original dtype is preserved (SmolLM2 ships BF16, so the fixture also
exercises the Pascal reader's BF16 decode end-to-end). Reference logits are
computed by HF transformers' LlamaForCausalLM built on the SLICED config
and run in float64 (the oracle convention of the committed fixtures).

Coded by Claude (AI).

Usage:
  python3 make_pico_llama_fixture.py <src_dir> <out_prefix>
e.g.
  python3 make_pico_llama_fixture.py /tmp/smollm2 tests/fixtures/tiny_smollm2
reads <src_dir>/model.safetensors + config.json; writes
<out_prefix>.safetensors, <out_prefix>_config.json, <out_prefix>_logits.json.
Needs torch + transformers + safetensors.
"""
import json
import os
import sys

import torch
from safetensors.torch import load_file, save_file
from transformers import LlamaConfig, LlamaForCausalLM

N_LAYER = 2
N_HEAD = 2
N_KV_HEAD = 1
HEAD_DIM = 4
D_MODEL = N_HEAD * HEAD_DIM  # 8
D_FF = 12
VOCAB = 12
MAX_POS = 16
N_SEQUENCES = 3

if len(sys.argv) != 3:
    sys.exit(__doc__)
SRC_DIR, OUT_PREFIX = sys.argv[1], sys.argv[2]

cfg = json.load(open(os.path.join(SRC_DIR, 'config.json')))
sd = load_file(os.path.join(SRC_DIR, 'model.safetensors'))
orig_hd = cfg['hidden_size'] // cfg['num_attention_heads']
orig_group = cfg['num_attention_heads'] // cfg['num_key_value_heads']
assert N_HEAD % N_KV_HEAD == 0
new_group = N_HEAD // N_KV_HEAD
assert new_group <= orig_group, 'cannot widen a GQA group'

# Query head g*new_group+j is original head g*orig_group+j: the kept q heads
# attended exactly the kept kv heads in the original model too.
q_idx = torch.cat([torch.arange(0, HEAD_DIM) +
                   (g * orig_group + j) * orig_hd
                   for g in range(N_KV_HEAD) for j in range(new_group)])
kv_idx = torch.cat([torch.arange(0, HEAD_DIM) + g * orig_hd
                    for g in range(N_KV_HEAD)])
hid_idx = torch.arange(D_MODEL)


def t(name):
    return sd['model.' + name]


out = {}
out['embed_tokens.weight'] = t('embed_tokens.weight')[:VOCAB][:, hid_idx]
for L in range(N_LAYER):
    p = f'layers.{L}.'
    out[p + 'input_layernorm.weight'] = \
        t(p + 'input_layernorm.weight')[hid_idx]
    out[p + 'post_attention_layernorm.weight'] = \
        t(p + 'post_attention_layernorm.weight')[hid_idx]
    out[p + 'self_attn.q_proj.weight'] = \
        t(p + 'self_attn.q_proj.weight')[q_idx][:, hid_idx].contiguous()
    out[p + 'self_attn.k_proj.weight'] = \
        t(p + 'self_attn.k_proj.weight')[kv_idx][:, hid_idx].contiguous()
    out[p + 'self_attn.v_proj.weight'] = \
        t(p + 'self_attn.v_proj.weight')[kv_idx][:, hid_idx].contiguous()
    out[p + 'self_attn.o_proj.weight'] = \
        t(p + 'self_attn.o_proj.weight')[hid_idx][:, q_idx].contiguous()
    out[p + 'mlp.gate_proj.weight'] = \
        t(p + 'mlp.gate_proj.weight')[:D_FF][:, hid_idx].contiguous()
    out[p + 'mlp.up_proj.weight'] = \
        t(p + 'mlp.up_proj.weight')[:D_FF][:, hid_idx].contiguous()
    out[p + 'mlp.down_proj.weight'] = \
        t(p + 'mlp.down_proj.weight')[hid_idx][:, :D_FF].contiguous()
out['norm.weight'] = t('norm.weight')[hid_idx]

st_path = OUT_PREFIX + '.safetensors'
save_file({('model.' + k): v.contiguous() for k, v in out.items()}, st_path)

new_cfg = {
    'architectures': ['LlamaForCausalLM'],
    'model_type': 'llama',
    'hidden_size': D_MODEL,
    'intermediate_size': D_FF,
    'num_hidden_layers': N_LAYER,
    'num_attention_heads': N_HEAD,
    'num_key_value_heads': N_KV_HEAD,
    'vocab_size': VOCAB,
    'max_position_embeddings': MAX_POS,
    'rms_norm_eps': cfg.get('rms_norm_eps', 1e-6),
    'rope_theta': cfg.get('rope_theta', 10000.0),
    'rope_scaling': None,
    'tie_word_embeddings': True,  # no lm_head tensor in the fixture
    'attention_bias': False,
    'mlp_bias': False,
    'hidden_act': 'silu',
}
with open(OUT_PREFIX + '_config.json', 'w') as f:
    json.dump(new_cfg, f, indent=1)

# ---- reference logits via transformers (float64 oracle) -------------------
model = LlamaForCausalLM(LlamaConfig(**new_cfg))
missing, unexpected = model.load_state_dict(
    {('model.' + k): v.to(torch.float32) for k, v in out.items()},
    strict=False)
model.tie_weights()
assert not unexpected, unexpected
assert all(m == 'lm_head.weight' for m in missing), missing
model.double().eval()

sequences = [[(7 * i + 3 * s + s * s) % VOCAB for i in range(MAX_POS)]
             for s in range(N_SEQUENCES)]
with torch.no_grad():
    logits = [model(input_ids=torch.tensor([seq])).logits[0].tolist()
              for seq in sequences]
with open(OUT_PREFIX + '_logits.json', 'w') as f:
    json.dump({'sequences': sequences, 'logits': logits}, f)

print(f'wrote {st_path} ({len(out)} tensors, layers={N_LAYER}, '
      f'heads={N_HEAD}/{N_KV_HEAD}, hidden={D_MODEL}, vocab={VOCAB}) + '
      f'config + logits ({N_SEQUENCES} sequences)')
