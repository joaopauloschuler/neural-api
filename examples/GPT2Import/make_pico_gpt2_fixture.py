#!/usr/bin/env python3
"""Slice a REAL GPT-2-family checkpoint (gpt2, distilgpt2, ...) down to a
pico parity fixture (~10 KB) for tests/TestNeuralPretrained.pas.

Unlike slice_gpt2.py (which only cuts layers/vocab and keeps the 768-wide
hidden dimension), this also slices the HIDDEN dimensions so the fixture is
small enough to commit, while every kept value is a genuine pretrained
weight (a consistent sub-slab, never random):

  layers   : first N_LAYER blocks
  hidden   : first D_MODEL channels of the residual stream
  heads    : first N_HEAD attention heads, first HEAD_DIM dims of each
             (q/k/v output channels h*orig_head_dim + 0..HEAD_DIM-1)
  MLP      : first D_FF rows of c_fc
  vocab    : first VOCAB rows of wte; first N_CTX rows of wpe

Reference logits are computed by HF transformers' GPT2LMHeadModel built on
the SLICED config and run in float64 (the oracle convention of the
committed fixtures), so the Pascal importer can be asserted against
transformers itself.

Coded by Claude (AI).

Usage:
  python3 make_pico_gpt2_fixture.py <src.safetensors> <out_prefix>
e.g.
  python3 make_pico_gpt2_fixture.py /tmp/distilgpt2/model.safetensors \
      tests/fixtures/tiny_distilgpt2
writes <out_prefix>.safetensors and <out_prefix>_logits.json.
Needs torch + transformers + safetensors.
"""
import json
import sys

import torch
from safetensors.torch import load_file, save_file
from transformers import GPT2Config, GPT2LMHeadModel

N_LAYER = 2
N_HEAD = 2
HEAD_DIM = 4
D_MODEL = N_HEAD * HEAD_DIM  # 8
D_FF = 4 * D_MODEL           # 32 (GPT-2's fixed 4x ratio)
VOCAB = 12
N_CTX = 16
N_SEQUENCES = 3

if len(sys.argv) != 3:
    sys.exit(__doc__)
SRC, OUT_PREFIX = sys.argv[1], sys.argv[2]

sd = load_file(SRC)
prefix = 'transformer.' if 'transformer.wte.weight' in sd else ''


def t(name):
    return sd[prefix + name].to(torch.float32)


d_orig = t('wte.weight').shape[1]
# Head h of the original model owns q/k/v channels h*orig_hd..(h+1)*orig_hd-1.
# The original head_dim is d_orig / n_head_orig; for the GPT-2 family it is
# always 64 (the n_embd/64 rule the importer also applies).
ORIG_HD = 64
head_idx = torch.cat([torch.arange(h * ORIG_HD, h * ORIG_HD + HEAD_DIM)
                      for h in range(N_HEAD)])          # attention channels
hid_idx = torch.arange(D_MODEL)                          # residual channels

out = {}
out['wte.weight'] = t('wte.weight')[:VOCAB][:, hid_idx]
out['wpe.weight'] = t('wpe.weight')[:N_CTX][:, hid_idx]
for L in range(N_LAYER):
    p = f'h.{L}.'
    for ln in ('ln_1', 'ln_2'):
        out[p + ln + '.weight'] = t(p + ln + '.weight')[hid_idx]
        out[p + ln + '.bias'] = t(p + ln + '.bias')[hid_idx]
    # c_attn: HF Conv1D [in, out]; out = Q|K|V thirds of d_orig each.
    qkv_idx = torch.cat([head_idx, head_idx + d_orig, head_idx + 2 * d_orig])
    out[p + 'attn.c_attn.weight'] = \
        t(p + 'attn.c_attn.weight')[hid_idx][:, qkv_idx].contiguous()
    out[p + 'attn.c_attn.bias'] = t(p + 'attn.c_attn.bias')[qkv_idx]
    out[p + 'attn.c_proj.weight'] = \
        t(p + 'attn.c_proj.weight')[head_idx][:, hid_idx].contiguous()
    out[p + 'attn.c_proj.bias'] = t(p + 'attn.c_proj.bias')[hid_idx]
    out[p + 'mlp.c_fc.weight'] = \
        t(p + 'mlp.c_fc.weight')[hid_idx][:, :D_FF].contiguous()
    out[p + 'mlp.c_fc.bias'] = t(p + 'mlp.c_fc.bias')[:D_FF]
    out[p + 'mlp.c_proj.weight'] = \
        t(p + 'mlp.c_proj.weight')[:D_FF][:, hid_idx].contiguous()
    out[p + 'mlp.c_proj.bias'] = t(p + 'mlp.c_proj.bias')[hid_idx]
out['ln_f.weight'] = t('ln_f.weight')[hid_idx]
out['ln_f.bias'] = t('ln_f.bias')[hid_idx]

# Keep the genuine "transformer." naming of GPT2LMHeadModel exports so the
# fixture also exercises the importer's prefix detection.
st_path = OUT_PREFIX + '.safetensors'
save_file({('transformer.' + k): v.contiguous() for k, v in out.items()},
          st_path)

# ---- reference logits via transformers (float64 oracle) -------------------
config = GPT2Config(vocab_size=VOCAB, n_positions=N_CTX, n_embd=D_MODEL,
                    n_layer=N_LAYER, n_head=N_HEAD,
                    activation_function='gelu_new', layer_norm_epsilon=1e-5,
                    attn_pdrop=0.0, embd_pdrop=0.0, resid_pdrop=0.0)
model = GPT2LMHeadModel(config)
missing, unexpected = model.load_state_dict(
    {('transformer.' + k): v for k, v in out.items()}, strict=False)
model.tie_weights()
assert not unexpected, unexpected
assert all(m == 'lm_head.weight' for m in missing), missing
model.double().eval()

sequences = [[(7 * i + 3 * s + s * s) % VOCAB for i in range(N_CTX)]
             for s in range(N_SEQUENCES)]
with torch.no_grad():
    logits = [model(input_ids=torch.tensor([seq])).logits[0].tolist()
              for seq in sequences]
with open(OUT_PREFIX + '_logits.json', 'w') as f:
    json.dump({'sequences': sequences, 'logits': logits}, f)

print(f'wrote {st_path} ({len(out)} tensors, layers={N_LAYER}, '
      f'heads={N_HEAD}, d_model={D_MODEL}, vocab={VOCAB}, ctx={N_CTX}) '
      f'and {OUT_PREFIX}_logits.json ({N_SEQUENCES} sequences)')
