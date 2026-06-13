#!/usr/bin/env python3
"""Slice the REAL bigscience/bloom-560m checkpoint down to a pico parity
fixture (~15 KB) for tests/TestNeuralPretrained.pas - the make_pico recipe
(see examples/LlamaImport/make_pico_llama_fixture.py): every kept value is a
genuine pretrained weight (a consistent sub-slab, never random).

Pinned in tests/fixtures/:

  tiny_bloom.safetensors + tiny_bloom_config.json + tiny_bloom_logits.json:
      BloomForCausalLM with the quirks that distinguish BLOOM:
        - ALiBi position scheme: NO positional embeddings at all, per-head
          FIXED linear attention biases. The sliced model keeps
          N_HEAD = 3 heads - deliberately NON-power-of-two, so the parity
          test exercises the extra-slope branch of the HF
          build_alibi_tensor recipe (closest_power_of_2 = 2 plus one head
          from the doubled table);
        - word_embeddings_layernorm: LayerNorm on the embedding output
          BEFORE the first block;
        - ONE fused query_key_value projection with the PER-HEAD [q|k|v]
          interleave (HF view(.., heads, 3, head_dim)) - the same h-major
          byte layout as GPT-NeoX but with NO rotary permutation;
        - sequential pre-LN residual, Megatron tanh GELU MLP (4*hidden);
        - ALWAYS-tied LM head (no lm_head tensor), final ln_f.

  layers   : first N_LAYER blocks
  hidden   : first D_MODEL channels of the residual stream
  heads    : first N_HEAD heads, first HEAD_DIM dims of each
  MLP      : first D_FF = 4*D_MODEL rows of dense_h_to_4h (HF's BloomMLP
             hardcodes 4*hidden, so the sliced ratio must stay 4x)
  vocab    : first VOCAB rows of word_embeddings

The original dtype (FP16 for bloom-560m) is preserved, so the fixture also
exercises the Pascal reader's FP16 decode end-to-end. Reference logits are
computed by HF transformers' BloomForCausalLM built on the SLICED config and
run in float64 (the oracle convention of the committed fixtures). The script
ASSERTS that ALiBi is non-vacuous (zeroing build_alibi_tensor changes the
logits far above the parity gate) so a missing/wrong slope FAILS the test.

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/bloom_tiny_fixture.py /tmp/bloom560 tests/fixtures/tiny_bloom
reads <src_dir>/model.safetensors + config.json. Needs torch + transformers
+ safetensors.
"""
import json
import os
import sys

import torch
from safetensors.torch import load_file, save_file
import transformers.models.bloom.modeling_bloom as modeling_bloom
from transformers import BloomConfig, BloomForCausalLM

N_LAYER = 2
N_HEAD = 3            # deliberately NON-power-of-two (extra-slope branch)
HEAD_DIM = 4
D_MODEL = N_HEAD * HEAD_DIM  # 12
D_FF = 4 * D_MODEL    # BloomMLP hardcodes 4*hidden
VOCAB = 12
SEQ_LEN = 16
N_SEQUENCES = 3

if len(sys.argv) != 3:
    sys.exit(__doc__)
SRC_DIR, OUT_PREFIX = sys.argv[1], sys.argv[2]

cfg = json.load(open(os.path.join(SRC_DIR, 'config.json')))
sd = load_file(os.path.join(SRC_DIR, 'model.safetensors'))
HID = cfg.get('hidden_size', cfg.get('n_embed'))
SRC_HEADS = cfg.get('n_head', cfg.get('num_attention_heads'))
orig_hd = HID // SRC_HEADS
assert N_HEAD <= SRC_HEADS and HEAD_DIM <= orig_hd

# Residual-stream channels: first D_MODEL. Head-structured channels (the
# attention concat axis, i.e. dense.weight columns): first HEAD_DIM dims of
# each kept head.
hid_idx = torch.arange(D_MODEL)
head_idx = torch.cat([torch.arange(HEAD_DIM) + h * orig_hd
                      for h in range(N_HEAD)])
# Fused query_key_value rows: orig row of (head h, third t, dim d) is
# h*3*orig_hd + t*orig_hd + d  (per-head [q|k|v] interleave). The sliced
# checkpoint must keep the SAME interleave with the new head_dim.
qkv_idx = torch.cat([torch.arange(HEAD_DIM) + h * 3 * orig_hd + t * orig_hd
                     for h in range(N_HEAD) for t in range(3)])


def t(name):
    key = 'transformer.' + name
    return sd[key if key in sd else name]


out = {}
out['word_embeddings.weight'] = \
    t('word_embeddings.weight')[:VOCAB][:, hid_idx]
out['word_embeddings_layernorm.weight'] = \
    t('word_embeddings_layernorm.weight')[hid_idx]
out['word_embeddings_layernorm.bias'] = \
    t('word_embeddings_layernorm.bias')[hid_idx]
for L in range(N_LAYER):
    p = f'h.{L}.'
    for ln in ('input_layernorm', 'post_attention_layernorm'):
        out[p + ln + '.weight'] = t(p + ln + '.weight')[hid_idx]
        out[p + ln + '.bias'] = t(p + ln + '.bias')[hid_idx]
    out[p + 'self_attention.query_key_value.weight'] = \
        t(p + 'self_attention.query_key_value.weight')[qkv_idx][:, hid_idx]
    out[p + 'self_attention.query_key_value.bias'] = \
        t(p + 'self_attention.query_key_value.bias')[qkv_idx]
    out[p + 'self_attention.dense.weight'] = \
        t(p + 'self_attention.dense.weight')[hid_idx][:, head_idx]
    out[p + 'self_attention.dense.bias'] = \
        t(p + 'self_attention.dense.bias')[hid_idx]
    out[p + 'mlp.dense_h_to_4h.weight'] = \
        t(p + 'mlp.dense_h_to_4h.weight')[:D_FF][:, hid_idx]
    out[p + 'mlp.dense_h_to_4h.bias'] = t(p + 'mlp.dense_h_to_4h.bias')[:D_FF]
    out[p + 'mlp.dense_4h_to_h.weight'] = \
        t(p + 'mlp.dense_4h_to_h.weight')[hid_idx][:, :D_FF]
    out[p + 'mlp.dense_4h_to_h.bias'] = t(p + 'mlp.dense_4h_to_h.bias')[hid_idx]
out['ln_f.weight'] = t('ln_f.weight')[hid_idx]
out['ln_f.bias'] = t('ln_f.bias')[hid_idx]

st_path = OUT_PREFIX + '.safetensors'
save_file({('transformer.' + k): v.contiguous() for k, v in out.items()},
          st_path)

new_cfg = {
    'architectures': ['BloomForCausalLM'],
    'model_type': 'bloom',
    'hidden_size': D_MODEL,
    'n_layer': N_LAYER,
    'n_head': N_HEAD,
    'n_inner': None,
    'vocab_size': VOCAB,
    'seq_length': SEQ_LEN,
    'layer_norm_epsilon': cfg.get('layer_norm_epsilon', 1e-5),
    'apply_residual_connection_post_layernorm': False,
    'tie_word_embeddings': True,
}
with open(OUT_PREFIX + '_config.json', 'w') as f:
    json.dump(new_cfg, f, indent=1)

# ---- reference logits via transformers (float64 oracle) -------------------
model = BloomForCausalLM(BloomConfig(**new_cfg))
missing, unexpected = model.load_state_dict(
    {('transformer.' + k): v.to(torch.float32) for k, v in out.items()},
    strict=False)
model.tie_weights()
assert not unexpected, unexpected
assert all(m == 'lm_head.weight' for m in missing), missing
model.double().eval()

sequences = [[(7 * i + 3 * s + s * s) % VOCAB for i in range(SEQ_LEN)]
             for s in range(N_SEQUENCES)]
with torch.no_grad():
    logits = [model(input_ids=torch.tensor([seq])).logits[0].tolist()
              for seq in sequences]
with open(OUT_PREFIX + '_logits.json', 'w') as f:
    json.dump({'sequences': sequences, 'logits': logits}, f)

# ---- non-vacuity: zero ALiBi must change the logits materially -------------
# (well above the 1e-4 parity gate, so wrong/missing slopes FAIL the test)
real_build = modeling_bloom.build_alibi_tensor
modeling_bloom.build_alibi_tensor = \
    lambda *a, **kw: torch.zeros_like(real_build(*a, **kw))
with torch.no_grad():
    no_alibi = [model(input_ids=torch.tensor([seq])).logits[0]
                for seq in sequences]
modeling_bloom.build_alibi_tensor = real_build
alibi_diff = max((na - torch.tensor(lg)).abs().max().item()
                 for na, lg in zip(no_alibi, logits))
assert alibi_diff > 5e-4, f'ALiBi vacuous in the fixture: {alibi_diff}'

print(f'wrote {st_path} ({len(out)} tensors, layers={N_LAYER}, '
      f'heads={N_HEAD} (non-power-of-two), head_dim={HEAD_DIM}, '
      f'hidden={D_MODEL}, vocab={VOCAB}) + config + logits '
      f'({N_SEQUENCES} sequences); zero-ALiBi max |dlogit| = {alibi_diff:.3f}')
