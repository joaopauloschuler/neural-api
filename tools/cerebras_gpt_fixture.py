#!/usr/bin/env python3
"""Slice the REAL cerebras/Cerebras-GPT-111M checkpoint down to a pico
parity fixture for tests/TestNeuralPretrained.pas.

Cerebras-GPT is the truest open GPT-3 reproduction (the exact GPT-3 recipe:
dense attention, learned absolute positions, GPT-2 BPE, Chinchilla-scaled)
and ships in plain GPT2LMHeadModel format with model_type "gpt2", so it
loads through BuildGPT2FromSafeTensors. Two deviations from the OpenAI
GPT-2 checkpoints, both pinned by this fixture:

  - activation_function is "gelu" (the EXACT erf form), NOT "gelu_new"
    (the tanh approximation) - the importer's pExactGelu flag / the
    config-driven BuildFromPretrained route handles it;
  - the upstream repo carries only pytorch_model.bin (NO safetensors),
    so this script (and any full-model load) converts via torch.

Slicing convention is examples/GPT2Import/make_pico_gpt2_fixture.py: every
kept value is a genuine pretrained weight (a consistent sub-slab, never
random):

  layers   : first N_LAYER of 10 blocks
  hidden   : first D_MODEL of 768 residual channels
  heads    : first N_HEAD of 12 heads, first HEAD_DIM of 64 dims each
  MLP      : first D_FF rows of c_fc (keeps GPT-2's fixed 4x ratio)
  vocab    : first VOCAB rows of wte; first N_CTX of 2048 rows of wpe

Writes (committed, ~10 KB total):
  tests/fixtures/tiny_cerebras_gpt.safetensors
  tests/fixtures/tiny_cerebras_gpt_config.json   (the REAL config keys with
      the sliced dimensions - model_type "gpt2", activation_function "gelu")
  tests/fixtures/tiny_cerebras_gpt_logits.json   (float64 HF transformers
      GPT2LMHeadModel oracle on the sliced model, pinned token sequences)

The script ASSERTS that gelu-vs-gelu_new genuinely moves the FULL 111M
model's logits (measured: max |logit diff| 0.040, far above the 1e-4 parity
gate). At pico width the quirk is numerically invisible (the sliced
preactivations are ~0.1 where the two gelu forms agree to ~1e-6; their
global gap peaks at 4.7e-4 near |x|=2.7), so the Pascal test pins the
config-driven activation route STRUCTURALLY (the net built from the
Cerebras config must contain the TNNetErf exact-gelu composition) on top of
the logit parity.

Coded by Claude (AI).

Usage (from the repo root; downloads ~450 MB into the HF cache):
  python3 tools/cerebras_gpt_fixture.py
Needs torch + transformers + safetensors + huggingface_hub.
"""
import json
import os

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file
from transformers import GPT2Config, GPT2LMHeadModel

REPO = 'cerebras/Cerebras-GPT-111M'
N_LAYER = 2
N_HEAD = 2
HEAD_DIM = 4
D_MODEL = N_HEAD * HEAD_DIM  # 8
D_FF = 4 * D_MODEL           # 32 (n_inner=3072 = 4*768 upstream)
VOCAB = 12
N_CTX = 16
N_SEQUENCES = 3

cfg_path = hf_hub_download(REPO, 'config.json')
bin_path = hf_hub_download(REPO, 'pytorch_model.bin')
with open(cfg_path) as f:
    real_cfg = json.load(f)
assert real_cfg['model_type'] == 'gpt2', real_cfg['model_type']
assert real_cfg['activation_function'] == 'gelu', \
    real_cfg['activation_function']

sd = torch.load(bin_path, map_location='cpu', weights_only=True)
prefix = 'transformer.' if 'transformer.wte.weight' in sd else ''


def t(name):
    return sd[prefix + name].to(torch.float32)


d_orig = t('wte.weight').shape[1]
ORIG_HD = d_orig // real_cfg['n_head']  # 768 / 12 = 64
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

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fixdir = os.path.join(root, 'tests', 'fixtures')
st_path = os.path.join(fixdir, 'tiny_cerebras_gpt.safetensors')
# Keep the genuine "transformer." naming of GPT2LMHeadModel exports so the
# fixture also exercises the importer's prefix detection.
save_file({('transformer.' + k): v.contiguous() for k, v in out.items()},
          st_path)

# The REAL config keys with only the dimensions sliced - in particular
# model_type "gpt2" and activation_function "gelu" stay verbatim.
sliced_cfg = dict(real_cfg)
sliced_cfg.update({'n_embd': D_MODEL, 'n_head': N_HEAD, 'n_layer': N_LAYER,
                   'n_positions': N_CTX, 'n_inner': D_FF,
                   'vocab_size': VOCAB})
cfg_out = os.path.join(fixdir, 'tiny_cerebras_gpt_config.json')
with open(cfg_out, 'w') as f:
    json.dump(sliced_cfg, f, indent=1)


# ---- reference logits via transformers (float64 oracle) -------------------
def make_model(activation):
    config = GPT2Config(vocab_size=VOCAB, n_positions=N_CTX, n_embd=D_MODEL,
                        n_layer=N_LAYER, n_head=N_HEAD, n_inner=D_FF,
                        activation_function=activation,
                        layer_norm_epsilon=real_cfg['layer_norm_epsilon'],
                        attn_pdrop=0.0, embd_pdrop=0.0, resid_pdrop=0.0)
    model = GPT2LMHeadModel(config)
    missing, unexpected = model.load_state_dict(
        {('transformer.' + k): v for k, v in out.items()}, strict=False)
    model.tie_weights()
    assert not unexpected, unexpected
    assert all(m == 'lm_head.weight' for m in missing), missing
    return model.double().eval()


sequences = [[(7 * i + 3 * s + s * s) % VOCAB for i in range(N_CTX)]
             for s in range(N_SEQUENCES)]


def logits_of(model):
    with torch.no_grad():
        return [model(input_ids=torch.tensor([seq])).logits[0]
                for seq in sequences]


ref = logits_of(make_model('gelu'))  # the Cerebras activation
with open(os.path.join(fixdir, 'tiny_cerebras_gpt_logits.json'), 'w') as f:
    json.dump({'sequences': sequences,
               'logits': [l.tolist() for l in ref]}, f)

# Pico-slice gelu-vs-gelu_new effect (documentation only - numerically
# invisible at width 8, see the module docstring).
pico_effect = max((a - b).abs().max().item()
                  for a, b in zip(ref, logits_of(make_model('gelu_new'))))

# The exact-vs-tanh gelu choice must matter on the FULL checkpoint, or the
# activation handling would not be worth pinning at all (f32 is plenty to
# resolve a >1e-3 gap).
def full_model(activation):
    config = GPT2Config(
        vocab_size=real_cfg['vocab_size'], n_positions=real_cfg['n_positions'],
        n_embd=real_cfg['n_embd'], n_layer=real_cfg['n_layer'],
        n_head=real_cfg['n_head'], n_inner=real_cfg['n_inner'],
        activation_function=activation,
        layer_norm_epsilon=real_cfg['layer_norm_epsilon'],
        attn_pdrop=0.0, embd_pdrop=0.0, resid_pdrop=0.0)
    model = GPT2LMHeadModel(config)
    model.load_state_dict(sd, strict=False)
    model.tie_weights()
    return model.eval()


probe = torch.tensor([[(7 * i + 3) % real_cfg['vocab_size']
                       for i in range(N_CTX)]])
with torch.no_grad():
    full_effect = (full_model('gelu')(input_ids=probe).logits -
                   full_model('gelu_new')(input_ids=probe).logits
                   ).abs().max().item()
assert full_effect > 1e-3, \
    f'gelu vs gelu_new had no effect on the FULL 111M logits ({full_effect})'

print(f'wrote {st_path} ({len(out)} tensors, layers={N_LAYER}, '
      f'heads={N_HEAD}, d_model={D_MODEL}, vocab={VOCAB}, ctx={N_CTX})')
print(f'wrote {cfg_out} + logits ({N_SEQUENCES} sequences of {N_CTX})')
print(f'gelu-vs-gelu_new on the pico slice: max |diff| = {pico_effect:.2e}')
print(f'gelu-vs-gelu_new on the FULL 111M:  max |diff| = {full_effect:.4f}')
