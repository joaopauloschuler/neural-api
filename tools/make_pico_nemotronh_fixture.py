#!/usr/bin/env python3
"""Generate a tiny RANDOM Nemotron-H parity fixture for
tests/TestNeuralPretrained.pas (no network access: the model is randomly
initialized from a pico config, never downloaded).

Nemotron-H (HF model_type "nemotron_h", architectures
["NemotronHForCausalLM"]: nvidia/Nemotron-H-8B-Base and the 4B/47B siblings)
is the suite's SECOND HYBRID importer after Jamba, and architecturally
DISTINCT from it. Where Jamba is Mamba-1 + attention + MoE on a periodic
arithmetic schedule, Nemotron-H interleaves THREE block types on an EXPLICIT
per-layer string schedule (config "hybrid_override_pattern", here "M*-M":
'M'=Mamba-2 mixer, '*'=attention, '-'=non-gated relu2 MLP) and each block is
a SINGLE pre-norm residual x := x + mixer(norm(x)) (no separate FFN sub-block
- the MLP IS a block type on the schedule).

Quirks exercised (vs the landed Mamba-2 and Jamba importers):
  - the multi-head SSD Mamba-2 mixer ('M' block) - the SAME in_proj packing
    [gate | (x|B|C) | dt] and gated-RMSNorm-before-out_proj as standalone
    Mamba-2, but here on the .mixer. key namespace and as ONE of three
    interleaved block types;
  - full softmax attention ('*' block) - bias-free GQA via
    num_key_value_heads, NoPE (Nemotron-H attention has NO positional
    encoding; the Mamba-2 mixers carry order), and an EXPLICIT config head_dim
    that need NOT equal hidden/num_heads, with scaling = head_dim**-0.5;
  - the genuinely-new NON-GATED relu2 MLP ('-' block) -
    down(ReLU(up(x))**2) with NO gate projection (ACT2FN "relu2"); the Pascal
    importer reproduces it as up_proj -> ReLU -> Square -> down_proj.

The pico config picks head_dim != hidden//num_heads so the explicit-head_dim
path is genuinely exercised, and n_groups < mamba_num_heads so B/C sharing is
real.

The reference logits are computed by HF transformers in float64 running the
naive CPU path (the CUDA/mamba-ssm fast kernels are not installed and the
device is CPU, both checked below). The naive SSD path is mathematically EQUAL
to the head-wise recurrent scan the Pascal TNNetMamba2 leaf implements.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/make_pico_nemotronh_fixture.py
writes tests/fixtures/tiny_nemotronh{.safetensors,_config.json,_logits.json}.
"""
import copy
import json

import torch
from safetensors.torch import save_file
from transformers import NemotronHConfig
from transformers.models.nemotron_h import modeling_nemotron_h
from transformers.models.nemotron_h.modeling_nemotron_h import (
    NemotronHForCausalLM,
)

HIDDEN = 8
VOCAB = 13
PATTERN = "M*-M"          # Mamba-2, attention, relu2 MLP, Mamba-2
# attention
NUM_HEADS = 4
KV_HEADS = 2              # GQA: 2 query heads share each kv head
HEAD_DIM = 3             # != HIDDEN//NUM_HEADS (==2): exercise explicit head_dim
# relu2 MLP
INTER = 12
# mamba-2 mixer
STATE = 4
MAMBA_HEADS = 2
MAMBA_HEAD_DIM = 4       # d_inner = 2*4 = 8
N_GROUPS = 1            # < mamba_heads: B/C shared across the group's heads
CONV_K = 4
CHUNK = 8
EPS = 1e-5
SEQ_LEN = 7
N_SEQUENCES = 3

torch.manual_seed(20260614)

cfg_dict = {
    'architectures': ['NemotronHForCausalLM'],
    'model_type': 'nemotron_h',
    'vocab_size': VOCAB,
    'hidden_size': HIDDEN,
    'hybrid_override_pattern': PATTERN,
    'num_attention_heads': NUM_HEADS,
    'num_key_value_heads': KV_HEADS,
    'head_dim': HEAD_DIM,
    'attention_bias': False,
    'intermediate_size': INTER,
    'mlp_hidden_act': 'relu2',
    'mlp_bias': False,
    'ssm_state_size': STATE,
    'mamba_num_heads': MAMBA_HEADS,
    'mamba_head_dim': MAMBA_HEAD_DIM,
    'n_groups': N_GROUPS,
    'conv_kernel': CONV_K,
    'chunk_size': CHUNK,
    'use_bias': False,
    'use_conv_bias': True,
    'mamba_proj_bias': False,
    'layer_norm_epsilon': EPS,
    'tie_word_embeddings': False,
    'use_cache': True,
    'bos_token_id': 1,
    'eos_token_id': 2,
    'pad_token_id': 0,
}
config = NemotronHConfig(**cfg_dict)
assert config.num_hidden_layers == len(PATTERN), \
    (config.num_hidden_layers, len(PATTERN))
assert config.layers_block_type == ['mamba', 'attention', 'mlp', 'mamba'], \
    config.layers_block_type
model = NemotronHForCausalLM(config)

# The oracle MUST run HF's naive CPU path, not a fused kernel.
assert not any((modeling_nemotron_h.selective_state_update,
                modeling_nemotron_h.causal_conv1d_fn,
                modeling_nemotron_h.causal_conv1d_update)), \
    'CUDA fast-path kernels installed - oracle would not be the naive path'
assert model.model.embeddings.weight.device.type == 'cpu'

# Pinned inputs (3+ rows: the Pascal parity helper requires it).
sequences = [[(5 * i + 3 * s + s * s) % (VOCAB - 1) + 1
              for i in range(SEQ_LEN)] for s in range(N_SEQUENCES)]


# Re-randomize away from the HF std-0.02 init (vacuous at pico width) to O(1)
# scale so every block type and quirk is visible in the oracle.
def block_type(i):
    return config.layers_block_type[i]


with torch.no_grad():
    model.model.embeddings.weight.normal_(0.0, 0.8)
    model.model.norm_f.weight.normal_(1.0, 0.25)
    if model.lm_head.weight.data_ptr() != \
            model.model.embeddings.weight.data_ptr():
        model.lm_head.weight.normal_(0.0, 0.5)
    for i, layer in enumerate(model.model.layers):
        layer.norm.weight.normal_(1.0, 0.25)
        mx = layer.mixer
        bt = block_type(i)
        if bt == 'mamba':
            mx.A_log.copy_(torch.empty_like(mx.A_log).uniform_(-1.0, 1.5))
            mx.D.normal_(0.0, 0.8)
            mx.dt_bias.normal_(0.0, 0.8)
            mx.conv1d.weight.normal_(0.0, 0.5)
            mx.conv1d.bias.normal_(0.0, 0.4)
            mx.in_proj.weight.normal_(0.0, 0.5)
            mx.norm.weight.normal_(1.0, 0.25)
            mx.out_proj.weight.normal_(0.0, 0.5)
        elif bt == 'attention':
            mx.q_proj.weight.normal_(0.0, 0.5)
            mx.k_proj.weight.normal_(0.0, 0.5)
            mx.v_proj.weight.normal_(0.0, 0.5)
            mx.o_proj.weight.normal_(0.0, 0.5)
        elif bt == 'mlp':
            mx.up_proj.weight.normal_(0.0, 0.5)
            mx.down_proj.weight.normal_(0.0, 0.5)
        else:
            raise SystemExit('unexpected block type ' + bt)

model = model.double().eval()

sd = {k: v.to(torch.float32).clone().contiguous()
      for k, v in model.state_dict().items()}
save_file(sd, 'tests/fixtures/tiny_nemotronh.safetensors')
with open('tests/fixtures/tiny_nemotronh_config.json', 'w') as f:
    json.dump(cfg_dict, f, indent=1)


def run(m):
    with torch.no_grad():
        return torch.stack([m(input_ids=torch.tensor([s])).logits[0]
                            for s in sequences])


logits = run(model)
with open('tests/fixtures/tiny_nemotronh_logits.json', 'w') as f:
    json.dump({'sequences': sequences, 'logits': logits.tolist()}, f)
print(f'wrote tiny_nemotronh.safetensors ({len(sd)} tensors) + config + oracle')
print('layer schedule:', config.layers_block_type)
for k in sorted(sd):
    print(f'  {k} {list(sd[k].shape)}')

# ---- fixture self-checks: every block type / quirk visible in the oracle ----
base = logits

# 0. All three block types present.
kinds = set(config.layers_block_type)
assert kinds == {'mamba', 'attention', 'mlp'}, kinds
print('all three block types present:', sorted(kinds))

# 1. Per-head SCALAR A_log on the Mamba-2 blocks must carry signal.
shared_a = copy.deepcopy(model)
with torch.no_grad():
    for i, layer in enumerate(shared_a.model.layers):
        if block_type(i) == 'mamba':
            layer.mixer.A_log.fill_(layer.mixer.A_log[0].item())
d = (run(shared_a) - base).abs().max().item()
assert d > 1e-2, f'per-head A_log had no effect ({d})'
print(f'per-head A_log effect on logits: max |diff| = {d:.4f}')

# 2. relu2 MLP must matter: zeroing the up_proj of the MLP block kills it.
nomlp = copy.deepcopy(model)
with torch.no_grad():
    for i, layer in enumerate(nomlp.model.layers):
        if block_type(i) == 'mlp':
            layer.mixer.up_proj.weight.zero_()
d = (run(nomlp) - base).abs().max().item()
assert d > 1e-2, f'relu2 MLP had no effect ({d})'
print(f'relu2 MLP effect on logits: max |diff| = {d:.4f}')

# 3. relu2 is the SQUARED ReLU, not plain ReLU or square: a known scalar
# must map x -> relu(x)**2.
act = modeling_nemotron_h.ACT2FN['relu2']
probe = torch.tensor([-2.0, -0.5, 0.0, 1.5, 3.0])
exp = torch.relu(probe) ** 2
assert torch.allclose(act(probe), exp), (act(probe), exp)
print('relu2 == ReLU(x)^2 confirmed')

# 4. Attention block must matter: zeroing o_proj of the attn block kills it.
noattn = copy.deepcopy(model)
with torch.no_grad():
    for i, layer in enumerate(noattn.model.layers):
        if block_type(i) == 'attention':
            layer.mixer.o_proj.weight.zero_()
d = (run(noattn) - base).abs().max().item()
assert d > 1e-2, f'attention block had no effect ({d})'
print(f'attention effect on logits: max |diff| = {d:.4f}')

# 5. NoPE: attention must be position-encoding-free. The model has no rotary /
# learned position module.
assert not hasattr(model.model, 'rotary_emb'), 'unexpected rotary on NoPE'
for i, layer in enumerate(model.model.layers):
    if block_type(i) == 'attention':
        assert not hasattr(layer.mixer, 'rotary_emb')
print('NoPE attention confirmed (no rotary module)')

# 6. Explicit head_dim != hidden//num_heads is genuinely exercised.
assert HEAD_DIM != HIDDEN // NUM_HEADS, (HEAD_DIM, HIDDEN // NUM_HEADS)
assert KV_HEADS < NUM_HEADS and NUM_HEADS % KV_HEADS == 0
print('explicit head_dim + GQA confirmed')

# 7. Bias-free projections.
for i, layer in enumerate(model.model.layers):
    bt = block_type(i)
    if bt == 'mamba':
        assert layer.mixer.in_proj.bias is None
        assert layer.mixer.out_proj.bias is None
    elif bt == 'attention':
        assert layer.mixer.q_proj.bias is None
        assert layer.mixer.o_proj.bias is None
    elif bt == 'mlp':
        assert layer.mixer.up_proj.bias is None
        assert layer.mixer.down_proj.bias is None
print('bias-free projections confirmed')
print('all fixture self-checks passed')
