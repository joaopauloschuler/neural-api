#!/usr/bin/env python3
"""Generate a tiny RANDOM Mamba-2 / SSD parity fixture for
tests/TestNeuralPretrained.pas (no network access: the model is randomly
initialized from a pico config, never downloaded).

One fixture, KB-scale, pinned in tests/fixtures/:

  tiny_mamba2.*: Mamba2ForCausalLM (HF model_type "mamba2", the
      state-spaces/mamba2-* / Falcon-Mamba / Codestral-Mamba architecture) -
      the architecturally DISTINCT successor to Mamba-1: a MULTI-HEAD SSM
      (State-Space Duality, Dao & Gu 2024). Quirks exercised:
        - the per-HEAD SCALAR state transition A = -exp(A_log) (A_log is
          NumHeads-long, NOT the per-(channel,state) matrix of Mamba-1),
          discretized as a = exp(dt*A);
        - B/C SHARED across the heads of a group (n_groups < num_heads,
          GQA-like): here n_groups=1 < num_heads=2, so both heads share the
          single group's B_t/C_t;
        - the per-head dt = softplus(dt_in + dt_bias) and the D[h] head skip;
        - in_proj packs [gate | (x|B|C) | dt]; only the (x|B|C) slab
          (conv_dim = intermediate + 2*n_groups*state) goes through the
          depthwise CAUSAL conv1d (k=4) WITH bias + SiLU;
        - the EXTRA grouped gated RMSNorm BEFORE out_proj:
          out = rms(y * silu(gate)) * norm_weight  (HF MambaRMSNormGated);
        - bias-free in/out_proj (use_bias false) and the TIED lm_head.

The reference logits are computed by HF transformers in float64 running
torch_forward - the CPU naive SSD path (the CUDA/mamba-ssm fast kernels are
not installed and the device is CPU, both checked below). The naive path is
the chunked-SSD reference, which is mathematically EQUAL to the head-wise
recurrent scan the Pascal TNNetMamba2 leaf implements.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/make_pico_mamba2_fixture.py
writes tests/fixtures/tiny_mamba2{.safetensors,_config.json,_logits.json}.
"""
import copy
import json

import torch
from safetensors.torch import save_file
from transformers import Mamba2Config, Mamba2ForCausalLM
from transformers.models.mamba2 import modeling_mamba2

HIDDEN = 8
LAYERS = 2
VOCAB = 13
STATE = 4          # d_state per head
NUM_HEADS = 2
HEAD_DIM = 4       # intermediate = num_heads*head_dim = 8
N_GROUPS = 1       # < num_heads: B/C shared across the group's heads
EXPAND = 1         # intermediate = expand*hidden = 8 = num_heads*head_dim
CONV_K = 4
CHUNK = 8
SEQ_LEN = 8
N_SEQUENCES = 3

torch.manual_seed(20260613)

cfg_dict = {
    'architectures': ['Mamba2ForCausalLM'],
    'model_type': 'mamba2',
    'vocab_size': VOCAB,
    'hidden_size': HIDDEN,
    'num_hidden_layers': LAYERS,
    'state_size': STATE,
    'num_heads': NUM_HEADS,
    'head_dim': HEAD_DIM,
    'n_groups': N_GROUPS,
    'expand': EXPAND,
    'conv_kernel': CONV_K,
    'chunk_size': CHUNK,
    'use_bias': False,
    'use_conv_bias': True,
    'layer_norm_epsilon': 1e-5,
    'rms_norm': True,
    # time_step_limit omitted from the saved config: HF serializes it as
    # [0.0, Infinity], which is not valid JSON for the Pascal reader (and the
    # importer does not consume it - the default (0, inf) clamp is a no-op).
    'tie_word_embeddings': True,
    'use_cache': True,
    'bos_token_id': 0,
    'eos_token_id': 0,
    'pad_token_id': 0,
}
model = Mamba2ForCausalLM(Mamba2Config(**cfg_dict))

# The oracle MUST run HF's naive CPU path, not a fused kernel.
assert not any((modeling_mamba2.selective_state_update,
                modeling_mamba2.causal_conv1d_fn,
                modeling_mamba2.causal_conv1d_update)), \
    'CUDA fast-path kernels installed - oracle would not be the naive path'
assert model.backbone.layers[0].mixer.in_proj.weight.device.type == 'cpu'

# Pinned inputs (3+ rows: the Pascal parity helper requires it).
sequences = [[(5 * i + 3 * s + s * s) % (VOCAB - 1) + 1
              for i in range(SEQ_LEN)] for s in range(N_SEQUENCES)]

# Re-randomize away from the HF init so every quirk is visible in the oracle:
# dt pre-activations straddling zero, O(1) per-head decays with real spread
# ACROSS heads, non-trivial conv taps/biases, non-one D head skips and
# non-one RMSNorm gains.
with torch.no_grad():
    for layer in model.backbone.layers:
        mx = layer.mixer
        mx.A_log.copy_(torch.empty_like(mx.A_log).uniform_(-1.0, 1.5))
        mx.D.normal_(0.0, 0.8)
        mx.dt_bias.normal_(0.0, 0.8)
        mx.conv1d.weight.normal_(0.0, 0.5)
        mx.conv1d.bias.normal_(0.0, 0.4)
        mx.in_proj.weight.normal_(0.0, 0.5)
        mx.norm.weight.normal_(1.0, 0.25)
        layer.norm.weight.normal_(1.0, 0.25)
    model.backbone.norm_f.weight.normal_(1.0, 0.25)
    model.backbone.embeddings.weight.normal_(0.0, 0.8)
model = model.double().eval()

# Tied head: lm_head.weight is NOT saved (it aliases the embeddings).
sd = {k: v.to(torch.float32).clone().contiguous()
      for k, v in model.state_dict().items() if k != 'lm_head.weight'}
save_file(sd, 'tests/fixtures/tiny_mamba2.safetensors')
with open('tests/fixtures/tiny_mamba2_config.json', 'w') as f:
    json.dump(cfg_dict, f, indent=1)


def run(m):
    with torch.no_grad():
        return torch.stack([m(input_ids=torch.tensor([s])).logits[0]
                            for s in sequences])


logits = run(model)
with open('tests/fixtures/tiny_mamba2_logits.json', 'w') as f:
    json.dump({'sequences': sequences, 'logits': logits.tolist()}, f)
print(f'wrote tiny_mamba2.safetensors ({len(sd)} tensors) + config + oracle')
for k in sorted(sd):
    print(f'  {k} {list(sd[k].shape)}')

# ---- fixture self-checks: every quirk must be visible in the oracle ----
base = logits

# 1. Per-head SCALAR A_log: forcing both heads to one shared A_log must FAIL
# parity (the per-head spread must carry signal).
shared_a = copy.deepcopy(model)
with torch.no_grad():
    for layer in shared_a.backbone.layers:
        layer.mixer.A_log.fill_(layer.mixer.A_log[0].item())
d = (run(shared_a) - base).abs().max().item()
assert d > 1e-2, f'per-head A_log had no effect ({d})'
print(f'per-head A_log effect on logits: max |diff| = {d:.4f}')

# 2. conv1d bias: zeroing it must FAIL parity.
nocb = copy.deepcopy(model)
with torch.no_grad():
    for layer in nocb.backbone.layers:
        layer.mixer.conv1d.bias.zero_()
d = (run(nocb) - base).abs().max().item()
assert d > 1e-2, f'conv bias had no effect ({d})'
print(f'conv-bias effect on logits: max |diff| = {d:.4f}')

# 3. dt_bias: zeroing it must FAIL parity.
nodtb = copy.deepcopy(model)
with torch.no_grad():
    for layer in nodtb.backbone.layers:
        layer.mixer.dt_bias.zero_()
d = (run(nodtb) - base).abs().max().item()
assert d > 1e-2, f'dt bias had no effect ({d})'
print(f'dt-bias effect on logits: max |diff| = {d:.4f}')

# 4. D head skip: zeroing D must FAIL parity.
noD = copy.deepcopy(model)
with torch.no_grad():
    for layer in noD.backbone.layers:
        layer.mixer.D.zero_()
d = (run(noD) - base).abs().max().item()
assert d > 1e-2, f'D skip had no effect ({d})'
print(f'D-skip effect on logits: max |diff| = {d:.4f}')

# 5. gated RMSNorm gate: zeroing the gate slice of in_proj (gate=silu(0)=0,
# the gated norm output collapses) must FAIL parity.
nogate = copy.deepcopy(model)
with torch.no_grad():
    for layer in nogate.backbone.layers:
        layer.mixer.in_proj.weight[:layer.mixer.intermediate_size, :] = 0.0
d = (run(nogate) - base).abs().max().item()
assert d > 1e-2, f'gated-norm gate had no effect ({d})'
print(f'gate effect on logits: max |diff| = {d:.4f}')

# 6. norm.weight (the grouped RMSNorm gain): perturbing it must FAIL parity.
nonorm = copy.deepcopy(model)
with torch.no_grad():
    for layer in nonorm.backbone.layers:
        layer.mixer.norm.weight.fill_(1.0)
d = (run(nonorm) - base).abs().max().item()
assert d > 1e-2, f'gated-norm gain had no effect ({d})'
print(f'gated-norm gain effect on logits: max |diff| = {d:.4f}')

# 7. Bias-free in/out_proj + tied head.
for layer in model.backbone.layers:
    assert layer.mixer.in_proj.bias is None
    assert layer.mixer.out_proj.bias is None
assert model.lm_head.weight.data_ptr() == \
    model.backbone.embeddings.weight.data_ptr(), 'head not tied'
print('bias-free in/out_proj + tied-head checks passed')
print('all fixture self-checks passed')
