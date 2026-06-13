#!/usr/bin/env python3
"""Generate a tiny RANDOM Mamba parity fixture for
tests/TestNeuralPretrained.pas (no network access needed: the model is
randomly initialized from a pico config, never downloaded).

One fixture, KB-scale, pinned in tests/fixtures/:

  tiny_mamba.*: MambaForCausalLM (HF model_type "mamba", the
      state-spaces/mamba-130m-hf architecture) - the suite's SECOND
      NON-TRANSFORMER importer target after RWKV-4: a selective-SSM
      (S6) mixer with constant decode memory and no KV cache. Quirks
      exercised:
        - the A_log convention: the checkpoint stores A_log with
          A = -exp(A_log), discretized as abar = exp(delta * A);
          TNNetSelectiveSSM stores A_raw with a = exp(-delta *
          exp(A_raw)) - IDENTICAL formulas, so A_raw = A_log raw;
        - d_state = 4 > 1 states per channel with B_t/C_t in
          R^d_state SHARED across channels (x_proj split
          [time_step_rank | d_state | d_state]);
        - the LOW-RANK delta path: delta = softplus(dt_proj(
          x_proj_dt(x)) + dt_proj.bias), which the importer folds
          into one d_inner x d_inner W_d matrix plus the b_d bias;
        - the depthwise CAUSAL conv1d (k=4) WITH bias (use_conv_bias
          true) + SiLU before the scan;
        - the D skip (y += D * x) and the SiLU(z) gate from the
          in_proj x|z split (x first, z second);
        - bias-free in_proj/out_proj (use_bias false), per-token
          RMSNorm (norm/norm_f, gain only, eps=1e-5) and the TIED
          lm_head (the published Mamba checkpoints carry no separate
          lm_head.weight tensor).

The reference logits are computed by HF transformers in float64 (the
oracle convention of the committed fixtures) running slow_forward - the
CPU "slow path" sequential scan (the CUDA/mamba-ssm fast kernels are not
installed and the device is CPU, both checked below).

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/mamba_tiny_fixture.py
writes tests/fixtures/tiny_mamba{.safetensors,_config.json,_logits.json}.
Needs torch + transformers + safetensors.
"""
import copy
import json

import torch
from safetensors.torch import save_file
from transformers import MambaConfig, MambaForCausalLM
from transformers.models.mamba import modeling_mamba

HIDDEN = 8
LAYERS = 2
VOCAB = 13
STATE = 4          # d_state > 1: the multi-state Mamba scan
DT_RANK = 2        # low-rank delta path (folded by the importer)
EXPAND = 2         # d_inner = 16
CONV_K = 4
SEQ_LEN = 8
N_SEQUENCES = 3

torch.manual_seed(20260612)

cfg_dict = {
    'architectures': ['MambaForCausalLM'],
    'model_type': 'mamba',
    'vocab_size': VOCAB,
    'hidden_size': HIDDEN,
    'num_hidden_layers': LAYERS,
    'state_size': STATE,
    'time_step_rank': DT_RANK,
    'expand': EXPAND,
    'conv_kernel': CONV_K,
    'use_bias': False,
    'use_conv_bias': True,
    'layer_norm_epsilon': 1e-5,
    'tie_word_embeddings': True,
    'use_cache': True,
    'bos_token_id': 0,
    'eos_token_id': 0,
}
model = MambaForCausalLM(MambaConfig(**cfg_dict))

# The oracle MUST run HF's sequential CPU slow path, not a fused kernel.
assert not all((modeling_mamba.selective_state_update,
                modeling_mamba.selective_scan_fn,
                modeling_mamba.causal_conv1d_fn,
                modeling_mamba.causal_conv1d_update,
                modeling_mamba.mamba_inner_fn)), \
    'CUDA fast-path kernels installed - oracle would not be the slow path'
assert model.backbone.layers[0].mixer.x_proj.weight.device.type == 'cpu'
assert not model.backbone.layers[0].mixer.use_mambapy

# Pinned inputs (3+ rows: the Pascal parity helper requires it).
sequences = [[(5 * i + 3 * s + s * s) % (VOCAB - 1) + 1
              for i in range(SEQ_LEN)] for s in range(N_SEQUENCES)]

# Re-randomize away from the HF init so every quirk is visible in the
# oracle: dt pre-activations straddling zero (softplus != identity and
# != ReLU), O(1) decays per (channel,state) with real spread ACROSS the
# states of one channel (d_state > 1 must matter), non-trivial conv
# taps/biases, a non-one D skip and non-one RMSNorm gains.
with torch.no_grad():
    for layer in model.backbone.layers:
        mx = layer.mixer
        mx.A_log.copy_(torch.empty_like(mx.A_log).uniform_(-1.0, 1.5))
        mx.D.normal_(0.0, 0.8)
        mx.conv1d.weight.normal_(0.0, 0.5)
        mx.conv1d.bias.normal_(0.0, 0.4)
        mx.in_proj.weight.normal_(0.0, 0.5)
        mx.x_proj.weight.normal_(0.0, 0.5)
        mx.dt_proj.weight.normal_(0.0, 0.6)
        mx.dt_proj.bias.normal_(0.0, 0.8)
        layer.norm.weight.normal_(1.0, 0.25)
    model.backbone.norm_f.weight.normal_(1.0, 0.25)
    model.backbone.embeddings.weight.normal_(0.0, 0.8)
model = model.double().eval()

# Tied head: like the published Mamba checkpoints, lm_head.weight is NOT
# saved (it aliases backbone.embeddings.weight; safetensors refuses
# shared storage anyway).
sd = {k: v.to(torch.float32).clone().contiguous()
      for k, v in model.state_dict().items() if k != 'lm_head.weight'}
save_file(sd, 'tests/fixtures/tiny_mamba.safetensors')
with open('tests/fixtures/tiny_mamba_config.json', 'w') as f:
    json.dump(cfg_dict, f, indent=1)


def run(m):
    with torch.no_grad():
        return torch.stack([m(input_ids=torch.tensor([s])).logits[0]
                            for s in sequences])


logits = run(model)
with open('tests/fixtures/tiny_mamba_logits.json', 'w') as f:
    json.dump({'sequences': sequences, 'logits': logits.tolist()}, f)
print(f'wrote tiny_mamba.safetensors ({len(sd)} tensors) + config + oracle')
for k in sorted(sd):
    print(f'  {k} {list(sd[k].shape)}')

# ---- fixture self-checks: every quirk must be visible in the oracle ----
base = logits

# 0a. A_log convention: the Pascal layer's a = exp(-delta*exp(A_raw)) with
# A_raw = A_log copied RAW must equal HF's exp(delta * (-exp(A_log)))
# bit-tight for sample values.
for a_log in model.backbone.layers[0].mixer.A_log.flatten().tolist():
    for delta in (0.05, 0.7, 3.0):
        hf = torch.exp(torch.tensor(delta) * -torch.exp(torch.tensor(a_log)))
        ours = torch.exp(-torch.tensor(delta) * torch.exp(torch.tensor(a_log)))
        assert abs(hf - ours) == 0.0, (a_log, delta)
print('A_log convention round-trip check passed (identical formulas)')

# 0b. The dt fold W_d = dt_proj.weight @ x_proj.weight[:dt_rank] the
# importer performs must be mathematically exact (checked in float64).
mx = model.backbone.layers[0].mixer
x = torch.randn(5, 2 * HIDDEN, dtype=torch.float64)
two_step = mx.dt_proj.weight @ (mx.x_proj.weight[:DT_RANK] @ x.T)
folded_w = mx.dt_proj.weight @ mx.x_proj.weight[:DT_RANK]
assert (two_step - folded_w @ x.T).abs().max().item() < 1e-12
print('dt low-rank fold check passed')

# 1. d_state > 1: killing every state but s=0 (A_log[:,1:] = 12 decays
# them instantly AND C zeroed for s>0) must FAIL parity - the extra
# states must carry signal.
one_state = copy.deepcopy(model)
with torch.no_grad():
    for layer in one_state.backbone.layers:
        layer.mixer.A_log[:, 1:] = 12.0
        layer.mixer.x_proj.weight[DT_RANK + STATE + 1:, :] = 0.0  # C[1:]
d = (run(one_state) - base).abs().max().item()
assert d > 1e-2, f'd_state>1 had no effect ({d})'
print(f'd_state>1 effect on logits: max |diff| = {d:.4f}')

# 2. conv1d bias: zeroing it must FAIL parity.
nocb = copy.deepcopy(model)
with torch.no_grad():
    for layer in nocb.backbone.layers:
        layer.mixer.conv1d.bias.zero_()
d = (run(nocb) - base).abs().max().item()
assert d > 1e-2, f'conv bias had no effect ({d})'
print(f'conv-bias effect on logits: max |diff| = {d:.4f}')

# 3a. dt bias: zeroing dt_proj.bias must FAIL parity.
nodtb = copy.deepcopy(model)
with torch.no_grad():
    for layer in nodtb.backbone.layers:
        layer.mixer.dt_proj.bias.zero_()
d = (run(nodtb) - base).abs().max().item()
assert d > 1e-2, f'dt bias had no effect ({d})'
print(f'dt-bias effect on logits: max |diff| = {d:.4f}')

# 3b. dt SOFTPLUS: an importer that skipped the softplus (identity dt)
# must FAIL parity. softplus is referenced as nn.functional.softplus
# inside slow_forward, so patch it for one run.
real_softplus = torch.nn.functional.softplus
try:
    torch.nn.functional.softplus = lambda t: t
    d = (run(model) - base).abs().max().item()
finally:
    torch.nn.functional.softplus = real_softplus
assert d > 1e-2, f'dt softplus had no effect ({d})'
print(f'dt-softplus effect on logits: max |diff| = {d:.4f}')
assert (run(model) - base).abs().max().item() == 0.0  # patch undone

# 4. D skip: zeroing D must FAIL parity.
noD = copy.deepcopy(model)
with torch.no_grad():
    for layer in noD.backbone.layers:
        layer.mixer.D.zero_()
d = (run(noD) - base).abs().max().item()
assert d > 1e-2, f'D skip had no effect ({d})'
print(f'D-skip effect on logits: max |diff| = {d:.4f}')

# 5. SiLU(z) gate: zeroing the z half of in_proj (gate = silu(0) = 0,
# only the residual stream survives) must FAIL parity - the gate path
# must carry signal.
nogate = copy.deepcopy(model)
with torch.no_grad():
    for layer in nogate.backbone.layers:
        layer.mixer.in_proj.weight[EXPAND * HIDDEN:, :] = 0.0
d = (run(nogate) - base).abs().max().item()
assert d > 1e-2, f'SiLU(z) gate had no effect ({d})'
print(f'SiLU(z)-gate effect on logits: max |diff| = {d:.4f}')

# 6. Bias-free in_proj/out_proj (use_bias false) and the tied head.
for layer in model.backbone.layers:
    assert layer.mixer.in_proj.bias is None
    assert layer.mixer.out_proj.bias is None
assert model.lm_head.weight.data_ptr() == \
    model.backbone.embeddings.weight.data_ptr(), 'head not tied'
print('bias-free in/out_proj + tied-head checks passed')
print('all fixture self-checks passed')
