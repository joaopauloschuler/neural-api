#!/usr/bin/env python3
"""Generate a tiny RANDOM Falcon-Mamba parity fixture for
tests/TestNeuralPretrained.pas (no network access needed: the model is
randomly initialized from a pico config, never downloaded).

One fixture, KB-scale, pinned in tests/fixtures/:

  tiny_falcon_mamba.*: FalconMambaForCausalLM (HF model_type
      "falcon_mamba", the tiiuae/falcon-mamba-7b architecture) -
      Mamba-1 plus a WEIGHTLESS per-vector RMSNorm on the dt / B / C
      selection vectors between the x_proj split and the scan
      (rms_forward, config mixer_rms_eps). Quirks exercised on top of
      the plain-Mamba set (see tools/mamba_tiny_fixture.py):
        - the inner RMSNorms themselves (rms over dt_rank for dt and
          over d_state for B/C, eps INSIDE the sqrt, NO learned gain) -
          the importer maps them onto TNNetSelectiveSSM's Jamba
          inner-norm mode with unit gains;
        - the UNFOLDED low-rank dt path they force: dt = softplus(
          dt_proj(rms(x_proj_dt(x))) + dt_proj.bias) - the plain-Mamba
          W_d fold would be mathematically WRONG here (asserted below);
        - a non-default mixer_rms_eps (1e-3) so an importer that
          hardcodes 1e-6 FAILS parity;
        - the UNTIED lm_head (tie_word_embeddings false, like the real
          falcon-mamba-7b checkpoint, which ships lm_head.weight).

The reference logits are computed by HF transformers in float64 (the
oracle convention of the committed fixtures) running slow_forward - the
CPU "slow path" sequential scan (fast CUDA kernels absent, CPU device,
both checked below).

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/falcon_mamba_tiny_fixture.py
writes tests/fixtures/tiny_falcon_mamba{.safetensors,_config.json,_logits.json}.
Needs torch + transformers + safetensors.
"""
import copy
import json

import torch
from safetensors.torch import save_file
from transformers import FalconMambaConfig, FalconMambaForCausalLM
from transformers.models.falcon_mamba import modeling_falcon_mamba

HIDDEN = 8
LAYERS = 2
VOCAB = 13
STATE = 4          # d_state > 1: the multi-state scan
DT_RANK = 2        # low-rank delta path (UNFOLDED: rms sits inside it)
EXPAND = 2         # d_inner = 16
CONV_K = 4
MIXER_RMS_EPS = 1e-3   # non-default so a hardcoded 1e-6 fails parity
SEQ_LEN = 8
N_SEQUENCES = 3

torch.manual_seed(20260715)

cfg_dict = {
    'architectures': ['FalconMambaForCausalLM'],
    'model_type': 'falcon_mamba',
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
    'mixer_rms_eps': MIXER_RMS_EPS,
    'tie_word_embeddings': False,
    'use_cache': True,
    'bos_token_id': 0,
    'eos_token_id': 0,
}
model = FalconMambaForCausalLM(FalconMambaConfig(**cfg_dict))

# The oracle MUST run HF's sequential CPU slow path, not a fused kernel.
assert not all(getattr(modeling_falcon_mamba, name, None)
               for name in ('selective_state_update', 'selective_scan_fn',
                            'causal_conv1d_fn', 'causal_conv1d_update',
                            'mamba_inner_fn')), \
    'CUDA fast-path kernels installed - oracle would not be the slow path'
assert model.backbone.layers[0].mixer.x_proj.weight.device.type == 'cpu'
assert model.backbone.layers[0].mixer.rms_eps == MIXER_RMS_EPS

# Pinned inputs (3+ rows: the Pascal parity helper requires it).
sequences = [[(5 * i + 3 * s + s * s) % (VOCAB - 1) + 1
              for i in range(SEQ_LEN)] for s in range(N_SEQUENCES)]

# Re-randomize away from the HF init so every quirk is visible in the
# oracle (same rationale as the plain-Mamba fixture), including an
# UNTIED lm_head distinct from the embedding table.
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
    model.lm_head.weight.normal_(0.0, 0.6)
assert model.lm_head.weight.data_ptr() != \
    model.backbone.embeddings.weight.data_ptr(), 'head unexpectedly tied'
model = model.double().eval()

sd = {k: v.to(torch.float32).clone().contiguous()
      for k, v in model.state_dict().items()}
save_file(sd, 'tests/fixtures/tiny_falcon_mamba.safetensors')
with open('tests/fixtures/tiny_falcon_mamba_config.json', 'w') as f:
    json.dump(cfg_dict, f, indent=1)


def run(m):
    with torch.no_grad():
        return torch.stack([m(input_ids=torch.tensor([s])).logits[0]
                            for s in sequences])


logits = run(model)
with open('tests/fixtures/tiny_falcon_mamba_logits.json', 'w') as f:
    json.dump({'sequences': sequences, 'logits': logits.tolist()}, f)
print(f'wrote tiny_falcon_mamba.safetensors ({len(sd)} tensors) + '
      'config + oracle')
for k in sorted(sd):
    print(f'  {k} {list(sd[k].shape)}')

# ---- fixture self-checks: every quirk must be visible in the oracle ----
base = logits

# 1. The inner RMSNorms: patching rms_forward to identity (= plain Mamba)
# must FAIL parity - the norms must carry signal.
real_rms = modeling_falcon_mamba.rms_forward
try:
    modeling_falcon_mamba.rms_forward = \
        lambda t, variance_epsilon=1e-6: t
    d = (run(model) - base).abs().max().item()
finally:
    modeling_falcon_mamba.rms_forward = real_rms
assert d > 1e-2, f'inner RMSNorms had no effect ({d})'
print(f'inner-RMSNorm effect on logits: max |diff| = {d:.4f}')
assert (run(model) - base).abs().max().item() == 0.0  # patch undone

# 2. mixer_rms_eps: running with eps hardcoded to the 1e-6 default must
# FAIL parity (the fixture pins 1e-3 precisely to catch that).
try:
    modeling_falcon_mamba.rms_forward = \
        lambda t, variance_epsilon=1e-6: real_rms(t, 1e-6)
    d = (run(model) - base).abs().max().item()
finally:
    modeling_falcon_mamba.rms_forward = real_rms
assert d > 1e-4, f'mixer_rms_eps had no effect ({d})'
print(f'mixer_rms_eps(1e-3 vs 1e-6) effect on logits: max |diff| = {d:.6f}')
assert (run(model) - base).abs().max().item() == 0.0  # patch undone

# 3. The W_d fold plain Mamba uses is INVALID here: dt_proj(rms(ts)) !=
# (dt_proj @ x_proj_dt) applied to x for the fixture weights.
mx = model.backbone.layers[0].mixer
x = torch.randn(5, EXPAND * HIDDEN, dtype=torch.float64)
ts = x @ mx.x_proj.weight[:DT_RANK].T
unfolded = real_rms(ts, MIXER_RMS_EPS) @ mx.dt_proj.weight.T
folded = x @ (mx.dt_proj.weight @ mx.x_proj.weight[:DT_RANK]).T
assert (unfolded - folded).abs().max().item() > 1e-2, \
    'W_d fold accidentally valid - inner norm not between the factors?'
print('W_d-fold-invalid check passed (dt must stay unfolded)')

# 4. dt softplus (shared with plain Mamba, re-checked on this arch).
real_softplus = torch.nn.functional.softplus
try:
    torch.nn.functional.softplus = lambda t: t
    d = (run(model) - base).abs().max().item()
finally:
    torch.nn.functional.softplus = real_softplus
assert d > 1e-2, f'dt softplus had no effect ({d})'
print(f'dt-softplus effect on logits: max |diff| = {d:.4f}')
assert (run(model) - base).abs().max().item() == 0.0  # patch undone

# 5. d_state > 1 must matter (kill states s>0: instant decay + C[1:] = 0).
one_state = copy.deepcopy(model)
with torch.no_grad():
    for layer in one_state.backbone.layers:
        layer.mixer.A_log[:, 1:] = 12.0
        layer.mixer.x_proj.weight[DT_RANK + STATE + 1:, :] = 0.0  # C[1:]
d = (run(one_state) - base).abs().max().item()
assert d > 1e-2, f'd_state>1 had no effect ({d})'
print(f'd_state>1 effect on logits: max |diff| = {d:.4f}')

# 6. The UNTIED head: swapping lm_head for the embedding rows must FAIL
# parity (a tied-head importer shortcut would be wrong here).
tied = copy.deepcopy(model)
with torch.no_grad():
    tied.lm_head.weight.copy_(tied.backbone.embeddings.weight)
d = (run(tied) - base).abs().max().item()
assert d > 1e-2, f'untied head had no effect ({d})'
print(f'untied-head effect on logits: max |diff| = {d:.4f}')

# 7. Bias-free in_proj/out_proj (use_bias false).
for layer in model.backbone.layers:
    assert layer.mixer.in_proj.bias is None
    assert layer.mixer.out_proj.bias is None
print('bias-free in/out_proj check passed')
print('all fixture self-checks passed')
