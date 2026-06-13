#!/usr/bin/env python3
"""Generate a tiny RANDOM Gemma-1 parity fixture for
tests/TestNeuralPretrained.pas (no network access needed: the model is
randomly initialized from a pico config, never downloaded).

One fixture, ~5 KB, pinned in tests/fixtures/:

  tiny_gemma.*: GemmaForCausalLM with every Gemma-1 delta on the Llama
      path genuinely exercised:
      - GeGLU MLP (gated tanh-GELU, gelu_pytorch_tanh): the script ASSERTS
        that the same weights under a SiLU gate (the Llama/SwiGLU default)
        give different logits, so the activation switch is not vacuous.
        The committed config carries the LEGACY spelling "hidden_act":
        "gelu" (no "hidden_activation" key) - older HF Gemma configs say
        "gelu" but Gemma means the tanh approximation; transformers
        special-cases this and the Pascal reader must match.
      - zero-centered RMSNorm ((1+w)*xhat): HF zero-inits the gains, which
        already distinguishes 1+w from w (w=0 -> gain 1 vs gain 0), but the
        script re-randomizes them anyway and ASSERTS that zeroing them back
        changes the logits, so the w-loading path is exercised too.
      - embedding output scaled by sqrt(hidden_size), TIED lm_head reading
        the UNSCALED rows (Gemma always ties).
      - MQA: num_key_value_heads=1 < num_attention_heads=2.
      - DECOUPLED head_dim=6 with hidden=8 (num_heads*head_dim = 12 != 8,
        the Gemma-7B shape quirk: head_dim=256 with hidden=3072, heads=16).

Reference logits are computed by HF transformers in float64 (the oracle
convention of the committed fixtures).

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/gemma_tiny_fixture.py
writes tests/fixtures/tiny_gemma{.safetensors,_config.json,_logits.json}.
Needs torch + transformers (>= the Gemma release) + safetensors.
"""
import json

import torch
from safetensors.torch import save_file
from transformers import GemmaConfig, GemmaForCausalLM

N_LAYER = 2
N_HEAD = 2
N_KV_HEAD = 1          # MQA (Gemma-2B uses num_key_value_heads=1)
HEAD_DIM = 6           # != D_MODEL // N_HEAD = 4: decoupled head_dim
D_MODEL = 8
D_FF = 12
MAX_POS = 16
N_SEQUENCES = 3
VOCAB = 13

torch.manual_seed(20260612)
gemma_cfg = {
    'architectures': ['GemmaForCausalLM'],
    'model_type': 'gemma',
    'hidden_size': D_MODEL,
    'intermediate_size': D_FF,
    'num_hidden_layers': N_LAYER,
    'num_attention_heads': N_HEAD,
    'num_key_value_heads': N_KV_HEAD,
    'head_dim': HEAD_DIM,
    'vocab_size': VOCAB,
    'max_position_embeddings': MAX_POS,
    'rms_norm_eps': 1e-6,
    'rope_theta': 10000.0,
    'attention_bias': False,
    'tie_word_embeddings': True,
}
# transformers 5.x dropped GemmaConfig's 'hidden_activation' field; the
# activation is config.hidden_act (default 'gelu_pytorch_tanh'). The oracle
# is constructed with the EXPLICIT tanh spelling - the committed config's
# legacy "gelu" spelling means exactly this activation for Gemma.
model = GemmaForCausalLM(GemmaConfig(
    **gemma_cfg, hidden_act='gelu_pytorch_tanh',
    attn_implementation='eager'))
assert N_HEAD * HEAD_DIM != D_MODEL, 'head_dim decoupling not exercised'

# HF zero-inits the (zero-centered) RMSNorm gains; carry NON-ZERO gains so
# the fixture also exercises the w-loading path, not just the folded +1.
# The gate_proj weights are amplified so the gate pre-activations land in
# the |x| ~ 2 region where tanh-GELU and SiLU differ measurably - at the
# default pico scale the two activations agree to ~7e-5 on the logits and
# the GeGLU-vs-SiLU sanity assert below would be vacuous.
with torch.no_grad():
    for layer in model.model.layers:
        layer.input_layernorm.weight.normal_(0.0, 0.5)
        layer.post_attention_layernorm.weight.normal_(0.0, 0.5)
        layer.mlp.gate_proj.weight.mul_(8.0)
    model.model.norm.weight.normal_(0.0, 0.5)

model = model.double().eval()
sd = {k: v.to(torch.float32).contiguous()
      for k, v in model.state_dict().items()}
save_file(sd, 'tests/fixtures/tiny_gemma.safetensors')
# The committed config uses the LEGACY "hidden_act": "gelu" spelling (and NO
# "hidden_activation" key) on purpose: the importer must map it to the tanh
# approximation the way transformers' Gemma special-case does.
with open('tests/fixtures/tiny_gemma_config.json', 'w') as f:
    json.dump({**gemma_cfg, 'hidden_act': 'gelu'}, f, indent=1)
sequences = [[(5 * i + 2 * s + s * s) % VOCAB for i in range(MAX_POS)]
             for s in range(N_SEQUENCES)]
with torch.no_grad():
    logits = [model(input_ids=torch.tensor([seq])).logits[0].tolist()
              for seq in sequences]
with open('tests/fixtures/tiny_gemma_logits.json', 'w') as f:
    json.dump({'sequences': sequences, 'logits': logits}, f)
print(f'wrote tiny_gemma.safetensors ({len(sd)} tensors) '
      f'+ config + logits ({N_SEQUENCES} sequences of {MAX_POS})')

# Sanity 1: the GeGLU gate must MATTER - the same weights under the
# Llama-default SiLU gate must give different logits, otherwise the
# activation parameterization would be vacuously "verified".
silu = GemmaForCausalLM(GemmaConfig(
    **gemma_cfg, hidden_act='silu', attn_implementation='eager'))
silu.load_state_dict(model.state_dict())
silu = silu.double().eval()
with torch.no_grad():
    lg = model(input_ids=torch.tensor([sequences[0]])).logits
    ls = silu(input_ids=torch.tensor([sequences[0]])).logits
act_effect = (lg - ls).abs().max().item()
assert act_effect > 1e-3, \
    f'GeGLU vs SiLU had no effect on the logits ({act_effect})'
print(f'GeGLU-vs-SiLU effect on logits: max |diff| = {act_effect:.4f}')

# Sanity 2: zeroing the RMSNorm gains back (the HF init) must change the
# logits, otherwise the w-loading path would be untested.
zeros = GemmaForCausalLM(GemmaConfig(
    **gemma_cfg, hidden_act='gelu_pytorch_tanh',
    attn_implementation='eager'))
sd_zero = {k: v.clone() for k, v in model.state_dict().items()}
for k in sd_zero:
    if 'layernorm' in k or k == 'model.norm.weight':
        sd_zero[k] = torch.zeros_like(sd_zero[k])
zeros.load_state_dict(sd_zero)
zeros = zeros.double().eval()
with torch.no_grad():
    lz = zeros(input_ids=torch.tensor([sequences[0]])).logits
norm_effect = (lg - lz).abs().max().item()
assert norm_effect > 1e-3, \
    f'RMSNorm gains had no effect on the logits ({norm_effect})'
print(f'RMSNorm gain effect on logits: max |diff| = {norm_effect:.4f}')
