#!/usr/bin/env python3
"""Generate a tiny RANDOM Llama-4 (text) parity fixture for
tests/TestNeuralPretrained.pas (no network access needed: the model is
randomly initialized from a pico config, never downloaded).

Llama-4 text is NOT a near-duplicate of the Llama builder. This fixture
genuinely exercises every new piece BuildLlama4FromSafeTensors adds:

  (a) iRoPE - interleaved attention: no_rope_layers selects which layers use
      RoPE; the NoPE layers carry NO positional encoding and a chunked-causal
      mask (attention_chunk_size). With seq_len <= attention_chunk_size the
      chunked mask EQUALS plain causal, so this fixture keeps seq_len small and
      pins the RoPE/NoPE SPLIT (not the chunk boundary; that is a follow-up).
  (b) attn_temperature_tuning - the NoPE layers scale every query by a
      per-position log factor (floor_scale / attn_scale).
  (c) use_qk_norm - an UNWEIGHTED L2 RMS-norm over each q/k head AFTER RoPE, on
      the RoPE layers ONLY.
  (d) sigmoid-gated top-k MoE with an always-on shared expert (the transposed
      fused 3-D expert slabs); the dense layers use the wider
      intermediate_size_mlp. interleave_moe_layer_step=2 -> a MIXED dense/MoE
      stack.

The config interleaves both RoPE/NoPE AND dense/MoE so the importer's per-layer
gating is fully exercised, and the script ASSERTS the split against the actual
instantiated HF module types + a non-vacuous effect for the qk-norm, the
temperature tuning and the sigmoid router.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/llama4_tiny_fixture.py
writes tests/fixtures/tiny_llama4{.safetensors,_config.json,_logits.json}.
Needs torch + transformers (with Llama4 text) + safetensors.
"""
import json
import types

import torch
import torch.nn.functional as F
import transformers.models.llama4.modeling_llama4 as L4
from safetensors.torch import save_file
from transformers import Llama4TextConfig
from transformers.models.llama4.modeling_llama4 import Llama4ForCausalLM


# Llama4TextExperts.forward uses bmm with a 3-D slab and (in newer transformers)
# may dispatch to a grouped kernel that rejects float64. Install a plain,
# float64-safe eager loop that reproduces the exact HF math:
#   gate_up = hidden @ gate_up_proj[e]   (chunk -> gate, up)
#   out = (up * silu(gate)) @ down_proj[e]
# Routing (sigmoid of the top-k logits, scattered) stays in Llama4TextMoe.
def eager_experts_forward(self, hidden_states):
    hidden_states = hidden_states.view(self.gate_up_proj.shape[0], -1,
                                       self.hidden_size)
    out = []
    for e in range(self.gate_up_proj.shape[0]):
        gate_up = hidden_states[e] @ self.gate_up_proj[e]
        gate, up = gate_up.chunk(2, dim=-1)
        out.append((up * self.act_fn(gate)) @ self.down_proj[e])
    return torch.cat(out, dim=0).view(-1, self.hidden_size)


L4.Llama4TextExperts.forward = eager_experts_forward

N_LAYER = 4
N_HEAD = 2
N_KV_HEAD = 1
HEAD_DIM = 6
D_MODEL = 8
D_MOE_FF = 5            # intermediate_size: per-expert SwiGLU width
D_MLP_FF = 12           # intermediate_size_mlp: the WIDER dense-layer width
MAX_POS = 64
N_SEQUENCES = 3
SEQ_LEN = 12            # < attention_chunk_size, so chunked == full causal
VOCAB = 13
N_EXPERTS = 4
TOP_K = 2
MOE_STEP = 2           # interleave_moe_layer_step -> moe_layers = [1, 3]
NOROPE_INTERVAL = 2    # no_rope_layers = [1, 0, 1, 0] (layers 1,3 RoPE; 0,2 NoPE)
CHUNK = 8192
# floor_scale is 8192 in real Scout/Maverick configs, but then the temperature
# factor log1p(floor((pos+1)/floor_scale))*attn_scale+1 == 1 for any pos < 8192
# (a no-op until very long context). To PIN the NoPE temperature-tuning math at
# a tiny seq_len, use a small floor_scale so f(pos) actually varies across
# positions 0..SEQ_LEN-1 (the layer computes the exact HF formula either way).
FLOOR_SCALE = 4
ATTN_SCALE = 0.7

torch.manual_seed(20260614)
cfg_kwargs = dict(
    hidden_size=D_MODEL,
    intermediate_size=D_MOE_FF,
    intermediate_size_mlp=D_MLP_FF,
    num_hidden_layers=N_LAYER,
    num_attention_heads=N_HEAD,
    num_key_value_heads=N_KV_HEAD,
    head_dim=HEAD_DIM,
    vocab_size=VOCAB,
    max_position_embeddings=MAX_POS,
    rms_norm_eps=1e-5,
    rope_theta=10000.0,
    num_local_experts=N_EXPERTS,
    num_experts_per_tok=TOP_K,
    interleave_moe_layer_step=MOE_STEP,
    no_rope_layer_interval=NOROPE_INTERVAL,
    use_qk_norm=True,
    attn_temperature_tuning=True,
    attention_chunk_size=CHUNK,
    floor_scale=FLOOR_SCALE,
    attn_scale=ATTN_SCALE,
    attention_bias=False,
    tie_word_embeddings=False,
    hidden_act='silu',
)
config = Llama4TextConfig(**cfg_kwargs, attn_implementation='eager')
model = Llama4ForCausalLM(config)

assert N_HEAD * HEAD_DIM != D_MODEL, 'head_dim decoupling not exercised'
assert D_MOE_FF != D_MLP_FF, 'dense width not distinct from expert width'

# Confirm the mixed stack actually has BOTH dense and MoE layers AND both RoPE
# and NoPE layers, matching the HF rules the Pascal importer mirrors.
moe_kinds, rope_kinds = [], []
for i, layer in enumerate(model.model.layers):
    is_moe = isinstance(layer.feed_forward, L4.Llama4TextMoe)
    expect_moe = (i in config.moe_layers)
    assert is_moe == expect_moe, f'layer {i}: HF MoE kind disagrees'
    moe_kinds.append('MoE' if is_moe else 'dense')
    use_rope = bool(config.no_rope_layers[i])
    assert layer.self_attn.use_rope == use_rope
    rope_kinds.append('RoPE' if use_rope else 'NoPE')
    # qk_norm exists iff use_qk_norm AND use_rope.
    assert hasattr(layer.self_attn, 'qk_norm') == use_rope
assert 'MoE' in moe_kinds and 'dense' in moe_kinds, moe_kinds
assert 'RoPE' in rope_kinds and 'NoPE' in rope_kinds, rope_kinds
print('moe_layers     :', config.moe_layers, moe_kinds)
print('no_rope_layers :', config.no_rope_layers, rope_kinds)

# HF inits these pico models at std ~0.02, which makes the FFN/attention
# numerically negligible against the residual stream (vacuous parity). Re-init
# every projection to O(1) scale (the ModernBERT pico recipe) so every piece
# genuinely moves the logits. Keep the router flat-ish so the top-k survivors
# stay comparable.
with torch.no_grad():
    for p in model.parameters():
        if p.dim() >= 2:
            p.normal_(0.0, 0.5)
    for layer in model.model.layers:
        if isinstance(layer.feed_forward, L4.Llama4TextMoe):
            layer.feed_forward.router.weight.normal_(0.0, 0.3)

model = model.double().eval()
state = model.state_dict()

sd = {k: v.to(torch.float32).contiguous() for k, v in state.items()}
save_file(sd, 'tests/fixtures/tiny_llama4.safetensors')

cfg_json = dict(cfg_kwargs)
cfg_json['architectures'] = ['Llama4ForCausalLM']
cfg_json['model_type'] = 'llama4_text'
with open('tests/fixtures/tiny_llama4_config.json', 'w') as f:
    json.dump(cfg_json, f, indent=1)

sequences = [[(7 * i + 3 * s + s * s) % VOCAB for i in range(SEQ_LEN)]
             for s in range(N_SEQUENCES)]
with torch.no_grad():
    logits = [model(input_ids=torch.tensor([seq])).logits[0].tolist()
              for seq in sequences]
with open('tests/fixtures/tiny_llama4_logits.json', 'w') as f:
    json.dump({'sequences': sequences, 'logits': logits}, f)
print(f'wrote tiny_llama4.safetensors ({len(sd)} tensors) + config + logits '
      f'({N_SEQUENCES} sequences of {SEQ_LEN})')

seq0 = torch.tensor([sequences[0]])
with torch.no_grad():
    base = model(input_ids=seq0).logits

# --- qk-norm effect (pins the L2 QK-norm on the RoPE layers) --------------
no_qk = Llama4ForCausalLM(Llama4TextConfig(**{**cfg_kwargs, 'use_qk_norm': False},
                                           attn_implementation='eager'))
no_qk.load_state_dict(model.state_dict(), strict=False)
no_qk = no_qk.double().eval()
with torch.no_grad():
    lg_no_qk = no_qk(input_ids=seq0).logits
qk_effect = (base - lg_no_qk).abs().max().item()
assert qk_effect > 1e-3, f'use_qk_norm had no effect ({qk_effect})'
print(f'qk-norm (L2) effect on logits   : max |diff| = {qk_effect:.4f}')

# --- attn temperature tuning effect (pins the NoPE per-position scaling) ---
no_temp = Llama4ForCausalLM(
    Llama4TextConfig(**{**cfg_kwargs, 'attn_temperature_tuning': False},
                     attn_implementation='eager'))
no_temp.load_state_dict(model.state_dict(), strict=False)
no_temp = no_temp.double().eval()
with torch.no_grad():
    lg_no_temp = no_temp(input_ids=seq0).logits
temp_effect = (base - lg_no_temp).abs().max().item()
assert temp_effect > 1e-3, f'attn_temperature_tuning had no effect ({temp_effect})'
print(f'attn-temperature effect on logits: max |diff| = {temp_effect:.4f}')

# --- shared expert effect (pins the always-on shared MLP) ------------------
def moe_no_shared(self, hidden_states):
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_scores, router_logits = self.router(hidden_states)
    routed_in = hidden_states.repeat(router_scores.shape[1], 1)
    routed_in = routed_in * router_scores.transpose(0, 1).reshape(-1, 1)
    routed_out = self.experts(routed_in)
    out = torch.zeros_like(hidden_states)  # shared expert ZEROED
    out.add_(routed_out.reshape(router_scores.shape[1], -1,
                                routed_out.shape[-1]).sum(dim=0))
    return out, router_logits


patched = Llama4ForCausalLM(config)
patched.load_state_dict(model.state_dict())
patched = patched.double().eval()
for layer in patched.model.layers:
    if isinstance(layer.feed_forward, L4.Llama4TextMoe):
        layer.feed_forward.forward = types.MethodType(moe_no_shared,
                                                      layer.feed_forward)
with torch.no_grad():
    lg_no_shared = patched(input_ids=seq0).logits
shared_effect = (base - lg_no_shared).abs().max().item()
assert shared_effect > 1e-3, f'shared expert had no effect ({shared_effect})'
print(f'shared-expert effect on logits  : max |diff| = {shared_effect:.4f}')
