#!/usr/bin/env python3
"""Generate a tiny RANDOM Qwen3.5/3.6 (dense, model_type qwen3_5) parity
fixture for tests/TestNeuralPretrained.pas (no network access needed: the
model is randomly initialized from a pico config, never downloaded).

One fixture, ~40 KB, pinned in tests/fixtures/:

  tiny_qwen3_5.*: Qwen3_5ForCausalLM - the HYBRID text decoder of
      Qwen3.6-27B, 8 layers covering the (L L L F) layer_types pattern
      TWICE:
      - "linear_attention" layers (0,1,2, 4,5,6) are the gated-DeltaNet
        mixer: in_proj_qkv -> causal depthwise conv-4 (NO bias) + SiLU over
        the q|k|v slab, z/b/a side projections (NOT conv'd), the gated
        delta-rule recurrence (A_log / dt_bias / plain-gain gated RMSNorm),
        out_proj.
      - "full_attention" layers (3, 7) are Qwen3 attention with PARTIAL
        rotary (partial_rotary_factor=0.25 of head_dim=8 -> rotary_dim=2)
        and the per-head OUTPUT GATE: q_proj is DOUBLE width, per head
        [query(Dh)|gate(Dh)], attn := attn * sigmoid(gate) before o_proj.
      - EVERY RMSNorm gain is ZERO-CENTERED (Qwen3_5RMSNorm computes
        (1 + w) * xhat) EXCEPT the DeltaNet gated norm linear_attn.norm
        (plain ones-init gain) - the fixture randomizes BOTH families so a
        misplaced +1 offset breaks parity in either direction.

  The safetensors mimics the REAL multimodal checkpoint layout
  (Qwen3_5ForConditionalGeneration): the text backbone is renamed under
  "model.language_model.", lm_head.weight stays top-level, and DUMMY
  model.visual.* / mtp.* tensors are added - the importer must SKIP them
  (HF's ForCausalLM route ignores them too). The config nests the text
  fields under "text_config" (top-level model_type "qwen3_5", inner
  "qwen3_5_text") with rope_theta / partial_rotary_factor inside
  rope_parameters - exactly the transformers 5.x on-disk shape.

Reference logits are computed by HF transformers in float64 (the oracle
convention of the committed fixtures).

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/qwen3_5_tiny_fixture.py
writes tests/fixtures/tiny_qwen3_5{.safetensors,_config.json,_logits.json}.
Needs torch + transformers (with qwen3_5) + safetensors.
"""
import json

import torch
from safetensors.torch import save_file
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

N_LAYER = 8
LAYER_TYPES = ['linear_attention', 'linear_attention', 'linear_attention',
               'full_attention'] * 2
N_HEAD = 2
N_KV_HEAD = 1
HEAD_DIM = 8           # rotary_dim = 8 * 0.25 = 2 (even, partial rotary)
D_MODEL = 8
D_FF = 12
MAX_POS = 16
N_SEQUENCES = 3
VOCAB = 13
LIN_K_HEADS = 2
LIN_V_HEADS = 4
LIN_K_DIM = 4
LIN_V_DIM = 4
CONV_K = 4

torch.manual_seed(20260718)
text_cfg = {
    'model_type': 'qwen3_5_text',
    'hidden_size': D_MODEL,
    'intermediate_size': D_FF,
    'num_hidden_layers': N_LAYER,
    'num_attention_heads': N_HEAD,
    'num_key_value_heads': N_KV_HEAD,
    'head_dim': HEAD_DIM,
    'vocab_size': VOCAB,
    'max_position_embeddings': MAX_POS,
    'rms_norm_eps': 1e-6,
    'attention_bias': False,
    'tie_word_embeddings': False,
    'hidden_act': 'silu',
    'layer_types': LAYER_TYPES,
    'linear_num_key_heads': LIN_K_HEADS,
    'linear_num_value_heads': LIN_V_HEADS,
    'linear_key_head_dim': LIN_K_DIM,
    'linear_value_head_dim': LIN_V_DIM,
    'linear_conv_kernel_dim': CONV_K,
    'rope_parameters': {
        'rope_type': 'default',
        'rope_theta': 10000.0,
        'partial_rotary_factor': 0.25,
        'mrope_section': [1, 0, 0],  # rotary_dim/2 = 1 frequency pair
        'mrope_interleaved': True,   # a NO-OP for 1-D text positions
    },
}
config = Qwen3_5TextConfig(
    **{k: v for k, v in text_cfg.items() if k != 'model_type'})
model = Qwen3_5ForCausalLM(config)
model.config._attn_implementation = 'eager'

# HF inits at std ~0.02 (numerically negligible pico blocks), zero-inits the
# zero-centered RMSNorm offsets (gain exactly 1) and ones-inits dt_bias /
# the gated-norm gain. Re-randomize EVERYTHING that has a dedicated loading
# path so each path is genuinely pinned:
#   - projections at std 0.2 (large enough that every path moves the logits,
#     small enough that the f32 forward tracks the f64 oracle well inside
#     the 1e-4 parity gate - the 8-layer recurrent stack amplifies rounding,
#     see the f32-consistency assertion below);
#   - zero-centered norm offsets ~N(0, 0.3): a loader that forgets the +1
#     (or adds it twice) breaks parity;
#   - the PLAIN DeltaNet gated-norm gain ~N(1, 0.3): a loader that treats
#     it as zero-centered (adds +1) breaks parity;
#   - A_log / dt_bias / conv taps randomized so the decay/beta gates and the
#     causal conv are exercised with non-trivial values.
with torch.no_grad():
    for p in model.parameters():
        if p.dim() >= 2:
            p.normal_(0.0, 0.2)
    for layer in model.model.layers:
        ln = [layer.input_layernorm.weight,
              layer.post_attention_layernorm.weight]
        for w in ln:
            w.normal_(0.0, 0.3)
        if hasattr(layer, 'self_attn'):
            layer.self_attn.q_norm.weight.normal_(0.0, 0.3)
            layer.self_attn.k_norm.weight.normal_(0.0, 0.3)
        if hasattr(layer, 'linear_attn'):
            layer.linear_attn.norm.weight.normal_(1.0, 0.3)
            layer.linear_attn.A_log.uniform_(-2.0, 1.0)
            layer.linear_attn.dt_bias.normal_(0.0, 0.5)
    model.model.norm.weight.normal_(0.0, 0.3)

model = model.double().eval()
state = model.state_dict()

sequences = [[(7 * i + 3 * s + s * s) % VOCAB for i in range(MAX_POS)]
             for s in range(N_SEQUENCES)]
with torch.no_grad():
    logits = [model(input_ids=torch.tensor([seq])).logits[0].tolist()
              for seq in sequences]

# --- output-gate assertion (pins the per-head [query|gate] q_proj split) --
gated = Qwen3_5ForCausalLM(config)
sd_zero_gate = {k: v.clone() for k, v in state.items()}
for name, w in sd_zero_gate.items():
    if name.endswith('self_attn.q_proj.weight'):
        w2 = w.view(N_HEAD, 2 * HEAD_DIM, D_MODEL)
        w2[:, HEAD_DIM:, :] = 0.0  # gate logits -> 0 -> sigmoid = 0.5
gated.load_state_dict(sd_zero_gate)
gated.config._attn_implementation = 'eager'
gated = gated.double().eval()
with torch.no_grad():
    lg = model(input_ids=torch.tensor([sequences[0]])).logits
    l0 = gated(input_ids=torch.tensor([sequences[0]])).logits
gate_effect = (lg - l0).abs().max().item()
assert gate_effect > 1e-3, \
    f'attention output gate had no effect on the logits ({gate_effect})'
print(f'attn output-gate effect on logits: max |diff| = {gate_effect:.4f}')

# --- DeltaNet plain-norm assertion (a +1 offset there must move logits) ---
offset = Qwen3_5ForCausalLM(config)
sd_off = {k: v.clone() for k, v in state.items()}
for name, w in sd_off.items():
    if name.endswith('linear_attn.norm.weight'):
        w += 1.0
offset.load_state_dict(sd_off)
offset.config._attn_implementation = 'eager'
offset = offset.double().eval()
with torch.no_grad():
    l1 = offset(input_ids=torch.tensor([sequences[0]])).logits
norm_effect = (lg - l1).abs().max().item()
assert norm_effect > 1e-3, \
    f'DeltaNet gated-norm gain had no effect on the logits ({norm_effect})'
print(f'DeltaNet plain-norm gain effect: max |diff| = {norm_effect:.4f}')

# --- f32-consistency assertion: HF's OWN float32 forward must track the
# f64 oracle far inside the Pascal test's 1e-4 gate, or the fixture is too
# numerically hot for any faithful f32 importer to pass. ---
m32 = Qwen3_5ForCausalLM(config)
m32.load_state_dict({k: v.float() for k, v in state.items()})
m32.config._attn_implementation = 'eager'
m32 = m32.float().eval()
with torch.no_grad():
    f32_drift = max(
        (m32(input_ids=torch.tensor([seq])).logits[0].double() -
         torch.tensor(lgt)).abs().max().item()
        for seq, lgt in zip(sequences, logits))
assert f32_drift < 3e-5, \
    f'fixture too hot: HF f32 drifts {f32_drift} from the f64 oracle'
print(f'HF f32 vs f64 oracle drift: max |diff| = {f32_drift:.2e}')

# --- serialize in the REAL multimodal checkpoint layout -------------------
sd = {}
for k, v in state.items():
    if k == 'lm_head.weight':
        sd[k] = v.to(torch.float32).contiguous()
    elif k.startswith('model.'):
        sd['model.language_model.' + k[len('model.'):]] = \
            v.to(torch.float32).contiguous()
    else:
        raise AssertionError(f'unexpected key {k}')
# Dummy vision-tower / MTP tensors the TEXT importer must SKIP.
sd['model.visual.patch_embed.proj.weight'] = torch.zeros(2, 3)
sd['mtp.layers.0.dummy.weight'] = torch.zeros(2, 2)
save_file(sd, 'tests/fixtures/tiny_qwen3_5.safetensors')

wrapper_cfg = {
    'architectures': ['Qwen3_5ForConditionalGeneration'],
    'model_type': 'qwen3_5',
    'text_config': text_cfg,
}
with open('tests/fixtures/tiny_qwen3_5_config.json', 'w') as f:
    json.dump(wrapper_cfg, f, indent=1)
with open('tests/fixtures/tiny_qwen3_5_logits.json', 'w') as f:
    json.dump({'sequences': sequences, 'logits': logits}, f)
print(f'wrote tiny_qwen3_5.safetensors ({len(sd)} tensors) '
      f'+ config + logits ({N_SEQUENCES} sequences of {MAX_POS})')
