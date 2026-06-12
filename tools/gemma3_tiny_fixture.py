#!/usr/bin/env python3
"""Generate a tiny RANDOM Gemma-3 TEXT-ONLY parity fixture for
tests/TestNeuralPretrained.pas (no network access needed: the model is
randomly initialized from a pico config, never downloaded - the same
recipe as tools/gemma2_tiny_fixture.py).

One fixture, ~10 KB, pinned in tests/fixtures/:

  tiny_gemma3.*: Gemma3ForCausalLM (model_type "gemma3_text") with every
      Gemma-3 delta on top of the Gemma-2 skeleton genuinely exercised
      (each one is ASSERTED non-vacuous below by re-running the oracle
      with that single delta disabled):
      - PER-HEAD QK-NORM: learnable-scale RMSNorm on q and k applied AFTER
        the projection and BEFORE RoPE (HF modeling_gemma3 ordering:
        q_norm(q_proj(x)) then apply_rotary_pos_emb). The shared
        [head_dim] q_norm/k_norm gains are zero-centered (1+w) and are
        re-randomized to non-zero w so a skipped/unloaded gain breaks
        parity. This REPLACES the Gemma-2 soft-caps (both None here, the
        Gemma3TextConfig default).
      - layer_types PATTERN: sliding_window_pattern=3 over 3 layers, so
        layers 0,1 are "sliding_attention" and layer 2 ((i+1) % 3 == 0) is
        "full_attention" - the pico analogue of the production 5:1 ratio
        (pattern 6); a NON-DEFAULT pattern so the config read is exercised.
        sliding_window=4 < MAX_POS=16 makes the local mask bite.
      - PER-LAYER-TYPE ROPE THETA: rope_local_base_freq=10.0 on the
        sliding layers vs rope_theta=1000.0 on the global layer (distinct
        values at pico scale so a single-theta import breaks parity).
      - query_pre_attn_scalar=12 != head_dim=6 COMBINED with the q-side
        RMSNorm: the Pascal side must fold the scale into the q_norm GAINS
        (a W_q fold would be erased by the norm).
      Gemma-1/2 carried over: GeGLU MLP (gelu_pytorch_tanh), zero-centered
      RMSNorm, embedding output scaled by sqrt(hidden_size) with the TIED
      lm_head reading the UNSCALED rows, MQA (1 kv head), DECOUPLED
      head_dim=6 with hidden=8, sandwich norms (all four per-block gains
      re-randomized to non-zero w).

Reference logits are computed by HF transformers in float64 (the oracle
convention of the committed fixtures).

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/gemma3_tiny_fixture.py
writes tests/fixtures/tiny_gemma3{.safetensors,_config.json,_logits.json}.
Needs torch + transformers (>= the Gemma-3 release) + safetensors.
"""
import json

import torch
from safetensors.torch import save_file
from transformers import Gemma3TextConfig, Gemma3ForCausalLM

N_LAYER = 3            # layers 0,1 sliding; layer 2 global (pattern 3)
N_HEAD = 2
N_KV_HEAD = 1          # MQA
HEAD_DIM = 6           # != D_MODEL // N_HEAD = 4: decoupled head_dim
D_MODEL = 8
D_FF = 12
MAX_POS = 16
SLIDING_WINDOW = 4     # < MAX_POS so the local mask actually bites
SLIDING_PATTERN = 3    # non-default (HF default 6); every 3rd layer global
QUERY_PRE_ATTN_SCALAR = 12     # != HEAD_DIM and combined with QK-norm
ROPE_THETA = 1000.0            # global layers
ROPE_LOCAL_BASE_FREQ = 10.0    # sliding layers (distinct: per-layer theta)
N_SEQUENCES = 3
VOCAB = 13

torch.manual_seed(20260612)
gemma3_cfg = {
    'architectures': ['Gemma3ForCausalLM'],
    'model_type': 'gemma3_text',
    'hidden_size': D_MODEL,
    'intermediate_size': D_FF,
    'num_hidden_layers': N_LAYER,
    'num_attention_heads': N_HEAD,
    'num_key_value_heads': N_KV_HEAD,
    'head_dim': HEAD_DIM,
    'vocab_size': VOCAB,
    'max_position_embeddings': MAX_POS,
    'rms_norm_eps': 1e-6,
    'rope_theta': ROPE_THETA,
    'rope_local_base_freq': ROPE_LOCAL_BASE_FREQ,
    'attention_bias': False,
    'tie_word_embeddings': True,
    'sliding_window': SLIDING_WINDOW,
    'sliding_window_pattern': SLIDING_PATTERN,
    'query_pre_attn_scalar': QUERY_PRE_ATTN_SCALAR,
    'attn_logit_softcapping': None,
    'final_logit_softcapping': None,
    'hidden_activation': 'gelu_pytorch_tanh',
}


def make_model(**overrides):
    cfg = {**gemma3_cfg, **overrides}
    cfg.pop('architectures')
    return Gemma3ForCausalLM(Gemma3TextConfig(
        **cfg, attn_implementation='eager'))


model = make_model()
assert N_HEAD * HEAD_DIM != D_MODEL, 'head_dim decoupling not exercised'
assert model.config.layer_types == [
    'sliding_attention', 'sliding_attention', 'full_attention'], \
    'expected 2 sliding + 1 global layer (pattern 3)'
assert model.model.layers[0].self_attn.sliding_window == SLIDING_WINDOW, \
    'layer 0 must be sliding (local)'
assert model.model.layers[2].self_attn.sliding_window is None, \
    'layer 2 must be the global layer'

# HF zero-inits the (zero-centered) RMSNorm gains; carry NON-ZERO gains for
# all FOUR per-block sandwich norms AND the per-head q_norm/k_norm so every
# gain-loading path is exercised (a skipped norm or unloaded gain breaks
# parity). The q/k projections are NOT amplified here - no soft-cap to
# excite, and the QK-norm keeps the score scale bounded anyway.
with torch.no_grad():
    for layer in model.model.layers:
        layer.input_layernorm.weight.normal_(0.0, 0.5)
        layer.post_attention_layernorm.weight.normal_(0.0, 0.5)
        layer.pre_feedforward_layernorm.weight.normal_(0.0, 0.5)
        layer.post_feedforward_layernorm.weight.normal_(0.0, 0.5)
        layer.self_attn.q_norm.weight.normal_(0.0, 0.5)
        layer.self_attn.k_norm.weight.normal_(0.0, 0.5)
        layer.mlp.gate_proj.weight.mul_(8.0)
    model.model.norm.weight.normal_(0.0, 0.5)

model = model.double().eval()
sd = {k: v.to(torch.float32).contiguous()
      for k, v in model.state_dict().items()}
save_file(sd, 'tests/fixtures/tiny_gemma3.safetensors')
with open('tests/fixtures/tiny_gemma3_config.json', 'w') as f:
    json.dump(gemma3_cfg, f, indent=1)
sequences = [[(5 * i + 2 * s + s * s) % VOCAB for i in range(MAX_POS)]
             for s in range(N_SEQUENCES)]
with torch.no_grad():
    logits = [model(input_ids=torch.tensor([seq])).logits[0].tolist()
              for seq in sequences]
with open('tests/fixtures/tiny_gemma3_logits.json', 'w') as f:
    json.dump({'sequences': sequences, 'logits': logits}, f)
print(f'wrote tiny_gemma3.safetensors ({len(sd)} tensors) '
      f'+ config + logits ({N_SEQUENCES} sequences of {MAX_POS})')


def logits_with(state_dict=None, **overrides):
    """Oracle logits for sequence 0 with one delta disabled."""
    other = make_model(**overrides)
    other.load_state_dict(state_dict or model.state_dict())
    other = other.double().eval()
    with torch.no_grad():
        return other(input_ids=torch.tensor([sequences[0]])).logits


with torch.no_grad():
    ref = model(input_ids=torch.tensor([sequences[0]])).logits

# Zeroed q/k norm gains (gain == 1 everywhere): proves the LOADED gains
# matter, not just the structural norm.
flat_sd = {k: v.clone() for k, v in model.state_dict().items()}
for k in flat_sd:
    if k.endswith('q_norm.weight') or k.endswith('k_norm.weight'):
        flat_sd[k].zero_()

# Each Gemma-3 delta must MATTER: re-run the oracle with that single delta
# disabled and assert the logits move, otherwise the fixture would
# vacuously "verify" the corresponding import path.
checks = {
    'sliding window (4 -> 16 = full)':
        logits_with(sliding_window=MAX_POS),
    'layer pattern (3 -> 4 = no global layer)':
        logits_with(sliding_window_pattern=4),
    'per-layer rope theta (local 10.0 -> global 1000.0)':
        logits_with(rope_local_base_freq=ROPE_THETA),
    'query_pre_attn_scalar (12 -> head_dim 6)':
        logits_with(query_pre_attn_scalar=HEAD_DIM),
    'q/k norm gains (random -> zero, gain=1)':
        logits_with(state_dict=flat_sd),
}
for name, alt in checks.items():
    effect = (ref - alt).abs().max().item()
    assert effect > 1e-3, f'{name} had no effect on the logits ({effect})'
    print(f'{name}: max |diff| = {effect:.4f}')
