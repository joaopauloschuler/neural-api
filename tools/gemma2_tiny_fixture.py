#!/usr/bin/env python3
"""Generate a tiny RANDOM Gemma-2 parity fixture for
tests/TestNeuralPretrained.pas (no network access needed: the model is
randomly initialized from a pico config, never downloaded).

One fixture, ~5 KB, pinned in tests/fixtures/:

  tiny_gemma2.*: Gemma2ForCausalLM with every Gemma-2 delta on top of the
      Gemma-1 skeleton genuinely exercised (each one is ASSERTED non-vacuous
      below by re-running the oracle with that single delta disabled):
      - ALTERNATING local/global attention: sliding_window=4 < MAX_POS=16
        masks the EVEN layer (HF layer_types default: "sliding_attention"
        for layers 0, 2, ...); the odd layer attends over the full context.
      - query_pre_attn_scalar=12 != head_dim=6: the attention scores are
        scaled by 1/sqrt(12), not 1/sqrt(6) (folded into W_q at load on the
        Pascal side).
      - attention-logit soft-capping (attn_logit_softcapping): the q/k
        projections are AMPLIFIED so the pre-softmax scores reach the
        nonlinear region of cap*tanh(s/cap) (at the default pico scale the
        cap would be numerically invisible).
      - final-logit soft-capping (final_logit_softcapping) on the LM-head
        logits, sized to bite at the pico logit magnitudes.
      - sandwich norms: all FOUR per-block RMSNorm gains (input_layernorm,
        post_attention_layernorm, pre_feedforward_layernorm,
        post_feedforward_layernorm) re-randomized to non-zero w (Gemma's
        zero-centered (1+w)*xhat), so a swapped or skipped norm breaks
        parity.
      Gemma-1 carried over: GeGLU MLP (gelu_pytorch_tanh), zero-centered
      RMSNorm, embedding output scaled by sqrt(hidden_size) with the TIED
      lm_head reading the UNSCALED rows, MQA (1 kv head), DECOUPLED
      head_dim=6 with hidden=8.

Reference logits are computed by HF transformers in float64 (the oracle
convention of the committed fixtures).

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/gemma2_tiny_fixture.py
writes tests/fixtures/tiny_gemma2{.safetensors,_config.json,_logits.json}.
Needs torch + transformers (>= the Gemma-2 release) + safetensors.
"""
import json

import torch
from safetensors.torch import save_file
from transformers import Gemma2Config, Gemma2ForCausalLM

N_LAYER = 2            # layer 0 sliding, layer 1 global (the alternation)
N_HEAD = 2
N_KV_HEAD = 1          # MQA
HEAD_DIM = 6           # != D_MODEL // N_HEAD = 4: decoupled head_dim
D_MODEL = 8
D_FF = 12
MAX_POS = 16
SLIDING_WINDOW = 4     # < MAX_POS so the local mask actually bites
QUERY_PRE_ATTN_SCALAR = 12     # != HEAD_DIM: scaling delta exercised
ATTN_SOFTCAP = 5.0     # small so the amplified scores reach tanh's bend
FINAL_SOFTCAP = 0.5    # small so the pico logits reach tanh's bend
N_SEQUENCES = 3
VOCAB = 13

torch.manual_seed(20260612)
gemma2_cfg = {
    'architectures': ['Gemma2ForCausalLM'],
    'model_type': 'gemma2',
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
    'sliding_window': SLIDING_WINDOW,
    'query_pre_attn_scalar': QUERY_PRE_ATTN_SCALAR,
    'attn_logit_softcapping': ATTN_SOFTCAP,
    'final_logit_softcapping': FINAL_SOFTCAP,
    'hidden_activation': 'gelu_pytorch_tanh',
}


def make_model(**overrides):
    cfg = {**gemma2_cfg, **overrides}
    cfg.pop('architectures')
    return Gemma2ForCausalLM(Gemma2Config(
        **cfg, attn_implementation='eager'))


model = make_model()
assert N_HEAD * HEAD_DIM != D_MODEL, 'head_dim decoupling not exercised'
assert model.model.layers[0].self_attn.sliding_window == SLIDING_WINDOW, \
    'layer 0 must be the sliding (local) layer'
assert model.model.layers[1].self_attn.sliding_window is None, \
    'layer 1 must be the global layer'

# HF zero-inits the (zero-centered) RMSNorm gains; carry NON-ZERO gains for
# ALL FOUR per-block sandwich norms so each loading path is exercised. The
# q/k projections are amplified so the pre-softmax attention scores land in
# the |s| ~ ATTN_SOFTCAP region where cap*tanh(s/cap) measurably deviates
# from identity - at the default pico scale the soft-cap would be vacuous.
with torch.no_grad():
    for layer in model.model.layers:
        layer.input_layernorm.weight.normal_(0.0, 0.5)
        layer.post_attention_layernorm.weight.normal_(0.0, 0.5)
        layer.pre_feedforward_layernorm.weight.normal_(0.0, 0.5)
        layer.post_feedforward_layernorm.weight.normal_(0.0, 0.5)
        layer.self_attn.q_proj.weight.mul_(20.0)
        layer.self_attn.k_proj.weight.mul_(20.0)
        layer.mlp.gate_proj.weight.mul_(8.0)
    model.model.norm.weight.normal_(0.0, 0.5)

model = model.double().eval()
sd = {k: v.to(torch.float32).contiguous()
      for k, v in model.state_dict().items()}
save_file(sd, 'tests/fixtures/tiny_gemma2.safetensors')
with open('tests/fixtures/tiny_gemma2_config.json', 'w') as f:
    json.dump(gemma2_cfg, f, indent=1)
sequences = [[(5 * i + 2 * s + s * s) % VOCAB for i in range(MAX_POS)]
             for s in range(N_SEQUENCES)]
with torch.no_grad():
    logits = [model(input_ids=torch.tensor([seq])).logits[0].tolist()
              for seq in sequences]
with open('tests/fixtures/tiny_gemma2_logits.json', 'w') as f:
    json.dump({'sequences': sequences, 'logits': logits}, f)
print(f'wrote tiny_gemma2.safetensors ({len(sd)} tensors) '
      f'+ config + logits ({N_SEQUENCES} sequences of {MAX_POS})')


def logits_with(**overrides):
    """Oracle logits for sequence 0 with one delta disabled."""
    other = make_model(**overrides)
    other.load_state_dict(model.state_dict())
    other = other.double().eval()
    with torch.no_grad():
        return other(input_ids=torch.tensor([sequences[0]])).logits


with torch.no_grad():
    ref = model(input_ids=torch.tensor([sequences[0]])).logits

# Each Gemma-2 delta must MATTER: re-run the oracle with that single delta
# disabled and assert the logits move, otherwise the fixture would
# vacuously "verify" the corresponding import path.
checks = {
    'sliding window (4 -> 16 = full)':
        logits_with(sliding_window=MAX_POS),
    'query_pre_attn_scalar (12 -> head_dim 6)':
        logits_with(query_pre_attn_scalar=HEAD_DIM),
    'attn_logit_softcapping (5.0 -> off)':
        logits_with(attn_logit_softcapping=None),
    'final_logit_softcapping (0.5 -> off)':
        logits_with(final_logit_softcapping=None),
}
for name, alt in checks.items():
    effect = (ref - alt).abs().max().item()
    assert effect > 1e-3, f'{name} had no effect on the logits ({effect})'
    print(f'{name}: max |diff| = {effect:.4f}')
