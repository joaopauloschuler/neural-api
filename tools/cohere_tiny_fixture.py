#!/usr/bin/env python3
"""Generate tiny RANDOM Cohere Command-R / Aya parity fixtures for
tests/TestNeuralPretrained.pas (no network access: the models are randomly
initialized from pico configs, never downloaded - the same recipe as
tools/gemma3_tiny_fixture.py / tools/modernbert_tiny_fixture.py).

Two fixtures, ~10 KB each, pinned in tests/fixtures/:

  tiny_cohere.*: CohereForCausalLM (model_type "cohere") exercising every
      Cohere delta that distinguishes it from a Llama clone:
      - PARALLEL residual: one input_layernorm feeds BOTH attention and MLP,
        summed into the residual (x = x + Attn(LN(x)) + MLP(LN(x))).
      - mean-subtracting, bias-free, weight-only CohereLayerNorm (NOT the
        Llama RMSNorm); the gains are re-randomized non-zero so a skipped
        norm or unloaded gain breaks parity.
      - logit_scale != 1 multiplying the final logits (folded into the tied
        LM head on the Pascal side).
      - tied input/output embeddings.
      - INTERLEAVED RoPE pair layout (NO rotate_half row permutation).
      - GQA (num_key_value_heads=1 < num_attention_heads=2; MQA).
      - PER-HEAD q_norm/k_norm CohereLayerNorm over head_dim, applied AFTER
        the q/k projection and BEFORE RoPE (use_qk_norm=True), with PER-HEAD
        DISTINCT gains re-randomized non-zero.

  tiny_cohere2.*: Cohere2ForCausalLM (model_type "cohere2", Command-R7B)
      adding the alternating sliding-window/global attention pattern AND -
      the crucial cohere2 distinction - RoPE applied ONLY on the SLIDING
      layers; the GLOBAL layers use NoPE (no positional embedding):
      - sliding_window=4 < MAX_POS=16 so the local mask bites.
      - sliding_window_pattern=4 over 4 layers -> layers 0,1,2 sliding
        (RoPE) and layer 3 global (NoPE).
      - cohere2 has NO qk_norm.

Reference logits are computed by HF transformers in float64 (the oracle
convention of the committed fixtures), so the Pascal f32 import is checked
against an exact float64 forward on the same (f32-rounded) weights.

Coded by Claude (AI).

Usage (from the repo root):
  /home/bpsa/x/bin/python tools/cohere_tiny_fixture.py
writes tests/fixtures/tiny_cohere{,2}{.safetensors,_config.json,_logits.json}.
Needs torch + transformers (>= the Cohere release) + safetensors.
"""
import json

import torch
from safetensors.torch import save_file
from transformers import (CohereConfig, CohereForCausalLM,
                          Cohere2Config, Cohere2ForCausalLM)

N_HEAD = 2
N_KV_HEAD = 1          # MQA (genuine GQA sharing)
HEAD_DIM = 4
D_MODEL = 8
D_FF = 12
MAX_POS = 16
VOCAB = 13
LOGIT_SCALE = 0.0625
ROPE_THETA = 1000.0
LN_EPS = 1e-5
N_SEQUENCES = 3

sequences = [[(5 * i + 2 * s + s * s) % VOCAB for i in range(MAX_POS)]
             for s in range(N_SEQUENCES)]


def randomize(model, with_qk_norm):
    """HF inits norm gains to 1 and weights at std 0.02 - vacuous at pico
    scale (attention near-uniform). Re-randomize to O(1) so every loaded
    weight matters; gains get non-zero w so a skipped/unloaded norm breaks
    parity."""
    with torch.no_grad():
        for layer in model.model.layers:
            layer.input_layernorm.weight.normal_(1.0, 0.3)
            layer.self_attn.q_proj.weight.normal_(0.0, 0.5)
            layer.self_attn.k_proj.weight.normal_(0.0, 0.5)
            layer.self_attn.v_proj.weight.normal_(0.0, 0.5)
            layer.self_attn.o_proj.weight.normal_(0.0, 0.5)
            layer.mlp.gate_proj.weight.normal_(0.0, 0.5)
            layer.mlp.up_proj.weight.normal_(0.0, 0.5)
            layer.mlp.down_proj.weight.normal_(0.0, 0.5)
            if with_qk_norm:
                layer.self_attn.q_norm.weight.normal_(1.0, 0.3)
                layer.self_attn.k_norm.weight.normal_(1.0, 0.3)
        model.model.norm.weight.normal_(1.0, 0.3)
        model.model.embed_tokens.weight.normal_(0.0, 1.0)


def dump(model, cfg, name):
    model = model.double().eval()
    sd = {k: v.to(torch.float32).contiguous()
          for k, v in model.state_dict().items()}
    save_file(sd, f'tests/fixtures/tiny_{name}.safetensors')
    with open(f'tests/fixtures/tiny_{name}_config.json', 'w') as f:
        json.dump(cfg, f, indent=1)
    with torch.no_grad():
        logits = [model(input_ids=torch.tensor([seq])).logits[0].tolist()
                  for seq in sequences]
    with open(f'tests/fixtures/tiny_{name}_logits.json', 'w') as f:
        json.dump({'sequences': sequences, 'logits': logits}, f)
    print(f'wrote tiny_{name}.safetensors ({len(sd)} tensors) + config + '
          f'logits ({N_SEQUENCES}x{MAX_POS})')
    with torch.no_grad():
        return model(input_ids=torch.tensor([sequences[0]])).logits


# ---------------------------------------------------------------- cohere ---
torch.manual_seed(20260613)
cohere_cfg = {
    'architectures': ['CohereForCausalLM'],
    'model_type': 'cohere',
    'hidden_size': D_MODEL,
    'intermediate_size': D_FF,
    'num_hidden_layers': 3,
    'num_attention_heads': N_HEAD,
    'num_key_value_heads': N_KV_HEAD,
    'head_dim': HEAD_DIM,
    'vocab_size': VOCAB,
    'max_position_embeddings': MAX_POS,
    'layer_norm_eps': LN_EPS,
    'rope_theta': ROPE_THETA,
    'logit_scale': LOGIT_SCALE,
    'attention_bias': False,
    'tie_word_embeddings': True,
    'use_qk_norm': True,
    'hidden_act': 'silu',
}


def make_cohere(**ov):
    cfg = {**cohere_cfg, **ov}
    cfg.pop('architectures')
    return CohereForCausalLM(CohereConfig(**cfg, attn_implementation='eager'))


m1 = make_cohere()
assert N_HEAD * HEAD_DIM != D_MODEL or N_KV_HEAD != N_HEAD, 'GQA not exercised'
randomize(m1, with_qk_norm=True)
ref1 = dump(m1, cohere_cfg, 'cohere')


def cohere_logits_with(state_dict=None, **ov):
    other = make_cohere(**ov)
    other.load_state_dict(state_dict or m1.state_dict())
    other = other.double().eval()
    with torch.no_grad():
        return other(input_ids=torch.tensor([sequences[0]])).logits


# Each delta must MATTER. logit_scale and qk_norm are the genuinely new
# pieces for this family - assert both move the logits.
flat_qk = {k: v.clone() for k, v in m1.state_dict().items()}
for k in flat_qk:
    if k.endswith('q_norm.weight') or k.endswith('k_norm.weight'):
        flat_qk[k].fill_(1.0)  # CohereLayerNorm gain 1 (structural norm only)
checks1 = {
    'logit_scale (0.0625 -> 1.0)': cohere_logits_with(logit_scale=1.0),
    'qk_norm gains (random -> 1)': cohere_logits_with(state_dict=flat_qk),
}
for nm, alt in checks1.items():
    eff = (ref1 - alt).abs().max().item()
    assert eff > 1e-3, f'cohere: {nm} had no effect ({eff})'
    print(f'cohere {nm}: max |diff| = {eff:.4f}')

# --------------------------------------------------------------- cohere2 ---
torch.manual_seed(20260614)
cohere2_cfg = {
    'architectures': ['Cohere2ForCausalLM'],
    'model_type': 'cohere2',
    'hidden_size': D_MODEL,
    'intermediate_size': D_FF,
    'num_hidden_layers': 4,
    'num_attention_heads': N_HEAD,
    'num_key_value_heads': N_KV_HEAD,
    'head_dim': HEAD_DIM,
    'vocab_size': VOCAB,
    'max_position_embeddings': MAX_POS,
    'layer_norm_eps': LN_EPS,
    'rope_theta': ROPE_THETA,
    'logit_scale': LOGIT_SCALE,
    'attention_bias': False,
    'tie_word_embeddings': True,
    'sliding_window': 4,
    'sliding_window_pattern': 4,
    'hidden_act': 'silu',
}


def make_cohere2(**ov):
    cfg = {**cohere2_cfg, **ov}
    cfg.pop('architectures')
    return Cohere2ForCausalLM(
        Cohere2Config(**cfg, attn_implementation='eager'))


m2 = make_cohere2()
assert m2.config.layer_types == [
    'sliding_attention', 'sliding_attention', 'sliding_attention',
    'full_attention'], 'expected 3 sliding + 1 global (pattern 4)'
assert m2.model.layers[0].self_attn.sliding_window == 4, 'layer 0 sliding'
assert m2.model.layers[3].self_attn.sliding_window is None, 'layer 3 global'
randomize(m2, with_qk_norm=False)
ref2 = dump(m2, cohere2_cfg, 'cohere2')


def cohere2_logits_with(**ov):
    other = make_cohere2(**ov)
    other.load_state_dict(m2.state_dict())
    other = other.double().eval()
    with torch.no_grad():
        return other(input_ids=torch.tensor([sequences[0]])).logits


checks2 = {
    'sliding window (4 -> 16 = full)': cohere2_logits_with(sliding_window=MAX_POS),
    'sliding pattern (4 -> 5 = all sliding, no global NoPE layer)':
        cohere2_logits_with(sliding_window_pattern=5),
}
for nm, alt in checks2.items():
    eff = (ref2 - alt).abs().max().item()
    assert eff > 1e-3, f'cohere2: {nm} had no effect ({eff})'
    print(f'cohere2 {nm}: max |diff| = {eff:.4f}')
