#!/usr/bin/env python3
"""Generate a Q8_0-CAPABLE tiny RANDOM Gemma-2 fixture for the GGUF
export-drift test in tests/TestNeuralPretrained.pas (no network: the model
is randomly initialized from a pico config, never downloaded).

WHY a second gemma2 fixture: the committed tiny_gemma2.* fixture is pico-
WIDTH (hidden_size=8, head_dim=6, intermediate=12) so that the Gemma-2
deltas land in their nonlinear regions at tiny magnitudes. Those widths are
NOT multiples of the Q8_0 block size (32), so the GGUF writer refuses to
emit them as Q8_0. This fixture mirrors the tiny_llama_q8.* convention: a
gemma2 whose every contiguous (last) dimension is a multiple of 32, so the
Q8_0 export path is exercisable. F16 export uses the narrow fixture (any
width); Q8_0 export uses this one.

All the Gemma-2 deltas are kept ON (alternating local/global, decoupled
head_dim, query_pre_attn_scalar, attn/final logit soft-capping, the four
sandwich RMSNorms, GeGLU, tied head) so the importer still goes through the
full gemma2 path; the Q8_0 quantization drift is what the test bounds.

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/gemma2_q8_tiny_fixture.py
writes tests/fixtures/tiny_gemma2_q8{.safetensors,_config.json}.
Needs torch + transformers (>= the Gemma-2 release) + safetensors.
"""
import json

import torch
from safetensors.torch import save_file
from transformers import Gemma2Config, Gemma2ForCausalLM

# Every contiguous dimension a multiple of the Q8_0 block size 32.
N_LAYER = 2            # layer 0 sliding, layer 1 global (the alternation)
N_HEAD = 2
N_KV_HEAD = 1          # MQA
HEAD_DIM = 32          # multiple of 32; != D_MODEL // N_HEAD: still decoupled
D_MODEL = 32
D_FF = 64
MAX_POS = 16
SLIDING_WINDOW = 4     # < MAX_POS so the local mask actually bites
QUERY_PRE_ATTN_SCALAR = 48     # != HEAD_DIM: scaling delta exercised
ATTN_SOFTCAP = 5.0
FINAL_SOFTCAP = 0.5
VOCAB = 32             # multiple of 32 (embed/head rows are length D_MODEL)

torch.manual_seed(20260614)
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

cfg = {k: v for k, v in gemma2_cfg.items() if k != 'architectures'}
model = Gemma2ForCausalLM(Gemma2Config(**cfg, attn_implementation='eager'))
assert N_HEAD * HEAD_DIM != D_MODEL, 'head_dim decoupling not exercised'
assert model.model.layers[0].self_attn.sliding_window == SLIDING_WINDOW
assert model.model.layers[1].self_attn.sliding_window is None

with torch.no_grad():
    for layer in model.model.layers:
        layer.input_layernorm.weight.normal_(0.0, 0.5)
        layer.post_attention_layernorm.weight.normal_(0.0, 0.5)
        layer.pre_feedforward_layernorm.weight.normal_(0.0, 0.5)
        layer.post_feedforward_layernorm.weight.normal_(0.0, 0.5)
        layer.self_attn.q_proj.weight.mul_(4.0)
        layer.self_attn.k_proj.weight.mul_(4.0)
    model.model.norm.weight.normal_(0.0, 0.5)

model = model.eval()
sd = {k: v.to(torch.float32).contiguous()
      for k, v in model.state_dict().items()
      if k != 'lm_head.weight'}  # tied head: stored once via embed_tokens
save_file(sd, 'tests/fixtures/tiny_gemma2_q8.safetensors')
with open('tests/fixtures/tiny_gemma2_q8_config.json', 'w') as f:
    json.dump(gemma2_cfg, f, indent=1)
print(f'wrote tiny_gemma2_q8.safetensors ({len(sd)} tensors) + config '
      f'(hidden={D_MODEL}, head_dim={HEAD_DIM}, ff={D_FF}, vocab={VOCAB})')
