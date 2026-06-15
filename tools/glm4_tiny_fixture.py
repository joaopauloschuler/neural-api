#!/usr/bin/env python3
"""Generate a tiny RANDOM GLM-4 parity fixture for
tests/TestNeuralPretrained.pas (no network access needed: the model is
randomly initialized from a pico config, never downloaded).

One fixture, ~10 KB, pinned in tests/fixtures/:

  tiny_glm4.*: Glm4ForCausalLM (model_type "glm4") with every GLM-4 delta on
      the Llama path genuinely exercised:
      - FOUR-norm SANDWICH block: input_layernorm (attention pre-norm) and
        post_attention_layernorm (FFN pre-norm) PLUS post_self_attn_layernorm
        INSIDE the attention residual branch and post_mlp_layernorm INSIDE
        the FFN residual branch. ALL FOUR gains are re-randomized (HF
        ones-inits every RMSNorm) and the script ASSERTS that resetting the
        two IN-BRANCH post-norms back to ones changes the logits, so a
        two-norm (plain pre-norm) import cannot pass;
      - PARTIAL rotary (partial_rotary_factor 0.5: head_dim 4 -> rotary_dim 2,
        the first pair of each head rotated, the tail passes through);
      - INTERLEAVED rotary (GLM-4 rotates consecutive channel pairs over
        x[0::2]/x[1::2]) - a rotate_half (half-split) import mis-permutes the
        q/k rows, so the partial+interleaved path is load-bearing;
      - SEPARATE q/k/v projections WITH bias (attention_bias true) and a
        bias-free o_proj; the q/k/v biases are re-randomized and asserted to
        move the logits;
      - FUSED bias-free mlp.gate_up_proj (gate|up rows, the Phi-3 packing).
      GQA (1 kv head < 2 query heads) and an UNTIED lm_head are also on.

Reference logits are computed by HF transformers (Glm4ForCausalLM is present
in the venv) in float64, the oracle convention of the committed fixtures.

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/glm4_tiny_fixture.py
writes tests/fixtures/tiny_glm4{.safetensors,_config.json,_logits.json}.
Needs torch + transformers (>= the GLM-4 release) + safetensors.
"""
import json

import torch
from safetensors.torch import save_file
from transformers import Glm4Config, Glm4ForCausalLM

N_LAYER = 2
N_HEAD = 2
N_KV_HEAD = 1
D_MODEL = 8
D_FF = 12
HEAD_DIM = 4            # decoupled head_dim (2*4 = 8 = hidden, but explicit)
PARTIAL_ROTARY = 0.5   # rotary_dim = int(4*0.5) = 2 (interleaved pair)
MAX_POS = 16
N_SEQUENCES = 3
VOCAB = 13

torch.manual_seed(20260613)
glm4_cfg = {
    'architectures': ['Glm4ForCausalLM'],
    'model_type': 'glm4',
    'hidden_size': D_MODEL,
    'intermediate_size': D_FF,
    'num_hidden_layers': N_LAYER,
    'num_attention_heads': N_HEAD,
    'num_key_value_heads': N_KV_HEAD,
    'head_dim': HEAD_DIM,
    'partial_rotary_factor': PARTIAL_ROTARY,
    'vocab_size': VOCAB,
    'max_position_embeddings': MAX_POS,
    'rms_norm_eps': 1e-5,
    'rope_theta': 10000.0,
    'attention_bias': True,
    'tie_word_embeddings': False,
    'hidden_act': 'silu',
    'pad_token_id': 0,
    'eos_token_id': 1,
    'bos_token_id': 2,
}
model = Glm4ForCausalLM(Glm4Config(**glm4_cfg, attn_implementation='eager'))

# HF ones-inits every RMSNorm gain and zero-inits the attention biases, which
# would make the sandwich-norm and qkv-bias load paths vacuous. Re-randomize
# all four block norms, the final norm, and the q/k/v biases.
with torch.no_grad():
    for layer in model.model.layers:
        layer.input_layernorm.weight.normal_(1.0, 0.5)
        layer.post_attention_layernorm.weight.normal_(1.0, 0.5)
        layer.post_self_attn_layernorm.weight.normal_(1.0, 0.5)
        layer.post_mlp_layernorm.weight.normal_(1.0, 0.5)
        layer.self_attn.q_proj.bias.normal_(0.0, 0.5)
        layer.self_attn.k_proj.bias.normal_(0.0, 0.5)
        layer.self_attn.v_proj.bias.normal_(0.0, 0.5)
    model.model.norm.weight.normal_(1.0, 0.5)

model = model.double().eval()
sd = {k: v.to(torch.float32).contiguous()
      for k, v in model.state_dict().items()}
save_file(sd, 'tests/fixtures/tiny_glm4.safetensors')
with open('tests/fixtures/tiny_glm4_config.json', 'w') as f:
    json.dump(glm4_cfg, f, indent=1)
sequences = [[(7 * i + 3 * s + s * s) % VOCAB for i in range(MAX_POS)]
             for s in range(N_SEQUENCES)]
with torch.no_grad():
    logits = [model(input_ids=torch.tensor([seq])).logits[0].tolist()
              for seq in sequences]
with open('tests/fixtures/tiny_glm4_logits.json', 'w') as f:
    json.dump({'sequences': sequences, 'logits': logits}, f)
print(f'wrote tiny_glm4.safetensors ({len(sd)} tensors) '
      f'+ config + logits ({N_SEQUENCES} sequences of {MAX_POS})')


def logits_with(sd_patch):
    """Logits of sequence 0 under a patched copy of the state dict."""
    patched = Glm4ForCausalLM(Glm4Config(**glm4_cfg,
                                         attn_implementation='eager'))
    sd_full = {k: v.clone() for k, v in model.state_dict().items()}
    sd_full.update(sd_patch)
    patched.load_state_dict(sd_full)
    patched = patched.double().eval()
    with torch.no_grad():
        return patched(input_ids=torch.tensor([sequences[0]])).logits


with torch.no_grad():
    base = model(input_ids=torch.tensor([sequences[0]])).logits

# Resetting the two IN-BRANCH post-norm gains to ones MUST change the logits
# (these distinguish the GLM-4 four-norm sandwich from a plain pre-norm).
post_ones = {k: torch.ones_like(v) for k, v in model.state_dict().items()
             if 'post_self_attn_layernorm' in k or 'post_mlp_layernorm' in k}
post_effect = (base - logits_with(post_ones)).abs().max().item()
assert post_effect > 1e-3, \
    f'in-branch sandwich post-norm gains had no effect ({post_effect})'
print(f'sandwich post-norm gain effect: max |diff| = {post_effect:.4f}')

# Zeroing the q/k/v biases MUST change the logits.
bias_zero = {k: torch.zeros_like(v) for k, v in model.state_dict().items()
             if k.endswith(('q_proj.bias', 'k_proj.bias', 'v_proj.bias'))}
bias_effect = (base - logits_with(bias_zero)).abs().max().item()
assert bias_effect > 1e-3, \
    f'q/k/v biases had no effect on the logits ({bias_effect})'
print(f'qkv-bias effect: max |diff| = {bias_effect:.4f}')
