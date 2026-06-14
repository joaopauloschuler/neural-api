#!/usr/bin/env python3
"""Generate a tiny RANDOM OLMo-2 parity fixture for
tests/TestNeuralPretrained.pas (no network access needed: the model is
randomly initialized from a pico config, never downloaded).

One fixture, ~10 KB, pinned in tests/fixtures/:

  tiny_olmo2.*: Olmo2ForCausalLM with the two OLMo-2 deltas on the Llama
      path genuinely exercised:
      - REORDERED post-norm: RMSNorm on the SUBLAYER OUTPUT before the
        residual add (x + Norm(Attn(x)); post_attention_layernorm /
        post_feedforward_layernorm; NO input_layernorm) - the gains are
        re-randomized (HF ones-inits every RMSNorm) and the script ASSERTS
        that resetting them back to ones changes the logits;
      - q/k RMSNorm over the FULL flattened projection width BEFORE the
        head split + RoPE (q_norm [num_heads*head_dim], k_norm
        [num_kv_heads*head_dim]) - re-randomized with PER-CHANNEL
        (non-head-periodic) gains so a per-head (Qwen3-style) mix-up
        cannot pass, and asserted to move the logits.
      GQA (1 kv head < 2 query heads) and an UNTIED lm_head are also on.

Reference logits are computed by HF transformers in float64 (the oracle
convention of the committed fixtures).

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/olmo2_tiny_fixture.py
writes tests/fixtures/tiny_olmo2{.safetensors,_config.json,_logits.json}.
Needs torch + transformers (>= the OLMo-2 release) + safetensors.
"""
import json

import torch
from safetensors.torch import save_file
from transformers import Olmo2Config, Olmo2ForCausalLM

N_LAYER = 2
N_HEAD = 2
N_KV_HEAD = 1
D_MODEL = 8
D_FF = 12
MAX_POS = 16
N_SEQUENCES = 3
VOCAB = 13

torch.manual_seed(20260612)
olmo2_cfg = {
    'architectures': ['Olmo2ForCausalLM'],
    'model_type': 'olmo2',
    'hidden_size': D_MODEL,
    'intermediate_size': D_FF,
    'num_hidden_layers': N_LAYER,
    'num_attention_heads': N_HEAD,
    'num_key_value_heads': N_KV_HEAD,
    'vocab_size': VOCAB,
    'max_position_embeddings': MAX_POS,
    'rms_norm_eps': 1e-5,
    'rope_theta': 10000.0,
    'attention_bias': False,
    'tie_word_embeddings': False,
    'hidden_act': 'silu',
}
model = Olmo2ForCausalLM(Olmo2Config(**olmo2_cfg,
                                     attn_implementation='eager'))

# HF ones-inits every RMSNorm gain, which would make BOTH delta-specific
# gain-loading paths vacuous. Re-randomize: the full-width q/k norms and
# the reordered post-norms (and the final norm for good measure).
with torch.no_grad():
    for layer in model.model.layers:
        layer.self_attn.q_norm.weight.normal_(1.0, 0.5)
        layer.self_attn.k_norm.weight.normal_(1.0, 0.5)
        layer.post_attention_layernorm.weight.normal_(1.0, 0.5)
        layer.post_feedforward_layernorm.weight.normal_(1.0, 0.5)
    model.model.norm.weight.normal_(1.0, 0.5)

model = model.double().eval()
sd = {k: v.to(torch.float32).contiguous()
      for k, v in model.state_dict().items()}
save_file(sd, 'tests/fixtures/tiny_olmo2.safetensors')
with open('tests/fixtures/tiny_olmo2_config.json', 'w') as f:
    json.dump(olmo2_cfg, f, indent=1)
sequences = [[(7 * i + 3 * s + s * s) % VOCAB for i in range(MAX_POS)]
             for s in range(N_SEQUENCES)]
with torch.no_grad():
    logits = [model(input_ids=torch.tensor([seq])).logits[0].tolist()
              for seq in sequences]
with open('tests/fixtures/tiny_olmo2_logits.json', 'w') as f:
    json.dump({'sequences': sequences, 'logits': logits}, f)
print(f'wrote tiny_olmo2.safetensors ({len(sd)} tensors) '
      f'+ config + logits ({N_SEQUENCES} sequences of {MAX_POS})')


def logits_with(sd_patch):
    """Logits of sequence 0 under a patched copy of the state dict."""
    patched = Olmo2ForCausalLM(Olmo2Config(**olmo2_cfg,
                                           attn_implementation='eager'))
    sd_full = {k: v.clone() for k, v in model.state_dict().items()}
    sd_full.update(sd_patch)
    patched.load_state_dict(sd_full)
    patched = patched.double().eval()
    with torch.no_grad():
        return patched(input_ids=torch.tensor([sequences[0]])).logits


with torch.no_grad():
    base = model(input_ids=torch.tensor([sequences[0]])).logits

# Resetting the full-width q/k norm gains to ones MUST change the logits.
qk_ones = {k: torch.ones_like(v) for k, v in model.state_dict().items()
           if k.endswith(('q_norm.weight', 'k_norm.weight'))}
qk_effect = (base - logits_with(qk_ones)).abs().max().item()
assert qk_effect > 1e-3, \
    f'full-width q/k norm gains had no effect on the logits ({qk_effect})'
print(f'full-width qk-norm gain effect: max |diff| = {qk_effect:.4f}')

# Resetting the reordered post-norm gains to ones MUST change the logits.
post_ones = {k: torch.ones_like(v) for k, v in model.state_dict().items()
             if 'post_attention_layernorm' in k
             or 'post_feedforward_layernorm' in k}
post_effect = (base - logits_with(post_ones)).abs().max().item()
assert post_effect > 1e-3, \
    f'post-norm gains had no effect on the logits ({post_effect})'
print(f'reordered post-norm gain effect: max |diff| = {post_effect:.4f}')
