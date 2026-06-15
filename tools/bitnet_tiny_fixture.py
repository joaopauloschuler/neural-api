#!/usr/bin/env python3
"""Generate a tiny RANDOM BitNet b1.58 parity fixture for
tests/TestNeuralPretrained.pas (no network access needed: the model is
randomly initialized from a pico config, never downloaded).

One fixture, ~30 KB, pinned in tests/fixtures/:

  tiny_bitnet.*: BitNetForCausalLM (model_type "bitnet",
      microsoft/bitnet-b1.58-2B-4T architecture) with every bitnet delta on
      the Llama path genuinely exercised:
      - SubLN ("norm-before-quantized-linear"): self_attn.attn_sub_norm
        (before o_proj) and mlp.ffn_sub_norm (before down_proj);
      - relu2 gated FFN (hidden_act "relu2"): down(ffn_sub_norm(
        relu2(gate(x)) * up(x))) with SEPARATE gate_proj/up_proj;
      - GQA: 4 query heads sharing 2 kv heads;
      - rope_parameters carrying rope_theta (transformers 5.x BitNetConfig).

  THE TERNARY ROUND-TRIP. The released BitNet ships ternary-quantized weights;
  the HF transformers checkpoint (BitNetForCausalLM) is the FP "shadow" form
  that stores the ALREADY-ternary effective weights (scale * {-1,0,+1}) and
  runs them through plain nn.Linear. This script reproduces that: every
  BitLinear-mapped projection (q/k/v/o, gate/up/down) is ternarized in place
  with the BitNet b1.58 per-output-channel absmean rule
      scale = mean(|W|);  W_eff = scale * round(clip(W / scale, -1, +1))
  and the EFFECTIVE weights are what we save. So the Pascal importer loading
  them straight is bit-for-bit identical to HF (the importer's load-time
  absmean ternarize is a no-op round-trip on already-ternary weights). The
  script ASSERTS the ternarization visibly moved the logits, so the fixture is
  non-vacuous (it really tests ternary weights, not the dense init).

Reference logits are computed by HF transformers (BitNetForCausalLM) in
float64 (the oracle convention of the committed fixtures).

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/bitnet_tiny_fixture.py
writes tests/fixtures/tiny_bitnet{.safetensors,_config.json,_logits.json}.
Needs torch + transformers (>= the BitNet release, 5.x) + safetensors.
"""
import json

import torch
from safetensors.torch import save_file
from transformers import BitNetConfig, BitNetForCausalLM

N_LAYER = 2
N_HEAD = 4
N_KV_HEAD = 2
D_MODEL = 16           # head_dim = 4
D_FF = 32
MAX_POS = 16
N_SEQUENCES = 3
VOCAB = 13
ROPE_THETA = 500000.0

# The projection tensors that BitNet runs through BitLinear (ternary weights).
BITLINEAR_SUFFIXES = (
    '.self_attn.q_proj.weight', '.self_attn.k_proj.weight',
    '.self_attn.v_proj.weight', '.self_attn.o_proj.weight',
    '.mlp.gate_proj.weight', '.mlp.up_proj.weight', '.mlp.down_proj.weight',
)

bitnet_cfg = {
    'architectures': ['BitNetForCausalLM'],
    'model_type': 'bitnet',
    'hidden_size': D_MODEL,
    'intermediate_size': D_FF,
    'num_hidden_layers': N_LAYER,
    'num_attention_heads': N_HEAD,
    'num_key_value_heads': N_KV_HEAD,
    'vocab_size': VOCAB,
    'max_position_embeddings': MAX_POS,
    'rms_norm_eps': 1e-5,
    'hidden_act': 'relu2',
    'attention_bias': False,
    'tie_word_embeddings': False,
    'rope_parameters': {'rope_theta': ROPE_THETA, 'rope_type': 'default'},
    'bos_token_id': 1,
    'eos_token_id': 2,
    'pad_token_id': 0,
}


def make_model():
    torch.manual_seed(20260613)  # identical weights every call
    return BitNetForCausalLM(BitNetConfig(**bitnet_cfg,
                                          attn_implementation='eager'))


def ternarize_absmean(w):
    """BitNet b1.58 per-output-channel (row) absmean ternarization."""
    scale = w.abs().mean(dim=1, keepdim=True)           # [out, 1]
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = torch.round((w / scale).clamp_(-1.0, 1.0))      # {-1, 0, +1}
    return scale * q


model = make_model()
# HF's std-0.02 init keeps a pico net almost linear; rescale every matrix to
# O(1) so the architecture (SubLN, relu2 FFN) and the ternarization move the
# logits well above the 1e-4 parity gate (the ModernBERT / Phi-3 lesson).
with torch.no_grad():
    for p in model.parameters():
        if p.dim() >= 2:
            p.mul_(8.0)

# Snapshot the DENSE (pre-ternary) state for the non-vacuous assertion.
sd_dense = {k: v.detach().clone() for k, v in model.state_dict().items()}

# Ternarize the BitLinear-mapped projections IN PLACE: the saved checkpoint is
# the ternary*scale effective form, exactly like the HF shadow checkpoint.
with torch.no_grad():
    sd = model.state_dict()
    for k, v in sd.items():
        if any(k.endswith(suf) for suf in BITLINEAR_SUFFIXES):
            sd[k] = ternarize_absmean(v)
    model.load_state_dict(sd)

sd_f32 = {k: v.to(torch.float32).contiguous()
          for k, v in model.state_dict().items()}
model = model.double().eval()

save_file(sd_f32, 'tests/fixtures/tiny_bitnet.safetensors')
with open('tests/fixtures/tiny_bitnet_config.json', 'w') as f:
    json.dump(bitnet_cfg, f, indent=1)
sequences = [[(7 * i + 3 * s + s * s) % VOCAB for i in range(MAX_POS)]
             for s in range(N_SEQUENCES)]
with torch.no_grad():
    logits = [model(input_ids=torch.tensor([seq])).logits[0].tolist()
              for seq in sequences]
with open('tests/fixtures/tiny_bitnet_logits.json', 'w') as f:
    json.dump({'sequences': sequences, 'logits': logits}, f)
print(f'wrote tiny_bitnet.safetensors ({len(sd_f32)} tensors) '
      f'+ config + logits ({N_SEQUENCES} sequences of {MAX_POS})')

# Non-vacuous check: the ternarization MUST move the logits, otherwise the
# fixture would be testing the dense init, not the ternary weights.
dense_model = make_model()
with torch.no_grad():
    for p in dense_model.parameters():
        if p.dim() >= 2:
            p.mul_(8.0)
dense_model.load_state_dict(sd_dense)
dense_model = dense_model.double().eval()
with torch.no_grad():
    dense_logits = torch.tensor(
        dense_model(input_ids=torch.tensor([sequences[0]])).logits.tolist())
tern_effect = (torch.tensor([logits[0]]) - dense_logits).abs().max().item()
assert tern_effect > 1e-3, \
    f'ternarization had no effect on the logits ({tern_effect})'
print(f'ternarization effect on logits: max |diff| = {tern_effect:.4f}')
