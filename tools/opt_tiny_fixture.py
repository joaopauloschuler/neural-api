#!/usr/bin/env python3
"""Generate a tiny RANDOM OPT (facebook/opt-architecture) parity fixture for
tests/TestNeuralPretrained.pas (no network access: the model is randomly
initialized from a pico config, never downloaded).

Pinned in tests/fixtures/ (~30 KB total):

  tiny_opt.safetensors + tiny_opt_config.json + tiny_opt_logits.json:
      OPTForCausalLM with the quirks that distinguish OPT from GPT-2:
        - LEARNED absolute positions with the +2 OFFSET
          (OPTLearnedPositionalEmbedding: position p reads embed_positions
          row p + 2; the embed_positions table has max_position_embeddings
          + 2 rows);
        - PRE-LN blocks (do_layer_norm_before=True, the 125m..175B default):
          x := x + out_proj(MHA(self_attn_layer_norm(x)));
          x := x + fc2(ReLU(fc1(final_layer_norm(x))));
        - plain biased nn.LayerNorm norms (NOT RMSNorm), bias=True
          (enable_bias) on every linear (q/k/v/out_proj + fc1/fc2);
        - ReLU FFN (activation_function 'relu', NOT gelu);
        - a decoder-level final_layer_norm (do_layer_norm_before and not
          _remove_final_layer_norm);
        - NO rotary; standard 1/sqrt(head_dim) attention scaling;
        - word_embed_proj_dim == hidden_size in this pico fixture (so
          project_in / project_out are absent); the Pascal config reader
          handles the != case (opt-350m) generally;
        - ffn_dim=24, deliberately NOT 4*hidden, to catch a hardcoded 4x.

Reference logits are computed by HF transformers in float64 (the oracle
convention of the committed fixtures).

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/opt_tiny_fixture.py
Needs torch + transformers + safetensors.
"""
import json

import torch
from safetensors.torch import save_file
from transformers import OPTConfig, OPTForCausalLM

N_LAYER = 2
N_HEAD = 2
HEAD_DIM = 8
D_MODEL = N_HEAD * HEAD_DIM  # 16
FFN_DIM = 24  # deliberately NOT 4*hidden
MAX_POS = 16
N_SEQUENCES = 3
VOCAB = 11

torch.manual_seed(20260615)

cfg_dict = {
    'architectures': ['OPTForCausalLM'],
    'model_type': 'opt',
    'hidden_size': D_MODEL,
    'word_embed_proj_dim': D_MODEL,
    'ffn_dim': FFN_DIM,
    'num_hidden_layers': N_LAYER,
    'num_attention_heads': N_HEAD,
    'vocab_size': VOCAB,
    'max_position_embeddings': MAX_POS,
    'layer_norm_elementwise_affine': True,
    'do_layer_norm_before': True,
    'enable_bias': True,
    '_remove_final_layer_norm': False,
    'activation_function': 'relu',
    'dropout': 0.0,
    'attention_dropout': 0.0,
    'layerdrop': 0.0,
    'pad_token_id': 1,
    'bos_token_id': 2,
    'eos_token_id': 2,
    'tie_word_embeddings': True,
}

model = OPTForCausalLM(OPTConfig(**cfg_dict, attn_implementation='eager'))
# HF inits with std 0.02; at pico width attention scores are then ~0 and the
# softmax is near-uniform, making the per-head mixing numerically invisible.
# Boost q/k rows so the scores are O(1) and the attention path genuinely
# matters (mirrors the gptneox fixture convention).
with torch.no_grad():
    for layer in model.model.decoder.layers:
        layer.self_attn.q_proj.weight.normal_(0.0, 0.5)
        layer.self_attn.k_proj.weight.normal_(0.0, 0.5)
model = model.double().eval()

sd = {k: v.to(torch.float32).contiguous()
      for k, v in model.state_dict().items()}
save_file(sd, 'tests/fixtures/tiny_opt.safetensors')
with open('tests/fixtures/tiny_opt_config.json', 'w') as f:
    json.dump(cfg_dict, f, indent=1)

sequences = [[(7 * i + 3 * s + s * s) % VOCAB for i in range(MAX_POS)]
             for s in range(N_SEQUENCES)]


def logits_of(m, seqs):
    with torch.no_grad():
        return [m(input_ids=torch.tensor([seq])).logits[0] for seq in seqs]


ref = logits_of(model, sequences)
with open('tests/fixtures/tiny_opt_logits.json', 'w') as f:
    json.dump({'sequences': sequences,
               'logits': [l.tolist() for l in ref]}, f)
print(f'wrote tiny_opt.safetensors ({len(sd)} tensors) '
      f'+ config + logits ({N_SEQUENCES} sequences of {MAX_POS})')

# ---- the +2 position OFFSET must matter: zeroing rows 0/1 of the position
# table (the offset slots OPT never reads) must NOT change the logits, while
# shifting the table by one row (reading p+1 instead of p+2) MUST change them.
# We verify the second: a model that drops the +2 offset diverges.
with torch.no_grad():
    shifted = OPTForCausalLM(OPTConfig(**cfg_dict, attn_implementation='eager'))
    shifted.load_state_dict(model.state_dict())
    # roll the position table down by one row -> position p now reads row p+1.
    pos = shifted.model.decoder.embed_positions.weight
    pos.copy_(torch.roll(pos, shifts=1, dims=0))
shifted = shifted.double().eval()
offset_effect = max((a - b).abs().max().item()
                    for a, b in zip(ref, logits_of(shifted, sequences)))
assert offset_effect > 1e-3, \
    f'the +2 position offset had no effect on the logits ({offset_effect})'
print(f'+2 position-offset effect on logits: '
      f'max |diff| = {offset_effect:.4f}')

# ---- ReLU (not gelu) must matter: swapping the activation must move logits.
gelu = OPTForCausalLM(OPTConfig(**{**cfg_dict, 'activation_function': 'gelu'},
                                attn_implementation='eager'))
gelu.load_state_dict(model.state_dict())
gelu = gelu.double().eval()
act_effect = max((a - b).abs().max().item()
                 for a, b in zip(ref, logits_of(gelu, sequences)))
assert act_effect > 1e-3, \
    f'the ReLU-vs-gelu activation had no effect on the logits ({act_effect})'
print(f'relu-vs-gelu effect on logits: max |diff| = {act_effect:.4f}')

print('tensor names:')
for k in sorted(sd):
    print(' ', k, list(sd[k].shape))
