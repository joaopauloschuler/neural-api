#!/usr/bin/env python3
"""Generate a tiny RANDOM ModernBERT parity fixture for
tests/TestNeuralPretrained.pas (no network access needed: the model is
randomly initialized from a pico config, never downloaded - the same
recipe as tools/gemma3_tiny_fixture.py).

One fixture, ~10 KB, pinned in tests/fixtures/:

  tiny_modernbert.*: ModernBertModel (model_type "modernbert") with every
      ModernBERT delta on top of the vanilla-BERT skeleton genuinely
      exercised (each one is ASSERTED non-vacuous below by re-running the
      oracle with that single delta disabled):
      - RoPE on a BIDIRECTIONAL encoder (no position table at all): the
        only position signal; rotate_half q/k convention.
      - ALTERNATING LOCAL/GLOBAL attention: global_attn_every_n_layers=3
        over 4 layers, so layers 0 and 3 are "full_attention" and layers
        1, 2 are "sliding_attention" (HF: sliding iff bool(i % n) - layer
        0 is ALWAYS global, the OPPOSITE phase of Gemma-3's (i+1) % n).
        local_attention=4 (total window; HF half-window = 2) << seq len 16
        makes the SYMMETRIC |i-j| <= 2 mask bite on BOTH sides.
      - PER-LAYER-TYPE ROPE THETA: global_rope_theta=1000.0 on the global
        layers vs local_rope_theta=10.0 on the sliding layers (distinct
        values at pico scale so a single-theta import breaks parity).
      - GeGLU MLP with the EXACT erf "gelu" (the ModernBERT default
        hidden_activation): Wi packs input|gate, out = Wo(gelu(input) *
        gate) - the packing is asserted by swapping the Wi halves.
      - PRE-LN blocks with BIAS-FREE norms (norm_bias=false) and NO linear
        biases (attention_bias=mlp_bias=false); layers[0].attn_norm is
        nn.Identity (structural - layer 1+ norms carry re-randomized
        non-trivial gains so a skipped/unloaded gamma breaks parity);
        embedding LayerNorm + final_norm.

Reference hidden states (ModernBertModel last_hidden_state) are computed
by HF transformers in float64 and stored under the "logits" key so the
test reuses AssertLogitParityWithFixture with Vocab=hidden_size.

Coded by Claude (AI).

Usage (from the repo root):
  python3 tools/modernbert_tiny_fixture.py
writes tests/fixtures/tiny_modernbert{.safetensors,_config.json,_logits.json}.
Needs torch + transformers (>= the ModernBERT release) + safetensors.
"""
import json

import torch
from safetensors.torch import save_file
from transformers import ModernBertConfig, ModernBertModel

N_LAYER = 4            # layers 0, 3 global; layers 1, 2 sliding (n=3)
N_HEAD = 2
D_MODEL = 8            # head_dim = 4 (even - RoPE pairs)
D_FF = 6               # Wi is 2*D_FF = 12 wide
MAX_POS = 16
LOCAL_ATTENTION = 4    # TOTAL window; HF half-window = 2 << 16: mask bites
GLOBAL_EVERY_N = 3     # non-trivial pattern over 4 layers: global at 0, 3
GLOBAL_ROPE_THETA = 1000.0
LOCAL_ROPE_THETA = 10.0   # distinct: per-layer-type theta is exercised
N_SEQUENCES = 3
VOCAB = 13

torch.manual_seed(20260612)
modernbert_cfg = {
    'architectures': ['ModernBertModel'],
    'model_type': 'modernbert',
    'hidden_size': D_MODEL,
    'intermediate_size': D_FF,
    'num_hidden_layers': N_LAYER,
    'num_attention_heads': N_HEAD,
    'vocab_size': VOCAB,
    'max_position_embeddings': MAX_POS,
    'norm_eps': 1e-5,
    'norm_bias': False,
    'attention_bias': False,
    'mlp_bias': False,
    'local_attention': LOCAL_ATTENTION,
    'global_attn_every_n_layers': GLOBAL_EVERY_N,
    'global_rope_theta': GLOBAL_ROPE_THETA,
    'local_rope_theta': LOCAL_ROPE_THETA,
    'hidden_activation': 'gelu',  # the EXACT erf form (ModernBERT default)
    # the released ids (50281+) exceed the pico vocab - keep them in range
    'pad_token_id': 0,
    'bos_token_id': 1,
    'eos_token_id': 2,
    'cls_token_id': 1,
    'sep_token_id': 2,
}


def make_model(**overrides):
    cfg = {**modernbert_cfg, **overrides}
    cfg.pop('architectures')
    return ModernBertModel(ModernBertConfig(
        **cfg, attn_implementation='eager'))


model = make_model()
assert model.config.layer_types == [
    'full_attention', 'sliding_attention', 'sliding_attention',
    'full_attention'], 'expected global at layers 0 and 3 (n=3 phase)'
assert model.config.sliding_window == LOCAL_ATTENTION // 2, \
    'sliding_window must be the half-window'
assert isinstance(model.layers[0].attn_norm, torch.nn.Identity), \
    'layer 0 attn_norm must be nn.Identity'
assert not isinstance(model.layers[1].attn_norm, torch.nn.Identity), \
    'layer 1+ attn_norm must be a real LayerNorm'

# HF's _init_weights draws every Linear at std initializer_range = 0.02 -
# at pico scale the attention scores are then ~1e-3, the softmax rows
# near-uniform and the branch outputs ~1e-3 of the residual stream, so NO
# delta (window/theta/packing) could clear the 1e-3 non-vacuity floor.
# Re-randomize all the carried weights at O(1) scale (pre-LN keeps the
# stream bounded) and give every LayerNorm a NON-TRIVIAL gamma so a
# skipped/unloaded gain breaks parity (the norms are BIAS-FREE -
# norm_bias=false - so there is no beta tensor at all).
with torch.no_grad():
    model.embeddings.tok_embeddings.weight.normal_(0.0, 1.0)
    model.embeddings.norm.weight.normal_(1.0, 0.5)
    for layer in model.layers:
        if not isinstance(layer.attn_norm, torch.nn.Identity):
            layer.attn_norm.weight.normal_(1.0, 0.5)
        layer.mlp_norm.weight.normal_(1.0, 0.5)
        layer.attn.Wqkv.weight.normal_(0.0, 0.4)
        layer.attn.Wo.weight.normal_(0.0, 0.4)
        layer.mlp.Wi.weight.normal_(0.0, 0.4)
        layer.mlp.Wo.weight.normal_(0.0, 0.4)
    model.final_norm.weight.normal_(1.0, 0.5)

model = model.double().eval()
sd = {k: v.to(torch.float32).contiguous()
      for k, v in model.state_dict().items()}
save_file(sd, 'tests/fixtures/tiny_modernbert.safetensors')
with open('tests/fixtures/tiny_modernbert_config.json', 'w') as f:
    json.dump(modernbert_cfg, f, indent=1)
sequences = [[(5 * i + 2 * s + s * s) % VOCAB for i in range(MAX_POS)]
             for s in range(N_SEQUENCES)]
with torch.no_grad():
    hidden = [model(input_ids=torch.tensor([seq])).last_hidden_state[0]
              .tolist() for seq in sequences]
with open('tests/fixtures/tiny_modernbert_logits.json', 'w') as f:
    json.dump({'sequences': sequences, 'logits': hidden}, f)
print(f'wrote tiny_modernbert.safetensors ({len(sd)} tensors) '
      f'+ config + hidden states ({N_SEQUENCES} sequences of {MAX_POS})')


def hidden_with(state_dict=None, **overrides):
    """Oracle hidden states for sequence 0 with one delta disabled."""
    other = make_model(**overrides)
    other.load_state_dict(state_dict or model.state_dict())
    other = other.double().eval()
    with torch.no_grad():
        return other(input_ids=torch.tensor([sequences[0]])
                     ).last_hidden_state


with torch.no_grad():
    ref = model(input_ids=torch.tensor([sequences[0]])).last_hidden_state

# GeGLU packing check: swapping the Wi input|gate halves must move the
# output (proves act(input)*gate, NOT act(gate)*input, is on the parity
# path - a wrong-half import would silently pass otherwise).
swapped_sd = {k: v.clone() for k, v in model.state_dict().items()}
for k in swapped_sd:
    if k.endswith('mlp.Wi.weight'):
        w = swapped_sd[k]
        swapped_sd[k] = torch.cat([w[D_FF:], w[:D_FF]], dim=0)

# Each ModernBERT delta must MATTER: re-run the oracle with that single
# delta disabled and assert the hidden states move, otherwise the fixture
# would vacuously "verify" the corresponding import path.
checks = {
    'local window (4 -> 32 = full)':
        hidden_with(local_attention=2 * MAX_POS),
    'layer pattern (3 -> 2: global set {0,3} -> {0,2})':
        hidden_with(global_attn_every_n_layers=2),
    'per-layer rope theta (local 10.0 -> global 1000.0)':
        hidden_with(local_rope_theta=GLOBAL_ROPE_THETA),
    'rope theta global (1000.0 -> 10.0)':
        hidden_with(global_rope_theta=LOCAL_ROPE_THETA),
    'exact erf gelu (gelu -> gelu_pytorch_tanh)':
        hidden_with(hidden_activation='gelu_pytorch_tanh'),
    'GeGLU packing (Wi halves swapped)':
        hidden_with(state_dict=swapped_sd),
}
for name, alt in checks.items():
    effect = (ref - alt).abs().max().item()
    assert effect > 1e-3, f'{name} had no effect on the output ({effect})'
    print(f'{name}: max |diff| = {effect:.4f}')
